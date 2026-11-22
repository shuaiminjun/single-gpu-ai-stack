"""Wire protocol tests against a real HTTP server (no GPU, fake backend)."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import wave

import pytest

from core.backends import wav_header
from core.config import Config, ModelSpec
from core.server import serve


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def server():
    cfg = Config(
        vram_gb=20.0,
        models=[
            ModelSpec(id="chat-4b", kind="chat", vram_gb=6.0),
            ModelSpec(id="tts-a", kind="audio", vram_gb=3.0),
          ModelSpec(id="img", kind="image", vram_gb=11.0),
        ],
    )
    httpd = serve(cfg, host="127.0.0.1", port=free_port(), backend="fake",
                  word_delay_s=0.002, chunk_delay_s=0.002)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()


def http_post(url: str, payload: dict, timeout: float = 10.0):
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def http_get(url: str, timeout: float = 10.0):
    import urllib.request

    return urllib.request.urlopen(url, timeout=timeout)


def test_models_and_health(server):
    models = json.loads(http_get(server + "/v1/models").read())
    ids = {m["id"] for m in models["data"]}
    assert ids == {"chat-4b", "tts-a", "img"}
    health = json.loads(http_get(server + "/healthz").read())
    assert health["ok"] is True
    assert health["vram_gb"] == 20.0


def test_chat_non_stream(server):
    resp = http_post(server + "/v1/chat/completions",
                     {"model": "chat-4b", "messages": [{"role": "user", "content": "hi"}]})
    body = json.loads(resp.read())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("[chat-4b]")
    assert "X-Queue-Wait-Ms" in resp.headers


def test_chat_stream_delivers_incrementally(server):
    t0 = time.perf_counter()
    resp = http_post(server + "/v1/chat/completions",
                     {"model": "chat-4b", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]})
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    first_at = None
    chunks = 0
    for raw in resp:
        if raw.strip() == b"data: [DONE]":
            break
        if raw.startswith(b"data: "):
            payload = json.loads(raw[6:])
            if first_at is None:
                first_at = time.perf_counter() - t0
            chunks += 1
    total = time.perf_counter() - t0
    assert chunks > 5
    assert first_at is not None and first_at < total  # incremental, not buffered


def test_audio_is_a_valid_streaming_wav(server):
    resp = http_post(server + "/v1/audio/speech",
                     {"model": "tts-a", "input": "hello world", "sample_rate": 32000})
    assert resp.headers["Content-Type"] == "audio/wav"
    assert resp.headers["X-Sample-Rate"] == "32000"
    pcm = resp.read()
    assert len(pcm) % 2 == 0  # 16-bit frames
    assert len(pcm) > 1000
    # the body must parse as WAV once the (fake) sizes are honoured
    with wave.open(io.BytesIO(wav_header(32000, data_bytes=len(pcm)) + pcm), "rb") as w:
        assert w.getframerate() == 32000
        assert w.getnchannels() == 1
        assert w.getnframes() == len(pcm) // 2


def test_unknown_model_returns_404(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        http_post(server + "/v1/chat/completions",
                  {"model": "ghost", "messages": [{"role": "user", "content": "hi"}]})
    assert ei.value.code == 404
    body = json.loads(ei.value.read())
    assert "unknown model" in body["error"]["message"]


def test_missing_fields_return_400(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        http_post(server + "/v1/audio/speech", {"model": "tts-a"})
    assert ei.value.code == 400


def test_metrics_expose_counters(server):
    http_post(server + "/v1/chat/completions",
              {"model": "chat-4b", "messages": [{"role": "user", "content": "hi"}]}).read()
    text = http_get(server + "/metrics").read().decode()
    assert "orchestra_admitted_total 1" in text
    assert "orchestra_resident_gb" in text


def test_sequential_models_swap_within_budget(server):
    """chat-4b (6 GB) then tts-a (3 GB) both fit; no eviction needed."""
    http_post(server + "/v1/chat/completions",
              {"model": "chat-4b", "messages": [{"role": "user", "content": "a"}]}).read()
    http_post(server + "/v1/audio/speech", {"model": "tts-a", "input": "b"}).read()
    health = json.loads(http_get(server + "/healthz").read())
    assert health["resident_models"] == 2
    assert health["resident_gb"] == pytest.approx(9.0)


def test_image_generation_returns_valid_png(server):
    import base64
    import struct

    resp = http_post(server + "/v1/images/generations",
                     {"model": "img", "prompt": "a red fox", "size": "32x24"})
    body = json.loads(resp.read())
    assert "X-Gen-Ms" in resp.headers
    png = base64.b64decode(body["data"][0]["b64_json"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR: width/height live at bytes 16..24
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (32, 24)
    assert body["png_bytes"] == len(png)


def test_image_url_format(server):
    resp = http_post(server + "/v1/images/generations",
                     {"model": "img", "prompt": "cat", "response_format": "url"})
    url = json.loads(resp.read())["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_image_unknown_model(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        http_post(server + "/v1/images/generations", {"model": "nope", "prompt": "x"})
    assert ei.value.code == 404


def test_image_needs_prompt(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        http_post(server + "/v1/images/generations", {"model": "img"})
    assert ei.value.code == 400
