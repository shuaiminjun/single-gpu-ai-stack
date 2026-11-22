"""Backends: what actually produces tokens or audio.

The scheduler does not care how a model is served, only how much VRAM it holds.
A backend is any object with ``chat_stream`` and/or ``audio_stream``.

``FakeBackend`` exists so the whole HTTP surface (SSE, chunked WAV, first-chunk
metrics, queueing) can be exercised in CI on a machine without a GPU. That is
also how the tests run.
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
import wave
from typing import Iterator, Protocol

from .config import ModelSpec
from .pngio import demo_png


class Backend(Protocol):
    def chat_stream(self, spec: ModelSpec, prompt: str, **kw) -> Iterator[str]: ...
    def audio_stream(self, spec: ModelSpec, text: str, **kw) -> Iterator[bytes]: ...
    def image(self, spec: ModelSpec, prompt: str, **kw) -> bytes: ...


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


class FakeBackend:
    """Deterministic, dependency-free stand-in for a real model.

    Chat: yields word by word with a small delay so streaming is observable.
    Audio: synthesises a quiet sine tone as 16-bit PCM frames, so the WAV
    container and chunking can be validated byte for byte.
    """

    def __init__(self, word_delay_s: float = 0.01, chunk_delay_s: float = 0.01):
        self.word_delay_s = word_delay_s
        self.chunk_delay_s = chunk_delay_s

    def chat_stream(self, spec: ModelSpec, prompt: str, **kw) -> Iterator[str]:
        words = f"[{spec.id}] " + " ".join(f"echo{i}" for i in range(24))
        for w in words.split():
            time.sleep(self.word_delay_s)
            yield w + " "

    def image(self, spec: ModelSpec, prompt: str, size: str = "64x64", **kw) -> bytes:
        """A deterministic PNG whose colours depend on the prompt hash."""
        try:
            w, h = (int(v) for v in size.lower().split("x"))
        except Exception:
            w, h = 64, 64
        w, h = max(8, min(1024, w)), max(8, min(1024, h))
        return demo_png(w, h, seed=abs(hash(prompt)) % 251)

    def audio_stream(self, spec: ModelSpec, text: str, sample_rate: int = 32000, **kw) -> Iterator[bytes]:
        # 按 ~8 字/秒 估时长，上限 3 s：让 demo 里的音频长度看起来像真的
        seconds = min(3.0, max(0.3, 0.12 * max(1, len(text))))
        total_frames = int(sample_rate * seconds)
        block = 2048
        freq = 220.0
        produced = 0
        while produced < total_frames:
            n = min(block, total_frames - produced)
            buf = bytearray()
            for i in range(n):
                t = (produced + i) / sample_rate
                v = int(3000 * math.sin(2 * math.pi * freq * t))
                buf += struct.pack("<h", v)
            produced += n
            time.sleep(self.chunk_delay_s)
            yield bytes(buf)


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16, data_bytes: int = 0) -> bytes:
    """Build a 44-byte PCM WAV header.

    ``data_bytes == 0`` writes 0xFFFFFFFF placeholders: with chunked transfer the
    final size is unknown up front, and some players honour the length fields.
    """
    fake = data_bytes == 0
    riff_size = 0xFFFFFFFF if fake else 36 + data_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(b"")
    head = bytearray(buf.getvalue())
    # patch sizes: 'RIFF' <size> 'WAVEfmt ' ... 'data' <size>
    head[4:8] = struct.pack("<I", riff_size)
    idx = head.find(b"data")
    head[idx + 4: idx + 8] = struct.pack("<I", 0xFFFFFFFF if fake else data_bytes)
    return bytes(head[:44])


class ComfyUIBackend:
    """Talk to a running ComfyUI: submit a workflow, poll history, fetch the image.

    ``spec.target`` is the workflow JSON path. ``{prompt}`` inside the workflow's
    positive text node is substituted. Kept dependency-free (urllib).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout_s: float = 300.0,
                 poll_s: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    # -- transport -----------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get(self, path: str) -> dict:
        import urllib.request

        with urllib.request.urlopen(self.base_url + path, timeout=30) as r:
            return json.loads(r.read().decode())

    def _fetch(self, path: str) -> bytes:
        import urllib.request

        with urllib.request.urlopen(self.base_url + path, timeout=60) as r:
            return r.read()

    # -- workflow ------------------------------------------------------
    def _load_workflow(self, spec: ModelSpec, prompt: str) -> dict:
        from pathlib import Path

        wf = json.loads(Path(spec.target).read_text(encoding="utf-8"))
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for key, val in list(inputs.items()):
                if isinstance(val, str) and "{prompt}" in val:
                    inputs[key] = val.replace("{prompt}", prompt)
        return wf

    def image(self, spec: ModelSpec, prompt: str, **kw) -> bytes:
        wf = self._load_workflow(spec, prompt)
        submitted = self._post("/prompt", {"prompt": wf})
        pid = submitted.get("prompt_id")
        if not pid:
            raise RuntimeError(f"ComfyUI rejected the workflow: {submitted}")
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            hist = self._get(f"/history/{pid}")
            entry = hist.get(pid) or {}
            outputs = entry.get("outputs") or {}
            for node_out in outputs.values():
                for img in (node_out.get("images") or []):
                    q = f"/view?filename={img['filename']}&subfolder={img.get('subfolder','')}&type={img.get('type','output')}"
                    return self._fetch(q)
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI execution error")
            time.sleep(self.poll_s)
        raise TimeoutError(f"ComfyUI did not finish within {self.timeout_s}s")

    def chat_stream(self, spec: ModelSpec, prompt: str, **kw) -> Iterator[str]:
        raise NotImplementedError("use a chat backend for this model")

    def audio_stream(self, spec: ModelSpec, text: str, **kw) -> Iterator[bytes]:
        raise NotImplementedError("use an audio backend for this model")


class SubprocessBackend:
    """Run a CLI model (llama.cpp, a TTS script, ...) and stream its stdout.

    Kept deliberately small: the interesting part of this project is the
    scheduler and the wire protocol, not process plumbing. ``target`` is the
    argv template; ``{prompt}`` and ``{out}`` are substituted.
    """

    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size

    def _run(self, argv: list[str]) -> Iterator[bytes]:
        import subprocess

        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def chat_stream(self, spec: ModelSpec, prompt: str, **kw) -> Iterator[str]:
        argv = [p.replace("{prompt}", prompt) for p in spec.target.split()]
        for raw in self._run(argv):
            yield raw.decode("utf-8", "ignore")

    def audio_stream(self, spec: ModelSpec, text: str, **kw) -> Iterator[bytes]:
        argv = [p.replace("{prompt}", text) for p in spec.target.split()]
        yield from self._run(argv)


def make_backend(name: str, **kw) -> Backend:
    if name in ("fake", ""):
        return FakeBackend(**kw)
    if name in ("subprocess", "cli"):
        return SubprocessBackend(**kw)
    if name in ("comfyui", "comfy"):
        return ComfyUIBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")
