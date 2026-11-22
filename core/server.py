"""HTTP surface: OpenAI-compatible chat + audio, plus ops endpoints.

Implemented on the standard library (``http.server``) on purpose:

* zero runtime dependencies — a reviewer can ``python -m orchestra serve``
  without installing anything;
* it shows the wire format explicitly (SSE framing, chunked WAV, headers),
  which is where most "OpenAI-compatible" servers actually get it wrong.

Endpoints
---------
GET  /healthz                     liveness + scheduler snapshot
GET  /v1/models                   model list (OpenAI shape)
GET  /metrics                     scheduler counters (Prometheus text)
POST /v1/chat/completions         JSON or SSE (``stream: true``)
POST /v1/audio/speech             chunked 16-bit PCM WAV
POST /v1/images/generations       OpenAI-shaped b64 PNG
"""

from __future__ import annotations

import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends import Backend, make_backend, wav_header
from .config import Config
from .scheduler import QueueFull, Scheduler, SchedulerError, UnknownModel


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class App:
    """Holds config, scheduler and backends; the handler only does protocol."""

    def __init__(self, config: Config, backend_name: str = "fake", **backend_kw):
        self.config = config
        self.scheduler = Scheduler(config)
        self.backend: Backend = make_backend(backend_name, **backend_kw)

    def model_list(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": m.id, "object": "model", "owned_by": "local", "kind": m.kind,
                 "vram_gb": m.vram_gb, "tags": m.tags}
                for m in self.config.models
            ],
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "single-gpu-ai-stack/0.1"

    # ------------------------------------------------------------- plumbing
    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # quieter default
        if self.app.config.debug:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _json(self, code: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json_raw(self, code: int, body: bytes, content_type: str,
                  headers: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": message, "type": err_type}})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    # ----------------------------------------------------------------- verbs
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            st = self.app.scheduler.stats
            self._json(200, {"ok": True, "vram_gb": self.app.config.vram_gb, **st})
        elif self.path == "/v1/models":
            self._json(200, self.app.model_list())
        elif self.path == "/metrics":
            st = self.app.scheduler.stats
            lines = [
                "# HELP orchestra_admitted_total jobs granted the GPU",
                "# TYPE orchestra_admitted_total counter",
                f"orchestra_admitted_total {st['admitted']}",
                "# TYPE orchestra_rejected_total counter",
                f"orchestra_rejected_total {st['rejected']}",
                "# TYPE orchestra_evictions_total counter",
                f"orchestra_evictions_total {st['evictions']}",
                "# TYPE orchestra_resident_gb gauge",
                f"orchestra_resident_gb {st['resident_gb']:.3f}",
                "# TYPE orchestra_waiting gauge",
                f"orchestra_waiting {st['waiting']}",
                "# TYPE orchestra_wait_seconds_total counter",
                f"orchestra_wait_seconds_total {st['wait_total_s']:.3f}",
            ]
            body = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._error(404, f"no route for GET {self.path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._error(400, str(exc))
        if self.path == "/v1/chat/completions":
            self._chat(body)
        elif self.path == "/v1/audio/speech":
            self._audio(body)
        elif self.path == "/v1/images/generations":
            self._image(body)
        else:
            self._error(404, f"no route for POST {self.path}", "not_found_error")

    # --------------------------------------------------------------- routes
    def _chat(self, body: dict[str, Any]) -> None:
        model = body.get("model")
        messages = body.get("messages") or []
        if not model:
            return self._error(400, "missing 'model'")
        if not messages:
            return self._error(400, "missing 'messages'")
        prompt = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        stream = bool(body.get("stream"))
        try:
            lease = self.app.scheduler.acquire(model, timeout_s=body.get("queue_timeout_s"))
        except UnknownModel:
            return self._error(404, f"unknown model {model!r}", "not_found_error")
        except QueueFull as exc:
            return self._json(429, {"error": {"message": str(exc), "type": "queue_full"}})
        except SchedulerError as exc:
            return self._error(400, str(exc))
        except TimeoutError as exc:
            return self._json(504, {"error": {"message": str(exc), "type": "timeout"}})

        wait_ms = lease.wait_s * 1000.0
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        try:
            if not stream:
                text = "".join(self.app.backend.chat_stream(lease.spec, prompt))
                return self._json(200, {
                    "id": cid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())},
                }, headers={"X-Queue-Wait-Ms": f"{wait_ms:.1f}"})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            # SSE 没有 Content-Length，又不走 chunked，所以必须显式关连接：
            # 否则 HTTP/1.1 的客户端会一直等"还有没有数据"，表现为 curl 挂住。
            self.send_header("Connection", "close")
            self.close_connection = True
            self.send_header("X-Queue-Wait-Ms", f"{wait_ms:.1f}")
            self.end_headers()
            t0 = _now_ms()
            for piece in self.app.backend.chat_stream(lease.spec, prompt):
                first = False
                payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model,
                           "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            lease.release()

    def _audio(self, body: dict[str, Any]) -> None:
        model = body.get("model") or body.get("voice")
        text = body.get("input") or body.get("text")
        if not model:
            return self._error(400, "missing 'model'")
        if not text:
            return self._error(400, "missing 'input'")
        sample_rate = int(body.get("sample_rate") or 32000)
        # 默认非流式：缓冲完再写**正确的** WAV 长度字段（播放器兼容性最好）。
        # stream=true 才走分块，此时长度字段只能是 0xFFFFFFFF 占位。
        stream = bool(body.get("stream"))
        try:
            lease = self.app.scheduler.acquire(model, timeout_s=body.get("queue_timeout_s"))
        except UnknownModel:
            return self._error(404, f"unknown model {model!r}", "not_found_error")
        except QueueFull as exc:
            return self._json(429, {"error": {"message": str(exc), "type": "queue_full"}})
        except SchedulerError as exc:
            return self._error(400, str(exc))
        except TimeoutError as exc:
            return self._json(504, {"error": {"message": str(exc), "type": "timeout"}})

        wait_ms = lease.wait_s * 1000.0
        if not stream:
            try:
                pcm = b"".join(self.app.backend.audio_stream(lease.spec, text, sample_rate=sample_rate))
            finally:
                lease.release()
            body_wav = wav_header(sample_rate, data_bytes=len(pcm)) + pcm
            return self._json_raw(200, body_wav, "audio/wav", {"X-Sample-Rate": str(sample_rate),
                                                              "X-Queue-Wait-Ms": f"{wait_ms:.1f}"})
        try:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Sample-Rate", str(sample_rate))
            self.send_header("X-Channels", "1")
            self.send_header("X-Queue-Wait-Ms", f"{wait_ms:.1f}")
            self.end_headers()
            self._chunk(wav_header(sample_rate))
            for pcm in self.app.backend.audio_stream(lease.spec, text, sample_rate=sample_rate):
                self._chunk(pcm)
            self._chunk(b"")
        finally:
            lease.release()

    def _image(self, body: dict[str, Any]) -> None:
        """OpenAI 形状：{"created": ts, "data": [{"b64_json": ...}]}.

        ``response_format`` 支持 ``b64_json``（默认，便于测试）与 ``url``
        （返回 ``data:image/png;base64,...``，浏览器直接能显示）。
        """
        model = body.get("model")
        prompt = body.get("prompt")
        if not model:
            return self._error(400, "missing 'model'")
        if not prompt:
            return self._error(400, "missing 'prompt'")
        size = str(body.get("size") or "64x64")
        fmt = str(body.get("response_format") or "b64_json")
        try:
            lease = self.app.scheduler.acquire(model, timeout_s=body.get("queue_timeout_s"))
        except UnknownModel:
            return self._error(404, f"unknown model {model!r}", "not_found_error")
        except QueueFull as exc:
            return self._json(429, {"error": {"message": str(exc), "type": "queue_full"}})
        except SchedulerError as exc:
            return self._error(400, str(exc))
        except TimeoutError as exc:
            return self._json(504, {"error": {"message": str(exc), "type": "timeout"}})

        wait_ms = lease.wait_s * 1000.0
        try:
            t0 = _now_ms()
            png = self.app.backend.image(lease.spec, prompt, size=size)
            gen_ms = _now_ms() - t0
        finally:
            lease.release()
        import base64

        b64 = base64.b64encode(png).decode()
        item = {"b64_json": b64} if fmt == "b64_json" else {"url": "data:image/png;base64," + b64}
        self._json(200, {"created": int(time.time()), "data": [item], "size": size,
                         "png_bytes": len(png)},
                   headers={"X-Queue-Wait-Ms": f"{wait_ms:.1f}", "X-Gen-Ms": f"{gen_ms:.1f}"})

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode())
        if data:
            self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()


def serve(config: Config, host: str = "127.0.0.1", port: int = 9881,
          backend: str = "fake", **backend_kw) -> ThreadingHTTPServer:
    app = App(config, backend_name=backend, **backend_kw)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.app = app  # type: ignore[attr-defined]
    httpd.daemon_threads = True
    return httpd
