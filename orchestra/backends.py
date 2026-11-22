"""Backends: what actually produces tokens or audio.

The scheduler does not care how a model is served, only how much VRAM it holds.
A backend is any object with ``chat_stream`` and/or ``audio_stream``.

``FakeBackend`` exists so the whole HTTP surface (SSE, chunked WAV, first-chunk
metrics, queueing) can be exercised in CI on a machine without a GPU. That is
also how the tests run.
"""

from __future__ import annotations

import io
import math
import struct
import time
import wave
from typing import Iterator, Protocol

from .config import ModelSpec


class Backend(Protocol):
    def chat_stream(self, spec: ModelSpec, prompt: str, **kw) -> Iterator[str]: ...
    def audio_stream(self, spec: ModelSpec, text: str, **kw) -> Iterator[bytes]: ...


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
    raise ValueError(f"unknown backend {name!r}")
