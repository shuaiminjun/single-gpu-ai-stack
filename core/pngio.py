"""Minimal PNG encoder (RGB8) — no Pillow, no numpy.

Only used by the fake backend so the image endpoint can be exercised end to end
in CI. Real deployments return whatever bytes ComfyUI produced.
"""

from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_rgb(width: int, height: int, pixels: bytes) -> bytes:
    """``pixels`` is RGB, row-major, ``width*height*3`` bytes."""
    if len(pixels) != width * height * 3:
        raise ValueError("pixel buffer does not match width*height*3")
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += pixels[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _chunk(b"IEND", b""))


def gradient(width: int, height: int, seed: int = 0) -> bytes:
    """Deterministic gradient — different seeds give visibly different images."""
    px = bytearray(width * height * 3)
    r0, g0, b0 = (seed * 37) % 256, (seed * 91) % 256, (seed * 53) % 256
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            px[i] = (r0 + x * 255 // max(1, width - 1)) % 256
            px[i + 1] = (g0 + y * 255 // max(1, height - 1)) % 256
            px[i + 2] = (b0 + (x + y) % 64) % 256
    return bytes(px)


def demo_png(width: int = 64, height: int = 64, seed: int = 0) -> bytes:
    return encode_rgb(width, height, gradient(width, height, seed))
