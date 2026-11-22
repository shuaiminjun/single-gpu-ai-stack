"""Small helpers shared by the orchestrator: config loading and a tiny event bus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    """One servable model.

    ``vram_gb`` is what the model needs *resident*. ``kind`` decides which
    endpoint may use it (``chat`` / ``audio``). ``backend`` names a backend
    entry, ``target`` is backend-specific (argv, url, ...).
    """

    id: str
    kind: str
    vram_gb: float
    backend: str = "fake"
    target: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelSpec":
        missing = {"id", "kind", "vram_gb"} - raw.keys()
        if missing:
            raise ValueError(f"model spec missing fields: {sorted(missing)}")
        kind = raw["kind"]
        if kind not in ("chat", "audio", "image"):
            raise ValueError(f"model {raw['id']}: kind must be chat|audio|image, got {kind!r}")
        return cls(
            id=str(raw["id"]),
            kind=kind,
            vram_gb=float(raw["vram_gb"]),
            backend=str(raw.get("backend", "fake")),
            target=str(raw.get("target", "")),
            tags=list(raw.get("tags") or []),
        )


@dataclass
class Config:
    vram_gb: float
    models: list[ModelSpec]
    max_queue: int = 32
    evict_policy: str = "lru"  # lru | none
    first_chunk_timeout_s: float = 30.0
    debug: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("pyyaml is required for .yaml configs") from exc
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        models = [ModelSpec.from_dict(m) for m in raw.get("models", [])]
        if not models:
            raise ValueError("config has no models")
        ids = [m.id for m in models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model ids in config")
        return cls(
            vram_gb=float(raw["vram_gb"]),
            models=models,
            max_queue=int(raw.get("max_queue", 32)),
            evict_policy=str(raw.get("evict_policy", "lru")),
            first_chunk_timeout_s=float(raw.get("first_chunk_timeout_s", 30.0)),
        )

    def by_id(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)
