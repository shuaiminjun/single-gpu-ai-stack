"""VRAM-budget scheduler for a single GPU.

Why this exists
---------------
On a 20 GB card you cannot keep an LLM, a diffusion model and a TTS model
resident at the same time. Every naive "just load it" implementation ends up
with the driver aborting (on ROCm it surfaces as ``LLVM ERROR: Can't get
available size``), or with a 4x slowdown from constant swap-in/swap-out.

This scheduler keeps a *budget* of resident models, serialises work that needs
the GPU, evicts idle models in LRU order when a new one does not fit, and
exposes the wait time so the API layer can report it.

Design notes
------------
* **Single-flight**: one GPU job at a time. Concurrency on one card buys
  nothing and makes latency unpredictable.
* **Eviction only of idle models**: never unload a model that is serving.
* **Admission control**: a request that cannot *ever* fit (bigger than the
  whole card) is rejected immediately instead of waiting forever.
* **Queue with a cap**: overload returns a fast error instead of unbounded
  memory growth.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .config import Config, ModelSpec


class SchedulerError(RuntimeError):
    """Base class for admission failures."""


class UnknownModel(SchedulerError):
    pass


class QueueFull(SchedulerError):
    pass


class DoesNotFit(SchedulerError):
    pass


@dataclass
class Lease:
    """A granted right to run one job on ``spec``.

    Use as a context manager; releasing returns the GPU slot and updates the
    LRU order so the model is not evicted while it is hot.
    """

    spec: ModelSpec
    scheduler: "Scheduler"
    wait_s: float = 0.0
    evicted: list[str] = field(default_factory=list)
    _released: bool = False

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.scheduler._release(self)


class Scheduler:
    """Admit, evict, serialise. One instance per GPU."""

    def __init__(self, config: Config, clock=time.monotonic):
        self.config = config
        self._clock = clock
        self._cond = threading.Condition(threading.Lock())
        self._resident: "OrderedDict[str, float]" = OrderedDict()  # id -> vram_gb
        self._busy = False
        self._waiting = 0
        self._stats = {
            "admitted": 0,
            "rejected": 0,
            "evictions": 0,
            "wait_total_s": 0.0,
        }

    # ---------------------------------------------------------------- state
    @property
    def resident(self) -> dict[str, float]:
        with self._cond:
            return dict(self._resident)

    @property
    def used_gb(self) -> float:
        with self._cond:
            return sum(self._resident.values())

    @property
    def stats(self) -> dict[str, float]:
        with self._cond:
            out = dict(self._stats)
            out["resident_gb"] = sum(self._resident.values())
            out["resident_models"] = len(self._resident)
            out["waiting"] = self._waiting
            out["busy"] = 1 if self._busy else 0
            return out

    def _fits(self, spec: ModelSpec, keep: set[str] | None = None) -> bool:
        keep = keep or set()
        total = sum(v for k, v in self._resident.items() if k not in keep)
        return total + spec.vram_gb <= self.config.vram_gb + 1e-9

    # ------------------------------------------------------------- acquire
    def acquire(self, model_id: str, timeout_s: float | None = None) -> Lease:
        """Block until ``model_id`` can run, then take the GPU slot.

        Raises:
            UnknownModel: model is not in the registry.
            DoesNotFit: the model alone exceeds the card.
            QueueFull: too many requests already waiting.
            TimeoutError: ``timeout_s`` elapsed while queued.
        """
        try:
            spec = self.config.by_id(model_id)
        except KeyError as exc:
            raise UnknownModel(model_id) from exc
        if spec.vram_gb > self.config.vram_gb + 1e-9:
            with self._cond:
                self._stats["rejected"] += 1
            raise DoesNotFit(
                f"{model_id} needs {spec.vram_gb:.1f} GB but the card has {self.config.vram_gb:.1f} GB"
            )

        started = self._clock()
        deadline = None if timeout_s is None else started + timeout_s
        with self._cond:
            if self._waiting >= self.config.max_queue:
                self._stats["rejected"] += 1
                raise QueueFull(f"queue is full ({self.config.max_queue})")
            self._waiting += 1
            try:
                while True:
                    if not self._busy and self._fits(spec):
                        break
                    if not self._busy and self.config.evict_policy == "lru":
                        victims = self._plan_eviction(spec)
                        if victims is not None:
                            for vid in victims:
                                self._resident.pop(vid, None)
                                self._stats["evictions"] += 1
                            break
                    remaining = None if deadline is None else deadline - self._clock()
                    if remaining is not None and remaining <= 0:
                        self._stats["rejected"] += 1
                        raise TimeoutError(f"timed out waiting for {model_id}")
                    self._cond.wait(0.05 if remaining is None else min(remaining, 0.05))
                self._waiting -= 1
                wait = self._clock() - started
                self._stats["admitted"] += 1
                self._stats["wait_total_s"] += wait
                self._busy = True
                self._resident[spec.id] = spec.vram_gb
                self._resident.move_to_end(spec.id)
                lease = Lease(spec=spec, scheduler=self, wait_s=wait)
            except BaseException:
                self._waiting = max(0, self._waiting - 1)
                raise
        return lease

    def _plan_eviction(self, spec: ModelSpec) -> list[str] | None:
        """Return ids to evict (LRU first) so ``spec`` fits, or None if impossible.

        Nothing is mutated here; the caller applies the plan while holding the lock.
        """
        if self._fits(spec):
            return []
        freed = 0.0
        victims: list[str] = []
        total = sum(self._resident.values())
        for mid, gb in self._resident.items():  # OrderedDict: oldest first
            victims.append(mid)
            freed += gb
            if total - freed + spec.vram_gb <= self.config.vram_gb + 1e-9:
                return victims
        return None

    def _release(self, lease: Lease) -> None:
        with self._cond:
            self._busy = False
            if lease.spec.id in self._resident:
                self._resident.move_to_end(lease.spec.id)
            self._cond.notify_all()

    # ------------------------------------------------------------ utilities
    def preload(self, model_id: str) -> None:
        """Mark a model resident without running a job (warm start)."""
        spec = self.config.by_id(model_id)
        with self._cond:
            self._resident[spec.id] = spec.vram_gb
            self._resident.move_to_end(spec.id)
            self._cond.notify_all()

    def unload(self, model_id: str) -> bool:
        with self._cond:
            existed = self._resident.pop(model_id, None) is not None
            self._cond.notify_all()
            return existed
