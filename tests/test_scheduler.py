"""Scheduler behaviour: admission, eviction, serialisation, queueing."""

from __future__ import annotations

import threading
import time

import pytest

from orchestra.config import Config, ModelSpec
from orchestra.scheduler import DoesNotFit, QueueFull, Scheduler, UnknownModel


def make_config(total_gb: float = 20.0, **kw) -> Config:
    models = [
        ModelSpec(id="small", kind="chat", vram_gb=4.0),
        ModelSpec(id="big", kind="chat", vram_gb=14.0),
        ModelSpec(id="huge", kind="chat", vram_gb=40.0),
        ModelSpec(id="audio", kind="audio", vram_gb=3.0),
    ]
    return Config(vram_gb=total_gb, models=models, **kw)


def test_unknown_model_is_rejected():
    sch = Scheduler(make_config())
    with pytest.raises(UnknownModel):
        sch.acquire("nope")


def test_model_larger_than_card_is_rejected_immediately():
    sch = Scheduler(make_config())
    t0 = time.monotonic()
    with pytest.raises(DoesNotFit):
        sch.acquire("huge")
    assert time.monotonic() - t0 < 1.0  # no waiting for the impossible


def test_two_models_fit_side_by_side_then_third_evicts_lru():
    cfg = make_config()
    sch = Scheduler(cfg)
    with sch.acquire("small"):
        pass
    with sch.acquire("audio"):
        pass
    # both resident: 4 + 3 = 7 GB
    assert sch.used_gb == pytest.approx(7.0)
    # 'big' needs 14 GB -> 7 + 14 > 20, eviction of LRU ('small') frees enough
    with sch.acquire("big"):
        assert "big" in sch.resident
        assert "small" not in sch.resident
    assert sch.stats["evictions"] >= 1


def test_no_eviction_policy_waits_instead_of_unloading():
    cfg = make_config(total_gb=16.0, evict_policy="none")
    sch = Scheduler(cfg)
    lease = sch.acquire("small")
    lease.release()
    # big does not fit while 'small' is resident and eviction is disabled:
    with pytest.raises(TimeoutError):
        sch.acquire("big", timeout_s=0.2)


def test_single_flight_serialises_jobs():
    sch = Scheduler(make_config())
    order: list[str] = []
    started = threading.Event()

    def worker(name: str):
        with sch.acquire("small"):
            order.append(f"start-{name}")
            started.set()
            time.sleep(0.15)
            order.append(f"end-{name}")

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); started.wait(1.0); t2.start()
    t1.join(3.0); t2.join(3.0)
    # strictly nested, never interleaved
    assert order in (
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    )


def test_queue_is_capped():
    cfg = make_config(max_queue=1, evict_policy="none")
    sch = Scheduler(cfg)
    held = sch.acquire("big")
    errors: list[Exception] = []

    def worker():
        try:
            sch.acquire("small", timeout_s=2.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.15)
    with pytest.raises(QueueFull):
        sch.acquire("audio", timeout_s=1.0)
    held.release()
    t.join(3.0)
    assert not errors


def test_wait_time_is_reported_and_counted():
    sch = Scheduler(make_config())
    held = sch.acquire("big")
    result: dict[str, float] = {}

    def worker():
        lease = sch.acquire("small", timeout_s=3.0)
        result["wait"] = lease.wait_s
        lease.release()

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.25)
    held.release()
    t.join(3.0)
    assert result["wait"] >= 0.2
    assert sch.stats["wait_total_s"] >= 0.2


def test_preload_and_unload():
    sch = Scheduler(make_config())
    sch.preload("audio")
    assert "audio" in sch.resident
    assert sch.unload("audio") is True
    assert "audio" not in sch.resident
    assert sch.unload("audio") is False
