"""Deterministic regressions for the virtual driver's quiescence contract."""

import asyncio
import math
import threading
from types import SimpleNamespace

from .gateway_outbound_harness import GatewayClock


def clock_without_worker(monkeypatch):
    clock = object.__new__(GatewayClock)
    clock.loop = asyncio.get_running_loop()
    clock.now = clock.origin = clock.loop.time()
    clock.closing = False
    clock.parked = None
    clock.next_park = clock.loop.create_future()
    clock.parent = SimpleNamespace(pushes=[])
    monkeypatch.setattr(clock.loop, "time", lambda: clock.now)
    monkeypatch.setattr(clock.loop, "_clock_resolution", math.ulp(clock.now))
    return clock


async def test_settle_waits_for_late_wake_and_repark(monkeypatch):
    clock = clock_without_worker(monkeypatch)
    clock.register_wait(60, "select", threading.Event())
    callbacks = []

    def later(tick):
        callbacks.append(tick)
        if tick == 6:
            clock.wake()
        if tick == 20:
            clock.register_wait(60, "select", threading.Event())
        else:
            clock.loop.call_soon(later, tick + 1)

    clock.loop.call_soon(later, 1)
    try:
        await clock.settle()
        assert callbacks == list(range(1, 21))
        assert clock.parked is not None
        assert not clock.loop._ready
    finally:
        clock.wake()


async def test_select_registration_rechecks_readability(monkeypatch):
    clock = clock_without_worker(monkeypatch)
    gate = threading.Event()
    # The producer ran after the worker's empty check, before register_wait.
    clock.parent.pushes.append("synthetic push")
    clock.register_wait(60, "select", gate)
    assert gate.is_set()
    assert clock.parked is None
    assert not clock.next_park.done()


async def test_advance_processes_due_timers_and_callback_chains(monkeypatch):
    clock = clock_without_worker(monkeypatch)
    clock.register_wait(60, "select", threading.Event())
    observed = []

    def callback(tick):
        observed.append(clock.elapsed)
        if tick:
            clock.loop.call_soon(callback, tick - 1)

    clock.loop.call_at(clock.now, callback, 20)
    try:
        await clock.advance_to(1)
        assert observed == [0] * 21
        assert clock.elapsed == 1
        assert not clock.due_timers()
    finally:
        clock.wake()


async def test_clock_resolution_tracks_float_precision_boundaries(monkeypatch):
    clock = clock_without_worker(monkeypatch)
    clock.now = clock.origin = 2**23 - 0.25
    clock.loop._clock_resolution = math.ulp(clock.now)
    clock.register_wait(60, "select", threading.Event())
    observed = []
    clock.loop.call_at(clock.origin + 0.5, lambda: observed.append(clock.elapsed))
    try:
        await clock.advance_to(1)
        assert observed == [0.5]
        assert not clock.due_timers()
    finally:
        clock.wake()
