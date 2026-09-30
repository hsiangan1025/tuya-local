"""Outbound lifecycle, aligned polls and virtual soak; no real network access."""

import asyncio
import json
import threading
from collections import Counter

import pytest

from .gateway_outbound_harness import GatewayClock
from .test_gateway import eventually
from .test_gateway import gateway_env as gateway_env


async def initialized_children(gateway_env):
    children = [gateway_env(f"child-{index}") for index in range(13)]
    await asyncio.gather(*(child.async_refresh() for child in children))
    assert len({id(child._broker) for child in children}) == 1
    return children


@pytest.mark.parametrize("poll_delay", [0, 0.05, 0.25])
@pytest.mark.parametrize("handoff", ["worker_first", "loop_first"])
async def test_aligned_safety_polls_and_outbound_commands(
    gateway_env, poll_delay, handoff
):
    """20 repeated worst-case bursts, with controls before/during/after each."""
    children = await initialized_children(gateway_env)
    bound = 13 * (0.2 + poll_delay) + 0.4
    async with GatewayClock(children, poll_delay, handoff=handoff) as clock:
        for child in children:
            child.actually_start()
        for round_number in range(1, 21):
            deadline = round_number * 30
            clock.align_polls(deadline)
            before = [sum(c[0] == "status" for c in ch._api.calls) for ch in children]
            for index, offset in enumerate((-0.005, 0.001, bound + 0.1)):
                clock.command_at(
                    deadline + offset,
                    children[(round_number * 3 + index) % 13],
                    round_number * 3 + index,
                    f"burst-{round_number}-{index}",
                )
            for index, child in enumerate(children):
                clock.push_at(deadline + index * 0.05, child, round_number)
            await clock.advance_to(deadline + bound + 0.5)
            assert [
                sum(c[0] == "status" for c in ch._api.calls) for ch in children
            ] == [count + 1 for count in before]
            assert all(ch.get_property("2") == round_number for ch in children)
            clock.assert_drained()
            clock.assert_commands_once(bound)
        assert clock.broker.running
        assert clock.summary()["heartbeats"] >= 110
        assert clock.queue.maximum_depth <= 1
        assert clock.queue.maximum_pending <= 1
        assert not any(clock.pending_samples)
        assert len(clock.push_latencies) == 13 * 20
        clock.assert_pushes_once()
        print("ALIGNED_POLL_METRICS=" + json.dumps(clock.summary(), sort_keys=True))


async def test_virtual_gateway_soak(gateway_env):
    """15 virtual minutes, 13 children, heartbeat, pushes, controls every 5-15 s."""
    children = await initialized_children(gateway_env)
    async with GatewayClock(children) as clock:
        for child in children:
            child.actually_start()
        at, sequence = 5, 0
        while at < 896:
            child = children[sequence % 13]
            clock.command_at(at, child, sequence, f"soak-{sequence}")
            at += (5, 10, 15)[sequence % 3]
            sequence += 1
        for second in range(1, 900):
            clock.push_at(second, children[second % 13], second)
        for minute in range(1, 16):
            await clock.advance_to(minute * 60 - 0.01)
            # Other children's polls may be active at this exact instant. Check
            # every future eventually completes, then drain at the end below.
            assert clock.broker.running
            assert len(clock.broker._pending) <= 1
            clock.pending_samples.append(len(clock.broker._pending))
        await clock.advance_to(904)
        for child in children:
            child.pause()
        await clock.advance_to(907)
        clock.assert_drained()
        clock.assert_commands_once(3.1)
        assert clock.summary()["heartbeats"] >= 175
        assert all(
            sum(c[0] == "status" for c in ch._api.calls) >= 29 for ch in children
        )
        assert len({c[2] for ch in children for c in ch._api.calls}) == 1
        assert all(ch._api.version == 3.3 for ch in children)
        assert len(clock.push_latencies) == 899
        clock.assert_pushes_once()
        print("SOAK_METRICS=" + json.dumps(clock.summary(), sort_keys=True))


@pytest.mark.parametrize("first_result", ["success", "retry", "exhausted"])
async def test_older_nowait_completion_cannot_ack_newer_same_value(
    gateway_env, mocker, first_result
):
    """A later HA request must not become an empty send after an old callback."""
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()
    controls = []

    def control(values, nowait):
        child._api.record("control", dict(values), nowait)
        controls.append(dict(values))
        if len(controls) == 1:
            entered.set()
            assert release.wait(3)
        failed_attempts = {"success": 0, "retry": 1, "exhausted": 3}[first_result]
        if len(controls) <= failed_attempts:
            return {"Err": "901", "Error": "synthetic send failure"}

    mocker.patch.object(child._api, "set_multiple_values", side_effect=control)
    first = asyncio.create_task(child.async_set_property("1", True))
    await eventually(entered.is_set)
    original_debounce = child._debounce_sending_updates
    submitted, finish_debounce = asyncio.Event(), asyncio.Event()

    async def delayed_debounce():
        submitted.set()
        await finish_debounce.wait()
        await original_debounce()

    mocker.patch.object(child, "_debounce_sending_updates", delayed_debounce)
    second = asyncio.create_task(child.async_set_property("1", True))
    try:
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        await first
        # The first operation succeeded, but it did not submit this newer one.
        newer_was_acknowledged = submitted.is_set() and child._pending_updates.get(
            "1", {}
        ).get("sent", False)
        finish_debounce.set()
        await second
        print(
            "NOWAIT_GENERATION_REPRO="
            + json.dumps(
                {
                    "newer_marked_sent_by_older_call": newer_was_acknowledged,
                    "controls": controls,
                }
            )
        )
        attempts = {"success": 1, "retry": 2, "exhausted": 3}[first_result]
        assert controls == [{"1": True}] * (attempts + 1)
        assert not newer_was_acknowledged
        assert not child._gateway.lock.locked()
        assert child._broker._calls.empty()
        await asyncio.sleep(0)
        assert not child._broker._pending
    finally:
        release.set()
        finish_debounce.set()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_commands_waiting_on_poll_lock_do_not_duplicate_other_dps(
    gateway_env, mocker
):
    """Long poll lock waits must not turn older pending data into another send."""
    child = gateway_env()
    await child.async_refresh()
    control = mocker.spy(child._api, "set_multiple_values")
    entered = mocker.spy(child, "_send_pending_updates")
    async with child._gateway.lock:
        first = asyncio.create_task(child.async_set_property("101", 1))
        await eventually(lambda: entered.call_count == 1)
        second = asyncio.create_task(child.async_set_property("102", 2))
        await asyncio.sleep(0.02)
    await asyncio.gather(first, second)
    transmissions = Counter(
        (key, value)
        for call in control.call_args_list
        for key, value in call.args[0].items()
    )
    print("PENDING_SNAPSHOT_REPRO=" + repr(control.call_args_list))
    assert transmissions == Counter({("101", 1): 1, ("102", 2): 1})


async def test_cancelled_inflight_completion_cannot_ack_replacement(
    gateway_env, mocker
):
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()
    controls = []

    def control(values, nowait):
        child._api.record("control", dict(values), nowait)
        controls.append(dict(values))
        if len(controls) == 1:
            entered.set()
            assert release.wait(3)

    mocker.patch.object(child._api, "set_multiple_values", side_effect=control)
    marked = mocker.spy(child, "_mark_updates_sent")
    first = asyncio.create_task(child.async_set_property("1", True))
    await eventually(entered.is_set)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    original_debounce = child._debounce_sending_updates
    submitted, resume = asyncio.Event(), asyncio.Event()

    async def delayed_debounce():
        submitted.set()
        await resume.wait()
        await original_debounce()

    mocker.patch.object(child, "_debounce_sending_updates", delayed_debounce)
    replacement = asyncio.create_task(child.async_set_property("1", True))
    try:
        await submitted.wait()
        release.set()
        await eventually(lambda: marked.call_count == 1)
        assert child._pending_updates["1"]["sent"] is False
        resume.set()
        await replacement
        await child._broker.async_call(lambda: None)
        assert controls == [{"1": True}, {"1": True}]
        assert not child._gateway.lock.locked()
        assert not child._command_lock.locked()
        assert child._broker._calls.empty()
        await asyncio.sleep(0)
        assert not child._broker._pending
    finally:
        release.set()
        resume.set()
        await asyncio.gather(first, replacement, return_exceptions=True)


@pytest.mark.parametrize("phase", ["debounce", "gateway_lock", "broker_queue"])
async def test_cancelled_command_does_not_block_following_calls(
    gateway_env, mocker, phase
):
    child = gateway_env()
    await child.async_refresh()
    controls = mocker.spy(child._api, "set_multiple_values")
    entered, release = threading.Event(), threading.Event()
    active = None
    debounce_entered = asyncio.Event()
    original_debounce = child._debounce_sending_updates

    async def debounce():
        debounce_entered.set()
        if phase == "debounce":
            await asyncio.Event().wait()
        await original_debounce()

    def block_worker():
        entered.set()
        assert release.wait(3)

    mocker.patch.object(child, "_debounce_sending_updates", debounce)
    if phase == "gateway_lock":
        await child._gateway.lock.acquire()
    if phase == "broker_queue":
        active = asyncio.create_task(child._broker.async_call(block_worker))
        await eventually(entered.is_set)
    command = asyncio.create_task(child.async_set_property("101", 1))
    try:
        await debounce_entered.wait()
        if phase == "broker_queue":
            await eventually(lambda: child._broker._calls.qsize() == 1)
        elif phase == "gateway_lock":
            await eventually(lambda: bool(child._gateway.lock._waiters))
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
    finally:
        release.set()
        if phase == "gateway_lock":
            child._gateway.lock.release()
        if active:
            await active
    mocker.patch.object(child, "_debounce_sending_updates", original_debounce)
    await child.async_set_property("101", 2)
    await child._broker.async_call(lambda: None)
    controls.assert_called_once_with({"101": 2}, nowait=True)
    assert not child._gateway.lock.locked()
    assert not child._command_lock.locked()
    assert child._broker._calls.empty()
    await asyncio.sleep(0)
    assert not child._broker._pending


@pytest.mark.parametrize("state", ["healthy", "no_state", "paused", "poll_only"])
async def test_state_and_poll_mode_do_not_gate_later_commands(
    gateway_env, mocker, state
):
    child = gateway_env()
    await child.async_refresh()
    if state == "no_state":
        child._reset_cached_state()
        assert not child.has_returned_state
    elif state == "paused":
        child.pause()
    elif state == "poll_only":
        child._poll_only = True
    controls = mocker.spy(child._api, "set_multiple_values")
    async with asyncio.timeout(3):
        await child.async_set_property("1", True)
        await child.async_set_property("1", False)
    assert [c.args[0] for c in controls.call_args_list] == [{"1": True}, {"1": False}]
    assert not child._gateway.lock.locked()
    assert not child._command_lock.locked()


async def test_successful_nowait_without_push_ack_expires_and_can_repeat(
    gateway_env, mocker
):
    child = gateway_env()
    await child.async_refresh()
    clock = mocker.patch("custom_components.tuya_local.device.time", return_value=100)
    controls = mocker.spy(child._api, "set_multiple_values")
    await child.async_set_property("1", True)
    assert child._pending_updates["1"]["sent"] is True
    clock.return_value = 106
    assert not child._get_pending_updates()
    await child.async_set_property("1", True)
    assert [c.args[0] for c in controls.call_args_list] == [{"1": True}, {"1": True}]
    assert child._pending_updates["1"]["sent"] is True
