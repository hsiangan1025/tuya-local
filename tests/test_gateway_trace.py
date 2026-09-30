"""Structured canary diagnostics must remain private and observational."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from custom_components.tuya_local import trace
from custom_components.tuya_local.gateway_broker import GatewayBroker
from tools.analyze_gateway_trace import analyze

from .gateway_outbound_harness import GatewayClock
from .test_gateway import eventually
from .test_gateway import gateway_env as gateway_env
from .test_gateway_outbound import (
    test_cancelled_inflight_completion_cannot_ack_replacement as cancelled_completion,
)


@pytest.fixture(autouse=True)
def trace_logging(caplog):
    caplog.set_level(logging.DEBUG, logger=trace._LOGGER.name)


def records(caplog):
    return [
        json.loads(record.getMessage()[8:])
        for record in caplog.records
        if record.name == trace._LOGGER.name
        and record.getMessage().startswith("GBTRACE ")
    ]


def report(caplog):
    return analyze("GBTRACE " + json.dumps(e) for e in records(caplog))


def assert_schema(events):
    schema = json.loads(
        (Path(__file__).parents[1] / "tools/gateway_trace_schema.json").read_text()
    )
    for event in events:
        assert set(event) <= schema["properties"].keys()
        assert set(schema["required"]) <= event.keys()
        for rule in schema["allOf"]:
            if event["event"] in rule["if"]["properties"]["event"]["enum"]:
                assert set(rule["then"]["required"]) <= event.keys()


async def push(child, values):
    child._api.parent.pushes.append({"device": child._api, "dps": values})
    await eventually(lambda: all(key not in child._pending_updates for key in values))


async def test_complete_command_and_progressive_push_ack(gateway_env, caplog):
    child = gateway_env()
    await child.async_refresh()
    await child.async_set_properties({"1": True, "2": 42})
    await push(child, {"1": True})
    await push(child, {"2": 42})
    await child.async_stop()
    events = records(caplog)
    assert_schema(events)
    ack = [e for e in events if e["event"] == "push_ack"]
    assert [e["remaining_pending_count"] for e in ack] == [1, 0]
    assert all(e["ack_latency_ms"] >= 0 for e in ack)
    summary = report(caplog)
    assert summary["status"] == "PASS", summary
    assert summary["commands"]["logical_commands"] == 1
    assert summary["commands"]["acknowledged"] == 1
    assert summary["broker"]["final_queue_depth"] == 0
    assert summary["broker"]["final_pending_futures"] == 0
    assert events[-1]["event"] == "gateway_stop"


async def test_retry_keeps_command_id_and_is_not_duplicate(gateway_env, mocker, caplog):
    child = gateway_env()
    await child.async_refresh()
    control = mocker.patch.object(
        child._api,
        "set_multiple_values",
        side_effect=[
            {"Err": "901", "Error": "SYNTHETIC_ERROR_PRIVATE"},
            None,
        ],
    )
    await child.async_set_property("1", True)
    await push(child, {"1": True})
    await child.async_stop()
    events = records(caplog)
    sends = [e for e in events if e["event"] == "tuya_send_start"]
    retry = [e for e in events if e["event"] == "retry"]
    assert control.call_count == len(sends) == 2
    assert len(retry) == 1
    assert len({e["cmd_id"] for e in sends + retry}) == 1
    assert report(caplog)["status"] == "PASS"
    assert report(caplog)["commands"]["duplicate_sends"] == 0


async def test_command_ids_and_pending_generations_are_unique(gateway_env, caplog):
    first, second = gateway_env(), gateway_env("child-b")
    await asyncio.gather(first.async_refresh(), second.async_refresh())
    for child in (first, first, second):
        await child.async_set_property("1", True)
    events = records(caplog)
    requests = [e for e in events if e["event"] == "cmd_request"]
    snapshots = [e for e in events if e["event"] == "pending_snapshot"]
    ids = [e["cmd_id"] for e in requests]
    generations = [e["pending_generation"] for e in snapshots]
    assert len(set(ids)) == len(set(generations)) == 3
    assert ids == sorted(ids)
    assert all(
        type(e["gateway_slot"]) is int and type(e["child_slot"]) is int
        for e in requests
    )


async def test_cancelled_old_callback_is_observable(gateway_env, mocker, caplog):
    await cancelled_completion(gateway_env, mocker)
    events = records(caplog)
    stale = [e for e in events if e["event"] == "stale_mark_sent_ignored"]
    sent = [e for e in events if e["event"] == "mark_sent"]
    assert len(stale) == len(sent) == 1
    assert stale[0]["cmd_id"] != sent[0]["cmd_id"]
    assert stale[0]["pending_generation"] != sent[0]["pending_generation"]
    assert report(caplog)["commands"]["stale_callbacks_ignored"] == 1
    assert report(caplog)["status"] != "FAIL"


async def test_expiry_emits_timeout_without_changing_pending_semantics(
    gateway_env, mocker, caplog
):
    child = gateway_env()
    await child.async_refresh()
    wall = mocker.patch("custom_components.tuya_local.device.time", return_value=100)
    await child.async_set_property("1", True)
    pending = child._pending_updates["1"]
    assert set(pending) == {"value", "updated_at", "sent"}
    wall.return_value = 104.999
    assert child._get_pending_updates()["1"] is pending
    assert not any(e["event"] == "ack_timeout" for e in records(caplog))
    wall.return_value = 105
    assert not child._get_pending_updates()
    assert not child._trace_pending
    assert sum(e["event"] == "pending_expire" for e in records(caplog)) == 1
    assert sum(e["event"] == "ack_timeout" for e in records(caplog)) == 1
    await child.async_set_property("1", True)
    assert child._pending_updates["1"]["sent"]
    assert report(caplog)["status"] != "FAIL"


async def test_disabled_debug_does_not_build_or_encode_events(
    gateway_env, mocker, caplog
):
    caplog.set_level(logging.WARNING, logger=trace._LOGGER.name)
    encode = mocker.Mock(side_effect=AssertionError("encoded disabled trace"))
    mocker.patch.object(trace, "json", SimpleNamespace(dumps=encode))
    fields = mocker.patch.object(
        trace, "device_fields", side_effect=AssertionError("built disabled trace")
    )
    context = mocker.spy(trace, "Context")
    child = gateway_env()
    await child.async_refresh()
    await child.async_set_property("1", True)
    await child.async_stop()
    trace.emit("cmd_request", secret=object())
    encode.assert_not_called()
    fields.assert_not_called()
    context.assert_not_called()
    assert not records(caplog)


async def test_trace_never_contains_injected_private_data(gateway_env, mocker, caplog):
    secrets = {
        name: f"PRIVACY_SENTINEL_{name.upper()}_DO_NOT_LOG"
        for name in (
            "gateway_id",
            "device_id",
            "cid",
            "ip",
            "mac",
            "local_key",
            "device_name",
            "dp_value",
            "payload",
            "token",
        )
    }
    child = gateway_env(
        secrets["cid"],
        device_id=secrets["gateway_id"],
        host=secrets["ip"],
        local_key=secrets["local_key"],
        name=secrets["device_name"],
    )
    await child.async_refresh()
    child._api.id = secrets["device_id"]
    child._api.mac = secrets["mac"]
    value = {
        "value": secrets["dp_value"],
        "payload": secrets["payload"],
        "token": secrets["token"],
    }
    mocker.patch.object(
        child._api,
        "set_multiple_values",
        side_effect=[
            RuntimeError(" ".join(secrets.values())),
            None,
        ],
    )
    await child.async_set_property("99999", value)
    await push(child, {"99999": value})
    await child.async_stop()
    # Defense in depth: accidental non-schema fields/strings cannot cross emit.
    trace.emit("health", secrets, outcome=secrets["token"], payload=value)
    output = "\n".join(
        r.getMessage() for r in caplog.records if r.name == trace._LOGGER.name
    )
    assert "GBTRACE " in output
    assert all(secret not in output for secret in secrets.values())
    assert '"99999"' not in output  # DP identifiers are not part of the schema.
    assert all(e["v"] == 1 for e in records(caplog))


async def test_drain_batch_observes_multiple_serial_status_calls(gateway_env, caplog):
    children = [gateway_env(f"child-{i}") for i in range(3)]
    await asyncio.gather(*(c.async_refresh() for c in children))
    async with GatewayClock(children, 0.05, handoff="loop_first") as clock:
        for child in children:
            child.actually_start()
        clock.align_polls(0.1)
        await clock.advance_to(2)
        clock.assert_drained()
        assert len(clock.polls) == 3
    events = records(caplog)
    batches = [e for e in events if e["event"] == "broker_drain_done"]
    assert max(e["calls_executed"] for e in batches) >= 6
    assert sum(e["event"] == "poll_done" for e in events) == 3
    assert all(e["duration_ms"] >= 0 for e in batches)


async def test_health_is_low_rate_and_has_safe_counts(mocker, caplog):
    broker = GatewayBroker(
        SimpleNamespace(loop=asyncio.get_running_loop()), SimpleNamespace(socket=None)
    )
    start = trace.monotonic()
    now = mocker.patch.object(trace, "monotonic", return_value=start)
    for _ in range(100):
        broker._trace_activity()
    await asyncio.sleep(0)
    assert len(records(caplog)) == 1
    now.return_value = start + 59.999
    broker._trace_activity()
    await asyncio.sleep(0)
    assert len(records(caplog)) == 1
    now.return_value = start + 60
    broker._trace_activity()
    await asyncio.sleep(0)
    events = records(caplog)
    assert len(events) == 2
    assert all(
        e["event"] == "health"
        and e["gateway_members"] == 0
        and e["broker_pending_futures"] == 0
        and not e["worker_alive"]
        for e in events
    )


async def test_worker_failure_is_structural_failure(gateway_env, mocker, caplog):
    child = gateway_env()
    await child.async_refresh()
    mocker.patch.object(
        child._broker,
        "_send_heartbeat_if_due",
        side_effect=RuntimeError("PRIVATE_FAILURE_DETAIL"),
    )
    await eventually(lambda: not child._broker.running)
    assert sum(e["event"] == "worker_failure" for e in records(caplog)) == 1
    assert report(caplog)["status"] == "FAIL"
    assert "PRIVATE_FAILURE_DETAIL" not in json.dumps(records(caplog))


async def test_trace_sink_failure_does_not_change_command_behavior(gateway_env, mocker):
    mocker.patch.object(
        trace._LOGGER, "debug", side_effect=RuntimeError("broken diagnostic sink")
    )
    child = gateway_env()
    await child.async_refresh()
    await child.async_set_property("1", True)
    await push(child, {"1": True})
    assert child.get_property("1") is True
    assert not child._command_lock.locked()
    assert child._broker.running
    await child.async_stop()


async def test_select_timeouts_and_successful_heartbeats_are_not_traced(
    gateway_env, caplog
):
    child = gateway_env()
    await child.async_refresh()
    child._broker._heartbeat_interval = 0
    caplog.clear()
    await asyncio.sleep(0.03)
    assert any(c[0] == "heartbeat" for c in child._api.parent.calls)
    assert not records(caplog)


async def test_poll_retries_are_not_counted_as_command_retries(gateway_env, caplog):
    child = gateway_env()
    await child.async_refresh()
    child._api.responses.append(RuntimeError("synthetic poll retry"))
    child._last_full_poll = 0
    child.actually_start()
    await eventually(lambda: any(e["event"] == "poll_done" for e in records(caplog)))
    await child.async_stop()
    events = records(caplog)
    retries = [e for e in events if e["event"] == "retry"]
    assert len(retries) == 1 and "cmd_id" not in retries[0]
    assert_schema(events)
    summary = report(caplog)
    assert summary["status"] == "PASS", summary
    assert summary["commands"]["retries"] == 0
    assert summary["polling"]["polls"] == 1
