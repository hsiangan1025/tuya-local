"""Synthetic complete/broken canary captures, without HA or real identifiers."""

import json

import pytest

from tools.analyze_gateway_trace import analyze, parse


def valid_events():
    events = []

    def add(event, **fields):
        events.append({"event": event, "gateway_slot": 1, **fields})

    def command(event, **fields):
        add(event, child_slot=2, cmd_id=3, **fields)

    def call(event, **fields):
        command(event, call_id=7, operation="control", **fields)

    add("gateway_start")
    add("child_register", child_slot=2)
    add("poll_due", child_slot=2, poll_id=10)
    add("poll_start", child_slot=2, poll_id=10)
    add("poll_done", child_slot=2, poll_id=10, duration_ms=9)
    command("cmd_request", dp_count=1)
    command("command_lock_wait", lock_id=4)
    command("command_lock_acquired", lock_id=4, wait_ms=2)
    command("debounce_start")
    command("debounce_done", duration_ms=1)
    command(
        "pending_snapshot",
        pending_generation=5,
        pending_cmd_id=3,
        dp_count=1,
        pending_count=1,
    )
    command("gateway_lock_wait", lock_id=6)
    command("gateway_lock_acquired", lock_id=6, wait_ms=3)
    call("broker_enqueue", broker_queue_depth=1, broker_pending_futures=1)
    add("broker_drain_start", batch_id=8)
    call("broker_execute_start", wait_ms=4)
    call("tuya_send_start")
    call("tuya_send_result", outcome="ok", duration_ms=5)
    call("broker_execute_done", outcome="ok", duration_ms=6)
    add(
        "broker_drain_done",
        batch_id=8,
        calls_executed=1,
        duration_ms=7,
        queue_depth_after=0,
    )
    command("mark_sent", pending_generation=5, dp_count=1, pending_count=1)
    command("gateway_lock_released", lock_id=6)
    command("command_lock_released", lock_id=4)
    command("cmd_complete", outcome="ok", duration_ms=25)
    add("push_rx", child_slot=2, push_id=9, source="socket")
    add("push_dispatch", child_slot=2, push_id=9, source="socket")
    command(
        "push_ack", pending_generation=5, ack_latency_ms=8, remaining_pending_count=0
    )
    add(
        "health",
        broker_queue_depth=0,
        broker_pending_futures=0,
        registered_children=1,
        gateway_members=1,
        socket_present=True,
        worker_alive=True,
    )
    add("child_unregister", child_slot=2)
    add(
        "health",
        broker_queue_depth=0,
        broker_pending_futures=0,
        registered_children=0,
        gateway_members=0,
        socket_present=False,
        worker_alive=False,
    )
    add("gateway_stop")
    return events


def lines(events):
    return [
        "2026-01-01 DEBUG [custom_components.tuya_local.trace] GBTRACE "
        + json.dumps({"v": 1, "run_id": 123, "seq": i, "ts_ms": i * 5, **e})
        for i, e in enumerate(events, 1)
    ]


def test_valid_trace_has_deterministic_pass_summary():
    sample = lines(valid_events())
    result = analyze(sample)
    assert result == analyze(list(reversed(sample)))
    assert result["status"] == "PASS", result
    assert result["commands"] == {
        "logical_commands": 1,
        "completed": 1,
        "sends": 1,
        "retries": 0,
        "duplicate_sends": 0,
        "stale_callbacks_ignored": 0,
        "incomplete_chains": 0,
        "acknowledged": 1,
        "ack_timeouts": 0,
    }
    assert result["latency_ms"]["command"] == dict(p50=25, p95=25, p99=25, max=25)
    assert result["broker"]["final_pending_futures"] == 0
    assert result["polling"]["maximum_short_window_poll_burst"] == 1


@pytest.mark.parametrize(
    "missing",
    [
        "cmd_request",
        "cmd_complete",
        "command_lock_wait",
        "command_lock_acquired",
        "command_lock_released",
        "gateway_lock_wait",
        "gateway_lock_acquired",
        "gateway_lock_released",
        "debounce_start",
        "pending_snapshot",
        "broker_enqueue",
        "broker_execute_start",
        "broker_execute_done",
        "tuya_send_result",
        "broker_drain_done",
        "poll_done",
        "child_unregister",
    ],
)
def test_missing_lifecycle_event_fails(missing):
    result = analyze(lines(e for e in valid_events() if e["event"] != missing))
    assert result["status"] == "FAIL", (missing, result)


@pytest.mark.parametrize("retry", [False, True])
def test_duplicate_send_requires_retry(retry):
    events = valid_events()
    more = [dict(e, call_id=11) for e in events if e.get("call_id") == 7]
    if retry:
        more.insert(
            0,
            {
                "event": "retry",
                "gateway_slot": 1,
                "child_slot": 2,
                "cmd_id": 3,
                "attempt": 2,
            },
        )
    position = next(
        i for i, e in enumerate(events) if e["event"] == "gateway_lock_released"
    )
    events[position:position] = more
    result = analyze(lines(events))
    assert result["commands"]["sends"] == 2
    assert result["commands"]["duplicate_sends"] == (0 if retry else 1)
    assert result["status"] == ("PASS" if retry else "FAIL")


@pytest.mark.parametrize("field", ["broker_queue_depth", "broker_pending_futures"])
def test_monotonic_backlog_accumulation_fails(field):
    events = [
        dict(
            event="health",
            gateway_slot=1,
            broker_queue_depth=0,
            broker_pending_futures=0,
            registered_children=1,
            gateway_members=1,
            socket_present=True,
            worker_alive=True,
        )
        | {field: n}
        for n in (1, 2, 3)
    ]
    result = analyze(lines(events))
    assert result["status"] == "FAIL"
    assert "monotonically_accumulating_" + field in result["failures"]


def test_a_draining_burst_does_not_imply_accumulation():
    events = [
        dict(
            event="health",
            gateway_slot=1,
            broker_queue_depth=n,
            broker_pending_futures=n,
            registered_children=1,
            gateway_members=1,
            socket_present=True,
            worker_alive=True,
        )
        for n in (1, 2, 3, 0)
    ]
    assert analyze(lines(events))["status"] == "PASS"


@pytest.mark.parametrize("timeout", [False, True])
def test_missing_push_ack_is_not_a_failure(timeout):
    events = [e for e in valid_events() if e["event"] != "push_ack"]
    if timeout:
        events.append(
            dict(
                event="ack_timeout",
                gateway_slot=1,
                child_slot=2,
                cmd_id=3,
                pending_generation=5,
                duration_ms=5000,
            )
        )
    result = analyze(lines(events))
    assert result["status"] == "WARN"
    assert not result["failures"]
    assert result["commands"]["ack_timeouts"] == int(timeout)


def test_malformed_and_non_trace_input_is_ignored_without_echo():
    garbage = [
        "PRIVATE_SENTINEL_NOT_A_TRACE",
        "GBTRACE {broken}",
        "GBTRACE []",
        'GBTRACE {"v":1,"event":[]}',
        'GBTRACE {"v":1,"event":"health","seq":"PRIVATE_SENTINEL"}',
        'GBTRACE {"v":2,"event":"health","run_id":1,"seq":1,"ts_ms":0}',
        'GBTRACE {"v":1,"event":"health","run_id":1,"seq":1,"ts_ms":NaN}',
        'NOTGBTRACE {"v":1,"event":"health","run_id":1,"seq":1,"ts_ms":0}',
    ]
    assert parse(garbage) == []
    result = analyze(garbage + lines(valid_events()))
    assert result["status"] == "PASS"
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


@pytest.mark.parametrize("bad", ["gateway_start", "child_register", "worker_failure"])
def test_bad_lifecycle_or_worker_failure_fails(bad):
    events = valid_events()
    events.insert(2, {"event": bad, "gateway_slot": 1, "child_slot": 2})
    assert analyze(lines(events))["status"] == "FAIL"


def test_empty_trace_warns_and_unobserved_final_counts_are_null():
    result = analyze(["ordinary HA log"])
    assert result["status"] == "WARN"
    assert result["broker"]["final_pending_futures"] is None
    assert result["latency_ms"]["command"]["p50"] is None


def test_debounce_and_command_order_is_checked():
    events = valid_events()
    first = next(i for i, e in enumerate(events) if e["event"] == "debounce_start")
    second = next(i for i, e in enumerate(events) if e["event"] == "debounce_done")
    events[first], events[second] = events[second], events[first]
    result = analyze(lines(events))
    assert "malformed_command_order" in result["failures"]


def test_process_local_ids_are_scoped_by_run():
    first = lines(valid_events())
    second = [line.replace('"run_id": 123', '"run_id": 124') for line in first]
    result = analyze(first + second)
    assert result["status"] == "PASS"
    assert result["commands"]["logical_commands"] == 2
