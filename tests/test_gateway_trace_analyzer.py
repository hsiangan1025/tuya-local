"""Synthetic complete/broken canary captures, without HA or real identifiers."""

import json
import sys
from io import StringIO

import pytest

from tools.analyze_gateway_trace import analyze, main, parse


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


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("\x1b[36m", "\x1b[0m"),
        ("\x1b[1;36m", "\x1b[0m\x1b[K\r\n"),
        ("\x9b36m", "\x9b0m"),
        ("\x1b]0;PRIVATE_SENTINEL\x07\x1b[36m", "\x1b[0m"),
        ("\x1b]0;PRIVATE_SENTINEL\x1b\\", "\x1b[0m"),
        ("\x9d0;PRIVATE_SENTINEL\x9c", "\x9b0m"),
    ],
)
def test_terminal_coloring_preserves_records_and_analysis(prefix, suffix):
    plain = lines(valid_events())
    colored = [prefix + line + suffix for line in plain]
    assert parse(colored) == parse(plain)
    assert analyze(colored) == analyze(plain)
    assert "PRIVATE_SENTINEL" not in json.dumps(analyze(colored))


def test_color_changes_inside_ha_prefix_and_before_marker():
    plain = lines(valid_events())
    colored = [
        "\x1b[36m"
        + line.replace(" DEBUG ", " \x1b[1mDEBUG\x1b[22m ").replace(
            " GBTRACE ", " \x1b[0mGBTRACE \x1b[36m"
        )
        + "\x1b[0m\n"
        for line in plain
    ]
    assert parse(colored) == parse(plain)


@pytest.mark.parametrize(
    "line",
    [
        "GBTRACE {broken}",
        'GBTRACE {"v":1,"event":"gateway_start","run_id":1,"seq":1}',
        'GBTRACE {"v":1,"event":"gateway_start","run_id":1,"seq":true,"ts_ms":0}',
        'GBTRACE {"v":1,"event":"gateway_start","run_id":1,"seq":1,"ts_ms":NaN}',
        'GBTRACE {"v":2,"event":"gateway_start","run_id":1,"seq":1,"ts_ms":0}',
        'GBTRACE {"v":1,"event":"PRIVATE_SENTINEL","run_id":1,"seq":1,"ts_ms":0}',
        lines(valid_events())[0].replace("GBTRACE", "NOTGBTRACE"),
        lines(valid_events())[0] + " PRIVATE_SENTINEL",
        lines(valid_events())[0] + "\x1b[",
        lines(valid_events())[0] + "\x1b]PRIVATE_SENTINEL",
    ],
)
def test_colored_malformed_input_stays_ignored(line):
    colored = "\x1b[36m" + line + "\x1b[0m"
    assert parse([line]) == []
    assert parse([colored]) == []
    result = analyze([colored])
    assert result["warnings"] == ["no_valid_trace_events"]
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


def test_osc_contents_cannot_inject_trace_events():
    hidden = "\x1b]0;" + lines(valid_events())[0] + "\x07"
    assert parse([hidden]) == []
    visible = lines(valid_events())[1]
    assert parse([hidden + visible]) == parse([visible])


@pytest.mark.parametrize("stdin", [False, True])
def test_cli_reads_colored_ha_logs_without_echo(stdin, tmp_path, monkeypatch, capsys):
    plain = lines(valid_events())
    text = "\n".join("\x1b[36m" + line + "\x1b[0m" for line in plain)
    text += "\n\x1b[31mPRIVATE_SENTINEL_NOT_A_TRACE\x1b[0m\n"
    if stdin:
        argument = "-"
        monkeypatch.setattr(sys, "stdin", StringIO(text))
    else:
        path = tmp_path / "ha.log"
        path.write_text(text)
        argument = str(path)
    monkeypatch.setattr(sys, "argv", ["analyze_gateway_trace.py", argument])
    assert main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == analyze(plain)
    assert "PRIVATE_SENTINEL" not in output.out
    assert output.err == ""


@pytest.mark.parametrize(
    ("last_event", "failure"),
    [
        ("gateway_lock_wait", "lock_leak_or_unfinished_wait"),
        ("gateway_lock_acquired", "lock_leak_or_unfinished_wait"),
        ("broker_enqueue", "unfinished_broker_call"),
        ("broker_execute_start", "unfinished_broker_call"),
        ("poll_start", "unfinished_or_malformed_poll"),
        ("broker_drain_start", "unfinished_or_malformed_drain_batch"),
    ],
)
def test_strict_analysis_retains_tail_open_lifecycle_failures(last_event, failure):
    events = valid_events()
    boundary = next(i for i, e in enumerate(events) if e["event"] == last_event)
    result = analyze(lines(events[: boundary + 1]))
    assert result["status"] == "FAIL"
    assert failure in result["failures"]


def worker_wait_events(phase, *, outcome="ok"):
    """Five seconds outside drain, followed by a ten-millisecond status call."""
    return [
        dict(event="gateway_start", gateway_slot=1, ts_ms=0),
        dict(
            event="broker_enqueue",
            gateway_slot=1,
            child_slot=2,
            call_id=7,
            operation="request",
            ts_ms=100,
            broker_queue_depth=1,
            broker_pending_futures=1,
        ),
        dict(event=phase + "_start", gateway_slot=1, phase_id=9, ts_ms=100),
        dict(
            event=phase + "_done",
            gateway_slot=1,
            phase_id=9,
            ts_ms=5100,
            duration_ms=5000,
            outcome=outcome,
            socket_present=True,
        ),
        dict(
            event="broker_execute_start",
            gateway_slot=1,
            child_slot=2,
            call_id=7,
            operation="request",
            ts_ms=5100,
            wait_ms=5000,
        ),
        dict(
            event="broker_execute_done",
            gateway_slot=1,
            child_slot=2,
            call_id=7,
            operation="request",
            ts_ms=5110,
            duration_ms=10,
            outcome="ok",
        ),
    ]


@pytest.mark.parametrize("phase", ["connect", "heartbeat", "receive"])
@pytest.mark.parametrize("outcome", ["ok", "error"])
def test_five_second_queue_wait_is_attributed_to_observed_worker_phase(phase, outcome):
    sample = lines(worker_wait_events(phase, outcome=outcome))
    result = analyze(sample)
    assert result == analyze(list(reversed(sample)))
    assert result["status"] == ("PASS" if outcome == "ok" else "WARN")
    worker = result["worker_phases"]
    assert worker[phase] == {
        "count": 1,
        "completed": 1,
        "errors": int(outcome == "error"),
        "max_duration_ms": 5000,
    }
    wait = worker["queue_wait_attribution"][0]
    assert wait["call_id"] == 7 and wait["child_slot"] == 2
    assert (wait["enqueue_seq"], wait["execute_seq"]) == (2, 5)
    assert wait["observed_phase_ms"] == 5000
    assert wait["unattributed_ms"] == 0
    assert len(wait["overlaps"]) == 1
    span = wait["overlaps"][0]
    assert span["phase"] == phase and span["phase_id"] == 9
    assert span["start_seq"] == 3 and span["end_seq"] == 4
    assert span["overlap_ms"] == 5000
    execution = next(s for s in worker["intervals"] if s["phase"] == "broker_execute")
    assert execution["duration_ms"] == 10


def test_queue_attribution_clips_phase_to_wait_and_retains_unobserved_time():
    events = worker_wait_events("receive")
    events[2]["ts_ms"] = 0  # Receive was already active at enqueue.
    events[3].update(ts_ms=4100, duration_ms=4100)
    # Keep sequence order consistent with the receive starting earlier.
    events[1], events[2] = events[2], events[1]
    result = analyze(lines(events))
    wait = result["worker_phases"]["queue_wait_attribution"][0]
    assert wait["observed_phase_ms"] == 4000
    assert wait["unattributed_ms"] == 1000
    assert wait["overlaps"][0]["overlap_ms"] == 4000


def test_slow_select_can_be_attributed_without_logging_normal_selects():
    events = worker_wait_events("receive")
    events[2:4] = [
        dict(
            event="select_slow",
            gateway_slot=1,
            ts_ms=5100,
            start_ts_ms=100,
            duration_ms=5000,
            socket_present=True,
        )
    ]
    result = analyze(lines(events))
    worker = result["worker_phases"]
    assert worker["select_slow"] == {"count": 1, "max_duration_ms": 5000}
    wait = worker["queue_wait_attribution"][0]
    assert wait["overlaps"][0]["phase"] == "select"
    assert wait["observed_phase_ms"] == 5000


def test_multiple_phases_explain_parts_of_one_wait():
    events = worker_wait_events("connect")
    events[3].update(ts_ms=2100, duration_ms=2000)
    events[4:4] = [
        dict(event="receive_start", gateway_slot=1, phase_id=10, ts_ms=2100),
        dict(
            event="receive_done",
            gateway_slot=1,
            phase_id=10,
            ts_ms=5100,
            duration_ms=3000,
            socket_present=True,
            outcome="ok",
        ),
    ]
    wait = analyze(lines(events))["worker_phases"]["queue_wait_attribution"][0]
    assert [(s["phase"], s["overlap_ms"]) for s in wait["overlaps"]] == [
        ("connect", 2000),
        ("receive", 3000),
    ]
    assert wait["observed_phase_ms"] == 5000 and wait["unattributed_ms"] == 0


@pytest.mark.parametrize("scope", ["gateway_slot", "run_id"])
def test_attribution_never_joins_different_gateways_or_runs(scope):
    events = worker_wait_events("receive")
    events[2][scope] = events[3][scope] = 99
    result = analyze(lines(events))
    wait = result["worker_phases"]["queue_wait_attribution"][0]
    assert wait["overlaps"] == []
    assert wait["observed_phase_ms"] == 0 and wait["unattributed_ms"] == 5000


def test_old_capture_does_not_gain_a_guessed_five_second_attribution():
    events = worker_wait_events("receive")
    del events[2:4]
    result = analyze(lines(events))
    assert result["status"] == "PASS"
    worker = result["worker_phases"]
    assert worker["receive"]["count"] == 0
    assert worker["receive"]["max_duration_ms"] is None
    wait = worker["queue_wait_attribution"][0]
    assert wait["overlaps"] == [] and wait["unattributed_ms"] == 5000


@pytest.mark.parametrize(
    "change",
    [
        "missing_start",
        "missing_done",
        "mismatched_phase",
        "wrong_gateway",
        "reverse_time",
        "duplicate_start",
    ],
)
def test_incomplete_or_malformed_worker_phases_fail(change):
    events = worker_wait_events("receive")
    if change == "missing_start":
        del events[2]
    elif change == "missing_done":
        del events[3]
    elif change == "mismatched_phase":
        events[3]["event"] = "heartbeat_done"
    elif change == "wrong_gateway":
        events[3]["gateway_slot"] = 2
    elif change == "reverse_time":
        events[3]["ts_ms"] = 0
    else:
        events.insert(3, dict(events[2]))
    result = analyze(lines(events))
    assert result["status"] == "FAIL"
    assert set(result["failures"]) & {
        "unfinished_worker_phase",
        "malformed_worker_phase",
    }


def test_tail_open_worker_phase_is_not_silently_excused():
    events = worker_wait_events("connect")[:3]
    result = analyze(lines(events))
    assert "unfinished_worker_phase" in result["failures"]
    span = result["worker_phases"]["intervals"][0]
    assert span["phase_id"] == 9 and span["end_seq"] is None
    assert span["duration_ms"] is None


@pytest.mark.parametrize("bad", [True, -1, float("inf"), "PRIVATE_CONFIG_SENTINEL"])
def test_worker_configuration_validation_is_numeric_only(bad):
    event = dict(event="gateway_start", gateway_slot=1, socket_timeout_ms=bad)
    assert parse(lines([event])) == []


def test_worker_phase_outcome_validation_and_privacy():
    events = worker_wait_events("receive")
    for event in events:
        event["payload"] = "PRIVATE_PAYLOAD_SENTINEL"
        event["exception"] = "PRIVATE_EXCEPTION_SENTINEL"
    result = analyze(lines(events))
    assert result["status"] == "PASS"
    assert "PRIVATE_" not in json.dumps(result)
    events[3]["outcome"] = "cancelled"  # Valid for commands, not these phases.
    assert not any(e["event"] == "receive_done" for e in parse(lines(events)))


def test_another_broker_execution_can_occupy_the_wait():
    events = worker_wait_events("connect")
    del events[2:4]
    events[1:1] = [
        dict(
            event="broker_enqueue",
            gateway_slot=1,
            call_id=8,
            operation="request",
            ts_ms=0,
            broker_queue_depth=1,
            broker_pending_futures=1,
        ),
        dict(
            event="broker_execute_start",
            gateway_slot=1,
            call_id=8,
            operation="request",
            ts_ms=0,
            wait_ms=0,
        ),
    ]
    events.insert(
        4,
        dict(
            event="broker_execute_done",
            gateway_slot=1,
            call_id=8,
            operation="request",
            ts_ms=5100,
            duration_ms=5100,
            outcome="ok",
        ),
    )
    result = analyze(lines(events))
    assert result["status"] == "PASS"
    wait = next(
        w
        for w in result["worker_phases"]["queue_wait_attribution"]
        if w["call_id"] == 7
    )
    assert wait["observed_phase_ms"] == 5000 and wait["unattributed_ms"] == 0
    assert wait["overlaps"][0]["phase"] == "broker_execute"
    assert wait["overlaps"][0]["call_id"] == 8


def test_overlapping_worker_spans_fail_without_double_counting_wait():
    events = worker_wait_events("connect")
    events.insert(
        3, dict(event="receive_start", gateway_slot=1, phase_id=10, ts_ms=100)
    )
    events.insert(
        5,
        dict(
            event="receive_done",
            gateway_slot=1,
            phase_id=10,
            ts_ms=5100,
            duration_ms=5000,
            outcome="ok",
            socket_present=True,
        ),
    )
    result = analyze(lines(events))
    assert "overlapping_worker_phases" in result["failures"]
    wait = result["worker_phases"]["queue_wait_attribution"][0]
    assert len(wait["overlaps"]) == 2
    assert wait["observed_phase_ms"] == 5000 and wait["unattributed_ms"] == 0
