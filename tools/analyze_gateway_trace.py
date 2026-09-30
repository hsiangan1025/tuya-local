#!/usr/bin/env python3
"""Analyze only GBTRACE JSON in HA logs, using the Python standard library.

Usage: python tools/analyze_gateway_trace.py [log-file|-]
Exit codes: PASS=0, WARN=1, FAIL=2. A complete capture is needed for lifecycle
checks; truncated captures deliberately report incomplete chains. No raw input
or arbitrary strings are reproduced in the report.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATTERN = re.compile(r"(?:^|\s)GBTRACE (\{[^\n]*\})\s*$")
EVENTS = set(
    "gateway_start gateway_stop child_register child_unregister cmd_request "
    "command_lock_wait command_lock_acquired command_lock_released "
    "gateway_lock_wait gateway_lock_acquired gateway_lock_released lock_cancelled "
    "debounce_start debounce_done pending_snapshot broker_enqueue "
    "broker_execute_start broker_execute_done broker_cancelled tuya_send_start "
    "tuya_send_result cmd_complete retry mark_sent stale_mark_sent_ignored "
    "pending_expire ack_timeout push_rx push_dispatch push_ack poll_due "
    "poll_start poll_done poll_error disconnect reconnect worker_failure "
    "broker_drain_start broker_drain_done health".split()
)
IDS = set(
    "run_id seq gateway_slot child_slot cmd_id poll_id call_id lock_id batch_id push_id pending_generation pending_cmd_id".split()
)
COUNTS = set(
    "dp_count pending_count remaining_pending_count attempt calls_executed queue_depth_after broker_queue_depth broker_pending_futures registered_children gateway_members".split()
)
TIMES = set("ts_ms duration_ms wait_ms ack_latency_ms".split())
ENUMS = {
    "outcome": {"ok", "error", "cancelled", "stopped"},
    "source": {"socket", "cache"},
    "operation": {"call", "control", "request"},
}


SCHEMA = json.loads(Path(__file__).with_name("gateway_trace_schema.json").read_text())
REQUIRED = {event: set(SCHEMA["required"]) for event in EVENTS}
for rule in SCHEMA["allOf"]:
    for event in rule["if"]["properties"]["event"]["enum"]:
        REQUIRED[event].update(rule["then"]["required"])


def parse(lines):
    """Ignore malformed/foreign records and never reflect their contents."""
    records = []
    seen = set()
    for line in lines:
        match = PATTERN.search(line)
        if not match:
            continue
        try:
            raw = json.loads(match[1])
        except ValueError, RecursionError:
            continue
        if (
            not isinstance(raw, dict)
            or type(raw.get("v")) is not int
            or raw["v"] != 1
            or type(raw.get("event")) is not str
            or raw["event"] not in EVENTS
        ):
            continue
        clean = {"v": 1, "event": raw["event"]}
        valid = True
        for key, value in raw.items():
            if key in IDS | COUNTS:
                if value is None and key in {"gateway_slot", "child_slot"}:
                    clean[key] = None
                elif type(value) is int and value >= 0:
                    clean[key] = value
                else:
                    valid = False
            elif key in TIMES:
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    clean[key] = value
                else:
                    valid = False
            elif key in ENUMS:
                if type(value) is str and value in ENUMS[key]:
                    clean[key] = value
                else:
                    valid = False
            elif key in {"socket_present", "worker_alive"}:
                if type(value) is bool:
                    clean[key] = value
                else:
                    valid = False
        if not valid or not {"run_id", "seq", "ts_ms"} <= clean.keys():
            continue
        if not REQUIRED[clean["event"]] <= clean.keys():
            continue
        identity = clean["run_id"], clean["seq"]
        if identity in seen:
            continue
        seen.add(identity)
        records.append(clean)
    return sorted(records, key=lambda e: (e["run_id"], e["seq"]))


def distribution(values):
    values = sorted(values)
    if not values:
        return {key: None for key in ("p50", "p95", "p99", "max")}

    def percentile(p):
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return round(values[low] + (values[high] - values[low]) * (position - low), 6)

    return {
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": round(values[-1], 6),
    }


def analyze(lines, *, poll_window_seconds=1.0):
    events = parse(lines)
    counts = Counter(e["event"] for e in events)
    failures, warnings = set(), set()
    bad_commands = set()

    def fail(code, event=None):
        failures.add(code)
        if event is not None and "cmd_id" in event:
            bad_commands.add((event["run_id"], event["cmd_id"]))

    def groups(field):
        result = defaultdict(list)
        for event in events:
            if event.get(field) is not None:
                result[event["run_id"], event[field]].append(event)
        return result

    commands = groups("cmd_id")
    requested = completed = acknowledged = timeouts = duplicates = 0
    for chain in commands.values():
        names = Counter(e["event"] for e in chain)
        request = [e for e in chain if e["event"] == "cmd_request"]
        completion = [e for e in chain if e["event"] == "cmd_complete"]
        requested += bool(request)
        completed += bool(completion)
        acknowledged += any(
            e["event"] == "push_ack" and e.get("remaining_pending_count") == 0
            for e in chain
        )
        timeouts += bool(names["ack_timeout"])
        if len(request) != 1 or len(completion) != 1:
            fail("incomplete_command", chain[0])
        elif completion[0]["seq"] < request[0]["seq"]:
            fail("command_completion_before_request", chain[0])
        if completion and completion[0].get("outcome") != "ok":
            warnings.add("command_error_or_cancellation")
        if request and completion and completion[0].get("outcome") == "ok":
            for name in (
                "command_lock_wait",
                "command_lock_acquired",
                "command_lock_released",
                "debounce_start",
                "debounce_done",
                "pending_snapshot",
                "gateway_lock_wait",
                "gateway_lock_acquired",
                "gateway_lock_released",
                "broker_enqueue",
                "broker_execute_start",
                "tuya_send_start",
                "tuya_send_result",
            ):
                if not names[name]:
                    fail("missing_command_lifecycle_event", chain[0])
            if not names["push_ack"]:
                warnings.add("commands_without_observed_push_ack")
        positions = {}
        for e in chain:
            positions.setdefault(e["event"], e["seq"])
        for before, after in (
            ("cmd_request", "command_lock_wait"),
            ("command_lock_acquired", "debounce_start"),
            ("debounce_start", "debounce_done"),
            ("debounce_done", "pending_snapshot"),
            ("pending_snapshot", "gateway_lock_wait"),
            ("gateway_lock_acquired", "tuya_send_start"),
            ("command_lock_released", "cmd_complete"),
        ):
            if (
                before in positions
                and after in positions
                and positions[before] >= positions[after]
            ):
                fail("malformed_command_order", chain[0])
        last_send = None
        retry_since_send = False
        for e in chain:
            if e["event"] == "retry":
                retry_since_send = True
            elif e["event"] == "tuya_send_start":
                if last_send is not None and not retry_since_send:
                    duplicates += 1
                    fail("duplicate_send_without_retry", e)
                last_send = e
                retry_since_send = False
    if timeouts:
        warnings.add("ack_timeout_is_not_proof_of_device_failure")

    for chain in groups("lock_id").values():
        state = "new"
        for e in chain:
            name = e["event"]
            if name.endswith("_lock_wait") and state == "new":
                state = "waiting"
            elif name.endswith("_lock_acquired") and state == "waiting":
                state = "held"
            elif name.endswith("_lock_released") and state == "held":
                state = "released"
            elif name == "lock_cancelled" and state == "waiting":
                state = "cancelled"
            else:
                fail("malformed_lock_order", e)
        if state not in {"released", "cancelled"}:
            fail("lock_leak_or_unfinished_wait", chain[0])

    for chain in groups("call_id").values():
        state = "new"
        send_started = send_finished = False
        for e in chain:
            name = e["event"]
            if name == "broker_enqueue" and state == "new":
                state = "queued"
            elif name == "broker_execute_start" and state == "queued":
                state = "running"
            elif name == "tuya_send_start" and state == "running" and not send_started:
                send_started = True
            elif (
                name == "tuya_send_result"
                and state == "running"
                and send_started
                and not send_finished
            ):
                send_finished = True
            elif name == "broker_execute_done" and state == "running":
                if send_started and not send_finished:
                    fail("execution_without_send_result", e)
                if (
                    e.get("operation") == "control"
                    and not send_finished
                    and e.get("outcome") != "error"
                ):
                    fail("execution_without_send_result", e)
                state = "done"
            elif name == "broker_cancelled" and state == "queued":
                state = "cancelled"
            elif name in {
                "broker_enqueue",
                "broker_execute_start",
                "broker_execute_done",
                "broker_cancelled",
                "tuya_send_start",
                "tuya_send_result",
            }:
                fail("malformed_broker_call_order", e)
        if state not in {"done", "cancelled"}:
            fail("unfinished_broker_call", chain[0])

    for chain in groups("batch_id").values():
        if [e["event"] for e in chain] != ["broker_drain_start", "broker_drain_done"]:
            fail("unfinished_or_malformed_drain_batch")
    for chain in groups("poll_id").values():
        names = [e["event"] for e in chain if e["event"].startswith("poll_")]
        if names not in (
            ["poll_due", "poll_start", "poll_done"],
            ["poll_due", "poll_start", "poll_error"],
        ):
            fail("unfinished_or_malformed_poll")
    for chain in groups("push_id").values():
        names = [e["event"] for e in chain]
        if names not in (["push_rx"], ["push_rx", "push_dispatch"]):
            fail("malformed_push_order")

    final_queues, final_pending = [], []
    for chain in groups("gateway_slot").values():
        running = False
        saw_start = False
        children = set()
        health = []
        last_queue = last_pending = None
        for e in chain:
            name = e["event"]
            if name == "gateway_start":
                if running:
                    fail("duplicate_gateway_start")
                running = True
                saw_start = True
            elif name == "gateway_stop":
                if not running:
                    fail("gateway_stop_without_start")
                running = False
            elif name == "child_register":
                slot = e.get("child_slot")
                if not running or slot in children or slot is None:
                    fail("malformed_child_registration")
                children.add(slot)
            elif name == "child_unregister":
                slot = e.get("child_slot")
                if slot not in children:
                    fail("child_unregister_without_registration")
                children.discard(slot)
            if name == "health":
                # Registrations and health snapshots run on the HA loop.
                # This also detects an omitted unregister at final unload.
                if saw_start and e.get("registered_children") != len(children):
                    fail("registration_count_mismatch")
                health.append(e)
                last_queue = e.get("broker_queue_depth")
                last_pending = e.get("broker_pending_futures")
            elif name == "broker_drain_done":
                last_queue = e.get("queue_depth_after")
        for field in ("broker_queue_depth", "broker_pending_futures"):
            values = [e[field] for e in health if field in e]
            if len(values) >= 3:
                tail = values[-3:]
                if tail[-1] > tail[0] and all(
                    b >= a for a, b in zip(tail, tail[1:], strict=False)
                ):
                    fail("monotonically_accumulating_" + field)
        if last_queue is not None:
            final_queues.append(last_queue)
        if last_pending is not None:
            final_pending.append(last_pending)
    if counts["worker_failure"]:
        fail("worker_failure")
    if not events:
        warnings.add("no_valid_trace_events")

    def values(name, field):
        return [e[field] for e in events if e["event"] == name and field in e]

    def maximum(numbers):
        return max(numbers, default=0)

    burst = 0
    for chain in groups("gateway_slot").values():
        starts = sorted(e["ts_ms"] for e in chain if e["event"] == "poll_start")
        left = 0
        for right, started in enumerate(starts):
            while started - starts[left] > poll_window_seconds * 1000:
                left += 1
            burst = max(burst, right - left + 1)
    return {
        "status": "FAIL" if failures else "WARN" if warnings else "PASS",
        "failures": sorted(failures),
        "warnings": sorted(warnings),
        "events": len(events),
        "schema_version": 1,
        "commands": {
            "logical_commands": requested,
            "completed": completed,
            "sends": counts["tuya_send_start"],
            "retries": sum(e["event"] == "retry" and "cmd_id" in e for e in events),
            "duplicate_sends": duplicates,
            "stale_callbacks_ignored": counts["stale_mark_sent_ignored"],
            "incomplete_chains": len(bad_commands),
            "acknowledged": acknowledged,
            "ack_timeouts": timeouts,
        },
        "latency_ms": {
            label: distribution(values(event, field))
            for label, event, field in (
                ("command", "cmd_complete", "duration_ms"),
                ("command_lock_wait", "command_lock_acquired", "wait_ms"),
                ("gateway_lock_wait", "gateway_lock_acquired", "wait_ms"),
                ("broker_queue_wait", "broker_execute_start", "wait_ms"),
                ("tuya_send", "tuya_send_result", "duration_ms"),
                ("push_ack", "push_ack", "ack_latency_ms"),
            )
        },
        "broker": {
            "drain_batches": counts["broker_drain_done"],
            "max_calls_per_drain": maximum(
                values("broker_drain_done", "calls_executed")
            ),
            "max_drain_duration_ms": maximum(
                values("broker_drain_done", "duration_ms")
            ),
            "max_queue_depth": maximum(
                [e["broker_queue_depth"] for e in events if "broker_queue_depth" in e]
                + values("broker_drain_done", "queue_depth_after")
            ),
            "final_queue_depth": sum(final_queues) if final_queues else None,
            "max_pending_futures": maximum(
                [
                    e["broker_pending_futures"]
                    for e in events
                    if "broker_pending_futures" in e
                ]
            ),
            "final_pending_futures": sum(final_pending) if final_pending else None,
            "worker_failures": counts["worker_failure"],
            "disconnects": counts["disconnect"],
            "reconnects": counts["reconnect"],
        },
        "polling": {
            "polls": counts["poll_start"],
            "errors": counts["poll_error"],
            "max_duration_ms": maximum(
                values("poll_done", "duration_ms") + values("poll_error", "duration_ms")
            ),
            "maximum_short_window_poll_burst": burst,
            "window_seconds": poll_window_seconds,
        },
        "push": {
            "received": counts["push_rx"],
            "dispatched": counts["push_dispatch"],
            "acknowledged": counts["push_ack"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", default="-")
    args = parser.parse_args()
    if args.log == "-":
        result = analyze(sys.stdin)
    else:
        with Path(args.log).open(encoding="utf-8", errors="replace") as stream:
            result = analyze(stream)
    print(json.dumps(result, sort_keys=True, indent=2))
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
