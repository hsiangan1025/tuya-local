"""Opt-in, value-free diagnostics for the gateway canary.

Only this logger needs DEBUG. Slots/IDs are process-local counters, never device
identifiers. Observations must not control retries, pending state or scheduling.
"""

import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import count
from secrets import randbits
from time import monotonic

_LOGGER = logging.getLogger(__name__)
_IDS = count(1)
_SEQUENCE = count(1)
_RUN = randbits(63)
_ORIGIN = monotonic()
_CURRENT = ContextVar("gateway_trace", default=None)

EVENTS = frozenset(
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
_NUMBERS = frozenset(
    "gateway_slot child_slot cmd_id poll_id call_id lock_id batch_id push_id "
    "pending_generation pending_cmd_id dp_count pending_count remaining_pending_count "
    "duration_ms wait_ms ack_latency_ms attempt calls_executed queue_depth_after "
    "broker_queue_depth broker_pending_futures registered_children gateway_members "
    "socket_present worker_alive".split()
)
_ENUMS = {
    "outcome": {"ok", "error", "cancelled", "stopped"},
    "source": {"socket", "cache"},
    "operation": {"call", "control", "request"},
}


def enabled():
    return _LOGGER.isEnabledFor(logging.DEBUG)


def new_id():
    return next(_IDS)


def emit(event, fields=None, **values):
    """The privacy boundary: no arbitrary strings, identifiers or containers."""
    if not enabled() or event not in EVENTS:
        return
    payload = {
        "v": 1,
        "run_id": _RUN,
        "seq": next(_SEQUENCE),
        "ts_ms": round((monotonic() - _ORIGIN) * 1000, 6),
        "event": event,
    }
    for key, value in ((fields or {}) | values).items():
        if key in _NUMBERS and (
            value is None
            or type(value) in (int, bool)
            or (type(value) is float and math.isfinite(value))
        ):
            payload[key] = value
        elif key in _ENUMS and type(value) is str and value in _ENUMS[key]:
            payload[key] = value
    try:
        _LOGGER.debug(
            "GBTRACE %s", json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
    except Exception:
        # An optional diagnostic sink must never fail a command or its worker.
        # Do not report the exception through a potentially broken/raw logger.
        return


def device_fields(device):
    broker = device._broker
    return {
        "gateway_slot": broker.trace_slot if broker else None,
        "child_slot": broker.trace_child_slot(device._api) if broker else None,
    }


@dataclass
class Context:
    fields: dict
    device: object = None
    outcome: str = "ok"

    def snapshot(self):
        fields = dict(self.fields)
        if self.device is not None:
            fields.update(device_fields(self.device))
        return fields

    def event(self, event, **values):
        if enabled():
            emit(event, self.snapshot(), **values)


def current():
    return _CURRENT.get() if enabled() else None


@contextmanager
def command(device, dp_count):
    if not enabled():
        yield
        return
    ctx = Context({"cmd_id": new_id()}, device)
    token = _CURRENT.set(ctx)
    started = monotonic()
    ctx.event("cmd_request", dp_count=dp_count)
    try:
        yield
    except asyncio.CancelledError:
        ctx.outcome = "cancelled"
        raise
    except BaseException:
        ctx.outcome = "error"
        raise
    finally:
        ctx.event(
            "cmd_complete",
            duration_ms=(monotonic() - started) * 1000,
            outcome=ctx.outcome,
        )
        _CURRENT.reset(token)


@asynccontextmanager
async def measured_lock(lock, kind, device=None):
    ctx = current()
    token = None
    if ctx is None and device is not None and enabled():
        ctx = Context({}, device)
        token = _CURRENT.set(ctx)
    if ctx is None:
        async with lock:
            yield
        return
    lock_id = new_id()
    acquired = False
    started = monotonic()
    ctx.event(f"{kind}_lock_wait", lock_id=lock_id)
    try:
        async with lock:
            acquired = True
            ctx.event(
                f"{kind}_lock_acquired",
                lock_id=lock_id,
                wait_ms=(monotonic() - started) * 1000,
            )
            yield
    finally:
        ctx.event(
            f"{kind}_lock_released" if acquired else "lock_cancelled", lock_id=lock_id
        )
        if token is not None:
            _CURRENT.reset(token)


@contextmanager
def debounce():
    ctx = current()
    started = monotonic() if ctx else None
    if ctx:
        ctx.event("debounce_start")
    try:
        yield
    finally:
        if ctx:
            ctx.event("debounce_done", duration_ms=(monotonic() - started) * 1000)


@dataclass
class BrokerCall:
    fields: dict
    enqueued: float


def enqueue(broker, operation):
    if not enabled():
        return None
    ctx = current()
    fields = ctx.snapshot() if ctx else {}
    fields.update(gateway_slot=broker.trace_slot, call_id=new_id(), operation=operation)
    call = BrokerCall(fields, monotonic())
    emit(
        "broker_enqueue",
        fields,
        broker_queue_depth=broker._calls.qsize() + 1,
        broker_pending_futures=len(broker._pending),
    )
    return call


@contextmanager
def execute(call):
    if call is None:
        yield
        return
    started = monotonic()
    token = _CURRENT.set(Context(call.fields))
    emit("broker_execute_start", call.fields, wait_ms=(started - call.enqueued) * 1000)
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        emit(
            "broker_execute_done",
            call.fields,
            outcome=outcome,
            duration_ms=(monotonic() - started) * 1000,
        )
        _CURRENT.reset(token)


@contextmanager
def send():
    ctx = current()
    started = monotonic() if ctx else None
    if ctx:
        ctx.event("tuya_send_start")
    result = {"outcome": "error"} if ctx else None
    try:
        yield result
    finally:
        if ctx:
            ctx.event(
                "tuya_send_result",
                outcome=result["outcome"],
                duration_ms=(monotonic() - started) * 1000,
            )


@dataclass
class Pending:
    record: dict
    fields: dict
    sent_at: float | None = None


def pending_added(device, properties):
    ctx = current()
    if ctx is None:
        # DEBUG may be toggled while a previous traced command is pending.
        for key in properties:
            device._trace_pending.pop(key, None)
        return
    generation = new_id()
    fields = ctx.snapshot() | {"pending_generation": generation}
    for key in properties:
        device._trace_pending[key] = Pending(device._pending_updates[key], fields)


def pending_snapshot(device, properties):
    ctx = current()
    if ctx is None:
        return None
    snapshot = {
        k: device._trace_pending[k] for k in properties if k in device._trace_pending
    }
    groups = {}
    for pending in snapshot.values():
        generation = pending.fields["pending_generation"]
        groups.setdefault(generation, [pending, 0])[1] += 1
    for generation, (pending, size) in groups.items():
        ctx.event(
            "pending_snapshot",
            pending_generation=generation,
            pending_cmd_id=pending.fields["cmd_id"],
            dp_count=size,
            pending_count=len(device._pending_updates),
        )
    return snapshot


def marked(device, accepted_keys, snapshot, sent_at):
    if snapshot is None:
        return
    for key, pending in snapshot.items():
        accepted = key in accepted_keys
        if accepted:
            pending.sent_at = sent_at
        if enabled():
            pending.fields.update(
                (key, value)
                for key, value in device_fields(device).items()
                if value is not None
            )
            emit(
                "mark_sent" if accepted else "stale_mark_sent_ignored",
                pending.fields,
                dp_count=1,
                pending_count=len(device._pending_updates),
            )


def push_ack(device, data):
    if not enabled():
        return
    for key, pending in tuple(device._trace_pending.items()):
        if (
            device._pending_updates.get(key) is pending.record
            and pending.record["sent"]
            and key in data
            and data[key] == pending.record["value"]
        ):
            device._trace_pending.pop(key)
            if pending.sent_at is not None:
                remaining = sum(
                    p.fields["cmd_id"] == pending.fields["cmd_id"]
                    for p in device._trace_pending.values()
                )
                emit(
                    "push_ack",
                    pending.fields,
                    ack_latency_ms=(monotonic() - pending.sent_at) * 1000,
                    remaining_pending_count=remaining,
                )


def prune_pending(device, *, expired=False):
    for key, pending in tuple(device._trace_pending.items()):
        if device._pending_updates.get(key) is pending.record:
            continue
        device._trace_pending.pop(key)
        if expired and enabled():
            emit(
                "pending_expire",
                pending.fields,
                dp_count=1,
                pending_count=len(device._pending_updates),
            )
            if pending.sent_at is not None:
                emit(
                    "ack_timeout",
                    pending.fields,
                    duration_ms=(monotonic() - pending.sent_at) * 1000,
                )


def poll_start(device):
    if not enabled():
        return None
    ctx = Context({"poll_id": new_id()}, device)
    ctx.event("poll_due")
    ctx.event("poll_start")
    return ctx, monotonic()


def poll_done(observation, failed):
    if observation is None:
        return
    ctx, started = observation
    ctx.event(
        "poll_error" if failed else "poll_done",
        duration_ms=(monotonic() - started) * 1000,
    )
