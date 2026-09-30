"""Test-only virtual clock and measurements around the real gateway worker.

The worker remains a real thread. A fake select (and optional fake wire delay)
parks it on an Event; the event loop advances only while that worker is parked.
No socket is created. Device deadlines, asyncio timeouts and broker heartbeats
share the virtual clock; no production timer or scheduling code is replaced.
"""

import asyncio
import math
import queue
import threading
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from custom_components.tuya_local import device as device_module
from custom_components.tuya_local import gateway_broker as broker_module


@dataclass
class LockWait:
    task: str
    requested: float
    acquired: float | None = None
    released: float | None = None


class MeasuredLock:
    """Observe asyncio.Lock's actual FIFO wait/cancellation behavior."""

    def __init__(self, clock):
        self.clock = clock
        self.lock = asyncio.Lock()
        self.waits = []
        self.owner = None

    async def __aenter__(self):
        wait = LockWait(asyncio.current_task().get_name(), self.clock.now)
        self.waits.append(wait)
        await self.lock.acquire()
        assert self.owner is None
        self.owner = wait
        wait.acquired = self.clock.now
        return self

    async def __aexit__(self, *args):
        assert self.owner is not None
        self.owner.released = self.clock.now
        self.owner = None
        self.lock.release()

    def locked(self):
        return self.lock.locked()


class MeasuredQueue(queue.Queue):
    """Timestamp enqueue, execution, and future completion without reordering."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.calls = []
        self.maximum_depth = 0
        self.maximum_pending = 0

    def put(self, call, *args, **kwargs):
        record = {
            "task": asyncio.current_task().get_name(),
            "enqueued": self.clock.now,
            "started": None,
            "ended": None,
            "completed": None,
            "cancelled": None,
        }
        self.calls.append(record)
        original = call.func

        def execute():
            record["started"] = self.clock.now
            try:
                return original()
            finally:
                record["ended"] = self.clock.now

        def complete(future):
            record["completed"] = self.clock.now
            record["cancelled"] = future.cancelled()

        call.func = execute
        call.future.add_done_callback(complete)
        self.maximum_depth = max(self.maximum_depth, self.qsize() + 1)
        self.maximum_pending = max(
            self.maximum_pending, len(self.clock.broker._pending)
        )
        return super().put(call, *args, **kwargs)

    def get_nowait(self):
        if self.clock.handoff == "loop_first" and not self.clock.closing:
            # Let result callbacks release the gateway lock and enqueue the
            # next child before the worker checks the queue. No virtual time
            # elapses here: this is one legal OS scheduling order.
            gate = threading.Event()

            def ready():
                if self.clock.loop._ready or self.clock.due_timers():
                    self.clock.loop.call_soon(ready)
                else:
                    gate.set()

            self.clock.loop.call_soon_threadsafe(ready)
            assert gate.wait(10), "event-loop handoff did not settle"
        return super().get_nowait()


class GatewayClock:
    """Advance many gateway minutes deterministically, including 100 ms select."""

    def __init__(self, children, poll_delay=0, *, handoff=None):
        self.children = children
        self.loop = asyncio.get_running_loop()
        self.broker = children[0]._broker
        self.parent = children[0]._api.parent
        self.origin = self.loop.time()
        self.now = self.origin
        self.parked = None
        self.next_park = self.loop.create_future()
        self.closing = False
        self.patches = ExitStack()
        self.commands = []
        self.controls = []
        self.pushes = []
        self.pending_samples = []
        self.queue = MeasuredQueue(self)
        self.lock = MeasuredLock(self)
        self.watchdog = None
        self.scheduled = []
        self.debug = self.loop.get_debug()
        self.push_latencies = []
        self.push_times = {}
        self.poll_delay = poll_delay
        self.handoff = handoff
        self.deliveries = []
        self.polls = []

    @property
    def elapsed(self):
        return self.now - self.origin

    def wall_time(self):
        # Use one floating-point domain. Translating to a different-magnitude
        # wall clock can leave a positive residual deadline smaller than one
        # monotonic ULP, repeatedly scheduling an immediate asyncio timeout.
        return self.now

    async def __aenter__(self):
        self.loop.set_debug(False)
        self.patches.enter_context(patch.object(self.loop, "time", lambda: self.now))
        # asyncio may otherwise promote a timer slightly before its deadline.
        # With a frozen clock that can endlessly reschedule a not-yet-due poll.
        self.patches.enter_context(
            patch.object(self.loop, "_clock_resolution", math.ulp(self.now))
        )
        self.patches.enter_context(patch.object(device_module, "time", self.wall_time))
        self.patches.enter_context(
            patch.object(
                broker_module, "time", SimpleNamespace(monotonic=lambda: self.now)
            )
        )
        self.patches.enter_context(
            patch.object(broker_module.select, "select", self.select)
        )
        self.broker._calls = self.queue
        self.broker._last_heartbeat = self.now
        if self.handoff == "worker_first":
            complete = self.broker._set_future_result

            def after_worker_parks(future, result):
                # Exercise the other legal order: the worker checks its empty
                # queue and enters select before HA handles the result.
                if self.parked is None and not self.closing:
                    self.next_park.add_done_callback(lambda _: complete(future, result))
                else:
                    complete(future, result)

            self.patches.enter_context(
                patch.object(self.broker, "_set_future_result", after_worker_parks)
            )
        self.children[0]._gateway.lock = self.lock
        for child in self.children:
            child._api_lock = self.lock
            child._last_full_poll = self.wall_time()
            original = child._api.set_multiple_values

            def control(values, nowait, *, api=child._api, original=original):
                self.controls.append((api.cid, dict(values), self.elapsed))
                result = original(values, nowait)
                self.parent.received_wrong_cid_queue.append(
                    (api, {"dps": dict(values)})
                )
                return result

            self.patches.enter_context(
                patch.object(child._api, "set_multiple_values", control)
            )
            process_received = child._process_received

            def received(data, *, child=child, original=process_received):
                sent = self.push_times.get((child.dev_cid, data.get("2")))
                if sent is not None:
                    self.push_latencies.append(self.elapsed - sent)
                    self.deliveries.append((child.dev_cid, data["2"]))
                return original(data)

            self.patches.enter_context(
                patch.object(child, "_process_received", received)
            )
            status = child._api.status

            def delayed_status(*, original=status):
                poll = {"start": self.elapsed}
                self.polls.append(poll)
                if self.poll_delay:
                    self.park(self.poll_delay, "status")
                try:
                    return original()
                finally:
                    poll["end"] = self.elapsed

            self.patches.enter_context(
                patch.object(child._api, "status", delayed_status)
            )
        # A real-time guard also handles a broken test or orphaned worker future
        # when virtual time cannot advance. It never touches a real transport.
        task = asyncio.current_task()
        self.watchdog = threading.Timer(
            60, lambda: self.loop.call_soon_threadsafe(task.cancel)
        )
        self.watchdog.start()
        await self.settle()
        return self

    async def __aexit__(self, *args):
        self.closing = True
        self.watchdog.cancel()
        self.watchdog.join()
        for handle in self.scheduled:
            handle.cancel()
        if self.parked:
            self.wake()
        self.patches.close()
        for command in self.commands:
            if not command["task"].done():
                command["task"].cancel()
        await asyncio.gather(
            *(command["task"] for command in self.commands), return_exceptions=True
        )
        for child in self.children:
            await child.async_stop()
        self.loop.set_debug(self.debug)

    def park(self, delay, kind):
        """Called only by the worker thread; no busy-waiting or real I/O."""
        if self.closing:
            return
        gate = threading.Event()
        self.loop.call_soon_threadsafe(self.register_wait, delay, kind, gate)
        assert gate.wait(10), "virtual-time driver did not wake the worker"

    def register_wait(self, delay, kind, gate):
        if self.closing:
            gate.set()
            return
        assert self.parked is None
        timer = self.loop.call_later(delay, self.wake)
        self.parked = (gate, kind, timer)
        if not self.next_park.done():
            self.next_park.set_result(None)
        # Readiness is level-triggered. The producer may have run after the
        # worker's first check but before this registration reached the loop.
        if kind == "select" and self.parent.pushes:
            self.wake()

    def wake(self):
        if self.parked is None:
            return
        gate, _, timer = self.parked
        self.parked = None
        timer.cancel()
        self.next_park = self.loop.create_future()
        gate.set()

    def select(self, read, write, error, timeout):
        if not read[0].pushes:
            self.park(timeout, "select")
        return ([read[0]] if read[0].pushes else [], [], [])

    async def settle(self):
        # All worker wakeups and timer registrations belong to this loop. Once
        # parked AND the ready queue/due timers are empty, neither side can
        # progress until the driver advances time. A fixed number of loop turns
        # is insufficient for arbitrary callback chains and lock handoffs.
        while True:
            await asyncio.sleep(0)
            if self.parked is None:
                await asyncio.shield(self.next_park)
                continue
            if not self.loop._ready and not self.due_timers():
                return

    def due_timers(self):
        """The same due-time boundary used by asyncio's timer promotion."""
        return any(
            not handle.cancelled()
            and handle.when() < self.now + self.loop._clock_resolution
            for handle in self.loop._scheduled
        )

    async def advance_to(self, elapsed):
        target = self.origin + elapsed
        assert target >= self.now
        while self.now < target:
            await self.settle()
            assert self.parked is not None and not self.due_timers()
            deadlines = [
                handle.when()
                for handle in self.loop._scheduled
                if not handle.cancelled() and handle.when() > self.now
            ]
            self.now = min(target, min(deadlines, default=target))
            self.loop._clock_resolution = math.ulp(self.now)
            await self.settle()

    def push_latency_bound(self):
        """Finite burst: one select wait plus every blocking status ahead.

        There is one successful status per child and no retries in this model.
        Configure, control and heartbeat have zero modeled duration. A readable
        socket cannot incur further select waits. This is not a hardware SLA.
        """
        return self.broker._select_timeout + len(self.children) * self.poll_delay

    def assert_pushes_once(self):
        assert Counter(self.deliveries) == Counter(
            (cid, value) for cid, value, _ in self.pushes
        )
        # Tolerance is solely for floating-point virtual timestamp arithmetic.
        assert max(self.push_latencies, default=0) <= self.push_latency_bound() + 1e-6

    def align_polls(self, deadline):
        """Force a repeated worst-case burst; the separate soak never realigns."""
        for child in self.children:
            child._last_full_poll = self.origin + deadline - 30
            child._gateway_poll_event.set()

    def command_at(self, at, child, value, label):
        record = {"cid": child.dev_cid, "value": value, "label": label}

        async def send():
            record["submitted"] = self.elapsed
            await child.async_set_properties({"101": value})
            record["completed"] = self.elapsed

        def submit():
            record["task"] = self.loop.create_task(send(), name=label)
            self.commands.append(record)

        self.scheduled.append(self.loop.call_at(self.origin + at, submit))

    def push_at(self, at, child, value):
        def push():
            self.pushes.append((child.dev_cid, value, self.elapsed))
            self.push_times[child.dev_cid, value] = self.elapsed
            self.parent.pushes.append({"device": child._api, "dps": {"2": value}})
            if self.parked and self.parked[1] == "select":
                self.wake()

        self.scheduled.append(self.loop.call_at(self.origin + at, push))

    def assert_drained(self):
        assert not self.lock.locked()
        assert self.broker._calls.empty()
        self.pending_samples.append(len(self.broker._pending))
        assert not self.broker._pending
        assert all(
            w.acquired is None or w.released is not None for w in self.lock.waits
        )
        assert all(c["completed"] is not None for c in self.queue.calls)
        assert all(command["task"].done() for command in self.commands)
        assert all(command["task"].exception() is None for command in self.commands)

    def assert_commands_once(self, bound):
        expected = Counter((c["cid"], c["value"]) for c in self.commands)
        actual = Counter((cid, values.get("101")) for cid, values, _ in self.controls)
        assert actual == expected
        for command in self.commands:
            assert command["completed"] - command["submitted"] <= bound

    def summary(self):
        def maximum(values):
            return round(max(values, default=0), 6)

        return {
            "virtual_seconds": round(self.elapsed, 3),
            "status_response_delay": self.poll_delay,
            "handoff": self.handoff,
            "push_latency_bound": self.push_latency_bound(),
            "poll_timing": self.polls,
            "commands": len(self.commands),
            "control_calls": len(self.controls),
            "max_command_latency": maximum(
                c["completed"] - c["submitted"]
                for c in self.commands
                if "completed" in c
            ),
            "max_lock_wait": maximum(
                w.acquired - w.requested
                for w in self.lock.waits
                if w.acquired is not None
            ),
            "max_broker_queue_depth": self.queue.maximum_depth,
            "max_broker_pending": self.queue.maximum_pending,
            "max_enqueue_to_start": maximum(
                c["started"] - c["enqueued"]
                for c in self.queue.calls
                if c["started"] is not None
            ),
            "max_execution_duration": maximum(
                c["ended"] - c["started"]
                for c in self.queue.calls
                if c["ended"] is not None
            ),
            "settled_pending": self.pending_samples,
            "pushes_delivered": len(self.push_latencies),
            "max_push_latency": maximum(self.push_latencies),
            "heartbeats": sum(c[0] == "heartbeat" for c in self.parent.calls),
        }
