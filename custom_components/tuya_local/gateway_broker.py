"""
Single-owner transport broker for Tuya gateway sub-devices.

This module is intentionally side-effect free. Importing it does not open a
socket, start a thread, or modify Home Assistant state.

The broker owns all I/O for one TinyTuya gateway parent:
- exactly one persistent TCP socket
- exactly one receive owner
- heartbeat writes from the same worker thread
- child status / updatedps / control calls serialized through the same worker
- CID-routed push messages dispatched back onto the Home Assistant event loop

The first integration step will use this only for sub-devices. Standalone Wi-Fi
devices keep the existing tuya-local communication path.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import select
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from homeassistant.core import HomeAssistant

from . import trace

_LOGGER = logging.getLogger(__name__)

MessageCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class _BrokerCall:
    """A function that must execute on the gateway I/O thread."""

    func: Callable[[], Any]
    future: asyncio.Future[Any]
    observation: trace.BrokerCall | None = None


class GatewayBroker:
    """Own one TinyTuya gateway transport and serialize all socket I/O."""

    def __init__(
        self,
        hass: HomeAssistant,
        parent_api: Any,
        *,
        heartbeat_interval: float = 5.0,
        select_timeout: float = 0.1,
        reconnect_backoff: float = 1.0,
    ) -> None:
        self._hass = hass
        self._parent = parent_api
        self._heartbeat_interval = heartbeat_interval
        self._select_timeout = select_timeout
        self._reconnect_backoff = reconnect_backoff

        self._calls: queue.Queue[_BrokerCall] = queue.Queue()
        self._pending: set[asyncio.Future] = set()
        self._children: dict[int, MessageCallback] = {}
        self._children_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop_task: asyncio.Task | None = None
        self._next_connect_attempt = 0.0
        self._last_heartbeat = 0.0
        self.trace_slot = trace.new_id()
        self._trace_children = {}
        self.trace_gateway_members = 0
        self._trace_last_health = None
        self._trace_config_observed = False

    def trace_child_slot(self, child_api):
        with self._children_lock:
            return self._trace_children.get(id(child_api))

    def _trace_health(self):
        """Called on HA's loop so future/member counts have the same owner."""
        if trace.enabled():
            configuration = {}
            if not self._trace_config_observed:
                configuration = trace.socket_configuration(self)
                self._trace_config_observed = True
            trace.emit(
                "health",
                gateway_slot=self.trace_slot,
                broker_queue_depth=self._calls.qsize(),
                broker_pending_futures=len(self._pending),
                registered_children=self.child_count,
                gateway_members=self.trace_gateway_members,
                socket_present=getattr(self._parent, "socket", None) is not None,
                worker_alive=self.running,
                **configuration,
            )

    def _trace_activity(self):
        if not trace.enabled():
            return
        now = trace.monotonic()
        if self._trace_last_health is None or now - self._trace_last_health >= 60:
            self._trace_last_health = now
            self._hass.loop.call_soon_threadsafe(self._trace_health)

    @property
    def parent_api(self) -> Any:
        """Return the TinyTuya parent object owned by this broker."""
        return self._parent

    @property
    def running(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def child_count(self) -> int:
        """Return the number of registered child callbacks."""
        with self._children_lock:
            return len(self._children)

    def register_child(self, child_api: Any, callback: MessageCallback) -> None:
        """Register a child API object for CID-routed push delivery."""
        with self._children_lock:
            self._children[id(child_api)] = callback
            slot = self._trace_children.setdefault(id(child_api), trace.new_id())
        if trace.enabled():
            trace.emit("child_register", gateway_slot=self.trace_slot, child_slot=slot)

    def unregister_child(self, child_api: Any) -> None:
        """Remove a child from push delivery."""
        with self._children_lock:
            self._children.pop(id(child_api), None)
            slot = self._trace_children.pop(id(child_api), None)
        if slot is not None and trace.enabled():
            trace.emit(
                "child_unregister", gateway_slot=self.trace_slot, child_slot=slot
            )

    async def async_start(self) -> None:
        """Start the single gateway I/O owner."""
        if (
            self._stop_event.is_set()
            and self._thread is not None
            and self._stop_task is None
        ):
            await self.async_stop()
        if self._stop_task is not None:
            await asyncio.shield(self._stop_task)
            self._stop_task = None
        if self.running:
            return

        self._stop_event.clear()
        self._last_heartbeat = time.monotonic()
        self._next_connect_attempt = 0.0

        self._thread = threading.Thread(
            target=self._worker,
            name="tuya-local-gateway",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._stop_event.set()
            raise

    async def async_stop(self) -> None:
        """Stop the worker and close the persistent gateway connection."""
        if self._stop_task is None:
            self._stop_event.set()
            self._fail_queued_calls(RuntimeError("Gateway broker stopped"))
            self._stop_task = self._hass.loop.create_task(self._async_join())
        await asyncio.shield(self._stop_task)

    async def _async_join(self) -> None:
        """Never release ownership while the old thread can still do I/O."""
        thread = self._thread
        if thread is None:
            return

        await self._hass.async_add_executor_job(thread.join)
        self._thread = None
        self._trace_health()
        if trace.enabled():
            trace.emit("gateway_stop", gateway_slot=self.trace_slot)

    async def async_call(self, func: Callable[[], Any], *, trace_kind="call") -> Any:
        """Run one TinyTuya operation on the gateway I/O owner thread."""
        if not self.running or self._stop_event.is_set():
            raise RuntimeError("Gateway broker is not running")

        future = self._hass.loop.create_future()
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)
        observation = trace.enqueue(self, trace_kind)
        self._calls.put(_BrokerCall(func=func, future=future, observation=observation))
        return await future

    def _worker(self) -> None:
        """Own all gateway socket reads and writes."""
        if trace.enabled():
            trace.emit("gateway_start", gateway_slot=self.trace_slot)
        try:
            self._parent.set_socketRetryLimit(1)
            self._parent.set_socketPersistent(True)
            while not self._stop_event.is_set():
                self._drain_calls()
                self._trace_activity()
                if self._stop_event.is_set():
                    break

                self._ensure_connected()
                self._send_heartbeat_if_due()

                # TinyTuya may queue wrong-CID asynchronous updates while a
                # child status/control call is waiting for its own response.
                # Drain those cached messages before waiting on the socket.
                if self._drain_cached_messages():
                    continue

                sock = getattr(self._parent, "socket", None)
                if sock is None:
                    self._stop_event.wait(self._select_timeout)
                    continue

                try:
                    with trace.select_wait(self):
                        readable, _, _ = select.select(
                            [sock],
                            [],
                            [],
                            self._select_timeout,
                        )
                except OSError, ValueError:
                    self._disconnect("select failed")
                    continue

                if not readable:
                    continue

                try:
                    with trace.worker_phase(self, "receive") as observation:
                        data = self._parent.receive()
                        if observation is not None:
                            observation["outcome"] = trace.io_outcome(data)
                except Exception as exc:
                    _LOGGER.debug(
                        "Gateway receive failed: %s",
                        exc,
                        exc_info=True,
                    )
                    self._disconnect("receive failed")
                    continue

                self._dispatch(data)
        except Exception:
            if trace.enabled():
                trace.emit("worker_failure", gateway_slot=self.trace_slot)
            _LOGGER.exception("Gateway broker worker failed")
        finally:
            self._stop_event.set()
            try:
                self._parent.set_socketPersistent(False)
            except Exception:
                _LOGGER.debug(
                    "Error closing gateway socket on worker exit",
                    exc_info=True,
                )
            self._fail_queued_calls(RuntimeError("Gateway broker worker exited"))
            self._hass.loop.call_soon_threadsafe(self._worker_exited)

    def _worker_exited(self):
        for future in tuple(self._pending):
            self._set_future_exception(
                future, RuntimeError("Gateway broker worker exited")
            )

    def _drain_calls(self) -> None:
        """Execute pending status/control operations serially."""
        batch = None
        executed = 0
        try:
            while not self._stop_event.is_set():
                try:
                    call = self._calls.get_nowait()
                except queue.Empty:
                    return

                if call.future.cancelled():
                    if call.observation is not None:
                        trace.emit(
                            "broker_cancelled",
                            call.observation.fields,
                            outcome="cancelled",
                        )
                    continue

                if batch is None and trace.enabled():
                    batch = (trace.new_id(), trace.monotonic())
                    trace.emit(
                        "broker_drain_start",
                        gateway_slot=self.trace_slot,
                        batch_id=batch[0],
                    )
                executed += 1
                try:
                    with trace.execute(call.observation):
                        self._parent.set_socketPersistent(True)
                        result = call.func()
                except Exception as exc:
                    self._hass.loop.call_soon_threadsafe(
                        self._set_future_exception,
                        call.future,
                        exc,
                    )
                    # TinyTuya may have closed the parent socket after an error.
                    if getattr(self._parent, "socket", None) is None:
                        self._next_connect_attempt = 0.0
                else:
                    self._hass.loop.call_soon_threadsafe(
                        self._set_future_result,
                        call.future,
                        result,
                    )

                # A child call can cache async messages for another CID.
                self._drain_cached_messages()
                # A continuously replenished batch is activity too. Health
                # must not depend on eventually observing an empty queue.
                self._trace_activity()
        finally:
            if batch is not None and trace.enabled():
                trace.emit(
                    "broker_drain_done",
                    gateway_slot=self.trace_slot,
                    batch_id=batch[0],
                    calls_executed=executed,
                    duration_ms=(trace.monotonic() - batch[1]) * 1000,
                    queue_depth_after=self._calls.qsize(),
                )

    def _ensure_connected(self) -> None:
        """Open the parent persistent socket when needed."""
        if getattr(self._parent, "socket", None) is not None:
            return

        now = time.monotonic()
        if now < self._next_connect_attempt:
            return

        self._parent.set_socketPersistent(True)
        try:
            with trace.worker_phase(self, "connect") as observation:
                result = self._parent._get_socket(False)
                if observation is not None:
                    observation["outcome"] = (
                        "ok"
                        if result is True
                        and getattr(self._parent, "socket", None) is not None
                        else "error"
                    )
        except Exception:
            _LOGGER.debug("Gateway connection attempt failed", exc_info=True)
            result = None

        if result is True and getattr(self._parent, "socket", None) is not None:
            self._last_heartbeat = time.monotonic()
            self._next_connect_attempt = 0.0
            _LOGGER.debug("Gateway broker connected")
            if trace.enabled():
                trace.emit("reconnect", gateway_slot=self.trace_slot)
            return

        self._next_connect_attempt = now + self._reconnect_backoff

    def _send_heartbeat_if_due(self) -> None:
        """Send heartbeat from the same socket owner thread."""
        if getattr(self._parent, "socket", None) is None:
            return

        now = time.monotonic()
        if now - self._last_heartbeat < self._heartbeat_interval:
            return

        try:
            with trace.worker_phase(self, "heartbeat") as observation:
                result = self._parent.heartbeat(nowait=True)
                if observation is not None:
                    observation["outcome"] = trace.io_outcome(result)
        except Exception:
            _LOGGER.debug("Gateway heartbeat failed", exc_info=True)
            self._disconnect("heartbeat failed")
            return

        self._last_heartbeat = now
        if isinstance(result, dict) and result.get("Err"):
            _LOGGER.debug("Gateway heartbeat returned error: %s", result)
            self._disconnect("heartbeat returned error")

    def _drain_cached_messages(self) -> bool:
        """Dispatch TinyTuya's cached wrong-CID messages without rereading."""
        dispatched = False
        cached = getattr(self._parent, "received_wrong_cid_queue", None)

        while cached:
            item = cached.pop(0)

            # TinyTuya stores wrong-CID responses as
            # (child_device, processed_result). Calling parent.receive() here
            # would return that tuple rather than a normal result dict.
            if isinstance(item, tuple) and len(item) == 2:
                child_api, data = item
                if (
                    child_api is not None
                    and isinstance(data, dict)
                    and data.get("device") is None
                ):
                    data = dict(data)
                    data["device"] = child_api
            else:
                data = item

            if not data:
                continue

            dispatched = True
            self._dispatch(data, trace_source="cache")

        return dispatched

    def _dispatch(self, data: Any, *, trace_source="socket") -> None:
        """Dispatch one CID-routed TinyTuya result to its child."""
        if not isinstance(data, dict):
            return

        if data.get("Err"):
            _LOGGER.debug("Gateway receive returned error: %s", data)
            return

        child_api = data.get("device")
        if child_api is None:
            # Null heartbeat responses and gateway-level messages may not have
            # a child object. They are intentionally ignored here.
            return

        with self._children_lock:
            callback = self._children.get(id(child_api))
            slot = self._trace_children.get(id(child_api))

        observation = None
        if trace.enabled():
            observation = {
                "gateway_slot": self.trace_slot,
                "child_slot": slot,
                "push_id": trace.new_id(),
                "source": trace_source,
            }
            trace.emit("push_rx", observation)

        if callback is None:
            _LOGGER.debug(
                "Gateway push received for unregistered child %s",
                getattr(child_api, "cid", getattr(child_api, "id", "unknown")),
            )
            return

        self._hass.loop.call_soon_threadsafe(
            self._deliver, child_api, callback, data, observation
        )

    def _deliver(self, child_api, callback, data, observation=None) -> None:
        """Discard deliveries queued before a child was unregistered."""
        with self._children_lock:
            current = self._children.get(id(child_api))
        if current is callback and not self._stop_event.is_set():
            if observation is not None:
                trace.emit("push_dispatch", observation)
            callback(data)

    def _disconnect(self, reason: str) -> None:
        """Close a bad socket so the worker reconnects on the next cycle."""
        _LOGGER.debug("Gateway broker disconnecting: %s", reason)
        if trace.enabled():
            trace.emit("disconnect", gateway_slot=self.trace_slot)
        try:
            self._parent.set_socketPersistent(False)
        except Exception:
            _LOGGER.debug("Error closing gateway broker socket", exc_info=True)

        cached = getattr(self._parent, "received_wrong_cid_queue", None)
        if cached is not None:
            cached.clear()

        self._next_connect_attempt = time.monotonic() + self._reconnect_backoff

    def _fail_queued_calls(self, exc: Exception) -> None:
        """Fail any calls that were never executed."""
        while True:
            try:
                call = self._calls.get_nowait()
            except queue.Empty:
                return

            if call.observation is not None:
                trace.emit(
                    "broker_cancelled", call.observation.fields, outcome="stopped"
                )
            self._hass.loop.call_soon_threadsafe(
                self._set_future_exception,
                call.future,
                exc,
            )

    @staticmethod
    def _set_future_result(
        future: asyncio.Future[Any],
        result: Any,
    ) -> None:
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _set_future_exception(
        future: asyncio.Future[Any],
        exc: Exception,
    ) -> None:
        if not future.done():
            future.set_exception(exc)
