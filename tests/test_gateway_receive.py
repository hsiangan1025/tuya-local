"""Exercise installed TinyTuya receive logic without a network or wall-clock sleeps."""

import asyncio
import socket
import struct
import threading
from collections import deque
from importlib import import_module
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from tinytuya import Device
from tinytuya.core import (
    HEART_BEAT,
    STATUS,
    AESCipher,
    TuyaMessage,
    pack_message,
    parse_header,
)
from tinytuya.core.Monitor import Monitor
from tinytuya.core.XenonDevice import XenonDevice

from custom_components.tuya_local import gateway_broker
from custom_components.tuya_local.gateway import GatewayConnection
from custom_components.tuya_local.gateway_broker import GatewayBroker, _BrokerCall


def reply(payload=b"", *, command=HEART_BEAT, protocol=3.3):
    """Build synthetic replies with TinyTuya's own encryption/frame packer."""
    key = b"synthetic-key-01"
    if protocol == 3.5:
        return pack_message(
            TuyaMessage(1, command, 0, payload, 0, True, 0x6699, b"test-nonce01"),
            hmac_key=key,
        )
    if payload:
        header = str(protocol).encode() + b"\0" * 12
        cipher = AESCipher(key)
        payload = (
            header + cipher.encrypt(payload, use_base64=False)
            if protocol == 3.3
            else cipher.encrypt(header + payload, use_base64=False)
        )
    return pack_message(
        TuyaMessage(1, command, 0, b"\0" * 4 + payload, 0),
        hmac_key=key if protocol == 3.4 else None,
    )


class WireSocket:
    """Socket bytes/time only: TinyTuya does all framing and retry decisions."""

    def __init__(self, parent, chunks=()):
        self.parent = parent
        self.chunks = deque(chunks)
        self.now = 100.0
        self.reads = []
        self.sends = []
        self.closes = []
        self.on_send = lambda command: None
        self.before_recv = lambda: None

    def advance(self, seconds):
        self.now += seconds

    def recv(self, length):
        self.reads.append((length, self.parent.retry, threading.current_thread().name))
        self.before_recv()
        if not self.chunks:
            self.advance(self.parent.connection_timeout)
            raise socket.timeout()
        data = self.chunks.popleft()
        if isinstance(data, BaseException):
            raise data
        if len(data) > length:
            self.chunks.appendleft(data[length:])
        return data[:length]

    def sendall(self, data):
        command = parse_header(data).cmd
        self.sends.append((command, self.parent.retry))
        self.on_send(command)

    def close(self):
        self.closes.append(self.parent.retry)


@pytest.fixture
def real_parent(mocker):
    """Keep all protocol methods real; only the socket is synthetic."""
    assert version("tinytuya") == "1.20.0", "Recheck receive semantics on upgrades"
    xenon = import_module("tinytuya.core.XenonDevice")
    socket_module = SimpleNamespace(**vars(socket))
    socket_module.socket = mocker.Mock(side_effect=AssertionError("No real network"))
    mocker.patch.object(xenon, "socket", socket_module)
    parent = Device(
        "synthetic-gateway",
        "gateway.invalid",
        "synthetic-key-01",
        version=3.3,
        persist=True,
        connection_retry_limit=1,
    )
    parent.socket = mocker.Mock()
    yield parent
    parent.close()


@pytest.mark.parametrize("retry,reads", [(True, 2), (False, 1)])
async def test_real_receive_empty_frame_then_timeout(real_parent, mocker, retry, reads):
    """The unmodified receive/_send_receive loop decides whether to read twice."""
    parent = real_parent
    assert Device.receive is XenonDevice.receive
    assert Device._send_receive is XenonDevice._send_receive
    parent.set_retry(retry)
    empty = TuyaMessage(1, HEART_BEAT, 0, b"", 0)
    low_receive = mocker.patch.object(
        parent, "_receive", side_effect=[empty, socket.timeout()]
    )
    send_receive = mocker.spy(parent, "_send_receive")
    original_socket = parent.socket

    assert parent.receive() is None

    send_receive.assert_called_once_with(None)
    assert low_receive.call_count == reads
    assert parent.raw_recv == [empty]
    assert parent.socket is original_socket
    assert parent.retry is retry
    assert parent.socketRetryLimit == 1
    parent.socket.sendall.assert_not_called()


def wire_clock(parent, mocker, chunks=()):
    wire = WireSocket(parent, chunks)
    parent.socket = wire
    xenon = import_module("tinytuya.core.XenonDevice")
    # Replace this module's clock reference, never asyncio's real clock.
    mocker.patch.object(
        xenon, "time", SimpleNamespace(time=lambda: 0, sleep=wire.advance)
    )
    mocker.patch.object(
        gateway_broker, "time", SimpleNamespace(monotonic=lambda: wire.now)
    )
    return wire


@pytest.mark.parametrize(
    "suppress_retry,reads,wait", [(False, 2, 5.01), (True, 1, 0.01)]
)
async def test_heartbeat_reply_yields_to_queued_call(
    real_parent, mocker, suppress_retry, reads, wait
):
    """Baseline keeps retry enabled; candidate uses the real temporary setter."""
    parent = real_parent
    wire = wire_clock(parent, mocker)
    broker = GatewayBroker(SimpleNamespace(loop=mocker.Mock()), parent)
    completed = []
    enqueued = []
    future = mocker.Mock()
    future.cancelled.return_value = False

    def command():
        completed.append((wire.now, parent.retry))
        broker._stop_event.set()

    def heartbeat_sent(command_type):
        assert command_type == HEART_BEAT
        assert parent.retry is True
        wire.chunks.append(reply())
        enqueued.append(wire.now)
        broker._calls.put(_BrokerCall(command, future))

    wire.on_send = heartbeat_sent
    if not suppress_retry:
        # Emulate the old selector receive's retry=True, not TinyTuya's loop.
        mocker.patch.object(parent, "set_retry", return_value=None)
    receive = mocker.spy(parent, "_receive")
    heartbeat = mocker.spy(parent, "heartbeat")
    disconnect = mocker.spy(broker, "_disconnect")

    def readable(read, write, error, timeout):
        assert timeout == 0.1 and wire.chunks
        return read, [], []

    selected = mocker.patch.object(
        gateway_broker.select, "select", side_effect=readable
    )
    broker._worker()

    assert receive.call_count == reads
    assert len(wire.reads) == reads
    assert len(completed) == len(enqueued) == 1
    assert completed[0][0] - enqueued[0] == pytest.approx(wait)
    assert completed[0][1] is True
    assert parent.retry is True
    heartbeat.assert_called_once_with(nowait=True)
    selected.assert_called_once()
    disconnect.assert_not_called()
    assert wire.closes == [True]  # Restore before worker-finally closes.
    assert (
        parent.connection_timeout,
        parent.socketRetryLimit,
        parent.socketRetryDelay,
    ) == (5, 1, 5)
    assert (
        broker._heartbeat_interval,
        broker._select_timeout,
        broker._reconnect_backoff,
    ) == (5, 0.1, 1)


@pytest.mark.parametrize("cancel_stop_waiter", [False, True])
async def test_stop_and_cancel_wait_for_receive_restore(
    hass, real_parent, mocker, cancel_stop_waiter
):
    parent = real_parent
    wire = wire_clock(parent, mocker, [reply()])
    broker = GatewayBroker(hass, parent)
    entered, release = threading.Event(), threading.Event()

    def hold_receive():
        assert parent.retry is False
        entered.set()
        assert release.wait(3), "test failed to release worker"

    wire.before_recv = hold_receive
    mocker.patch.object(gateway_broker.select, "select", return_value=([wire], [], []))
    executed = mocker.Mock()
    queued = stop = None
    try:
        await broker.async_start()
        await wait_thread_event(entered)
        queued = asyncio.create_task(broker.async_call(executed))
        await asyncio.sleep(0)
        assert broker._calls.qsize() == 1
        # Cancelling a caller cannot inject cancellation into the I/O thread.
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        stop = asyncio.create_task(broker.async_stop())
        await asyncio.sleep(0)
        assert not stop.done() and broker.running
        if cancel_stop_waiter:
            stop.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stop
        release.set()
        await asyncio.wait_for(broker.async_stop(), 3)
    finally:
        release.set()
        await broker.async_stop()
        await asyncio.gather(*(t for t in (queued, stop) if t), return_exceptions=True)
    executed.assert_not_called()
    assert parent.retry is True
    assert wire.closes == [True]
    assert not broker.running and not broker._pending


async def test_decode_error_restores_retry_before_existing_reconnect(
    real_parent, mocker
):
    parent = real_parent
    broken = struct.pack(">4I", 0x55AA, 1, STATUS, 0xFFFFFFFF) + b"\0" * 12
    wire = wire_clock(parent, mocker, [broken])
    replacement = WireSocket(parent)
    broker = GatewayBroker(SimpleNamespace(loop=mocker.Mock()), parent)
    broker._last_heartbeat = wire.now
    original_get_socket = parent._get_socket
    reconnect_settings = []

    def get_socket(renew):
        if parent.socket is not None:
            return original_get_socket(renew)
        reconnect_settings.append((parent.retry, parent.socketRetryLimit))
        parent.socket = replacement
        return True

    mocker.patch.object(parent, "_get_socket", side_effect=get_socket)
    disconnect = mocker.spy(broker, "_disconnect")

    def selected(read, write, error, timeout):
        assert timeout == 0.1
        if read == [wire]:
            return read, [], []
        broker._stop_event.set()
        return [], [], []

    mocker.patch.object(gateway_broker.select, "select", side_effect=selected)
    broker._worker()
    assert reconnect_settings == [(True, 1)]
    assert parent.retry is True
    assert replacement.closes == [True]
    assert broker._next_connect_attempt == 0
    disconnect.assert_not_called()  # TinyTuya closes on DecodeError; existing loop reconnects.


@pytest.mark.parametrize("partial", [False, True])
async def test_real_monitor_buffers_without_a_second_blocking_read(
    real_parent, mocker, partial
):
    """Review Monitor's actual alternative without starting its reactor/sockets."""
    packet = reply(b'{"dps":{"1":true}}', command=STATUS) if partial else reply()
    wire = wire_clock(
        real_parent, mocker, [packet[:7], packet[7:]] if partial else [packet]
    )
    monitor = Monitor.__new__(Monitor)
    state = SimpleNamespace(device=real_parent, recv_buffer=b"")
    delivered = mocker.patch.object(monitor, "_fire_status")
    mocker.patch.object(
        monitor,
        "_handle_disconnect",
        side_effect=AssertionError("unexpected disconnect"),
    )
    monitor._handle_readable(state)
    assert len(wire.reads) == 1 and wire.now == 100
    delivered.assert_not_called()
    if partial:
        assert state.recv_buffer == packet[:7]
        monitor._handle_readable(state)
        assert len(wire.reads) == 2 and wire.now == 100
        delivered.assert_called_once_with(state, {"dps": {"1": True}})
    assert state.recv_buffer == b""


@pytest.mark.parametrize("original_retry", [True, False])
@pytest.mark.parametrize("finish", ["data", "none", "exception", "cancelled"])
async def test_retry_restored_before_dispatch_disconnect_and_worker_exit(
    real_parent, mocker, original_retry, finish
):
    parent = real_parent
    parent.set_retry(original_retry)
    broker = GatewayBroker(SimpleNamespace(loop=mocker.Mock()), parent)
    wire = wire_clock(parent, mocker)
    broker._last_heartbeat = wire.now
    result = {"dps": {"1": True}} if finish == "data" else None

    def receive():
        assert parent.retry is False
        if finish == "exception":
            raise RuntimeError("synthetic receive failure")
        if finish == "cancelled":
            raise asyncio.CancelledError
        return result

    mocker.patch.object(parent, "receive", side_effect=receive)

    def readable(*args):
        broker._stop_event.set()  # Shutdown requested while select becomes ready.
        return [wire], [], []

    mocker.patch.object(gateway_broker.select, "select", side_effect=readable)

    def restored(*args):
        assert parent.retry is original_retry

    mocker.patch.object(broker, "_dispatch", side_effect=restored)
    original_disconnect = broker._disconnect

    def disconnect(reason):
        restored()
        original_disconnect(reason)

    mocker.patch.object(broker, "_disconnect", side_effect=disconnect)
    if finish == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            broker._worker()
    else:
        broker._worker()
    assert parent.retry is original_retry
    assert wire.closes and all(value is original_retry for value in wire.closes)


@pytest.mark.parametrize("retry", [True, False])
@pytest.mark.parametrize(
    "kind",
    ["split_complete", "truncated_header", "truncated_body", "bad_header", "bad_json"],
)
async def test_real_frame_boundary_and_decode_errors(real_parent, mocker, retry, kind):
    """Disabling null-frame retry is not a nonblocking framing parser."""
    parent = real_parent
    parent.set_retry(retry)
    valid = reply(b'{"dps":{"1":true}}', command=STATUS)
    if kind == "split_complete":
        chunks = [valid[:7], valid[7:]]
    elif kind.startswith("truncated"):
        chunks = [valid[:7] if kind == "truncated_header" else valid[:28]]
    elif kind == "bad_header":
        chunks = [struct.pack(">4I", 0x55AA, 1, STATUS, 0xFFFFFFFF) + b"\0" * 12, valid]
    else:
        chunks = [reply(b"not-json", command=STATUS)]
    wire = wire_clock(parent, mocker, chunks)
    receive = mocker.spy(parent, "_receive")
    result = parent.receive()
    assert parent.retry is retry
    if kind == "split_complete" or (kind == "bad_header" and retry):
        assert result == {"dps": {"1": True}}
        assert not wire.closes
        assert receive.call_count == (2 if kind == "bad_header" else 1)
    elif kind.startswith("truncated"):
        assert result is None
        assert wire.now == 105  # Even retry=False can block inside the FIRST frame.
        assert len(wire.reads) == 2 and receive.call_count == 1
        assert parent.socket is wire  # Existing receive-timeout behavior persists.
    else:
        assert result["Err"] == ("900" if kind == "bad_json" else "904")
        assert receive.call_count == 1
        assert bool(wire.closes) == (kind == "bad_header")
        if kind == "bad_header":
            assert parent.socket is None
            assert list(wire.chunks) == [valid]  # No extra frame on a damaged stream.


async def wait_thread_event(event):
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("protocol", [3.3, 3.4, 3.5])
async def test_real_frames_after_empty_ack_remain_for_next_selector_iteration(
    real_parent, mocker, protocol
):
    parent = real_parent
    parent.set_version(protocol)
    packet = reply(protocol=protocol) + reply(
        b'{"dps":{"1":true}}', command=STATUS, protocol=protocol
    )
    wire = wire_clock(parent, mocker, [packet])
    broker = GatewayBroker(SimpleNamespace(loop=mocker.Mock()), parent)
    broker._last_heartbeat = wire.now
    received = []
    turns = []
    future = mocker.Mock()
    future.cancelled.return_value = False

    def dispatch(result):
        assert parent.retry is True
        received.append(result)
        if len(received) == 1:
            broker._calls.put(_BrokerCall(lambda: turns.append("queued call"), future))
        else:
            broker._stop_event.set()

    def selected(read, write, error, timeout):
        turns.append("select")
        assert wire.chunks
        return read, [], []

    mocker.patch.object(broker, "_dispatch", side_effect=dispatch)
    mocker.patch.object(gateway_broker.select, "select", side_effect=selected)
    receive = mocker.spy(parent, "_receive")
    broker._worker()
    assert received == [None, {"dps": {"1": True}}]
    assert turns == ["select", "queued call", "select"]
    assert receive.call_count == 2  # One complete frame per selector invocation.
    assert wire.now == 100  # No socket timeout or real sleep.
    assert not wire.chunks and parent.retry is True


@pytest.mark.parametrize("retry", [True, False])
async def test_bad_gcm_frame_is_not_dispatched_as_valid_data(
    real_parent, mocker, retry
):
    parent = real_parent
    parent.set_version(3.5)
    parent.set_retry(retry)
    valid = reply(b'{"dps":{"1":true}}', command=STATUS, protocol=3.5)
    corrupt = bytearray(valid)
    corrupt[-5] ^= 1  # GCM tag, not the framing prefix/header.
    wire = wire_clock(parent, mocker, [bytes(corrupt), valid])
    receive = mocker.spy(parent, "_receive")
    result = parent.receive()
    if retry:
        assert result == {"dps": {"1": True}}
        assert receive.call_count == 2 and not wire.closes
    else:
        assert result["Err"] == "904"
        assert receive.call_count == 1 and parent.socket is None
        assert list(wire.chunks) == [valid]


async def test_owner_serializes_real_status_control_and_configure_after_receive(
    hass, real_parent, mocker
):
    parent = real_parent
    wire = wire_clock(parent, mocker, [reply()])
    gateway = GatewayConnection(hass, parent, "gateway.invalid", "synthetic-key-01")
    gateway.protocol = 3.3
    child = Device("synthetic-child", cid="synthetic-child", parent=parent)
    gateway.apis[child.cid] = child
    broker = gateway.broker
    entered, release = threading.Event(), threading.Event()
    operations = []

    def before_recv():
        if not entered.is_set():
            assert parent.retry is False
            entered.set()
            assert release.wait(3), "test failed to release worker"

    wire.before_recv = before_recv

    def selected(read, write, error, timeout):
        assert timeout == 0.1
        if wire.chunks:
            return read, [], []
        broker._stop_event.wait(0.001)
        return [], [], []

    mocker.patch.object(gateway_broker.select, "select", side_effect=selected)
    status_reads = []

    def operation(kind):
        assert parent.retry is True
        assert threading.current_thread().name == "tuya-local-gateway"
        operations.append(kind)
        if kind == "status":
            before = len(wire.reads)
            wire.chunks.extend(
                [
                    reply(),
                    reply(
                        b'{"cid":"synthetic-child","dps":{"1":true}}', command=STATUS
                    ),
                ]
            )
            result = child.status()
            status_reads.append(len(wire.reads) - before)
            return result
        if kind == "control":
            return child.set_multiple_values({"1": False}, nowait=True)
        if kind == "updatedps":
            return child.updatedps([1], nowait=True)
        gateway.configure("gateway.invalid", "synthetic-key-01", 3.3)
        return parent.retry

    tasks = []
    try:
        await broker.async_start()
        await wait_thread_event(entered)
        for kind in ("status", "control", "updatedps", "configure"):
            tasks.append(
                asyncio.create_task(broker.async_call(lambda k=kind: operation(k)))
            )
        await asyncio.sleep(0)
        assert broker._calls.qsize() == 4
        assert not operations and not any(t.done() for t in tasks)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
        assert results[0]["dps"] == {"1": True}
        # status still skips an empty ACK and reads its following full reply.
        assert status_reads == [3]  # Two frames; full response requires header+body.
        assert results[-1] is True
        assert operations == ["status", "control", "updatedps", "configure"]
        assert all(retry is True for _, retry in wire.sends)
        assert all(thread == "tuya-local-gateway" for _, _, thread in wire.reads)
    finally:
        release.set()
        await broker.async_stop()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert parent.retry is True
