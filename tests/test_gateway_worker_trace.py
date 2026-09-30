"""Observe the real broker loop with deterministic, entirely local fake I/O."""

import json
import logging
from types import SimpleNamespace

import pytest

from custom_components.tuya_local import gateway_broker, trace
from custom_components.tuya_local.gateway_broker import GatewayBroker
from tools.analyze_gateway_trace import analyze, parse

from .test_gateway_trace import assert_schema, records


@pytest.fixture
def owner(mocker, caplog):
    caplog.set_level(logging.DEBUG, logger=trace._LOGGER.name)
    clock = SimpleNamespace(now=100.0)
    mocker.patch.object(trace, "monotonic", side_effect=lambda: clock.now)
    mocker.patch.object(trace, "_ORIGIN", 100.0)
    mocker.patch.object(
        gateway_broker, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    parent = SimpleNamespace(
        socket=object(),
        connection_timeout=5,
        socketRetryLimit=5,
        socketRetryDelay=5,
        received_wrong_cid_queue=[],
    )
    calls = []

    def persistent(value):
        calls.append(("persistent", value))
        if not value:
            parent.socket = None

    def retry_limit(value):
        calls.append(("retry_limit", value))
        parent.socketRetryLimit = value

    parent.set_socketPersistent = mocker.Mock(side_effect=persistent)
    parent.set_socketRetryLimit = mocker.Mock(side_effect=retry_limit)
    parent._get_socket = mocker.Mock()
    parent.heartbeat = mocker.Mock()
    parent.receive = mocker.Mock()
    broker = GatewayBroker(SimpleNamespace(loop=mocker.Mock()), parent)
    broker._last_heartbeat = clock.now
    select = mocker.patch.object(gateway_broker.select, "select")

    def select_once(delay=0.1, readable=False, error=False):
        def selected(read, write, exceptional, timeout):
            calls.append(("select", timeout))
            assert read == [parent.socket] and write == exceptional == []
            clock.now += delay
            broker._stop_event.set()
            if error:
                raise OSError("PRIVATE_SELECT_EXCEPTION")
            return (read if readable else [], [], [])

        select.side_effect = selected

    return SimpleNamespace(
        clock=clock,
        parent=parent,
        broker=broker,
        select=select,
        select_once=select_once,
        calls=calls,
    )


@pytest.mark.parametrize("phase", ["connect", "heartbeat", "receive"])
@pytest.mark.parametrize("result_kind", ["ok", "error_result", "exception"])
def test_five_second_owner_phase_timing_outcome_and_privacy(
    owner, phase, result_kind, caplog
):
    parent, broker, clock = owner.parent, owner.broker, owner.clock

    def operation(*args, **kwargs):
        clock.now += 5
        if result_kind == "exception":
            raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")
        if result_kind == "error_result":
            return {"Err": "PRIVATE_ERROR_SENTINEL", "payload": "PRIVATE_PAYLOAD"}
        if phase == "connect":
            parent.socket = object()
            return True
        return {"payload": "PRIVATE_SUCCESS_SENTINEL"}

    owner.select_once(readable=phase == "receive")
    if phase == "connect":
        parent.socket = None
        parent._get_socket.side_effect = operation
        broker._ensure_connected()
        parent._get_socket.assert_called_once_with(False)
        assert broker._next_connect_attempt == (0 if result_kind == "ok" else 101)
        assert broker._last_heartbeat == (105 if result_kind == "ok" else 100)
    elif phase == "heartbeat":
        broker._last_heartbeat = 0
        parent.heartbeat.side_effect = operation
        broker._send_heartbeat_if_due()
        parent.heartbeat.assert_called_once_with(nowait=True)
        assert broker._last_heartbeat == (0 if result_kind == "exception" else 100)
        assert broker._next_connect_attempt == (0 if result_kind == "ok" else 106)
    else:
        parent.receive.side_effect = operation
        broker._worker()
        parent.receive.assert_called_once_with()
        parent._get_socket.assert_not_called()
        parent.heartbeat.assert_not_called()
        # Returned Err follows the existing dispatch path, not disconnect.
        assert broker._next_connect_attempt == (
            106.1 if result_kind == "exception" else 0
        )

    events = records(caplog)
    phase_events = [e for e in events if e["event"].startswith(phase + "_")]
    assert [e["event"] for e in phase_events] == [phase + "_start", phase + "_done"]
    assert phase_events[0]["phase_id"] == phase_events[1]["phase_id"]
    assert phase_events[1]["duration_ms"] == 5000
    assert phase_events[1]["outcome"] == ("ok" if result_kind == "ok" else "error")
    assert phase_events[1]["socket_present"] == (
        phase != "connect" or result_kind == "ok"
    )
    assert "PRIVATE_" not in json.dumps(events)
    assert_schema(events)
    assert parse("GBTRACE " + json.dumps(e) for e in events) == events


@pytest.mark.parametrize(
    "delay,expected", [(0.1, False), (0.25, False), (0.3, True), (5, True)]
)
@pytest.mark.parametrize("raises", [False, True])
def test_select_emits_only_slow_observations(owner, caplog, delay, expected, raises):
    owner.select_once(delay, error=raises)
    owner.broker._worker()
    events = records(caplog)
    slow = [e for e in events if e["event"] == "select_slow"]
    assert bool(slow) == expected
    if expected:
        assert slow[0]["duration_ms"] == pytest.approx(delay * 1000)
        assert slow[0]["start_ts_ms"] == 0
        assert slow[0]["socket_present"]
    owner.parent.receive.assert_not_called()
    assert "PRIVATE_" not in json.dumps(events)
    assert_schema(events)


def test_select_threshold_respects_longer_configured_timeout(owner, caplog):
    owner.broker._select_timeout = 0.5
    owner.select_once(0.6)
    owner.broker._worker()
    assert not any(e["event"] == "select_slow" for e in records(caplog))


def test_no_connect_or_heartbeat_observation_when_skipped(owner, caplog):
    owner.broker._ensure_connected()  # Already connected.
    owner.broker._send_heartbeat_if_due()  # Not due.
    owner.parent.socket = None
    owner.broker._next_connect_attempt = 101
    owner.broker._ensure_connected()  # Backoff still in force.
    owner.broker._send_heartbeat_if_due()  # No socket.
    assert not records(caplog)
    owner.parent._get_socket.assert_not_called()
    owner.parent.heartbeat.assert_not_called()


@pytest.mark.parametrize("phase", ["heartbeat", "receive"])
def test_none_return_is_normal_completion_not_a_claim_of_device_success(
    owner, caplog, phase
):
    if phase == "heartbeat":
        owner.broker._last_heartbeat = 0
        owner.parent.heartbeat.return_value = None
        owner.broker._send_heartbeat_if_due()
    else:
        owner.parent.receive.return_value = None
        owner.select_once(readable=True)
        owner.broker._worker()
    done = next(e for e in records(caplog) if e["event"] == phase + "_done")
    assert done["outcome"] == "ok"


@pytest.mark.parametrize(
    "logging_mode", ["enabled", "disabled", "broken_sink", "broken_encoder"]
)
def test_instrumentation_preserves_worker_io_and_settings(
    owner, mocker, caplog, logging_mode
):
    if logging_mode == "disabled":
        caplog.set_level(logging.WARNING, logger=trace._LOGGER.name)
        mocker.patch.object(
            trace, "monotonic", side_effect=AssertionError("disabled timer")
        )
        mocker.patch.object(
            trace, "new_id", side_effect=AssertionError("disabled allocation")
        )
    elif logging_mode == "broken_sink":
        mocker.patch.object(
            trace._LOGGER, "debug", side_effect=RuntimeError("PRIVATE_SINK")
        )
    elif logging_mode == "broken_encoder":
        mocker.patch.object(
            trace,
            "json",
            SimpleNamespace(
                dumps=mocker.Mock(side_effect=RuntimeError("PRIVATE_ENCODER"))
            ),
        )
    owner.broker._last_heartbeat = 0
    owner.parent.heartbeat.return_value = None
    owner.parent.receive.return_value = None
    owner.select_once(readable=True)
    owner.broker._worker()
    assert owner.calls == [
        ("retry_limit", 1),
        ("persistent", True),
        ("select", 0.1),
        ("persistent", False),
    ]
    owner.parent._get_socket.assert_not_called()
    owner.parent.heartbeat.assert_called_once_with(nowait=True)
    owner.parent.receive.assert_called_once_with()
    assert owner.parent.connection_timeout == 5
    assert owner.parent.socketRetryLimit == 1  # The pre-existing worker setting.
    assert owner.parent.socketRetryDelay == 5
    assert owner.broker._heartbeat_interval == 5
    assert owner.broker._select_timeout == 0.1
    assert owner.broker._reconnect_backoff == 1
    assert not any(e["event"] == "worker_failure" for e in records(caplog))


def test_broken_sink_cannot_change_connect_success(owner, mocker):
    owner.parent.socket = None

    def connect(renew):
        assert renew is False
        owner.clock.now += 5
        owner.parent.socket = object()
        return True

    owner.parent._get_socket.side_effect = connect
    mocker.patch.object(
        trace._LOGGER, "debug", side_effect=RuntimeError("PRIVATE_SINK")
    )
    owner.broker._ensure_connected()
    owner.parent._get_socket.assert_called_once_with(False)
    assert owner.parent.socket is not None
    assert owner.broker._last_heartbeat == 105
    assert owner.broker._next_connect_attempt == 0


def test_first_health_observes_numeric_configuration_without_api_calls(owner, caplog):
    owner.select_once()
    owner.broker._worker()
    before = list(owner.calls)
    owner.broker._trace_health()
    owner.broker._trace_health()
    health = [e for e in records(caplog) if e["event"] == "health"]
    expected = {
        "socket_timeout_ms": 5000,
        "socket_retry_limit": 1,
        "socket_retry_delay_ms": 5000,
        "heartbeat_interval_ms": 5000,
        "select_timeout_ms": 100,
    }
    assert {k: health[0][k] for k in expected} == expected
    assert not expected.keys() & health[1].keys()
    assert owner.calls == before
    assert_schema(health)
    result = analyze("GBTRACE " + json.dumps(e) for e in health)
    snapshot = result["worker_phases"]["configuration_snapshots"][0]
    assert {k: snapshot[k] for k in expected} == expected


@pytest.mark.parametrize(
    "bad", ["PRIVATE_CONFIG_SENTINEL", None, True, -1, float("nan"), float("inf")]
)
def test_config_snapshot_omits_non_numeric_or_invalid_values(owner, caplog, bad):
    owner.parent.connection_timeout = bad
    owner.parent.socketRetryLimit = bad
    owner.parent.socketRetryDelay = bad
    owner.broker._trace_health()
    event = records(caplog)[0]
    assert (
        not {"socket_timeout_ms", "socket_retry_limit", "socket_retry_delay_ms"}
        & event.keys()
    )
    assert "PRIVATE_" not in json.dumps(event)
