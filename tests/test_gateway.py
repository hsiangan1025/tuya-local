"""Gateway integration regressions with synthetic devices and no network I/O."""

import asyncio
import hmac
import threading
from collections import deque
from types import SimpleNamespace

import pytest
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.exceptions import ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tinytuya import Device as TinyTuyaDevice
from tinytuya.core import (
    DP_QUERY_NEW,
    SESS_KEY_NEG_FINISH,
    SESS_KEY_NEG_RESP,
    SESS_KEY_NEG_START,
    AESCipher,
    TuyaMessage,
)

from custom_components.tuya_local import async_setup_entry
from custom_components.tuya_local.config_flow import async_test_connection
from custom_components.tuya_local.const import API_PROTOCOL_VERSIONS, DOMAIN
from custom_components.tuya_local.device import (
    TuyaLocalDevice,
    async_delete_device,
    setup_device,
)
from custom_components.tuya_local.gateway import get_gateway_registry
from custom_components.tuya_local.gateway_broker import GatewayBroker


class FakeAPI:
    """Model TinyTuya's child routing and assert the actual calling thread."""

    instances = []

    def __init__(
        self,
        dev_id,
        address=None,
        local_key=None,
        *,
        cid=None,
        parent=None,
        version=3.1,
        connection_retry_limit=5,
    ):
        self.id = dev_id
        self.cid = cid
        self.parent = parent
        self.address = address
        self.local_key = (local_key or "").encode()
        self.real_local_key = self.local_key
        self.children = {}
        self.received_wrong_cid_queue = []
        self.socket = None
        self.version = parent.version if parent else version
        self.disabledetect = True
        self.calls = []
        self.pushes = deque()
        self.responses = deque()
        self.dps_used = {}
        self.instances.append(self)
        if parent:
            parent.children[cid] = self

    def record(self, method, *args):
        thread = threading.current_thread()
        assert thread.name == "tuya-local-gateway", (method, thread.name)
        self.calls.append((method, args, thread))

    def set_socketRetryLimit(self, count):
        self.record("retry_limit", count)

    def set_socketPersistent(self, value):
        self.record("persistent", value)
        if not value:
            self.socket = None

    def set_version(self, version):
        self.record("version", version)
        self.version = version

    def set_dpsUsed(self, dps):
        self.record("dps_used", dps)
        self.dps_used = dps

    def _get_socket(self, renew):
        self.record("connect")
        self.socket = self
        return True

    def heartbeat(self, nowait):
        self.record("heartbeat", nowait)

    def receive(self):
        self.record("receive")
        assert self.parent is None, "A child became a socket reader"
        return self.pushes.popleft() if self.pushes else None

    def status(self):
        self.record("status")
        if self.responses:
            result = self.responses.popleft()
            if isinstance(result, Exception):
                raise result
            if callable(result):
                return result()
            return result
        return {"dps": {"1": False}}

    def updatedps(self, dps):
        self.record("updatedps", list(dps))
        return {"dps": {"2": 42}}

    def set_multiple_values(self, values, nowait):
        self.record("control", dict(values), nowait)


def config(cid="child-a", **overrides):
    return {
        "name": "Synthetic child",
        "device_id": "gateway-test",
        "device_cid": cid,
        "host": "gateway.invalid",
        "local_key": "synthetic-key-01",
        "protocol_version": 3.3,
        "poll_only": False,
        "type": "kogan_kahtp_heater",
        **overrides,
    }


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.fixture
async def gateway_env(hass, mocker):
    FakeAPI.instances = []
    mocker.patch("tinytuya.Device", FakeAPI)

    def select(read, write, error, timeout):
        threading.Event().wait(0.002)
        return ([read[0]] if read[0].pushes else [], [], [])

    mocker.patch("custom_components.tuya_local.gateway_broker.select.select", select)
    mocker.patch("custom_components.tuya_local.async_start_discovery")
    devices = []

    def make(cid="child-a", **kwargs):
        device = setup_device(hass, config(cid, **kwargs))
        devices.append(device)
        return device

    yield make
    for device in devices:
        await device.async_stop()
    registry = get_gateway_registry(hass)
    for gateway in list(registry.gateways.values()):
        await gateway.broker.async_stop()
    assert not any(t.name == "tuya-local-gateway" for t in threading.enumerate())


async def test_shared_parent_broker_registration_and_final_release(
    hass, gateway_env, mocker
):
    start = mocker.spy(GatewayBroker, "async_start")
    register = mocker.spy(GatewayBroker, "register_child")
    children = [gateway_env(f"child-{i}") for i in range(13)]
    await asyncio.gather(*(child.async_refresh() for child in children))
    parent = children[0]._api.parent
    broker = children[0]._broker
    assert all(
        child._api.parent is parent and child._broker is broker for child in children
    )
    assert len([api for api in FakeAPI.instances if api.parent is None]) == 1
    assert start.call_count == 1
    assert register.call_count == broker.child_count == 13
    await children[0].async_refresh()
    assert register.call_count == 13
    await async_delete_device(hass, config("child-0"))
    assert broker.running and broker.child_count == 12
    await children[1].async_set_property("1", True)
    for child in children[1:]:
        await async_delete_device(hass, config(child.dev_cid))
    assert not broker.running
    assert not parent.children
    assert not get_gateway_registry(hass).gateways
    assert all("/child-" not in key for key in hass.data[DOMAIN])


async def test_push_cache_pending_and_entity_semantics(hass, gateway_env, mocker):
    first, second = gateway_env(), gateway_env("child-b")
    await asyncio.gather(first.async_refresh(), second.async_refresh())
    entity = mocker.Mock()
    entity._config.dps.return_value = [SimpleNamespace(id="2", persist=False)]
    first._children.append(entity)
    first._cached_state["2"] = "transient"
    await first.async_set_property("1", True)
    first._api.parent.pushes.append({"device": first._api, "dps": {"1": True}})
    await eventually(lambda: not first._pending_updates)
    assert first.get_property("1") is True
    assert second.get_property("1") is False
    assert first.get_property("2") == "transient"
    entity.on_receive.assert_called_with({"1": True}, False)
    await first.async_refresh()
    entity.on_receive.assert_called_with({"1": False}, True)
    assert first.get_property("2") is None
    assert entity.schedule_update_ha_state.called


async def test_status_force_dps_control_and_heartbeat_have_one_owner(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    child._force_dps = [2]
    child._last_full_poll = 0
    child._running = True
    polls = child.async_receive()
    assert await anext(polls) == {"2": 42, "full_poll": False}
    result = await anext(polls)
    assert result == {"1": False, "full_poll": True}
    child._running = False
    await polls.aclose()
    await child.async_set_property("1", True)
    child._broker._heartbeat_interval = 0
    await eventually(lambda: any(c[0] == "heartbeat" for c in child._api.parent.calls))
    calls = child._api.calls + child._api.parent.calls
    assert {c[0] for c in calls} >= {"status", "updatedps", "control", "heartbeat"}
    assert len({c[2] for c in calls}) == 1
    assert not any(c[0] in {"receive", "heartbeat"} for c in child._api.calls)


async def test_safety_poll_not_reset_by_pushes(gateway_env, mocker):
    child = gateway_env()
    await child.async_refresh()
    clock = mocker.patch("custom_components.tuya_local.device.time", return_value=100)
    loop = asyncio.get_running_loop()
    monotonic = loop.time
    offset = 0
    mocker.patch.object(loop, "time", side_effect=lambda: monotonic() + offset)
    child._last_full_poll = 100
    child.actually_start()

    def statuses():
        return sum(c[0] == "status" for c in child._api.calls)

    await asyncio.sleep(0.02)
    assert statuses() == 1
    offset = 29
    clock.return_value = 129
    child._api.parent.pushes.append({"device": child._api, "dps": {"1": True}})
    await eventually(lambda: child.get_property("1") is True)
    assert statuses() == 1 and child._last_full_poll == 100
    offset = 30
    clock.return_value = 130
    await eventually(lambda: child._last_full_poll == 130)
    assert statuses() == 2
    offset = 59
    clock.return_value = 159
    await asyncio.sleep(0.02)
    assert statuses() == 2
    offset = 60
    clock.return_value = 160
    await eventually(lambda: child._last_full_poll == 160)
    assert statuses() == 3
    await child.async_stop()


async def test_healthy_poll_wait_is_idle_and_pushes_are_immediate(gateway_env, mocker):
    child = gateway_env()
    await child.async_refresh()
    clock = mocker.patch("custom_components.tuya_local.device.time", return_value=100)
    child._last_full_poll = 100
    child.actually_start()
    await asyncio.sleep(0.02)
    clock_calls = clock.call_count
    await asyncio.sleep(0.35)
    assert clock.call_count == clock_calls
    child._api.parent.pushes.append({"device": child._api, "dps": {"1": True}})
    await eventually(lambda: child.get_property("1") is True)
    assert sum(c[0] == "status" for c in child._api.calls) == 1
    assert child._last_full_poll == 100
    await child.async_stop()
    assert child._refresh_task is None


async def test_resume_wakes_full_safety_poll_promptly(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    child.actually_start()
    await asyncio.sleep(0.02)
    child.pause()
    await asyncio.sleep(0.02)
    assert sum(c[0] == "status" for c in child._api.calls) == 1
    child._api.responses.append({"dps": {"1": True}})
    child.resume()
    await eventually(lambda: child.get_property("1") is True)
    assert sum(c[0] == "status" for c in child._api.calls) == 2
    await child.async_stop()


async def test_uninitialized_poll_retries_after_five_seconds(gateway_env, mocker):
    child = gateway_env()
    await child._async_ensure_gateway()
    child._api.responses.extend([RuntimeError("offline")] * 3)
    loop = asyncio.get_running_loop()
    monotonic = loop.time
    offset = 0
    mocker.patch.object(loop, "time", side_effect=lambda: monotonic() + offset)
    wait = mocker.spy(child._gateway_poll_event, "wait")
    child.actually_start()
    await eventually(lambda: wait.call_count == 1)
    offset = 4
    await asyncio.sleep(0.02)
    assert not child.has_returned_state
    assert sum(c[0] == "status" for c in child._api.calls) == 3
    offset = 5
    await eventually(lambda: child.has_returned_state)
    assert sum(c[0] == "status" for c in child._api.calls) == 4
    await child.async_stop()


async def test_wrong_cid_tuple_dispatched_without_extra_receive(gateway_env):
    first, second = gateway_env(), gateway_env("child-b")
    await asyncio.gather(first.async_refresh(), second.async_refresh())
    parent = first._api.parent

    def response():
        parent.received_wrong_cid_queue.append((second._api, {"dps": {"1": True}}))
        return {"dps": {"1": False}}

    first._api.responses.append(response)
    await first.async_refresh()
    await eventually(lambda: second.get_property("1") is True)
    assert first.get_property("1") is False
    assert not any(c[0] == "receive" for c in parent.calls)
    assert not parent.received_wrong_cid_queue


async def test_unregister_discards_already_scheduled_push(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    broker, api = child._broker, child._api
    broker._dispatch({"device": api, "dps": {"1": True}})
    broker.unregister_child(api)
    await asyncio.sleep(0)
    assert child.get_property("1") is False


@pytest.mark.parametrize(
    "failure", ["offline", "exception", "missing_config", "platform"]
)
async def test_setup_failure_releases_only_failed_child(
    hass, gateway_env, mocker, failure
):
    sibling = gateway_env("sibling")
    await sibling.async_refresh()
    if failure == "offline":
        mocker.patch.object(
            FakeAPI, "status", return_value={"Error": "unreachable", "Err": "901"}
        )
    elif failure == "exception":
        mocker.patch.object(
            TuyaLocalDevice, "async_refresh", side_effect=RuntimeError("setup failed")
        )
    elif failure == "missing_config":
        mocker.patch("custom_components.tuya_local.get_config", return_value=None)
    else:
        mocker.patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            side_effect=RuntimeError("platform failed"),
        )
    entry = MockConfigEntry(domain=DOMAIN, title="Failed child", data=config("failed"))
    if failure == "missing_config":
        assert not await async_setup_entry(hass, entry)
    else:
        with pytest.raises((ConfigEntryNotReady, RuntimeError)):
            await async_setup_entry(hass, entry)
    assert "gateway-test/failed" not in hass.data[DOMAIN]
    assert sibling._broker.running
    assert sibling._broker.child_count == 1
    assert len(sibling._gateway.members) == 1


async def test_first_setup_failure_removes_gateway(hass, gateway_env, mocker):
    mocker.patch.object(
        FakeAPI, "status", return_value={"Error": "unreachable", "Err": "901"}
    )
    entry = MockConfigEntry(domain=DOMAIN, title="Failed child", data=config())
    with pytest.raises(ConfigEntryNotReady):
        await async_setup_entry(hass, entry)
    assert not get_gateway_registry(hass).gateways
    assert "gateway-test/child-a" not in hass.data[DOMAIN]


@pytest.mark.parametrize("failure", ["error", "no_state", "cancel", "configure"])
async def test_failed_new_child_restores_shared_transport(
    hass, gateway_env, mocker, failure
):
    sibling = gateway_env("sibling")
    await sibling.async_refresh()
    gateway, broker, parent = sibling._gateway, sibling._broker, sibling._api.parent
    entered, release = threading.Event(), threading.Event()
    original_status = FakeAPI.status

    def status(api):
        if api.cid != "failed":
            return original_status(api)
        api.record("status")
        if failure == "cancel":
            entered.set()
            assert release.wait(3)
        if failure == "no_state":
            return None
        return {"Err": "914", "Error": "synthetic failure"}

    mocker.patch.object(FakeAPI, "status", new=status)
    if failure == "configure":
        original_version = parent.set_version

        def set_version(version):
            original_version(version)
            if version == 3.4:
                raise RuntimeError("configuration interrupted")

        mocker.patch.object(parent, "set_version", side_effect=set_version)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Failed child",
        data=config(
            "failed",
            host="failed.invalid",
            local_key="synthetic-key-02",
            protocol_version=3.4,
        ),
    )
    setup = asyncio.create_task(async_setup_entry(hass, entry))
    if failure == "cancel":
        await eventually(entered.is_set)
        setup.cancel()
        release.set()
    with pytest.raises((ConfigEntryNotReady, RuntimeError, asyncio.CancelledError)):
        await setup
    assert gateway.settings == ("gateway.invalid", "synthetic-key-01")
    assert gateway.protocol == parent.version == sibling._api.version == 3.3
    assert gateway.protocol_working
    assert parent.address == "gateway.invalid"
    assert parent.local_key == parent.real_local_key == b"synthetic-key-01"
    assert parent.children == {"sibling": sibling._api}
    assert broker.child_count == len(gateway.members) == 1
    assert "gateway-test/failed" not in hass.data[DOMAIN]
    await sibling.async_refresh()
    await sibling.async_set_property("1", True)
    assert sibling.has_returned_state
    assert sibling._gateway is gateway and sibling._broker is broker
    assert sibling._api.parent is parent
    assert len({c[2] for api in FakeAPI.instances for c in api.calls}) == 1


async def test_successful_new_child_keeps_shared_transport(gateway_env):
    sibling = gateway_env("sibling", protocol_version="auto")
    await sibling.async_refresh()
    child = gateway_env(
        host="replacement.invalid", local_key="synthetic-key-02", protocol_version=3.4
    )
    await child.async_refresh()
    gateway, parent = sibling._gateway, sibling._api.parent
    assert child.has_returned_state
    assert gateway.settings == ("replacement.invalid", "synthetic-key-02")
    assert gateway.protocol == parent.version == 3.4
    assert parent.address == "replacement.invalid"
    assert parent.local_key == parent.real_local_key == b"synthetic-key-02"
    await child.async_stop()
    await sibling.async_refresh()
    await sibling.async_set_property("1", True)
    assert parent.version == 3.4
    assert parent.address == "replacement.invalid"


async def test_reachable_error_900_keeps_new_shared_settings(gateway_env):
    sibling = gateway_env("sibling")
    await sibling.async_refresh()
    child = gateway_env(host="replacement.invalid", protocol_version=3.4)
    await child._async_ensure_gateway()
    child._api.responses.append({"Err": "900", "Error": "no status data"})
    await child.async_refresh()
    assert child.has_returned_state
    assert child._gateway.protocol_working
    assert child._api.parent.address == "replacement.invalid"
    assert child._api.parent.version == 3.4
    await child.async_set_property("1", True)
    assert child._api.parent.address == "replacement.invalid"


async def test_runtime_failures_can_still_reset_working_protocol(gateway_env):
    child = gateway_env(protocol_version="auto")
    await child.async_refresh()
    child._api_working_protocol_failures = child._AUTO_FAILURE_RESET_COUNT
    child._api.responses.extend([RuntimeError("synthetic failure")] * 3)
    await child.async_refresh()
    assert not child._api_protocol_working
    assert not child._gateway.protocol_working


async def test_pause_probe_resume_uses_same_owner_and_restores_settings(
    hass, gateway_env
):
    child, sibling = gateway_env(), gateway_env("child-b")
    await asyncio.gather(child.async_refresh(), sibling.async_refresh())
    parent, broker = child._api.parent, child._broker
    child.pause()
    child._gateway_receive({"dps": {"1": True}})
    assert child.get_property("1") is False
    assert broker.running
    child.resume()
    probe = await async_test_connection(
        config(host="probe.invalid", protocol_version=3.4), hass
    )
    assert probe.has_returned_state
    assert probe._api is child._api
    assert parent.address == "gateway.invalid"
    assert parent.version == 3.3
    assert len([api for api in FakeAPI.instances if api.parent is None]) == 1
    assert broker.child_count == 2
    assert len(child._gateway.members) == 2
    assert not child._temporary_poll
    await sibling.async_set_property("1", True)
    assert len({c[2] for api in FakeAPI.instances for c in api.calls}) == 1


async def test_probe_without_entries_releases_resources(hass, gateway_env):
    probe = await async_test_connection(config(protocol_version="auto"), hass)
    assert probe.has_returned_state
    assert not get_gateway_registry(hass).gateways
    assert probe._gateway is None


async def test_auto_protocol_rotation_and_new_child_keeps_working_version(gateway_env):
    first = gateway_env(protocol_version="auto")
    await first._async_ensure_gateway()
    first._api.responses.extend([RuntimeError("bad protocol")] * 3)
    await first.async_refresh()
    assert first._api.parent.version == 3.4
    assert first._api_protocol_working
    second = gateway_env("child-b", protocol_version="auto")
    await second.async_refresh()
    assert second._api.parent.version == 3.4
    versions = [c[1][0] for c in first._api.parent.calls if c[0] == "version"]
    assert versions == API_PROTOCOL_VERSIONS[:4]


async def test_retry_exhaustion_clears_cache_and_recovers(gateway_env, caplog):
    child = gateway_env()
    await child.async_refresh()
    child._api.responses.extend([{"Error": "unreachable", "Err": "901"}] * 3)
    await child.async_refresh()
    assert not child.has_returned_state
    assert "901" in caplog.text
    assert child._api_working_protocol_failures == 1
    await child.async_refresh()
    assert child.has_returned_state
    assert child._api_working_protocol_failures == 0


async def test_homeassistant_stop_before_monitor_starts(hass, gateway_env):
    child = gateway_env()
    await child.async_refresh()
    broker = child._broker
    assert child._refresh_task is None
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    assert not broker.running
    assert not get_gateway_registry(hass).gateways


async def test_standalone_never_acquires_broker(hass, mocker):
    api = mocker.Mock(parent=None)
    api.status.return_value = {"dps": {"1": True}}
    mocker.patch("tinytuya.Device", return_value=api)
    broker = mocker.patch("custom_components.tuya_local.gateway.GatewayBroker")
    child = setup_device(hass, config(device_cid=None))
    await child.async_refresh()
    await child.async_set_property("1", False)
    assert child.get_property("1") is False
    api.status.assert_called_once()
    api.set_multiple_values.assert_called_once_with({"1": False}, nowait=True)
    broker.assert_not_called()
    await child.async_stop()


async def test_broker_stop_fails_queue_and_waits_for_inflight_owner(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    broker = child._broker
    entered, release = threading.Event(), threading.Event()
    executed = []

    def blocking_call():
        entered.set()
        assert release.wait(3)
        return "finished"

    active = asyncio.create_task(broker.async_call(blocking_call))
    await eventually(entered.is_set)
    queued = asyncio.create_task(broker.async_call(lambda: executed.append(True)))
    await asyncio.sleep(0)
    stop = asyncio.create_task(broker.async_stop())
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(queued, 1)
        assert broker.running
        assert not stop.done()
        with pytest.raises(RuntimeError, match="not running"):
            await broker.async_call(lambda: None)
    finally:
        release.set()
    assert await active == "finished"
    await stop
    assert not broker.running
    assert not executed


async def test_cancelled_queued_call_never_executes(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()
    called = []

    def blocking_call():
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(child._broker.async_call(blocking_call))
    await eventually(entered.is_set)
    queued = asyncio.create_task(child._broker.async_call(lambda: called.append(True)))
    await asyncio.sleep(0)
    queued.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await active
    await child._broker.async_call(lambda: None)
    assert not called


async def test_cancelled_setup_finishes_acquisition_then_cleans(
    hass, gateway_env, mocker
):
    entered, release = threading.Event(), threading.Event()

    def make_api(*args, **kwargs):
        if not kwargs.get("parent"):
            entered.set()
            assert release.wait(3)
        return FakeAPI(*args, **kwargs)

    mocker.patch("tinytuya.Device", side_effect=make_api)
    entry = MockConfigEntry(domain=DOMAIN, title="Cancelled child", data=config())
    setup = asyncio.create_task(async_setup_entry(hass, entry))
    await eventually(entered.is_set)
    setup.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await setup
    assert not get_gateway_registry(hass).gateways
    assert "gateway-test/child-a" not in hass.data[DOMAIN]


async def test_cancelled_probe_restores_settings_and_resumes_siblings(
    hass, gateway_env
):
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()

    def response():
        entered.set()
        assert release.wait(3)
        return {"dps": {"1": False}}

    child._api.responses.append(response)
    probe = asyncio.create_task(
        async_test_connection(config(host="probe.invalid"), hass)
    )
    await eventually(entered.is_set)
    probe.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await probe
    assert child._api.parent.address == "gateway.invalid"
    assert not child._temporary_poll
    assert len(child._gateway.members) == 1
    await child.async_refresh()
    assert child.has_returned_state


async def test_final_unload_and_reload_cannot_overlap_owners(hass, gateway_env):
    child = gateway_env()
    await child.async_refresh()
    old_broker = child._broker
    old_thread = old_broker._thread
    entered, release = threading.Event(), threading.Event()

    def blocking_call():
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(old_broker.async_call(blocking_call))
    await eventually(entered.is_set)
    stop = asyncio.create_task(child.async_stop())
    await asyncio.sleep(0)
    replacement = gateway_env("replacement")
    setup = asyncio.create_task(replacement.async_refresh())
    await asyncio.sleep(0.02)
    assert len([api for api in FakeAPI.instances if api.parent is None]) == 1
    release.set()
    await asyncio.gather(active, stop, setup)
    assert not old_thread.is_alive()
    assert replacement._broker is not old_broker
    assert replacement.has_returned_state


async def test_startup_listener_removed_by_early_unload(hass, gateway_env):
    child = gateway_env()
    await child.async_refresh()
    child.start()
    await child.async_stop()
    assert child._startup_listener is None
    child.actually_start()
    assert child._refresh_task is None
    assert not get_gateway_registry(hass).gateways


async def test_constructor_failure_does_not_leave_parent_registration(
    hass, gateway_env, mocker
):
    sibling = gateway_env("sibling")
    await sibling.async_refresh()

    def make_api(*args, **kwargs):
        api = FakeAPI(*args, **kwargs)
        if api.cid == "broken":
            raise RuntimeError("constructor failed after registration")
        return api

    mocker.patch("tinytuya.Device", side_effect=make_api)
    broken = gateway_env("broken")
    with pytest.raises(RuntimeError, match="constructor failed"):
        await broken.async_refresh()
    assert "broken" not in sibling._api.parent.children
    assert sibling._broker.child_count == 1
    await sibling.async_refresh()


async def test_poll_only_child_keeps_owner_but_ignores_pushes(gateway_env):
    child = gateway_env(poll_only=True)
    await child.async_refresh()
    broker = child._broker
    child._gateway_receive({"dps": {"1": True}})
    assert child.get_property("1") is False
    assert broker.running
    child._api.responses.append({"dps": {"1": True}})
    await child.async_refresh()
    assert child.get_property("1") is True


async def test_control_error_is_retried_and_pending_ack_is_cleared(gateway_env, mocker):
    child = gateway_env()
    await child.async_refresh()
    attempts = 0

    def control(values, nowait):
        nonlocal attempts
        child._api.record("control", values)
        attempts += 1
        if attempts == 1:
            return {"Err": "901", "Error": "unreachable"}
        child._api.parent.received_wrong_cid_queue.append((child._api, {"dps": values}))

    mocker.patch.object(child._api, "set_multiple_values", side_effect=control)
    await child.async_set_property("1", True)
    await eventually(lambda: not child._pending_updates)
    assert child.get_property("1") is True
    assert attempts == 2


async def test_runtime_entry_named_test_retains_lease(gateway_env):
    child = gateway_env(name="Test")
    await child.async_refresh()
    assert child._broker.running
    assert len(child._gateway.members) == 1


async def test_auto_probe_retains_detected_protocol(hass, gateway_env):
    probe = await async_test_connection(config(protocol_version="auto"), hass)
    assert probe._protocol_configured == 3.3
    assert probe._gateway is None


async def test_blank_host_is_discovered_without_replacing_resolved_address(
    hass, gateway_env, mocker
):
    discover = mocker.patch(
        "tinytuya.find_device",
        return_value={"ip": "discovered.invalid", "version": "3.2"},
    )
    child = gateway_env(host="")
    await child.async_refresh()
    discover.assert_called_once_with(dev_id="gateway-test")
    assert child._api.parent.address == "discovered.invalid"
    assert child._api.parent.auto_ip
    await async_test_connection(config(host="probe.invalid"), hass)
    assert child._api.parent.auto_ip
    assert child._api.parent.address == "discovered.invalid"


async def test_concurrent_refresh_same_child_registers_once(gateway_env, mocker):
    register = mocker.spy(GatewayBroker, "register_child")
    child = gateway_env()
    await asyncio.gather(child.async_refresh(), child.async_refresh())
    assert register.call_count == 1
    assert len(child._gateway.members) == 1


async def test_disconnect_clears_stale_wrong_cid_messages(gateway_env):
    child = gateway_env()
    await child.async_refresh()
    parent = child._api.parent

    def disconnect():
        parent.received_wrong_cid_queue.append((child._api, {"dps": {"1": True}}))
        child._broker._disconnect("synthetic failure")

    await child._broker.async_call(disconnect)
    assert not parent.received_wrong_cid_queue
    assert child.get_property("1") is False
    await child.async_refresh()
    assert child.has_returned_state


async def test_entity_exception_does_not_interrupt_other_entities(gateway_env, mocker):
    child = gateway_env()
    await child.async_refresh()
    broken, healthy = mocker.Mock(), mocker.Mock()
    broken.on_receive.side_effect = RuntimeError("entity failed")
    for entity in (broken, healthy):
        entity._config.dps.return_value = []
    child._children = [broken, healthy]
    event_loop_thread = threading.current_thread()

    def received(poll, full_poll):
        assert threading.current_thread() is event_loop_thread
        assert not full_poll

    healthy.on_receive.side_effect = received
    child._api.parent.pushes.append({"device": child._api, "dps": {"1": True}})
    await eventually(lambda: healthy.on_receive.called)
    assert child.get_property("1") is True
    healthy.schedule_update_ha_state.assert_called_once()


async def test_cancelled_stop_still_releases_registry(hass, gateway_env):
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()

    def blocking_call():
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(child._broker.async_call(blocking_call))
    await eventually(entered.is_set)
    stop = asyncio.create_task(child.async_stop())
    await asyncio.sleep(0)
    stop.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop
    await active
    await child.async_stop()
    assert not get_gateway_registry(hass).gateways


async def test_delete_does_not_remove_replacement_entry(hass, gateway_env):
    child = gateway_env()
    await child.async_refresh()
    entered, release = threading.Event(), threading.Event()

    def blocking_call():
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(child._broker.async_call(blocking_call))
    await eventually(entered.is_set)
    deletion = asyncio.create_task(async_delete_device(hass, config()))
    await asyncio.sleep(0)
    replacement = gateway_env()
    setup = asyncio.create_task(replacement.async_refresh())
    release.set()
    await asyncio.gather(active, deletion, setup)
    assert hass.data[DOMAIN]["gateway-test/child-a"]["device"] is replacement
    assert replacement._broker.running


async def test_failed_probe_preserves_sibling_and_restores_parameters(
    hass, gateway_env, mocker
):
    sibling = gateway_env()
    await sibling.async_refresh()
    status = mocker.patch.object(
        FakeAPI, "status", return_value={"Err": "914", "Error": "bad parameters"}
    )
    probe = await async_test_connection(
        config(protocol_version=3.4, host="probe.invalid"), hass
    )
    assert probe is None
    assert status.call_count == 3
    assert sibling._api.parent.version == 3.3
    assert sibling._api.parent.address == "gateway.invalid"
    assert sibling._broker.child_count == 1
    assert len(sibling._gateway.members) == 1
    assert not sibling._temporary_poll


@pytest.mark.parametrize("protocol", [3.2, 3.3, 3.4, 3.5])
async def test_real_tinytuya_registration_and_version_io_stay_on_owner(
    gateway_env, mocker, protocol
):
    calls = []

    def status(api):
        thread = threading.current_thread()
        assert thread.name == "tuya-local-gateway"
        calls.append(thread)
        return {"dps": {"1": True}}

    mocker.patch("tinytuya.Device", TinyTuyaDevice)
    mocker.patch.object(TinyTuyaDevice, "status", new=status)
    mocker.patch.object(GatewayBroker, "_ensure_connected")
    first = gateway_env(protocol_version=protocol)
    second = gateway_env("child-b", protocol_version=protocol)
    await asyncio.gather(first.async_refresh(), second.async_refresh())
    parent = first._api.parent
    assert parent is second._api.parent
    assert parent.children == {"child-a": first._api, "child-b": second._api}
    assert first._api.version == second._api.version == parent.version == protocol
    assert first._api.disabledetect and second._api.disabledetect
    assert len(set(calls)) == 1
    await first.async_stop()
    assert parent.children == {"child-b": second._api}
    await second.async_refresh()
    assert second.has_returned_state


@pytest.mark.parametrize("working_version", [3.4, 3.5])
async def test_real_tinytuya_auto_rotation_resets_protocol_and_session(
    gateway_env, mocker, working_version
):
    """Keep real status, payload generation and cryptographic negotiation."""
    attempts, negotiations, owners = [], [], set()
    remote_nonce = b"test-remote-0001"
    stale_nonce = b"stale-nonce-0001"

    def exchange_key(api, payload, recv_retries):
        owners.add(threading.current_thread())
        if payload.cmd == SESS_KEY_NEG_START:
            negotiations.append(
                (api.version, api.local_key, api.local_nonce, api.remote_nonce)
            )
            if api.version != working_version:
                api.local_key = b"stale-key-000001"
                api.remote_nonce = stale_nonce
                return None
            reply = remote_nonce + hmac.digest(
                api.real_local_key, payload.payload, "sha256"
            )
            if api.version == 3.4:
                reply = AESCipher(api.real_local_key).encrypt(reply, use_base64=False)
            return TuyaMessage(1, SESS_KEY_NEG_RESP, 0, reply, 0)
        assert payload.cmd == SESS_KEY_NEG_FINISH
        assert payload.payload == hmac.digest(
            api.real_local_key, remote_nonce, "sha256"
        )
        return None

    def exchange_status(api, payload, *args, **kwargs):
        owners.add(threading.current_thread())
        attempts.append((api.version, api.dev_type, payload.cmd))
        parent = api.parent or api
        if api.version >= 3.4:
            if parent._negotiate_session_key():
                return {"dps": {"1": True}}
        else:
            parent.local_nonce = stale_nonce
            parent.remote_nonce = stale_nonce
        return {"Err": "914", "Error": "synthetic protocol failure"}

    mocker.patch("tinytuya.Device", TinyTuyaDevice)
    mocker.patch.object(TinyTuyaDevice, "_send_receive", new=exchange_status)
    mocker.patch.object(TinyTuyaDevice, "_send_receive_quick", new=exchange_key)
    mocker.patch.object(GatewayBroker, "_ensure_connected")
    child = gateway_env(protocol_version="auto")
    await child.async_refresh()
    parent, broker = child._api.parent, child._broker
    # 3.2 also calls status internally to discover DPS; collapse repeats.
    assert list(dict.fromkeys(attempt[0] for attempt in attempts)) == (
        [3.3, 3.1, 3.2, 3.4] + ([3.5] if working_version == 3.5 else [])
    )
    assert child.has_returned_state
    assert attempts[-1] == (working_version, "default", DP_QUERY_NEW)
    for api in (parent, child._api):
        assert api.version == working_version
        assert api.version_str == f"v{working_version}"
        assert api.version_bytes == str(working_version).encode()
        assert api.version_header.startswith(api.version_bytes)
        assert api.dev_type == "default"
    assert negotiations
    for _, key, nonce, remote in negotiations:
        assert key == b"synthetic-key-01"
        assert nonce != stale_nonce and len(nonce) == 16
        assert remote == b""
    assert parent.remote_nonce == remote_nonce
    assert len(parent.local_key) == 16 and parent.local_key != parent.real_local_key
    previous_nonce = parent.local_nonce
    await child.async_refresh()
    assert parent.local_nonce != previous_nonce
    assert child._api.parent is parent and child._broker is broker
    assert len(owners) == 1
    assert next(iter(owners)).name == "tuya-local-gateway"


async def test_worker_start_failure_cleans_registry(hass, gateway_env, mocker):
    child = gateway_env()
    original_start = threading.Thread.start
    attempts = []

    def start(thread):
        if thread.name == "tuya-local-gateway":
            attempts.append(thread)
            raise RuntimeError("worker start failed")
        return original_start(thread)

    mocker.patch.object(threading.Thread, "start", new=start)
    with pytest.raises(RuntimeError, match="worker start failed"):
        await child.async_refresh()
    assert not get_gateway_registry(hass).gateways
    assert len(attempts) == 1


async def test_unload_during_parent_construction_cleans_registry(
    hass, gateway_env, mocker
):
    entered, release = threading.Event(), threading.Event()

    def make_api(*args, **kwargs):
        if not kwargs.get("parent"):
            entered.set()
            assert release.wait(3)
        return FakeAPI(*args, **kwargs)

    mocker.patch("tinytuya.Device", side_effect=make_api)
    child = gateway_env()
    refresh = asyncio.create_task(child.async_refresh())
    await eventually(entered.is_set)
    stop = asyncio.create_task(child.async_stop())
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="child is stopped"):
        await refresh
    await stop
    assert not get_gateway_registry(hass).gateways


async def test_probe_restores_gateway_still_in_initial_setup(hass, gateway_env, mocker):
    mocker.patch("tinytuya.find_device", return_value={"ip": "discovered.invalid"})
    child = gateway_env(host="")
    await child._async_ensure_gateway()
    assert child._gateway.protocol is None
    probe = await async_test_connection(
        config(host="probe.invalid", protocol_version=3.4), hass
    )
    assert probe.has_returned_state
    assert child._api.parent.address == "discovered.invalid"
    assert child._api.parent.auto_ip
    await child.async_refresh()
    assert child.has_returned_state
