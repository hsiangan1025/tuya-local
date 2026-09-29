"""Event-loop-owned gateway leases; transport belongs to GatewayBroker only."""

import asyncio

import tinytuya

from .const import DOMAIN
from .gateway_broker import GatewayBroker

DATA_GATEWAYS = "gateway_brokers"
# TinyTuya discovery sentinels, never local bind addresses.
_AUTO_ADDRESSES = ("Auto", "0.0.0.0")  # noqa: S104


def _create_parent(device_id, address, local_key):
    """Construct without protocol 3.2's implicit status/detection I/O."""
    auto_ip = not address or address in _AUTO_ADDRESSES
    if auto_ip:
        address = tinytuya.find_device(dev_id=device_id).get("ip")
        if not address:
            raise RuntimeError("Unable to find gateway on network (specify host)")
    parent = tinytuya.Device(
        device_id, address, local_key, version=3.3, connection_retry_limit=1
    )
    parent.auto_ip = auto_ip
    return parent


class GatewayConnection:
    """One parent, worker and protocol negotiation lock per physical gateway."""

    def __init__(self, hass, parent, address, local_key):
        self.parent = parent
        self.broker = GatewayBroker(hass, parent)
        self.lock = asyncio.Lock()
        self.members = {}
        self.apis = {}
        self.settings = (address, local_key)
        self.protocol = None
        self.protocol_working = False

    def configure(
        self, address, local_key, protocol, *, resolved_address=None, force=False
    ):
        """Apply transport settings on the owner thread, including session reset."""
        settings = (address, local_key)
        if force or settings != self.settings or protocol != self.protocol:
            self.parent.set_socketPersistent(False)
            self.parent.received_wrong_cid_queue.clear()
            self.parent.auto_ip = not address or address in _AUTO_ADDRESSES
            if not self.parent.auto_ip:
                self.parent.address = address
            elif resolved_address is not None:
                self.parent.address = resolved_address
            self.parent.local_key = local_key.encode("latin1")
            self.parent.real_local_key = self.parent.local_key
            version = {3.22: 3.3, 3.42: 3.4, 3.52: 3.5}.get(protocol, protocol)
            # TinyTuya sets device22 on 3.2 but never resets it on later
            # versions; that overrides 3.4/3.5's status command. Reset only
            # when changing protocols, preserving detection within a version.
            if protocol != self.protocol or version != self.parent.version:
                self.parent.dev_type = "default"
            self.parent.set_version(version)
            for api in self.apis.values():
                api.disabledetect = protocol not in (3.22, 3.42, 3.52)
                if protocol != self.protocol or version != api.version:
                    api.dev_type = "default"
                api.set_version(version)
            self.settings = settings
            self.protocol = protocol
            self.protocol_working = False
        self.parent.set_socketPersistent(True)

    def deliver(self, cid, data):
        """Fan out only to live leases, on Home Assistant's event loop."""
        for callback, member_cid in tuple(self.members.items()):
            if member_cid == cid:
                callback(data)


class GatewayRegistry:
    """Serialize acquisitions and final release, including in-flight shutdown."""

    def __init__(self, hass):
        self.hass = hass
        self.gateways = {}
        self.lock = asyncio.Lock()

    async def acquire(self, device_id, address, local_key, cid, callback):
        async with self.lock:
            gateway = self.gateways.get(device_id)
            if gateway is None:
                parent = await self.hass.async_add_executor_job(
                    _create_parent, device_id, address, local_key
                )
                gateway = GatewayConnection(self.hass, parent, address, local_key)
                self.gateways[device_id] = gateway
            try:
                if not gateway.broker.running:
                    await gateway.broker.async_start()
                async with gateway.lock:
                    if cid not in gateway.apis:

                        def make_child():
                            try:
                                api = tinytuya.Device(
                                    cid, cid=cid, parent=gateway.parent
                                )
                                api.set_socketRetryLimit(1)
                                api.disabledetect = gateway.protocol not in (
                                    3.22,
                                    3.42,
                                    3.52,
                                )
                            except Exception:
                                gateway.parent.children.pop(cid, None)
                                raise
                            gateway.apis[cid] = api
                            return api

                        api = await gateway.broker.async_call(make_child)
                        gateway.broker.register_child(
                            api, lambda data: gateway.deliver(cid, data)
                        )
                    gateway.members[callback] = cid
                    return gateway, gateway.apis[cid]
            except BaseException:
                if not gateway.members:
                    await gateway.broker.async_stop()
                    self.gateways.pop(device_id, None)
                raise

    async def release(self, device_id, callback):
        async with self.lock:
            gateway = self.gateways.get(device_id)
            if gateway is None or callback not in gateway.members:
                return
            async with gateway.lock:
                cid = gateway.members.pop(callback)
                if cid not in gateway.members.values():
                    api = gateway.apis.pop(cid)
                    gateway.broker.unregister_child(api)

                    def remove_child():
                        gateway.parent.children.pop(cid, None)
                        cached = gateway.parent.received_wrong_cid_queue
                        cached[:] = [
                            item
                            for item in cached
                            if not (isinstance(item, tuple) and item[0] is api)
                        ]

                    try:
                        await gateway.broker.async_call(remove_child)
                    except RuntimeError:
                        # A failed worker has already closed its transport.
                        # Join it before touching its former routing state.
                        await gateway.broker.async_stop()
                        remove_child()
                if not gateway.members:
                    await gateway.broker.async_stop()
                    self.gateways.pop(device_id, None)


def get_gateway_registry(hass):
    """Called only on the event loop; the empty registry itself is reusable."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if DATA_GATEWAYS not in domain_data:
        domain_data[DATA_GATEWAYS] = GatewayRegistry(hass)
    return domain_data[DATA_GATEWAYS]
