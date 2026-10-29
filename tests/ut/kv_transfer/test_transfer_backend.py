# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transport selection for PD KV transfer.

The dense connector used to hardcode both the Mooncake engine and its transport.
These tests pin the launch-time configuration that replaced it: the default is
still Mooncake with the ``ascend`` transport, MemFabric is opt-in, and the
process-wide engine rejects a second, conflicting protocol instead of silently
running on the first one (prefill and decode must match).
"""

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_ascend.distributed.kv_transfer.utils import transfer_backend as tb
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import GlobalMemfabricTE
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    DEFAULT_TRANSFER_PROTOCOL,
    GlobalTE,
)


def _config(**extra):
    def get_from_extra_config(name, default):
        return extra.get(name, default)

    return SimpleNamespace(kv_transfer_config=SimpleNamespace(get_from_extra_config=get_from_extra_config))


class _FakeTransferEngine:
    instances: list["_FakeTransferEngine"] = []

    def __init__(self):
        self.initialize_calls: list[tuple] = []
        self.rpc_port = 12345
        _FakeTransferEngine.instances.append(self)

    def initialize(self, hostname, metadata, protocol, device_name):
        self.initialize_calls.append((hostname, metadata, protocol, device_name))
        return 0

    def get_rpc_port(self):
        return self.rpc_port


@pytest.fixture
def fake_mooncake(monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=_FakeTransferEngine))
    _FakeTransferEngine.instances.clear()
    return _FakeTransferEngine


class _FakeMemfabricTE:
    def __init__(self, engine="memfabric-engine"):
        self.configured: dict[str, Any] | None = None
        self.engine = engine
        self.register_calls = []

    def configure(self, **kwargs):
        self.configured = kwargs

    def get_transfer_engine(self, hostname):
        return self.engine

    def register_buffer(self, ptrs, sizes):
        self.register_calls.append((list(ptrs), list(sizes)))


# --- configuration -----------------------------------------------------------


def test_defaults_keep_the_historical_behaviour():
    assert tb.resolve_transfer_backend(_config()) == tb.BACKEND_MOONCAKE
    assert tb.resolve_mooncake_protocol(_config()) == DEFAULT_TRANSFER_PROTOCOL == "ascend"


def test_backend_and_protocol_come_from_extra_config():
    assert tb.resolve_transfer_backend(_config(transfer_backend="MemFabric")) == tb.BACKEND_MEMFABRIC
    assert tb.resolve_mooncake_protocol(_config(transfer_protocol="  Ascend_Direct ")) == "ascend_direct"
    # An empty override must not disable the transport entirely.
    assert tb.resolve_transfer_backend(_config(transfer_backend="")) == tb.BACKEND_MOONCAKE
    assert tb.resolve_mooncake_protocol(_config(transfer_protocol="")) == "ascend"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Invalid transfer_backend"):
        tb.resolve_transfer_backend(_config(transfer_backend="hccl"))


# --- regressions the factory is meant to prevent ------------------------------


class _BoomTE:
    def __getattr__(self, name):
        raise AssertionError(f"memfabric must not be touched on the default path (accessed {name})")


def test_default_backend_never_touches_memfabric(monkeypatch, fake_mooncake):
    monkeypatch.setattr(tb, "global_memfabric_te", _BoomTE())
    monkeypatch.setattr(tb, "global_te", GlobalTE())

    registry, engine = tb.build_transfer_engine(_config(), "10.0.0.1", device_name=None, kv_role="kv_producer")

    assert registry is tb.global_te
    assert engine.initialize_calls == [("10.0.0.1", "P2PHANDSHAKE", "ascend", "")]


def test_memfabric_without_protocol_defers_to_its_default(monkeypatch):
    fake = _FakeMemfabricTE()
    monkeypatch.setattr(tb, "global_memfabric_te", fake)
    monkeypatch.setattr(tb.torch.npu, "current_device", lambda: 0)

    tb.build_transfer_engine(_config(transfer_backend="memfabric"), "10.0.0.1", device_name=None, kv_role="kv_consumer")

    # None lets MemFabric apply its own default (sdma) instead of us guessing.
    configured = fake.configured
    assert configured is not None
    assert configured["transfer_protocol"] is None


def test_connector_uses_the_backend_factory():
    """The worker must reach the engine through the factory, not a module global."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector

    assert mooncake_connector.build_transfer_engine is tb.build_transfer_engine
    # The connector still exposes the shared registry so existing tests and any
    # future call site patch one object, but it is passed explicitly instead of
    # being reached through a module global inside the factory.
    assert mooncake_connector.global_te is tb.global_te


def test_mooncake_only_connectors_reject_memfabric():
    # MooncakeConnectorV1 is the only dense path that can route through MemFabric.
    assert tb.ensure_mooncake_backend(_config(), "MooncakeConnector") is None
    with pytest.raises(ValueError, match="supports transfer_backend='mooncake' only"):
        tb.ensure_mooncake_backend(_config(transfer_backend="memfabric"), "MooncakeLayerwiseConnector")
    # A transport these connectors never read must not be accepted silently.
    with pytest.raises(ValueError, match="does not support transfer_protocol"):
        tb.ensure_mooncake_backend(_config(transfer_protocol="ascend_direct"), "MooncakeLayerwiseConnector")


def test_d2rh_worker_rejects_memfabric_before_base_init(monkeypatch):
    """The D2RH worker inherits the base worker that builds the engine, so it must
    reject a non-Mooncake backend before the base ``__init__`` runs. Otherwise it
    would register buffers with the wrong registry and fail at transfer time."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh

    base_init_called = []

    def fake_base_init(self, vllm_config, engine_id, kv_cache_config):
        base_init_called.append(True)

    monkeypatch.setattr(d2rh.BaseMooncakeConnectorWorker, "__init__", fake_base_init)

    worker = object.__new__(d2rh.MooncakeConnectorWorker)
    with pytest.raises(ValueError, match="supports transfer_backend='mooncake' only"):
        d2rh.MooncakeConnectorWorker.__init__(worker, _config(transfer_backend="memfabric"), "engine", None)

    assert not base_init_called, "the guard must fire before the base worker builds the engine"


# --- backend construction ----------------------------------------------------


def test_build_uses_mooncake_by_default(fake_mooncake, monkeypatch):
    monkeypatch.setattr(tb, "global_te", GlobalTE())
    registry, engine = tb.build_transfer_engine(
        _config(),
        "10.0.0.1",
        device_name=None,
        kv_role="kv_producer",
    )

    assert engine.initialize_calls == [("10.0.0.1", "P2PHANDSHAKE", "ascend", "")]
    assert registry is tb.global_te


@pytest.mark.parametrize(("kv_role", "expected"), [("kv_producer", "Prefill"), ("kv_consumer", "Decode")])
def test_build_configures_memfabric_role_and_protocol(monkeypatch, kv_role, expected):
    fake = _FakeMemfabricTE()
    monkeypatch.setattr(tb, "global_memfabric_te", fake)
    monkeypatch.setattr(tb.torch.npu, "current_device", lambda: 3)

    registry, engine = tb.build_transfer_engine(
        _config(transfer_backend="memfabric", memfabric_transfer_protocol="device_urma"),
        "10.0.0.1",
        device_name=None,
        kv_role=kv_role,
    )

    assert fake.configured == {"role": expected, "device_id": 3, "transfer_protocol": "device_urma"}
    assert engine == "memfabric-engine"
    assert registry is fake


def test_memfabric_rejects_a_role_it_cannot_map(monkeypatch):
    monkeypatch.setattr(tb, "global_memfabric_te", _FakeMemfabricTE())
    with pytest.raises(ValueError, match="producer/consumer role"):
        tb.build_transfer_engine(
            _config(transfer_backend="memfabric"),
            "10.0.0.1",
            device_name=None,
            kv_role="kv_both",
        )


def test_memfabric_rejects_a_retry_after_a_failed_init():
    """A failed native initialize() must not be retried into a second leaked
    engine: the singleton mirrors GlobalTE's no-retry-on-failure guard."""
    registry = GlobalMemfabricTE()
    registry.configure(role="Prefill", device_id=0)
    registry._initialization_attempted = True  # a prior initialize() failed

    with pytest.raises(RuntimeError, match="previously failed"):
        registry.get_transfer_engine("10.0.0.1")


# --- Mooncake engine contract ------------------------------------------------


def test_engine_initializes_with_the_configured_protocol(fake_mooncake):
    registry_holder = GlobalTE()

    engine = registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend_direct")

    assert engine.initialize_calls == [("10.0.0.1", "P2PHANDSHAKE", "ascend_direct", "")]
    assert registry_holder.transfer_protocol == "ascend_direct"


def test_same_protocol_reuses_the_engine(fake_mooncake):
    registry_holder = GlobalTE()

    first = registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend")
    second = registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend")

    assert first is second
    assert len(fake_mooncake.instances) == 1
    assert len(fake_mooncake.instances[0].initialize_calls) == 1


def test_unspecified_protocol_reuses_the_initialized_one(fake_mooncake):
    """Components sharing the singleton must not be forced to name a transport."""
    registry_holder = GlobalTE()
    first = registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend_direct")

    again = registry_holder.get_transfer_engine("10.0.0.1", device_name=None)

    assert again is first
    assert registry_holder.transfer_protocol == "ascend_direct"


def test_first_call_without_protocol_records_the_default(fake_mooncake):
    registry_holder = GlobalTE()

    registry_holder.get_transfer_engine("10.0.0.1", device_name=None)

    assert registry_holder.transfer_protocol == DEFAULT_TRANSFER_PROTOCOL


def test_conflicting_protocol_is_rejected(fake_mooncake):
    registry_holder = GlobalTE()
    registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend")

    with pytest.raises(RuntimeError, match="already initialized with protocol 'ascend'"):
        registry_holder.get_transfer_engine("10.0.0.1", device_name=None, protocol="ascend_direct")
