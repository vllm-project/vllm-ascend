"""Unit tests for the isolated MemFabric transfer-engine singleton."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (  # noqa: E402
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
    GlobalMemfabricTE,
)


def _fake_memfabric(raw_engine: MagicMock) -> ModuleType:
    module = ModuleType("memfabric_hybrid")
    module.TransferEngine = MagicMock(return_value=raw_engine)
    module.set_conf_store_tls = MagicMock()
    module.set_log_level = MagicMock()
    return module


def test_memfabric_initialization_publishes_session_from_engine_port():
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 0
    module = _fake_memfabric(raw_engine)
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=3)

    with patch.dict(sys.modules, {"memfabric_hybrid": module}):
        engine = manager.get_transfer_engine("192.168.1.10")

    assert manager.unique_id == "192.168.1.10:23456"
    assert engine.get_rpc_port() == 23456
    module.set_log_level.assert_called_once_with(2)
    module.set_conf_store_tls.assert_called_once_with(False, "")
    raw_engine.initialize.assert_called_once_with(
        "tcp://192.168.1.10",
        "192.168.1.10",
        MEMFABRIC_ROLE_PREFILL,
        3,
        store_server_role=MEMFABRIC_ROLE_PREFILL,
    )
    assert raw_engine.method_calls[:2] == [
        call.initialize(
            "tcp://192.168.1.10",
            "192.168.1.10",
            MEMFABRIC_ROLE_PREFILL,
            3,
            store_server_role=MEMFABRIC_ROLE_PREFILL,
        ),
        call.get_rpc_port(),
    ]


def test_memfabric_configuration_is_idempotent_but_role_bound():
    manager = GlobalMemfabricTE()

    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0)
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0)

    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=0)
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=1)


def test_memfabric_engine_is_bound_to_initial_hostname():
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0)

    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")
        with pytest.raises(RuntimeError, match="initialized for hostname"):
            manager.get_transfer_engine("127.0.0.2")


def test_memfabric_registers_each_region_only_once():
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 0
    raw_engine.register_memory.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=0)

    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")
    manager.register_buffer([100, 200], [10, 20])
    manager.register_buffer([300], [30])

    assert raw_engine.register_memory.call_args_list == [
        call(100, 10),
        call(200, 20),
    ]


def test_memfabric_rejects_registration_shape_mismatch():
    manager = GlobalMemfabricTE()

    with pytest.raises(ValueError, match="counts differ"):
        manager.register_buffer([100], [10, 20])


def test_memfabric_initialization_failure_does_not_publish_session():
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 1
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0)

    with (
        patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}),
        pytest.raises(RuntimeError, match="initialization failed"),
    ):
        manager.get_transfer_engine("127.0.0.1")
    with pytest.raises(RuntimeError, match="has not been initialized"):
        _ = manager.unique_id
