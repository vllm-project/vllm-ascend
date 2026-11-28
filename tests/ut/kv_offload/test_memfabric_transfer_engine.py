"""Unit tests for the isolated MemFabric transfer-engine singleton."""

import sys
from enum import Enum
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (  # noqa: E402
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
    GlobalMemfabricTE,
)


class _FakeTransDataOpType(Enum):
    SDMA = 1
    DEVICE_RDMA = 2
    DEVICE_URMA = 3
    HOST_RDMA = 4


def _fake_memfabric(raw_engine: MagicMock) -> ModuleType:
    module = ModuleType("memfabric_hybrid")
    module.TransferEngine = MagicMock(return_value=raw_engine)  # type: ignore[attr-defined]
    module.set_conf_store_tls = MagicMock()  # type: ignore[attr-defined]
    module.set_log_level = MagicMock()  # type: ignore[attr-defined]
    module.TransDataOpType = _FakeTransDataOpType  # type: ignore[attr-defined]
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
        data_op_type=_FakeTransDataOpType.SDMA,
    )
    assert raw_engine.method_calls[:2] == [
        call.initialize(
            "tcp://192.168.1.10",
            "192.168.1.10",
            MEMFABRIC_ROLE_PREFILL,
            3,
            store_server_role=MEMFABRIC_ROLE_PREFILL,
            data_op_type=_FakeTransDataOpType.SDMA,
        ),
        call.get_rpc_port(),
    ]


def test_memfabric_configuration_is_idempotent_but_role_bound():
    manager = GlobalMemfabricTE()

    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="device_urma")
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="device_urma")

    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=0, transfer_protocol="device_urma")
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=1, transfer_protocol="device_urma")
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="sdma")
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(
            role=MEMFABRIC_ROLE_DECODE,
            device_id=0,
            transfer_protocol="device_urma",
            store_server_role=MEMFABRIC_ROLE_DECODE,
        )


def test_push_initializes_decode_as_session_server():
    raw_engine = MagicMock()
    raw_engine.initialize.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=0, store_server_role=MEMFABRIC_ROLE_DECODE)
    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")
    assert raw_engine.initialize.call_args.kwargs["store_server_role"] == MEMFABRIC_ROLE_DECODE


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


def test_memfabric_transfer_protocol_defaults_to_sdma():
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0)

    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")

    assert raw_engine.initialize.call_args.kwargs["data_op_type"] == _FakeTransDataOpType.SDMA
    assert raw_engine.initialize.call_args.kwargs["store_server_role"] == MEMFABRIC_ROLE_PREFILL


@pytest.mark.parametrize(
    "protocol,expected",
    [
        ("sdma", _FakeTransDataOpType.SDMA),
        ("device_rdma", _FakeTransDataOpType.DEVICE_RDMA),
        ("device_urma", _FakeTransDataOpType.DEVICE_URMA),
        (" DEVICE_URMA ", _FakeTransDataOpType.DEVICE_URMA),
    ],
)
def test_memfabric_transfer_protocol_from_configure(protocol, expected):
    raw_engine = MagicMock()
    raw_engine.get_rpc_port.return_value = 23456
    raw_engine.initialize.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=1, transfer_protocol=protocol)

    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")

    assert raw_engine.initialize.call_args.kwargs["data_op_type"] == expected


def test_memfabric_transfer_protocol_rejects_unknown_value():
    manager = GlobalMemfabricTE()

    with pytest.raises(ValueError, match="Invalid MemFabric transfer_protocol"):
        manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="rocev2")


def test_host_rdma_initialization_uses_local_roce_nic():
    raw_engine = MagicMock()
    raw_engine.initialize.return_value = 0
    raw_engine.get_rpc_port.return_value = 23456
    manager = GlobalMemfabricTE()
    settings = dict(
        role=MEMFABRIC_ROLE_DECODE,
        device_id=3,
        transfer_protocol=" HOST_RDMA ",
        host_rdma_ip="192.0.2.10",
        host_rdma_port_base=41000,
    )
    manager.configure(**settings)
    manager.configure(**settings)
    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("198.51.100.10")

    assert manager.unique_id == "198.51.100.10:23456"
    raw_engine.initialize.assert_called_once_with(
        "tcp://198.51.100.10",
        "198.51.100.10",
        MEMFABRIC_ROLE_DECODE,
        3,
        store_server_role=MEMFABRIC_ROLE_PREFILL,
        data_op_type=_FakeTransDataOpType.HOST_RDMA,
        nic="tcp://192.0.2.10:41048",
    )
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure(**(settings | {"host_rdma_port_base": 42000}))


@pytest.mark.parametrize(
    "ip,port,match",
    [
        (None, 41000, "explicit local IPv4"),
        ("192.0.2.10", True, "integer HCOM base port"),
        ("127.0.0.1", 41000, "Unusable host_rdma"),
        ("192.0.2.10", 1023, "Unusable host_rdma"),
        ("192.0.2.10", 64500, "Unusable host_rdma"),
    ],
)
def test_host_rdma_rejects_invalid_local_endpoint(ip, port, match):
    with pytest.raises(ValueError, match=match):
        GlobalMemfabricTE().configure(
            role=MEMFABRIC_ROLE_DECODE,
            device_id=3,
            transfer_protocol="host_rdma",
            host_rdma_ip=ip,
            host_rdma_port_base=port,
        )


def test_device_transport_rejects_host_rdma_endpoint():
    with pytest.raises(ValueError, match="require explicit host_rdma"):
        GlobalMemfabricTE().configure(
            role=MEMFABRIC_ROLE_PREFILL,
            device_id=0,
            transfer_protocol="device_urma",
            host_rdma_ip="192.0.2.10",
            host_rdma_port_base=41000,
        )


def test_host_rdma_reports_sdk_without_protocol():
    manager = GlobalMemfabricTE()
    manager.configure(
        role=MEMFABRIC_ROLE_DECODE,
        device_id=0,
        transfer_protocol="host_rdma",
        host_rdma_ip="192.0.2.10",
        host_rdma_port_base=41000,
    )
    raw_engine = MagicMock()
    module = _fake_memfabric(raw_engine)
    module.TransDataOpType = Enum("LegacyTransDataOpType", {"SDMA": 1})
    with (
        patch.dict(sys.modules, {"memfabric_hybrid": module}),
        pytest.raises(ValueError, match="does not support host_rdma"),
    ):
        manager.get_transfer_engine("198.51.100.10")
    raw_engine.initialize.assert_not_called()
