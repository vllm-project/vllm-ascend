"""Unit tests for the isolated MemFabric transfer-engine singleton."""

import mmap
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


def test_memfabric_reuses_pinned_host_registration():
    raw_engine = MagicMock()
    raw_engine.initialize.return_value = 0
    raw_engine.register_memory.return_value = 0
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="device_rdma")
    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")
    host = MagicMock(nbytes=4096)
    host.data_ptr.return_value = 4096

    manager.register_pinned_host_buffer(host)
    with pytest.raises(ValueError, match="extent changed"):
        manager.register_buffer([8192, 4096], [64, 4095])
    manager.register_buffer([4096, 8192], [4096, 64])

    assert raw_engine.register_memory.call_args_list == [call(4096, 4096), call(8192, 64)]


def test_memfabric_pinned_host_registration_can_retry_after_failure():
    raw_engine = MagicMock()
    raw_engine.initialize.return_value = 0
    raw_engine.register_memory.side_effect = [1, 0]
    manager = GlobalMemfabricTE()
    manager.configure(role=MEMFABRIC_ROLE_DECODE, device_id=0, transfer_protocol="device_rdma")
    with patch.dict(sys.modules, {"memfabric_hybrid": _fake_memfabric(raw_engine)}):
        manager.get_transfer_engine("127.0.0.1")
    host = MagicMock(nbytes=4096)
    host.data_ptr.return_value = 4096

    with pytest.raises(RuntimeError, match="registration failed"):
        manager.register_pinned_host_buffer(host)
    manager.register_pinned_host_buffer(host)
    with pytest.raises(RuntimeError, match="already been registered"):
        manager.register_pinned_host_buffer(host)
    assert raw_engine.register_memory.call_args_list == [call(4096, 4096)] * 2


@pytest.mark.parametrize("lookup_status, device_ptr", [(0, 8192), (1, None), (0, None)])
def test_memfabric_pinned_host_reuses_device_mapping(monkeypatch, lookup_status, device_ptr):
    from vllm.utils import network_utils
    from vllm.v1.hisparse import runtime

    from vllm_ascend.distributed.kv_transfer.utils import memfabric_transfer_engine
    from vllm_ascend.hisparse import host_pool

    engine = MagicMock()
    host = MagicMock()
    host.data_ptr.return_value = mmap.PAGESIZE
    empty = MagicMock(return_value=host)
    memory_check = MagicMock()
    registration = MagicMock()
    acl = MagicMock()

    def get_device_pointer(address, mapped, flags):
        engine.register_pinned_host_buffer.assert_called_once_with(host)
        assert (address, flags) == (mmap.PAGESIZE, 0)
        mapped._obj.value = device_ptr
        return lookup_status

    acl.aclrtHostGetDevicePointer.side_effect = get_device_pointer
    monkeypatch.setattr(host_pool.ctypes, "CDLL", MagicMock(return_value=acl))
    monkeypatch.setattr(host_pool.torch, "empty", empty)
    monkeypatch.setattr(host_pool.torch, "npu", MagicMock(current_device=MagicMock(return_value=3)), raising=False)
    monkeypatch.setattr(host_pool, "MappedHostTensor", registration)
    monkeypatch.setattr(runtime, "check_hisparse_host_memory", memory_check)
    monkeypatch.setattr(network_utils, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(memfabric_transfer_engine, "global_memfabric_te", engine)

    if device_ptr is None:
        with pytest.raises(RuntimeError, match="device address lookup failed"):
            host_pool._allocate_memfabric_pinned_host(64)
        registration.assert_not_called()
    else:
        source = host_pool._allocate_memfabric_pinned_host(64)
        registration.assert_called_once_with(host, memfabric_device_ptr=device_ptr)
        host.__getitem__.assert_called_once_with(slice(None, 64))
        assert source._hisparse_registration is registration.return_value

    memory_check.assert_called_once_with(mmap.PAGESIZE)
    empty.assert_called_once_with(mmap.PAGESIZE, dtype=host_pool.torch.int8, pin_memory=True)
    engine.configure.assert_called_once_with(
        role=MEMFABRIC_ROLE_DECODE,
        device_id=3,
        transfer_protocol="device_rdma",
        store_server_role=MEMFABRIC_ROLE_DECODE,
    )
    engine.get_transfer_engine.assert_called_once_with("127.0.0.1")
    acl.aclrtHostRegister.assert_not_called()


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
