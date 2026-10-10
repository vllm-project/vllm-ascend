# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import gc
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_ascend.distributed.eplb import eplb_communicator


class _Tensor:
    def __init__(self, address, device, shape=(2, 4), nbytes=32):
        self._address = address
        self.device = device
        self.shape = shape
        self.ndim = len(shape)
        self.nbytes = nbytes

    def data_ptr(self):
        return self._address

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=self.data_ptr, nbytes=lambda: self.nbytes)

    def is_contiguous(self):
        return True


class _Engine:
    def __init__(self):
        self.registered = []
        self.connected = []
        self.transfers = []
        self.deregistered = []
        self.finalized = False

    def initialize(self, local_engine, options):
        self.local_engine = local_engine
        self.options = options
        return 0

    def register_mem(self, descriptor, _mem_type):
        self.registered.append(descriptor)
        return 0, len(self.registered)

    def connect(self, remote_engine, timeout):
        self.connected.append((remote_engine, timeout))
        return 0

    def transfer_async(self, remote_engine, operation, descriptors):
        self.transfers.append((remote_engine, operation, descriptors))
        return 0, 100 + len(self.transfers)

    def get_transfer_status(self, _request):
        return 0, 1

    def disconnect(self, _remote_engine):
        return 0

    def deregister_mem(self, handle):
        self.deregistered.append(handle)
        return 0

    def finalize(self):
        self.finalized = True


def _fake_hixl(engine):
    class MemDesc:
        def __init__(self, address, size):
            self.addr = address
            self.len = size

    class TransferOpDesc:
        def __init__(self, *, local_addr, remote_addr, len):
            self.local_addr = local_addr
            self.remote_addr = remote_addr
            self.len = len

    return SimpleNamespace(
        SUCCESS=0,
        Hixl=lambda: engine,
        MemDesc=MemDesc,
        MemType=SimpleNamespace(MEM_DEVICE=0),
        TransferOp=SimpleNamespace(READ=0),
        TransferOpDesc=TransferOpDesc,
        TransferStatus=SimpleNamespace(WAITING=0, COMPLETED=1),
    )


def test_hixl_reads_registered_remote_expert(monkeypatch):
    engine = _Engine()
    monkeypatch.setitem(sys.modules, "hixl", _fake_hixl(engine))
    context = object()
    set_context = MagicMock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        "acl",
        SimpleNamespace(rt=SimpleNamespace(get_context=lambda: (context, 0), set_context=set_context)),
    )
    monkeypatch.setattr(eplb_communicator, "is_weak_contiguous", lambda _tensor: True)
    set_device = MagicMock()
    memory_snapshot = MagicMock(
        return_value=[
            {
                "device": 0,
                "address": 0,
                "total_size": 8_388_608,
            }
        ]
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=set_device, memory_snapshot=memory_snapshot),
        raising=False,
    )
    monkeypatch.setattr(eplb_communicator, "get_ip", lambda: "192.0.2.1")
    monkeypatch.setattr(eplb_communicator, "get_open_port", lambda: 12345)

    group = MagicMock()
    group.rank.return_value = 0
    group.size.return_value = 2

    def all_gather(gathered, local_state, *, group):
        gathered[0] = local_state
        gathered[1] = (
            "192.0.2.2:12346",
            {
                key: (tuple(address + 10_000 for address in addresses), stride)
                for key, (addresses, stride) in local_state[1].items()
            },
        )

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather)
    work = MagicMock()

    def all_reduce(completed, *, group, async_op):
        completed.fill_(2)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    device = SimpleNamespace(type="npu", index=0)
    weights = [
        [_Tensor(1000, device), [_Tensor(2100, device, (4,), 16), _Tensor(2200, device, (4,), 16)]],
        [_Tensor(3000, device), [_Tensor(4100, device, (4,), 16), _Tensor(4200, device, (4,), 16)]],
    ]
    buffer_tensor_list = [_Tensor(6100, device, (4,), 16), _Tensor(6200, device, (4,), 16)]
    buffers = [
        _Tensor(5000, device),
        buffer_tensor_list,
    ]
    communicator = eplb_communicator.AscendHixlEplbCommunicator(group, weights, buffers)

    communicator.set_stream(None)
    set_context.assert_called_once_with(context)
    communicator.set_transfer_context(np.array([0, 1, 2, 3]), layer_idx=1)
    communicator.add_send([weights[1][0]], dst_rank=1, expert_id=0)
    with pytest.raises(RuntimeError, match="receive size"):
        communicator.add_recv(
            [_Tensor(5000, device, shape=(2,), nbytes=8), buffer_tensor_list[0]],
            src_rank=1,
            expert_id=3,
        )
    communicator.add_recv(
        [_Tensor(5000, device, shape=(4,), nbytes=16), buffer_tensor_list[0]],
        src_rank=1,
        expert_id=3,
    )
    communicator.execute()

    assert engine.local_engine == "192.0.2.1:12345"
    assert engine.options == {}
    assert engine.connected == [("192.0.2.2:12346", 300_000)]
    assert len(engine.registered) == 1
    assert (engine.registered[0].addr, engine.registered[0].len) == (0, 2_097_152)
    assert len(engine.transfers) == 1
    remote_engine, operation, descriptors = engine.transfers[0]
    assert remote_engine == "192.0.2.2:12346"
    assert operation == 0
    assert [(desc.local_addr, desc.remote_addr, desc.len) for desc in descriptors] == [
        (5000, 13016, 16),
        (6100, 14200, 16),
    ]
    assert memory_snapshot.call_count == 0
    assert work.wait.call_count == 6
    timing = communicator._eplb_hixl_phase_timings[0]
    assert (timing.request_count, timing.transfer_bytes) == (1, 32)

    communicator._close()
    assert engine.deregistered == [1]
    assert engine.finalized


def test_hixl_rejects_interleaved_expert_rows():
    communicator = object.__new__(eplb_communicator.AscendHixlEplbCommunicator)
    communicator._device = torch.device("cpu")
    communicator._num_local_experts = 4

    stacked = torch.arange(8).reshape(2, 4).T
    assert stacked.shape == (4, 2)
    assert stacked.stride() == (1, 4)
    assert eplb_communicator.is_weak_contiguous(stacked)

    with pytest.raises(ValueError, match="contiguous expert rows"):
        communicator._validate_view(stacked)
    with pytest.raises(ValueError, match="contiguous expert rows"):
        communicator._validate_view(torch.empty_like(stacked))

    communicator._validate_view(stacked.contiguous())


def test_single_per_expert_tensor_keeps_its_full_transfer_range(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._num_local_experts = 1
    communicator._world_size = 1
    communicator._cpu_group = object()
    monkeypatch.setattr(eplb_communicator, "is_weak_contiguous", lambda _tensor: True)
    view = [_Tensor(1000, communicator._device, shape=(8,), nbytes=32)]
    communicator._validate_view(view)
    gather = MagicMock()
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    communicator._exchange_remote_state("local", [[view]])
    assert gather.call_args.args[1] == ("local", {(0, 0): ((1000,), 32)})


def test_hixl_close_retains_resources_on_disconnect_failure(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._remote_engines = {1: "peer"}
    communicator._engine.disconnect = MagicMock(return_value=103900)
    communicator._registered_handles = [1]
    communicator._registered_regions = [(0, 2_097_152)]
    storage = object()
    communicator._storage_refs = [storage]

    with pytest.raises(RuntimeError, match="disconnect"):
        communicator.close()

    assert communicator._engine is not None
    assert communicator._registered_handles == [1]
    assert communicator._storage_refs == [storage]
    assert communicator._engine.deregistered == []
    assert not communicator._engine.finalized


def _bare_communicator(monkeypatch):
    communicator = object.__new__(eplb_communicator.AscendHixlEplbCommunicator)
    communicator._rank = 0
    communicator._engine = _Engine()
    communicator._hixl = _fake_hixl(communicator._engine)
    communicator._device = SimpleNamespace(type="npu", index=0)
    communicator._acl_context = None
    communicator._remote_engines = {}
    communicator._remote_send_meta = {}
    communicator._registered_handles = []
    communicator._registered_regions = []
    communicator._storage_refs = []
    communicator._requests = []
    communicator._pending_reads = {}
    communicator._usable = True
    communicator._confirm_all_ranks = lambda error, _operation: (_ for _ in ()).throw(error) if error else None
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _device: None), raising=False)
    return communicator


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ([(0, 84), (0, 86)], [(0, 86)]),
        ([(0, 4), (2, 4)], [(0, 6)]),
        ([(0, 2), (2, 2)], [(0, 4)]),
        ([(0, 2), (4, 2)], [(0, 2), (4, 2)]),
        ([(1, 1), (1, 1)], [(0, 2)]),
    ],
)
def test_registration_merges_live_ranges_without_bridging_holes(monkeypatch, ranges, expected):
    communicator = _bare_communicator(monkeypatch)
    monkeypatch.setattr(eplb_communicator, "_HIXL_MEMORY_ALIGNMENT", 2)
    tensors = [_Tensor(start, communicator._device, nbytes=size) for start, size in ranges]
    communicator._register_tensor_segments(tensors)
    assert [(d.addr, d.len) for d in communicator._engine.registered] == expected
    assert len(communicator._storage_refs) == len(tensors)
    communicator.close()
    assert communicator._storage_refs == []


def test_registration_keeps_old_storage_alive_after_tensor_rebinding(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    tensor = torch.empty(8)
    communicator._register_tensor_segments([tensor])
    old_storage = weakref.ref(communicator._storage_refs[0])
    old_address = tensor.data_ptr()
    tensor.data = torch.empty(8)
    del tensor
    storage = old_storage()
    assert storage is not None
    assert storage.data_ptr() == old_address
    del storage
    communicator.close()
    assert old_storage() is None


def test_partial_submission_is_drained_before_reporting_failure(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._remote_engines = {1: "one", 2: "two"}
    communicator._pending_reads = {1: [(0, 0, 8)], 2: [(8, 8, 8)]}
    communicator._engine.transfer_async = MagicMock(side_effect=[(0, 11), (103900, 0)])
    communicator._engine.get_transfer_status = MagicMock(return_value=(0, 1))
    communicator._pending_bytes = 16
    communicator._layer_idx = 0
    with pytest.raises(RuntimeError, match="read from rank 2"):
        communicator.execute()
    communicator._engine.get_transfer_status.assert_called_once_with(11)
    assert communicator._requests == []
    assert not communicator._usable


def test_close_preserves_failed_deregistration_handle(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._registered_handles = [1, 2]
    communicator._registered_regions = [(0, 2), (4, 2)]
    communicator._storage_refs = [object()]
    communicator._engine.deregister_mem = MagicMock(side_effect=[0, 103900])
    with pytest.raises(RuntimeError, match="deregister"):
        communicator.close()
    assert communicator._registered_handles == [1]
    assert communicator._registered_regions == [(0, 2)]
    assert communicator._storage_refs
    assert not communicator._engine.finalized


def test_close_is_idempotent(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    engine = communicator._engine
    communicator.close()
    communicator.close()
    assert engine.finalized
    assert communicator._engine is None


def test_initialization_rollback_participates_without_local_engine(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._engine = None
    communicator._initializing = True
    communicator._confirm_all_ranks = MagicMock()
    communicator.close()
    assert [call.args[1] for call in communicator._confirm_all_ranks.call_args_list] == [
        "drain",
        "disconnect",
        "deregistration",
        "finalization",
    ]
    communicator.close()
    assert communicator._confirm_all_ranks.call_count == 4


def test_failed_cpu_group_is_not_reused_for_cleanup(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._group_error = None
    communicator._world_size = 2
    communicator._cpu_group = object()
    del communicator._confirm_all_ranks
    collective = MagicMock(side_effect=TimeoutError("collective timeout"))
    monkeypatch.setattr(torch.distributed, "all_reduce", collective)
    with pytest.raises(TimeoutError, match="collective timeout"):
        communicator._confirm_all_ranks(None, "initialization")
    with pytest.raises(RuntimeError, match="CPU group failed"):
        communicator.close()
    assert collective.call_count == 1
    assert communicator._engine is not None


def test_rank_confirmation_uses_cpu_under_a_device_context(monkeypatch):
    communicator = _bare_communicator(monkeypatch)
    communicator._group_error = None
    communicator._world_size = 2
    communicator._cpu_group = object()
    del communicator._confirm_all_ranks

    def all_reduce(tensor, **_kwargs):
        assert tensor.device.type == "cpu"
        tensor.fill_(2)
        return MagicMock()

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    with torch.device("meta"):
        communicator._confirm_all_ranks(None, "initialization")


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_initialization_rollback_failure_bypasses_rpc_and_retains_storage(monkeypatch, rollback_fails):
    communicator = _bare_communicator(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "acl",
        SimpleNamespace(rt=SimpleNamespace(get_context=lambda: (None, 0), set_context=lambda _context: 0)),
    )
    monkeypatch.setattr(eplb_communicator, "get_ip", lambda: "192.0.2.1")
    monkeypatch.setattr(eplb_communicator, "get_open_port", lambda: 12345)
    communicator._validate_tensors = MagicMock()
    storage_ref = None

    def fail_registration(_tensors):
        nonlocal storage_ref
        storage = torch.empty(4).untyped_storage()
        storage_ref = weakref.ref(storage)
        communicator._storage_refs = [storage]
        communicator._registered_handles = [1]
        communicator._registered_regions = [(0, 2)]
        raise RuntimeError("registration failed")

    communicator._register_tensor_segments = fail_registration
    if rollback_fails:
        communicator._engine.deregister_mem = MagicMock(return_value=103900)
    rpc = SimpleNamespace(
        rank=0,
        worker=SimpleNamespace(rebuild=lambda: communicator._initialize([], [])),
        handle_output=MagicMock(),
    )
    if rollback_fails:
        with pytest.raises(SystemExit, match="worker must terminate") as fatal:
            WorkerProc._execute_worker_rpc(rpc, ("rebuild", (), {}, None))
        rpc.handle_output.assert_not_called()
        cause = fatal.value.__cause__
        assert cause is not None and hasattr(cause, "hixl_communicator")
        owner = cause.hixl_communicator
        assert owner._engine is not None
        assert owner._registered_handles == [1]
        del rpc, communicator, owner
        gc.collect()
        assert storage_ref is not None
        assert storage_ref() is not None
    else:
        WorkerProc._execute_worker_rpc(rpc, ("rebuild", (), {}, None))
        assert isinstance(rpc.handle_output.call_args.args[0], RuntimeError)
        assert communicator._engine is None
        assert communicator._storage_refs == []
