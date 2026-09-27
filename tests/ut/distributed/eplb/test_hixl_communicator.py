# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.eplb import hixl_communicator


class _Tensor:
    def __init__(self, address, device, shape=(2, 4), nbytes=32):
        self._address = address
        self.device = device
        self.shape = shape
        self.ndim = len(shape)
        self.nbytes = nbytes

    def data_ptr(self):
        return self._address


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

    def read_async_batch(self, batches):
        results = []
        for remote_engine, raw_descriptors in batches:
            descriptors = [
                SimpleNamespace(local_addr=local, remote_addr=remote, len=length)
                for local, remote, length in raw_descriptors
            ]
            results.append(self.transfer_async(remote_engine, 0, descriptors))
        return results

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
    monkeypatch.setitem(sys.modules, "vllm_ascend._hixl", _fake_hixl(engine))
    monkeypatch.setattr(hixl_communicator, "is_weak_contiguous", lambda _tensor: True)
    set_device = MagicMock()
    memory_snapshot = MagicMock(
        return_value=[
            {
                "device": 0,
                "address": 0,
                "total_size": 2_097_152,
            }
        ]
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(set_device=set_device, memory_snapshot=memory_snapshot),
        raising=False,
    )
    monkeypatch.setattr(hixl_communicator, "get_ip", lambda: "192.0.2.1")
    monkeypatch.setattr(hixl_communicator, "get_open_port", lambda: 12345)

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
    communicator = hixl_communicator.AscendHixlEplbCommunicator(group, weights, buffers)

    communicator.set_stream(None)
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
    assert work.wait.call_count == 1
    communicator.wait_for_transfer_safety()

    assert engine.local_engine == "192.0.2.1:12345"
    assert engine.options == {}
    assert engine.connected == [("192.0.2.2:12346", 300_000)]
    assert len(engine.registered) == 1
    assert all(descriptor.addr % 2_097_152 == 0 for descriptor in engine.registered)
    assert len(engine.transfers) == 1
    remote_engine, operation, descriptors = engine.transfers[0]
    assert remote_engine == "192.0.2.2:12346"
    assert operation == 0
    assert [(desc.local_addr, desc.remote_addr, desc.len) for desc in descriptors] == [
        (5000, 13016, 16),
        (6100, 14200, 16),
    ]
    assert memory_snapshot.call_count == 1
    assert work.wait.call_count == 2
    timing = communicator._eplb_hixl_phase_timings[0]
    assert (timing.request_count, timing.transfer_bytes) == (1, 32)

    communicator._close()
    assert engine.deregistered == [1]
    assert engine.finalized
