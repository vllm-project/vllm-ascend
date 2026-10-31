# SPDX-License-Identifier: Apache-2.0
import ctypes
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.expert_offload.utils import PreparedExpertTensor


def test_native_copy_resolves_chunk_boundaries_and_partial_tail(monkeypatch):
    size, experts = 8, 71
    raw = torch.arange(experts, dtype=torch.uint8)[:, None].expand(-1, size).contiguous()
    chunks = raw.split(32)
    destination = torch.zeros((2, size), dtype=torch.uint8)
    prepared = PreparedExpertTensor(
        data=chunks[0],
        extra_data=chunks[1:],
        shape=(size,),
        stride=destination.stride(),
        dtype=destination.dtype,
        device=destination.device,
        npu_format=2,
        expert_bytes=size,
    )
    runtime = SimpleNamespace(get_storage_size=lambda tensor: tensor.numel(), get_npu_format=lambda _: 2)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _: True)
    assert prepared.num_experts == experts
    for expert_id in (0, 31, 32, 63, 64, 70):
        dst, src, count = TorchExpertTransfer._host_copy_parameters(destination, prepared, expert_id, 1, runtime)
        ctypes.memmove(dst, src, count)
        torch.testing.assert_close(destination[1], raw[expert_id])
    for expert_id in (-1, experts, True):
        with pytest.raises(ValueError, match="out of range"):
            TorchExpertTransfer._host_copy_parameters(destination, prepared, expert_id, 0, runtime)


def test_preparation_bounds_pinned_allocations_and_owns_every_chunk(monkeypatch):
    raw = torch.arange(71 * 8, dtype=torch.int32).to(torch.uint8).reshape(71, 2, 4)
    store = object.__new__(HostExpertStore)
    store.num_experts, store.finalized = 71, True
    store.prepared, store.prototypes = {}, {}
    store.tensors = {"w13_weight": raw}
    runtime = SimpleNamespace(get_storage_size=lambda tensor: tensor.numel(), get_npu_format=lambda _: 2)
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda _: SimpleNamespace(synchronize=lambda: None)), raising=False
    )
    allocations = []
    empty = torch.empty

    def pinned_empty(shape, **kwargs):
        assert kwargs.pop("pin_memory")
        allocations.append(shape)
        return empty(shape, **kwargs)

    def memcpy(dst, capacity, src, count, direction):
        assert capacity == count == 8 and direction == 2
        ctypes.memmove(dst, src, count)
        return 0

    monkeypatch.setattr(torch, "empty", pinned_empty)
    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy=memcpy)))
    adapter = SimpleNamespace(
        output_names=("w13_weight",),
        materialize=lambda host, ids, device: {"w13_weight": host.tensors["w13_weight"][list(ids)].clone()},
    )
    store.prepare_native(adapter, {"w13_weight": torch.zeros((2, 2, 4), dtype=torch.uint8)}, "cpu")
    prepared = store.prepared["w13_weight"]
    assert allocations == [(32, 8), (32, 8), (7, 8)]
    assert not store.tensors
    expected = raw.reshape(71, 8).clone()
    raw.zero_()
    torch.testing.assert_close(torch.cat((prepared.data, *prepared.extra_data)), expected)
    store.close()
    assert not store.prepared and not store.prototypes
