# SPDX-License-Identifier: Apache-2.0
import ctypes
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.prefill import PrefillExpertPool
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer


def preparation_double(monkeypatch, *, fail_copy=None):
    store = object.__new__(HostExpertStore)
    store.names = MXFP4ExpertAdapter.tensor_names
    store.num_experts = 3
    store.tensors = {name: torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) for name in store.names}
    store.prepared = {}
    store.prototypes = {}
    store.finalized = True
    calls = []

    class Scheme:
        group_size = 32

        def process_weights_after_loading(self, layer):
            calls.append("convert")
            for name in store.names:
                getattr(layer, name).data = getattr(layer, name).data.transpose(-1, -2)

    adapter = MXFP4ExpertAdapter(SimpleNamespace(activation="silu"), Scheme())
    destination = adapter.materialize(store, [0, 1], "cpu")
    calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(get_storage_size=lambda t: t.numel(), get_npu_format=lambda t: 29),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda device: SimpleNamespace(synchronize=lambda: calls.append("sync"))),
        raising=False,
    )
    original_empty = torch.empty

    def empty(*args, **kwargs):
        assert kwargs.pop("pin_memory", False)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)

    def memcpy(dst, capacity, src, size, direction):
        assert direction == 2 and capacity == size
        calls.append("copy")
        if calls.count("copy") == fail_copy:
            return 42
        ctypes.memmove(dst, src, size)
        return 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy=memcpy)))
    return store, adapter, destination, calls


def test_prepare_once_owns_physical_bytes_and_retires_raw_layer(monkeypatch):
    store, adapter, destination, calls = preparation_double(monkeypatch)
    original = store.tensors["w13_weight"]
    expected = original.view(torch.uint8).reshape(3, -1).clone()
    store.prepare_native(adapter, destination, "cpu")
    assert calls == ["convert", "sync", "copy", "copy", "copy", "copy"] * 3
    assert not store.tensors
    for prepared in store.prepared.values():
        torch.testing.assert_close(prepared.data, expected)
        assert prepared.shape == (3, 2)
        assert prepared.stride == (6, 1, 3)
        assert prepared.npu_format == 29 and prepared.expert_bytes == 24
    original.zero_()
    torch.testing.assert_close(store.prepared["w13_weight"].data, expected)
    assert store.prototypes["w13_weight"].sum() > 0
    with pytest.raises(RuntimeError, match="copied directly"):
        adapter.materialize(store, [1], "cpu")
    with pytest.raises(RuntimeError, match="finalized raw"):
        store.prepare_native(adapter, destination, "cpu")
    store.close()
    assert not store.prepared and not store.prototypes


def test_preparation_failure_does_not_publish_or_retire_raw_weights(monkeypatch):
    store, adapter, destination, calls = preparation_double(monkeypatch, fail_copy=5)
    with pytest.raises(RuntimeError, match="ACL error 42"):
        store.prepare_native(adapter, destination, "cpu")
    assert not store.prepared and not store.prototypes
    assert set(store.tensors) == set(store.names)
    assert calls.count("convert") == 2


def test_native_pool_allocation_uses_only_one_raw_prototype(monkeypatch):
    store, adapter, destination, _ = preparation_double(monkeypatch)
    store.prepare_native(adapter, destination, "cpu")
    pool = adapter.allocate_pool(store, 3, "cpu")
    for name, tensor in pool.items():
        expected = store.prototypes[name].transpose(-1, -2).expand(3, 3, 2)
        torch.testing.assert_close(tensor, expected)
        assert tensor.stride() == destination[name].stride()


def test_prepared_bytes_roundtrip_into_transposed_slots_without_conversion(monkeypatch):
    store, adapter, destination, calls = preparation_double(monkeypatch)
    expected = store.tensors["w13_weight"][[2, 1]].transpose(-1, -2).clone()
    store.prepare_native(adapter, destination, "cpu")
    calls.clear()

    def memcpy_async(dst, capacity, src, size, direction, stream):
        assert direction == 1 and capacity == size and stream == 123
        ctypes.memmove(dst, src, size)
        calls.append("H2D")
        return 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy_async=memcpy_async)))
    monkeypatch.setattr(torch, "device", lambda device: SimpleNamespace(type="npu"))
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(npu_stream=123, synchronize=lambda: calls.append("sync"))
        ),
    )
    TorchExpertTransfer.copy_experts(adapter, store, destination, [(2, 0), (1, 1)], "npu:0")
    assert calls == ["H2D"] * 8 + ["sync"]
    for tensor in destination.values():
        torch.testing.assert_close(tensor, expected)


def test_native_prefill_refill_reuses_addresses_and_restores_every_tensor(monkeypatch):
    store, adapter, destination, calls = preparation_double(monkeypatch)
    expected = {name: tensor.transpose(-1, -2).clone() for name, tensor in store.tensors.items()}
    store.specs = ()
    store.prepare_native(adapter, destination, "cpu")
    binding = SimpleNamespace(adapter=adapter, host_store=store, num_experts=store.num_experts, device="cpu")
    pool = PrefillExpertPool()
    pool.initialize(binding)
    addresses = {name: tensor.data_ptr() for name, tensor in pool.tensors.items()}
    calls.clear()

    def memcpy_async(dst, capacity, src, size, direction, stream):
        assert direction == 1 and capacity == size and stream == 123
        ctypes.memmove(dst, src, size)
        calls.append("H2D")
        return 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy_async=memcpy_async)))
    monkeypatch.setattr(torch, "device", lambda device: SimpleNamespace(type="npu"))
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(npu_stream=123, synchronize=lambda: calls.append("sync"))
        ),
    )
    for _ in range(2):
        for tensor in pool.tensors.values():
            tensor.zero_()
        actual = pool.fill(binding)
        assert {name: tensor.data_ptr() for name, tensor in actual.items()} == addresses
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor, expected[name])
    assert calls == (["H2D"] * (store.num_experts * len(store.names)) + ["sync"]) * 2
    pool.close()
    assert not pool.tensors and pool.signature is None
