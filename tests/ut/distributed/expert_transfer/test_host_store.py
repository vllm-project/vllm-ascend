# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.expert_transfer.expert_offload_manager import HostExpertStore
from vllm_ascend.distributed.expert_transfer.expert_weight_transfer import TorchExpertTransfer


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_native_storage_chunking_transfer_and_full_pool(expert_factory, manager_factory, native_runtime, kind):
    layer, _ = expert_factory(kind, experts=71)
    manager = manager_factory(layer)
    binding = layer._expert_offload_binding
    store = binding.host_store
    expected = {name: tensor.clone() for name, tensor in manager.full_pool.tensors.items()}
    raw = store.tensors["w13_weight"]
    store.convert_to_native_layout(binding.adapter, "cpu")
    assert not store.tensors
    assert [shape[0] for shape in native_runtime.allocations] == [32, 32, 7] * len(binding.adapter.output_names)
    assert all(prepared.num_experts == 71 for prepared in store.prepared.values())
    raw.zero_()
    for expert_ids in ((0, 1), (31, 32), (64, 70)):
        native_runtime.calls.clear()
        TorchExpertTransfer.copy_experts(
            binding.adapter, store, binding.resident, tuple(zip(expert_ids, (0, 1))), "cpu"
        )
        assert native_runtime.calls[-1] == "sync"
        assert len(native_runtime.calls) == 2 * len(binding.adapter.output_names) + 1
        for name in binding.adapter.output_names:
            torch.testing.assert_close(binding.resident[name].float(), expected[name][list(expert_ids)].float())
    addresses = {name: tensor.data_ptr() for name, tensor in manager.full_pool.tensors.items()}
    for _ in range(2):
        for tensor in manager.full_pool.tensors.values():
            tensor.zero_()
        actual = manager.full_pool.load_layer(binding)
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor.float(), expected[name].float())
            assert tensor.data_ptr() == addresses[name]


def test_nz_transfer_counts_physical_padding_and_slot_offset(native_runtime):
    source = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)[..., :3]
    adapter = SimpleNamespace(load_experts=lambda *_: {"weight": source})
    store = HostExpertStore(SimpleNamespace(local_num_experts=1), ("weight",))
    store.tensors["weight"] = source.clone()
    store.convert_to_native_layout(adapter, "cpu")
    assert source.numel() * source.element_size() == 24
    assert store.prepared["weight"].expert_bytes == 48
    destination = torch.zeros(3, 2, 6, dtype=torch.float32)
    TorchExpertTransfer.copy_experts(adapter, store, {"weight": destination}, ((0, 2),), "cpu")
    assert not destination[:2].count_nonzero()
    torch.testing.assert_close(destination[2].reshape(-1), torch.arange(12, dtype=torch.float32))


def test_native_storage_rejects_offset_views_before_copy(native_runtime, monkeypatch):
    source = torch.arange(16, dtype=torch.float32)[4:].reshape(1, 2, 6)
    adapter = SimpleNamespace(load_experts=lambda *_: {"weight": source})
    store = HostExpertStore(SimpleNamespace(local_num_experts=1), ("weight",))
    store.tensors["weight"] = source.clone()
    copy = Mock(return_value=0)
    monkeypatch.setattr(sys.modules["acl"].rt, "memcpy", copy)

    with pytest.raises(ValueError, match="zero-offset.*weight.*expert 0"):
        store.convert_to_native_layout(adapter, "cpu")

    copy.assert_not_called()
    assert store.prepared == {}
    assert "weight" in store.tensors


@pytest.mark.parametrize("storage_elements", [0, 13])
def test_native_storage_rejects_invalid_copy_size_before_copy(native_runtime, monkeypatch, storage_elements):
    source = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    adapter = SimpleNamespace(load_experts=lambda *_: {"weight": source})
    store = HostExpertStore(SimpleNamespace(local_num_experts=1), ("weight",))
    store.tensors["weight"] = source.clone()
    copy = Mock(return_value=0)
    monkeypatch.setattr(sys.modules["acl"].rt, "memcpy", copy)
    monkeypatch.setattr(sys.modules["torch_npu"], "get_storage_size", lambda _: storage_elements)

    with pytest.raises(ValueError, match="invalid native storage size.*weight.*expert 0"):
        store.convert_to_native_layout(adapter, "cpu")

    copy.assert_not_called()
    assert store.prepared == {}
    assert "weight" in store.tensors


def test_native_storage_rejects_inconsistent_expert_sizes_before_copy(native_runtime, monkeypatch):
    sources = (
        torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
        torch.arange(10, dtype=torch.float32).reshape(1, 2, 5),
    )
    adapter = SimpleNamespace(load_experts=lambda _, ids, device: {"weight": sources[ids[0]]})
    store = HostExpertStore(SimpleNamespace(local_num_experts=2), ("weight",))
    store.tensors["weight"] = torch.zeros(2, 2, 6)
    copy = Mock(return_value=0)
    monkeypatch.setattr(sys.modules["acl"].rt, "memcpy", copy)

    with pytest.raises(ValueError, match="inconsistent native expert storage size.*weight.*expert 1"):
        store.convert_to_native_layout(adapter, "cpu")

    assert copy.call_count == 1
    assert copy.call_args.args[1] == 48
    assert store.prepared == {}
    assert "weight" in store.tensors
