# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.transfer import TorchExpertTransfer


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_native_storage_chunking_transfer_and_prefill(expert_factory, manager_factory, native_runtime, kind):
    layer, _ = expert_factory(kind, experts=71)
    manager = manager_factory(layer)
    binding = layer._expert_offload_binding
    store = binding.host_store
    expected = {name: tensor.clone() for name, tensor in manager.prefill_pool.tensors.items()}
    raw = store.tensors["w13_weight"]
    store.prepare_native(binding.adapter, binding.resident, "cpu")
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
    addresses = {name: tensor.data_ptr() for name, tensor in manager.prefill_pool.tensors.items()}
    for _ in range(2):
        for tensor in manager.prefill_pool.tensors.values():
            tensor.zero_()
        actual = manager.prefill_pool.fill(binding)
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor.float(), expected[name].float())
            assert tensor.data_ptr() == addresses[name]


def test_nz_transfer_counts_physical_padding_and_slot_offset():
    def tensor(experts, pointer):
        return SimpleNamespace(
            device="npu:0",
            dtype=torch.float32,
            shape=(experts, 2, 3),
            stride=lambda: (6, 3, 1),
            storage_offset=lambda: 0,
            element_size=lambda: 4,
            untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 48 * experts, data_ptr=lambda: pointer),
        )

    runtime = SimpleNamespace(get_storage_size=lambda value: 12 * value.shape[0], get_npu_format=lambda _: 29)
    assert TorchExpertTransfer._physical_copy_parameters(tensor(3, 2000), tensor(1, 1000), 2, runtime) == (
        2096,
        1000,
        48,
    )
