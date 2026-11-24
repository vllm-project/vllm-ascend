# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.expert_transfer.expert_offload_utils import PreparedExpertTensor
from vllm_ascend.distributed.expert_transfer.expert_weight_transfer import (
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_HOST_TO_DEVICE,
    TorchExpertTransfer,
)


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_full_pool_uses_current_residents_and_reuses_unchanged_layer(
    expert_factory, manager_factory, monkeypatch, kind
):
    first, _ = expert_factory(kind)
    second, _ = expert_factory(kind, layer_name="model.layers.1.mlp.experts")
    third, _ = expert_factory(kind, layer_name="model.layers.2.mlp.experts")
    for parameter in second.parameters():
        parameter.data = (parameter.data.float() + 3).to(parameter.dtype)
    manager = manager_factory(first, second, third)
    binding = first._expert_offload_binding
    expected = binding.adapter.load_experts(binding.host_store, range(binding.num_experts), "cpu")
    slots, loads = binding.plan_expert_loads({1, 4})
    binding.load_resident(slots, loads)
    transfer = Mock(wraps=TorchExpertTransfer.copy_experts)
    monkeypatch.setattr(TorchExpertTransfer, "copy_experts", transfer)

    result = manager.full_pool.load_layer(binding)
    assert transfer.call_args.args[3] == ((0, 0), (2, 2), (3, 3), (5, 5))
    assert transfer.call_args.kwargs["resident_loads"] == ((0, 4), (1, 1))
    for name in expected:
        torch.testing.assert_close(result[name].float(), expected[name].float())

    # A decode update changes the slots, but not the immutable full pool.
    slots, loads = binding.plan_expert_loads({2, 5})
    binding.load_resident(slots, loads)
    transfer.reset_mock()
    manager.full_pool.load_layer(binding)
    transfer.assert_not_called()
    manager.full_pool.load_layer(second._expert_offload_binding)
    transfer.reset_mock()
    manager.full_pool.load_layer(binding)
    transfer.assert_not_called()
    # The third layer reuses the first buffer and forces a reload of current residents.
    manager.full_pool.load_layer(third._expert_offload_binding)
    manager.full_pool.load_layer(binding)
    assert transfer.call_args.kwargs["resident_loads"] == tuple(enumerate(binding.slot_to_expert))
    for name in expected:
        torch.testing.assert_close(manager.full_pool.tensors[name].float(), expected[name].float())


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize(
    "resident_ids,host_copies,device_copies",
    [
        ((70, 31), 3, 2),
        ((0, 1), 3, 1),
        ((31, 32), 3, 1),
        ((32, 31), 3, 2),
        ((69, 70), 3, 1),
        ((1, 33), 5, 2),
    ],
)
def test_full_pool_copies_physical_bytes_only_for_missing_experts(
    expert_factory, manager_factory, native_runtime, kind, resident_ids, host_copies, device_copies
):
    layer, _ = expert_factory(kind, experts=71)
    manager = manager_factory(layer)
    binding = layer._expert_offload_binding
    expected = {name: tensor.clone() for name, tensor in manager.full_pool.tensors.items()}
    binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
    TorchExpertTransfer.copy_experts(
        binding.adapter,
        binding.host_store,
        binding.resident,
        tuple((expert_id, slot) for slot, expert_id in enumerate(resident_ids)),
        "cpu",
    )
    binding.slot_to_expert = resident_ids
    native_runtime.calls.clear()

    result = manager.full_pool.load_layer(binding)
    copies = native_runtime.calls[:-1]
    components = len(binding.adapter.output_names)
    # Cover chunk boundaries, reordered residents, and contiguous copy merging.
    assert sum(direction == ACL_MEMCPY_HOST_TO_DEVICE for direction, _ in copies) == host_copies * components
    assert sum(direction == ACL_MEMCPY_DEVICE_TO_DEVICE for direction, _ in copies) == device_copies * components
    bytes_per_expert = sum(t.expert_bytes for t in binding.host_store.prepared.values())
    assert sum(size for direction, size in copies if direction == ACL_MEMCPY_HOST_TO_DEVICE) == 69 * bytes_per_expert
    assert sum(size for direction, size in copies if direction == ACL_MEMCPY_DEVICE_TO_DEVICE) == 2 * bytes_per_expert
    assert native_runtime.calls[-1] == "sync"
    for name in expected:
        torch.testing.assert_close(result[name].float(), expected[name].float())


@pytest.mark.parametrize("has_prefill", [False, True])
def test_graph_residency_is_fenced_once_across_layer_loads(
    expert_factory, manager_factory, native_runtime, has_prefill
):
    first, _ = expert_factory()
    second, _ = expert_factory(layer_name="model.layers.1.mlp.experts")
    for parameter in second.parameters():
        parameter.data.add_(3)
    manager = manager_factory(first, second)
    bindings = [first._expert_offload_binding, second._expert_offload_binding]
    expected = []
    for binding in bindings:
        expected.append(binding.adapter.load_experts(binding.host_store, range(binding.num_experts), "cpu"))
        binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
        binding.device = SimpleNamespace(type="npu")
    manager.graph_streams.add(object())

    def finish_graph(_device):
        for binding, weights in zip(bindings, expected):
            for name in binding.resident:
                binding.resident[name][0].copy_(weights[name][4])
                binding.resident[name][1].copy_(weights[name][5])
            binding.slot_to_expert = (4, 5)

    native_runtime.npu.synchronize = Mock(side_effect=finish_graph)
    manager.has_prefill = has_prefill
    manager.run_fullgraph(lambda: None)
    manager.has_prefill = True
    for binding, weights in zip(bindings, expected):
        actual = manager.full_pool.load_layer(binding)
        for name in weights:
            torch.testing.assert_close(actual[name].float(), weights[name].float())
    native_runtime.npu.synchronize.assert_called_once_with(bindings[0].device)

    manager.has_prefill = has_prefill
    manager.run_fullgraph(lambda: None)
    manager.has_prefill = True
    manager.full_pool.load_layer(bindings[0])
    assert native_runtime.npu.synchronize.call_count == 2


@pytest.mark.parametrize("failed_direction", [ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_DEVICE])
def test_failed_prefill_transfer_invalidates_previous_owner(
    expert_factory, manager_factory, native_runtime, monkeypatch, failed_direction
):
    import acl  # type: ignore[import-not-found]

    first, _ = expert_factory()
    second, _ = expert_factory(layer_name="model.layers.1.mlp.experts")
    third, _ = expert_factory(layer_name="model.layers.2.mlp.experts")
    for parameter in third.parameters():
        parameter.data.add_(3)
    manager = manager_factory(first, second, third)
    bindings = [layer._expert_offload_binding for layer in (first, second, third)]
    expected = {name: tensor.clone() for name, tensor in manager.full_pool.tensors.items()}
    for binding in bindings:
        binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
    manager.full_pool.load_layer(bindings[0])
    manager.full_pool.load_layer(bindings[1])
    original_copy = acl.rt.memcpy_async

    seen = 0

    def fail_copy(*args):
        nonlocal seen
        if args[4] == failed_direction:
            seen += 1
        if seen == 2:
            return 1
        return original_copy(*args)

    monkeypatch.setattr(acl.rt, "memcpy_async", fail_copy)
    with pytest.raises(RuntimeError, match="asynchronous expert .* copy failed"):
        manager.full_pool.load_layer(bindings[2])
    assert manager.full_pool.owners[0] is None
    assert manager.full_pool.owners[1] is bindings[1]
    assert native_runtime.calls[-1] == "sync"
    monkeypatch.setattr(acl.rt, "memcpy_async", original_copy)
    result = manager.full_pool.load_layer(bindings[0])
    assert manager.full_pool.owners[0] is bindings[0]
    for name in expected:
        torch.testing.assert_close(result[name].float(), expected[name].float())


def test_prefill_coalesces_only_contiguous_slots_within_each_source_chunk(native_runtime):
    physical = torch.arange(7 * 2 * 6, dtype=torch.float32).reshape(7, 2, 6)
    chunks = [part.contiguous().view(torch.uint8).reshape(len(part), -1).clone() for part in physical.split(3)]
    prepared = PreparedExpertTensor(data=chunks[0], extra_data=tuple(chunks[1:]), expert_bytes=48)
    resident_storage = physical[[2, 4, 5]].clone()
    destination_storage = torch.zeros_like(physical)
    store = SimpleNamespace(prepared={"weight": prepared})

    native_runtime.calls.clear()
    TorchExpertTransfer.copy_experts(
        None,
        store,
        {"weight": destination_storage[..., :3]},
        ((0, 0), (1, 1), (3, 3), (6, 6)),
        "cpu",
        resident_weights={"weight": resident_storage[..., :3]},
        resident_loads=((0, 2), (1, 4), (2, 5)),
    )
    torch.testing.assert_close(destination_storage, physical)
    assert native_runtime.calls == [
        (ACL_MEMCPY_HOST_TO_DEVICE, 96),
        (ACL_MEMCPY_HOST_TO_DEVICE, 48),
        (ACL_MEMCPY_HOST_TO_DEVICE, 48),
        (ACL_MEMCPY_DEVICE_TO_DEVICE, 48),
        (ACL_MEMCPY_DEVICE_TO_DEVICE, 96),
        "sync",
    ]
