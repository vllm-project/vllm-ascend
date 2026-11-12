# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_loading_decode_prefill_and_capacity_fallback(expert_factory, manager_factory, kind):
    layer, scheme = expert_factory(kind)
    binding = layer._expert_offload_binding
    reference = torch.nn.Module()
    reference.activation = layer.activation
    for name in binding.adapter.tensor_names:
        reference.register_parameter(
            name, torch.nn.Parameter(getattr(layer, name).detach().clone(), requires_grad=False)
        )
    scheme.process_weights_after_loading(reference)
    checkpoint = layer.w13_weight.detach()
    manager = manager_factory(layer)
    checkpoint.zero_()
    assert binding.host_store.tensors["w13_weight"].float().sum() > 0
    addresses = {name: value.data_ptr() for name, value in binding.resident.items()}
    map_addresses = (binding.mapping.host_map.data_ptr(), binding.mapping.device_map.data_ptr())
    for prefill, logical in (
        (False, [[4, 5]]),
        (False, [[1, 4]]),
        (True, [[0, 5]]),
        (False, [[0, 1], [2, 3]]),
        (False, [[2, 2]]),
    ):
        manager.has_prefill = prefill
        ids = torch.tensor(logical, dtype=torch.int32)
        before = binding.slot_to_expert
        view, physical, _ = binding.prepare(layer, ids)
        full_pool = prefill or ids.unique().numel() > binding.num_slots
        assert manager.get_comm.call_args.args[-1] == (binding.num_experts if full_pool else binding.num_slots)
        if full_pool:
            assert torch.equal(physical, ids)
            assert binding.slot_to_expert == before
        for name in binding.adapter.output_names:
            torch.testing.assert_close(
                getattr(view, name)[physical.long()].float(), getattr(reference, name)[ids.long()].float()
            )
    assert addresses == {name: value.data_ptr() for name, value in binding.resident.items()}
    assert map_addresses == (binding.mapping.host_map.data_ptr(), binding.mapping.device_map.data_ptr())


def test_full_pool_is_shared_between_layers(expert_factory, manager_factory):
    first, _ = expert_factory()
    second, _ = expert_factory(layer_name="model.layers.1.mlp.experts")
    for parameter in second.parameters():
        parameter.data.add_(3)
    manager = manager_factory(first, second)
    addresses = {name: value.data_ptr() for name, value in manager.full_pool.tensors.items()}
    for layer in (first, second, first):
        binding = layer._expert_offload_binding
        view, physical, _ = binding.prepare(layer, torch.tensor([[1, 4]]))
        expected = binding.adapter.load_experts(binding.host_store, (1, 4), "cpu")
        for name in binding.adapter.output_names:
            torch.testing.assert_close(getattr(view, name)[physical.long()].reshape_as(expected[name]), expected[name])
        assert addresses == {name: value.data_ptr() for name, value in manager.full_pool.tensors.items()}


@pytest.mark.parametrize("mismatch", ["experts", "shape", "stride", "dtype", "format", "bytes", "quantization"])
def test_full_pool_rejects_incompatible_layers(expert_factory, native_runtime, monkeypatch, mismatch):
    first, _ = expert_factory()
    second, _ = expert_factory(experts=7 if mismatch == "experts" else 6, layer_name="model.layers.1.mlp.experts")
    bindings = [layer._expert_offload_binding for layer in (first, second)]
    for layer, binding in zip((first, second), bindings):
        binding.initialize_storage(layer)
    name = "w13_weight"
    tensor = bindings[1].resident[name]
    if mismatch == "shape":
        bindings[1].resident[name] = tensor[..., :-1]
    elif mismatch == "stride":
        bindings[1].resident[name] = torch.empty_strided(
            tensor.shape, (tensor.stride(0) + 1, *tensor.stride()[1:]), dtype=tensor.dtype
        )
    elif mismatch == "dtype":
        bindings[1].resident[name] = tensor.to(torch.int8)
    elif mismatch == "format":
        for binding in bindings:
            binding.device = SimpleNamespace(type="npu")
        monkeypatch.setattr("torch_npu.get_npu_format", lambda value: 2 if value is tensor else 29)
    elif mismatch == "bytes":
        for binding in bindings:
            binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
        prepared = bindings[1].host_store.prepared[name]
        bindings[1].host_store.prepared[name] = replace(prepared, expert_bytes=prepared.expert_bytes + 1)
    elif mismatch == "quantization":
        bindings[1].adapter.quant_scheme.quant_type = QuantType.W8A8MXFP
    allocation = Mock(return_value={})
    monkeypatch.setattr(bindings[0].adapter, "allocate_weights", allocation)
    manager = ExpertOffloadManager()
    with pytest.raises(ValueError, match="identical expert layouts.*model.layers.1"):
        manager.initialize(torch.nn.ModuleList((first, second)))
    assert manager.full_pool.tensors == {}
    allocation.assert_not_called()


def test_prefill_does_not_read_routes_on_cpu(expert_factory, manager_factory):
    layer, _ = expert_factory()
    manager = manager_factory(layer)

    class DeviceRoutes:
        def detach(self):
            pytest.fail("prefill must not copy routes to the CPU")

    routes = DeviceRoutes()
    view, physical, _ = layer._expert_offload_binding.prepare(layer, routes)
    assert physical is routes
    assert view.w13_weight is manager.full_pool.tensors["w13_weight"]
