# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


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


def test_prefill_pool_is_shared_between_layers(expert_factory, manager_factory):
    first, _ = expert_factory()
    second, _ = expert_factory(layer_name="model.layers.1.mlp.experts")
    for parameter in second.parameters():
        parameter.data.add_(3)
    manager = manager_factory(first, second)
    addresses = {name: value.data_ptr() for name, value in manager.prefill_pool.tensors.items()}
    for layer in (first, second, first):
        binding = layer._expert_offload_binding
        view, physical, _ = binding.prepare(layer, torch.tensor([[1, 4]]))
        expected = binding.adapter.materialize(binding.host_store, (1, 4), "cpu")
        for name in binding.adapter.output_names:
            torch.testing.assert_close(getattr(view, name)[physical.long()].reshape_as(expected[name]), expected[name])
        assert addresses == {name: value.data_ptr() for name, value in manager.prefill_pool.tensors.items()}
