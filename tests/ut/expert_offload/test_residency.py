# SPDX-License-Identifier: Apache-2.0


def test_decode_replacement_protects_hits_and_updates_mapping(expert_factory, manager_factory):
    layer, _ = expert_factory()
    binding = layer._expert_offload_binding
    manager_factory(layer)
    assert binding.slot_to_expert == (0, 1)
    slots, loads = binding.plan_expert_loads({1, 4})
    assert slots == (4, 1) and loads == ((4, 0),)
    assert binding.slot_to_expert == (0, 1)
    binding.load_resident(slots, loads)
    assert binding.slot_to_expert == (4, 1)
    assert binding.mapping.host_map.tolist() == [-1, 1, -1, -1, 0, -1]
    assert not binding.plan_expert_loads({1, 4})[1]
