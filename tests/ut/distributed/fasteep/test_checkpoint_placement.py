# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise checkpoint expert IDs against non-contiguous serving layouts."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.distributed.fasteep.policy import local_physical_mask
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.patch.platform import patch_eplb


@pytest.mark.parametrize("use_v2", [False, True])
def test_checkpoint_load_and_gathered_routing_match_logical_experts(use_v2):
    placements = torch.tensor([[0, 1, -1, -1], [-1, 0, 1, -1], [-1, -1, 0, 1], [0, -1, -1, 1]])
    checkpoint = torch.tensor([2.0, 3.0, 5.0, 7.0])
    weights = []
    for row in placements:
        layer = SimpleNamespace(
            _fasteep=True,
            _use_v2_model_runner=use_v2,
            moe_config=SimpleNamespace(num_logical_experts=4),
            ascend_expert_map=row,
        )
        local = torch.full((2,), float("nan"))
        # Exercise the physical IDs emitted by the actual V1 mapping, including
        # redundant copies, against the configured checkpoint destination.
        for physical in range(8):
            slot = AscendRoutedExperts._map_global_expert_id_to_local_expert_id(layer, physical)
            if slot >= 0:
                local[slot] = checkpoint[physical % 4]
        weights.append(local)
    for active in (4, 3):
        ids = generate_log2phy_map(placements[:active], 0, tp_size=1).long()
        contributions = []
        for rank in range(active):
            mask = local_physical_mask(ids, rank, 2)
            contributions.append(weights[rank][ids % 2] * mask)
        torch.testing.assert_close(torch.stack(contributions).sum(0), checkpoint)


def test_v2_checkpoint_mapping_uses_the_same_logical_ids_as_slot_loading():
    patch_eplb._patch_initial_expert_layout()
    layer = SimpleNamespace(
        _use_v2_model_runner=True,
        _fasteep=True,
        moe_config=SimpleNamespace(num_experts=8, num_logical_experts=4, ep_size=4, is_act_and_mul=True),
        expert_map_manager=SimpleNamespace(num_fused_shared_experts=0),
        build_expert_params_mapping=RoutedExperts.build_expert_params_mapping,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        lora_base_layer_prefix="",
    )
    mappings = RoutedExperts.get_expert_mapping(layer)
    for _, checkpoint_name, physical, _ in mappings:
        assert checkpoint_name.startswith(f"experts.{physical % 4}.")


def test_checkpoint_loading_rejects_missing_explicit_map():
    layer = SimpleNamespace(
        _fasteep=True,
        moe_config=SimpleNamespace(num_logical_experts=4),
        ascend_expert_map=None,
    )
    with pytest.raises(AssertionError, match="requires an expert map"):
        AscendRoutedExperts._map_global_expert_id_to_local_expert_id(layer, 0)
