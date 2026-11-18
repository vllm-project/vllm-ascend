# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch.nn as nn
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.distributed.eplb.eplb_state import EXPERT_MAPPING_EP_SIZE
from vllm_ascend.patch.platform.patch_eplb import (
    _build_distributed_initial_expert_map,
    _patch_initial_expert_layout,
    _with_expert_mapping_ep_size,
)


def test_initial_replicas_are_spread_across_ep_ranks():
    layout = _build_distributed_initial_expert_map(5, 3, ep_size=2)
    assert layout == [0, 1, 2, 3, 2, 3, 4, 0]
    assert [layout[:4].count(0), layout[4:].count(0)] == [1, 1]


def test_initial_layout_handles_single_rank_and_zero_redundancy():
    assert _build_distributed_initial_expert_map(5, 3, ep_size=1) == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
    ]
    assert _build_distributed_initial_expert_map(5, 0, ep_size=5) == list(range(5))
    with pytest.raises(ValueError, match="divisible"):
        _build_distributed_initial_expert_map(5, 2, ep_size=2)


def test_routed_weight_mapping_uses_distributed_initial_layout():
    _patch_initial_expert_layout()
    routed = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(ep_size=2, num_experts=8, num_logical_experts=5, is_act_and_mul=True)
    routed.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
    routed.ckpt_gate_proj_name = "gate_proj"
    routed.ckpt_down_proj_name = "down_proj"
    routed.ckpt_up_proj_name = "up_proj"
    routed.lora_base_layer_prefix = ""

    mapping = routed.get_expert_mapping()

    gate_mapping = [(expert_id, weight_name) for _, weight_name, expert_id, shard_id in mapping if shard_id == "w1"]
    assert gate_mapping == [
        (physical, f"experts.{logical}.gate_proj.") for physical, logical in enumerate([0, 1, 2, 3, 2, 3, 4, 0])
    ]
    assert EXPERT_MAPPING_EP_SIZE.get() == 1


def test_weight_mapping_ep_size_is_scoped_to_the_call():
    wrapped = _with_expert_mapping_ep_size(
        lambda _model: EXPERT_MAPPING_EP_SIZE.get(),
        lambda model: model.moe_config.ep_size,
    )
    first_model = SimpleNamespace(moe_config=SimpleNamespace(ep_size=4))
    second_model = SimpleNamespace(moe_config=SimpleNamespace(ep_size=2))
    assert wrapped(first_model) == 4
    assert wrapped(second_model) == 2
    assert EXPERT_MAPPING_EP_SIZE.get() == 1
