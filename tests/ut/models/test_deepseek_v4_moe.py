# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)

from vllm_ascend.models import deepseek_v4 as deepseek_v4_module


class _FakeGate(nn.Module):
    pass


class _FakeMoERunner(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.is_internal_router = True
        self.input_ids = None

    def forward(self, hidden_states, router_logits, input_ids=None):
        self.input_ids = input_ids
        return hidden_states


def test_deepseek_v4_hash_layer_uses_upstream_hash_router(monkeypatch):
    gate = _FakeGate()

    def build_runner(**kwargs):
        router = create_fused_moe_router(
            top_k=kwargs["top_k"],
            global_num_experts=kwargs["num_experts"],
            renormalize=kwargs["renormalize"],
            use_grouped_topk=kwargs.get("use_grouped_topk", False),
            num_expert_group=kwargs.get("num_expert_group"),
            topk_group=kwargs.get("topk_group"),
            scoring_func=kwargs["scoring_func"],
            routed_scaling_factor=kwargs["routed_scaling_factor"],
            e_score_correction_bias=kwargs["e_score_correction_bias"],
            hash_indices_table=kwargs["hash_indices_table"],
        )
        return _FakeMoERunner(router)

    fused_moe = MagicMock(side_effect=build_runner)
    ep_group = SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 1),
        rank_in_group=0,
    )
    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=None,
        norm_topk_prob=True,
        num_experts_per_tok=2,
        num_hash_layers=1,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        swiglu_limit=10.0,
        vocab_size=32,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    monkeypatch.setattr(deepseek_v4_module, "FusedMoE", fused_moe)
    monkeypatch.setattr(deepseek_v4_module, "ReplicatedLinear", lambda *args, **kwargs: gate)
    monkeypatch.setattr(deepseek_v4_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        deepseek_v4_module,
        "get_ascend_config",
        lambda: SimpleNamespace(mix_placement=False),
    )
    monkeypatch.setattr(deepseek_v4_module.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(
        deepseek_v4_module.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )

    moe = deepseek_v4_module.DeepseekV4MoE(
        config=config,
        parallel_config=parallel_config,
        prefix="model.layers.0.mlp",
    )

    kwargs = fused_moe.call_args.kwargs
    assert "use_grouped_topk" not in kwargs
    assert "num_expert_group" not in kwargs
    assert "topk_group" not in kwargs
    assert kwargs["hash_indices_table"] is moe.gate.tid2eid
    assert isinstance(moe.experts.router, FusedTopKBiasRouter)
    assert moe.experts.router._hash_indices_table is moe.gate.tid2eid

    input_ids = torch.tensor([11, 22])
    moe(torch.randn(2, config.hidden_size), input_ids=input_ids)
    assert moe.experts.input_ids is input_ids

    with pytest.raises(ValueError, match="hash MoE routing requires input_ids"):
        moe(torch.randn(2, config.hidden_size))
