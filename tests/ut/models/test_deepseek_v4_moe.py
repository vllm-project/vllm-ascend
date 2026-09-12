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

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_module
from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
    select_deepseek_v4_vision_experts,
)


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

    monkeypatch.setattr(deepseek_v4_module, "FusedMoEFactory", fused_moe)
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


def test_deepseek_v4_vision_router_keeps_text_hash_and_applies_bias_vl():
    router_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    input_ids = torch.tensor([7, 129257, 129259])
    tid2eid = torch.zeros(32, 2, dtype=torch.long)
    tid2eid[7] = torch.tensor([1, 3])
    bias_vl = torch.tensor([0.0, 0.0, 10.0, 0.0])

    weights, expert_ids = select_deepseek_v4_vision_experts(
        router_logits=router_logits,
        input_ids=input_ids,
        tid2eid=tid2eid,
        bias_vl=bias_vl,
        text_bias=None,
        top_k=2,
        renormalize=True,
    )

    # In-vocabulary text keeps the checkpoint's deterministic hash route.
    assert expert_ids[0].tolist() == [1, 3]
    # All five sentinel ids use the dynamic vision route.
    assert expert_ids[1].tolist() == [2, 0]
    assert expert_ids[2].tolist() == [2, 3]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))


def test_deepseek_v4_hash_vision_layer_exposes_bias_vl(monkeypatch):
    gate = _FakeGate()
    fused_moe = MagicMock(return_value=_FakeMoERunner(MagicMock()))
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
        vision_n_layers=2,
        vocab_size=32,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    monkeypatch.setattr(deepseek_v4_module, "FusedMoEFactory", fused_moe)
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

    assert moe.gate.bias_vl.shape == (config.n_routed_experts,)
    assert fused_moe.call_args.kwargs["bias_vl"] is moe.gate.bias_vl
    assert fused_moe.call_args.kwargs["e_score_correction_bias"] is None


def test_deepseek_v4_load_weights_skips_hash_layer_gate_bias(monkeypatch):
    model = deepseek_v4_module.AscendDeepseekV4ForCausalLM.__new__(deepseek_v4_module.AscendDeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=4,
        n_shared_experts=0,
        num_attention_heads=8,
    )
    model.num_redundant_experts = 0
    model.moe_mlp_layers = [
        SimpleNamespace(layer_idx=1, hash=True),
        SimpleNamespace(layer_idx=2, hash=False),
    ]

    gate = nn.Module()
    gate.e_score_correction_bias = nn.Parameter(torch.zeros(4))
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = gate
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module(), layer])

    monkeypatch.setattr(
        deepseek_v4_module.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "get_ascend_config",
        lambda: SimpleNamespace(mix_placement=False),
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(deepseek_v4_module, "is_pp_missing_parameter", lambda name, model: False)

    # Hash layers route text tokens through tid2eid, so the checkpoint's
    # text correction bias has no destination parameter there. Both the
    # e_score_correction_bias naming and the legacy .gate.bias naming that
    # gets remapped to it must be skipped without raising KeyError.
    hash_bias = torch.full((4,), 7.0)
    legacy_hash_bias = torch.full((4,), 8.0)
    # Regular layers still load the bias under both namings.
    text_bias = torch.full((4,), 9.0)
    legacy_text_bias = torch.full((4,), 10.0)

    loaded = model.load_weights(
        iter(
            [
                ("model.layers.1.mlp.gate.e_score_correction_bias", hash_bias),
                ("model.layers.1.mlp.gate.bias", legacy_hash_bias),
                ("model.layers.2.mlp.gate.e_score_correction_bias", text_bias),
                ("model.layers.2.mlp.gate.bias", legacy_text_bias),
            ]
        )
    )

    assert loaded == {"model.layers.2.mlp.gate.e_score_correction_bias"}
    torch.testing.assert_close(gate.e_score_correction_bias.data, legacy_text_bias)
