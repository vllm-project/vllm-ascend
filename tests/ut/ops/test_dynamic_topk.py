# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.experts_selector import _apply_token_top_ks, select_experts
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType


def test_apply_token_top_ks_masks_routes_per_token():
    topk_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    topk_weights = torch.ones(2, 4)
    token_top_ks = torch.tensor([3, 2], dtype=torch.int32)

    _apply_token_top_ks(
        topk_indices,
        topk_weights,
        invalid_expert_id=8,
        token_top_ks=token_top_ks,
    )

    assert topk_indices.tolist() == [[0, 1, 2, 8], [4, 5, 8, 8]]
    assert topk_weights.tolist() == [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]


def test_apply_token_top_ks_rejects_misaligned_shape():
    with pytest.raises(ValueError, match="shape must match"):
        _apply_token_top_ks(
            torch.zeros(2, 4, dtype=torch.int32),
            torch.ones(2, 4),
            invalid_expert_id=4,
            token_top_ks=torch.ones(3, dtype=torch.int32),
        )


def test_select_experts_uses_id_after_shared_experts_as_sentinel():
    topk_weights = torch.ones(1, 2)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    with (
        patch(
            "vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k",
            return_value=False,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.experts_selector._native_select_experts",
            return_value=(topk_weights, topk_ids),
        ),
    ):
        _, selected_ids = select_experts(
            hidden_states=torch.ones(1, 4),
            router_logits=torch.ones(1, 8),
            top_k=2,
            use_grouped_topk=False,
            renormalize=False,
            mix_placement=True,
            num_logical_experts=8,
            num_shared_experts=2,
            num_experts=8,
            token_top_ks=torch.tensor([1]),
        )

    assert selected_ids.tolist() == [[0, 10, 8, 9]]


@pytest.mark.parametrize("full_top_k", [False, True])
def test_runner_passes_spec_k_budget_without_patching_router(monkeypatch, full_top_k):
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 8)
    token_top_ks = torch.tensor([2, 3], dtype=torch.int32)
    expected_top_ks = None if full_top_k else token_top_ks
    quant_method = MagicMock()
    quant_method.apply.return_value = FusedExpertsResult(hidden_states)
    moe_comm_method = MagicMock()
    moe_comm_method.prepare.return_value = MoEPrepareOutput(
        hidden_states,
        router_logits,
        None,
        None,
        None,
        expected_top_ks,
    )
    moe_comm_method.finalize.return_value = hidden_states

    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.router = SimpleNamespace()
    runner.routed_experts = SimpleNamespace(quant_method=quant_method, activation="silu")
    runner._spec_k_full_top_k = full_top_k
    runner.enable_npugraph_ex_static_kernel = False
    runner.multistream_overlap_gate = False
    runner.moe_config = SimpleNamespace(num_experts=8)
    runner.top_k = 4
    runner.use_grouped_topk = False
    runner.renormalize = False
    runner.topk_group = None
    runner.num_expert_group = None
    runner.custom_routing_function = None
    runner.scoring_func = "softmax"
    runner._original_routed_scaling_factor = 1.0
    runner.e_score_correction_bias = None
    runner.tid2eid = None
    runner._expert_map = None
    runner.apply_router_weight_on_input = False
    runner.enable_shared_expert_dp = False
    runner.quant_type = QuantType.NONE
    runner.log2phy = None
    runner.global_redundant_expert_num = 0
    runner.dynamic_eplb = False

    forward_context = SimpleNamespace(
        all_moe_layers=None,
        moe_layer_index=0,
    )
    extra_context = SimpleNamespace(
        in_profile_run=False,
        moe_comm_method=moe_comm_method,
        flash_comm_v1_enabled=False,
        token_top_ks=token_top_ks,
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.fused_moe.get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX",
        extra_context,
    )

    result = runner.no_shared_forward_impl(hidden_states, router_logits)

    assert result is hidden_states
    assert not hasattr(runner.router, "moe_layer_idx")
    prepared_top_ks = moe_comm_method.prepare.call_args.kwargs["token_top_ks"]
    assert prepared_top_ks is expected_top_ks
    apply_kwargs = quant_method.apply.call_args.kwargs
    if full_top_k:
        assert "token_top_ks" not in apply_kwargs
    else:
        assert apply_kwargs["token_top_ks"] is token_top_ks


def test_moe_prepare_output_preserves_pertoken_scale_position():
    hidden_states = torch.ones(2, 4)
    router_logits = torch.ones(2, 8)
    pertoken_scale = torch.ones(2)

    output = MoEPrepareOutput(
        hidden_states,
        router_logits,
        None,
        None,
        pertoken_scale,
    )

    assert output.pertoken_scale is pertoken_scale
    assert output.token_top_ks is None
