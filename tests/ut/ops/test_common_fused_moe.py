import os
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from vllm.config import CompilationLevel, ModelConfig, get_current_vllm_config

from vllm.model_executor.layers.fused_moe.config import (  # isort: skip
    FusedMoEConfig, FusedMoEParallelConfig)
from vllm.model_executor.layers.fused_moe.layer import (  # isort: skip
    FusedMoE, UnquantizedFusedMoEMethod)

NUM_EXPERTS = 256
TOPK = 8
TP_SIZE = 1
DP_SIZE = 1


def mock_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    apply_router_weight_on_input: bool = False,
    max_num_tokens: Optional[int] = None,
) -> torch.Tensor:
    return hidden_states + 1


def mock_fused_experts_moge(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    return 2 * hidden_states


def mock_npu_moe_gating_top_k_softmax(x: torch.Tensor,
                                      finished: torch.Tensor = None,
                                      k: int = 0):
    topk_weights = x[:, :k]
    topk_ids = torch.range(0, k - 1).unsqueeze(0)
    row_idx = torch.range(0, k - 1).unsqueeze(0)
    return topk_weights, topk_ids, row_idx


def create_fused_moe_method(vllm_config):
    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=TP_SIZE,
        dp_size_=DP_SIZE,
        vllm_parallel_config=vllm_config.parallel_config)
    moe_config = FusedMoEConfig.make(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=32,
        num_local_experts=NUM_EXPERTS,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.float16,
        max_num_tokens=NUM_EXPERTS,
        quant_config=None,
    )
    layer = UnquantizedFusedMoEMethod(moe=moe_config)
    return layer


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("compilation_level", [0, 1, 2, 3])
def test_AscendUnquantizedFusedMoEMethod_init(enforce_eager,
                                              compilation_level):
    vllm_config = get_current_vllm_config()
    vllm_config.model_config = ModelConfig()
    vllm_config.model_config.enforce_eager = enforce_eager
    vllm_config.compilation_config.level = compilation_level
    with patch("vllm.config._current_vllm_config", vllm_config):
        layer = create_fused_moe_method(vllm_config)

        # check initialization
        assert hasattr(layer, "use_aclgraph")
        assert hasattr(layer, "max_num_batched_tokens")
        assert layer.max_num_batched_tokens == vllm_config.scheduler_config.max_num_batched_tokens
        expected_use_aclgraph = vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and not vllm_config.model_config.enforce_eager
        assert layer.use_aclgraph == expected_use_aclgraph


@pytest.mark.parametrize("select_gating_topk_softmax_experts", ["0", "1"])
@pytest.mark.parametrize("is_310p_return", [True, False])
@patch("vllm_ascend.ops.common_fused_moe.fused_experts_moge",
       side_effect=mock_fused_experts_moge)
@patch("vllm_ascend.ops.common_fused_moe.fused_experts",
       side_effect=mock_fused_experts)
@patch("torch_npu.npu_moe_gating_top_k_softmax",
       side_effect=mock_npu_moe_gating_top_k_softmax)
def test_AscendUnquantizedFusedMoEMethod_forward(
        mock_npu_moe_gating_top_k_softmax, mock_fused_experts,
        mock_fused_experts_moge, select_gating_topk_softmax_experts,
        is_310p_return):
    vllm_config = get_current_vllm_config()
    vllm_config.model_config = ModelConfig()
    vllm_config.model_config.enforce_eager = False
    vllm_config.compilation_config.level = 3
    with patch("vllm.config._current_vllm_config", vllm_config), patch(
            "vllm_ascend.utils.is_310p",
            return_value=is_310p_return), patch.dict(os.environ, {
                'SELECT_GATING_TOPK_SOTFMAX_EXPERTS':
                select_gating_topk_softmax_experts
            }):
        # prepare input and create layer
        layer = create_fused_moe_method(vllm_config)
        fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                             top_k=TOPK,
                             hidden_size=32,
                             intermediate_size=32,
                             dp_size=DP_SIZE,
                             tp_size=TP_SIZE)
        x = torch.randn(32, NUM_EXPERTS)
        router_logits = torch.randn(32, 128)
        # invoke forward
        layer.forward(
            fused_moe,
            x,
            use_grouped_topk=False,
            top_k=TOPK,
            router_logits=router_logits,
            renormalize=True,
            global_num_experts=NUM_EXPERTS,
        )
        # check 310p
        if is_310p_return:
            mock_fused_experts_moge.assert_called_once()
        else:
            mock_fused_experts.assert_called_once()
        # check SELECT_GATING_TOPK_SOTFMAX_EXPERTS
        if os.environ["SELECT_GATING_TOPK_SOTFMAX_EXPERTS"] == "1":
            mock_npu_moe_gating_top_k_softmax.assert_called_once()
        else:
            mock_npu_moe_gating_top_k_softmax.assert_not_called()
