# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops.fused_moe.catccos_adapter import CatccosLayerCapability
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType


def test_fused_mc2_passes_mxfp4_weights_and_swiglu_limit_to_catccos():
    config = SimpleNamespace(ep_size=4, num_local_experts=4)
    ascend_config = SimpleNamespace(
        enable_fused_mc2=1,
        ascend_fusion_config=SimpleNamespace(fusion_ops_gmmswigluquant=False),
    )
    hidden = torch.empty((2, 256), dtype=torch.bfloat16)
    topk_ids = torch.empty((2, 2), dtype=torch.int64)
    topk_weights = torch.empty((2, 2), dtype=torch.float32)
    w1 = torch.empty((4, 128, 512), dtype=torch.uint8)
    w1_scale = torch.empty((4, 4, 512, 2), dtype=torch.uint8)
    w2 = torch.empty((4, 128, 256), dtype=torch.uint8)
    w2_scale = torch.empty((4, 4, 256, 2), dtype=torch.uint8)
    expected_output = torch.empty_like(hidden)
    apply = Mock(return_value=expected_output)

    class TokenDispatcher:
        pass

    fused_input = MoEFusedExpertsInput(
        hidden_states=hidden,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=MoEWeights(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale),
        routing=MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W4A8MXFP),
        swiglu_limit=10.0,
    )

    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.set_gmmswigluquant_method", return_value=False),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.catccos_backend_enabled", return_value=True),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2", TokenDispatcher),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2"),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.apply_catccos", apply),
    ):
        comm_method = FusedMC2CommImpl(config, catccos_capability=CatccosLayerCapability(True))
        result = comm_method.fused_experts(fused_input)

    assert result.routed_out is expected_output
    assert result.swiglu_limit == 10.0
    apply.assert_called_once_with(hidden, topk_ids, topk_weights, w1, w1_scale, w2, w2_scale, swiglu_limit=10.0)
