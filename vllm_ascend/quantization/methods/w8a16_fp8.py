# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend W8A16_FP8 quantization helpers and fused MoE method."""

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


@register_scheme("W8A16_FP8", "moe")
class AscendW8A16FP8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A16_FP8 (weight-only FP8 quantization).

    Key characteristics:
    - Weights are stored in torch.float8_e4m3fn, activations remain bf16/fp16
    - Uses antiquant_scale path in npu_grouped_matmul (fused dequant + matmul)
    - Per-channel scale in bfloat16 (not E8M0, unlike MXFP4/MXFP8)
    - No activation quantization (mxfp_act_quant_type=None)
    """

    quant_type: QuantType = QuantType.W8A16FP

    def __init__(self) -> None:
        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create weight tensors for w13 (gate+up) and w2 (down) projections.

        W8A16 uses native float8_e4m3fn with full K dimension.
        """
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes,
            dtype=torch.float8_e4m3fn,
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition,
            dtype=torch.float8_e4m3fn,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create per-channel weight scale tensors.

        For W8A16 per-channel dequant, scale shape is [E, N] (one scale per
        output channel per expert), dtype matches activation (bfloat16/float16).
        """
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=params_dtype,  # bfloat16 or float16
        )
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts,
            hidden_sizes,
            dtype=params_dtype,  # bfloat16 or float16
        )
        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, (
            "Number of global experts mismatch (excluding redundancy): "
            f"router_logits.shape[1]={router_logits.shape[1]}, num_logical_experts={num_logical_experts}"
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                mxfp_act_quant_type=None,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=None,
                mxfp_per_token_scale_dtype=None,
                mxfp_use_bf16=(x.dtype == torch.bfloat16),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer):
        """Prepare weights for NPU grouped matmul.

        - Only transpose + contiguous for per-channel antiquant_scale alignment
        """
        # w13_weight: [E, 2N, K] → [E, K, 2N] (transpose for per-channel dequant)

        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.contiguous()
