#
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

from collections.abc import Callable

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.quantization.utils import pack_to_int32, unpack_from_int32


class AscendW4A16FusedMoEMethod:
    """FusedMoE method for Ascend W4A16."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
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
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[1] == global_num_experts - global_redundant_expert_num, (
            "Number of global experts mismatch (excluding redundancy)"
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
            global_num_experts=global_num_experts,
        )

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_offset=layer.w13_weight_offset,
            w2_offset=layer.w2_weight_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int4_w4a16=True,
            expert_map=expert_map,
            log2phy=log2phy,
            dynamic_eplb=dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask"),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert weights from Marlin format to Ascend NPU format.

        vllm create_weights weights:
        - w13_weight_packed: [e, k//8, 2*n] (Marlin)
        - w2_weight_packed:  [e, n//8, k]
        - w13_weight_scale:  [e, num_groups, 2*n]
        - w2_weight_scale:   [e, num_groups, k]

        Needed by Ascend ops:
        - w13_weight_packed: [e, 2*n, k//8] (NPU int4pack)
        - w2_weight_packed:  [e, k, n//8]
        - w13_weight_scale:  [e, num_groups, 2*n]
        - w2_weight_scale:   [e, num_groups, k]
        """

        num_bits = 4
        pack_factor = 8

        layer.w13_weight_packed.data = layer.w13_weight_packed.data.transpose(1, 2).contiguous()
        layer.w2_weight_packed.data = layer.w2_weight_packed.data.transpose(1, 2).contiguous()

        w13_shape = layer.w13_weight_packed.data.shape
        w2_shape = layer.w2_weight_packed.data.shape
        unpacked_w13_weight = (
            unpack_from_int32(
                layer.w13_weight_packed.data.flatten(0, 1),
                torch.Size([w13_shape[0] * w13_shape[1], w13_shape[2] * pack_factor]),
                num_bits,
            )
            .view(w13_shape[0], w13_shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            unpack_from_int32(
                layer.w2_weight_packed.data.flatten(0, 1),
                torch.Size([w2_shape[0] * w2_shape[1], w2_shape[2] * pack_factor]),
                num_bits,
            )
            .view(w2_shape[0], w2_shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)
        layer.w2_weight_packed.data = pack_to_int32(unpacked_w2_weight)

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.contiguous()

        # Only symmetric quantization is supported, the offset is for quant_apply_mlp function branch
        layer.w13_weight_offset = torch.nn.Parameter(
            torch.zeros_like(layer.w13_weight_scale.data),
            requires_grad=False,
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            torch.zeros_like(layer.w2_weight_scale.data),
            requires_grad=False,
        )
