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
from typing import Any

import numpy as np
import torch
import torch_npu
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts


def _normalize_weight_strategy(strategy: Any) -> str:
    if strategy is None:
        return "group"
    if hasattr(strategy, "value"):
        strategy = strategy.value
    if isinstance(strategy, str):
        lowered = strategy.lower()
        if "group" in lowered:
            return "group"
        if "channel" in lowered:
            return "channel"
    raise ValueError(f"Unsupported weight strategy: {strategy}")


def _process_scale_compressed_tensors(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.transpose(1, 2).to(torch.float32).contiguous()
    scale_np = scale.cpu().numpy()
    scale_np.dtype = np.uint32
    scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
    return scale_uint64_tensor


def _update_bias_compressed_tensors(weight: torch.Tensor, scale: torch.Tensor, strategy: str) -> torch.Tensor:
    group_num, k, n = weight.shape
    scale = scale.transpose(1, 2).contiguous()
    scale = scale.reshape(group_num, -1, n)
    group_num, quantgroup_num, n = scale.shape

    if strategy == "group":
        tmp = weight.to(torch.float32).reshape([group_num, quantgroup_num, -1, n]) * scale.reshape(
            [group_num, quantgroup_num, 1, n]
        )
        tmp = tmp.reshape([group_num, k, n])
        return 8 * tmp.sum(axis=1)
    if strategy == "channel":
        return 8 * (weight.to(torch.float32) * scale).sum(axis=1)
    raise ValueError(f"Unsupported weight strategy: {strategy}")


class CompressedTensorsAscendW4A8DynamicFusedMoEMethod:
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        # Validate scheme: weights=W4 (channel or group),
        # activations=dynamic TOKEN (A8)

        # Must be dynamic per-token activations
        if input_quant.strategy != QuantizationStrategy.TOKEN or not input_quant.dynamic:
            raise ValueError("W4A8-int MoE needs dynamic per-token activation quantization.")

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = weight_quant.group_size if (weight_quant.group_size is not None) else -1
        if weight_quant.num_bits != 4:
            raise ValueError("This method only supports 4-bit weights (num_bits=4).")

        self.static_input_scales = False  # always dynamic per token

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
        enable_force_load_balance: bool = False,
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
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), global_num_experts - global_redundant_expert_num, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=[layer.w13_weight],
            w2=[layer.w2_weight],
            w1_scale=[layer.w13_weight_scale],
            w2_scale=[layer.w2_weight_scale],
            w1_scale_bias=layer.w13_scale_bias if hasattr(layer, "w13_scale_bias") else None,
            w2_scale_bias=layer.w2_scale_bias if hasattr(layer, "w2_scale_bias") else None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int4_w4a8=True,
            expert_map=expert_map,
            log2phy=log2phy,
            dynamic_eplb=get_ascend_config().eplb_config.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask"),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        weight_quant = getattr(self, "weight_quant", None)
        strategy = _normalize_weight_strategy(weight_quant.strategy if weight_quant is not None else None)
        w13_bias = _update_bias_compressed_tensors(layer.w13_weight.data, layer.w13_weight_scale.data, strategy)
        w2_bias = _update_bias_compressed_tensors(layer.w2_weight.data, layer.w2_weight_scale.data, strategy)

        layer.w13_weight_scale.data = _process_scale_compressed_tensors(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = _process_scale_compressed_tensors(layer.w2_weight_scale.data)

        w13_scale_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        w2_scale_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
        layer.register_parameter("w2_scale_bias", w2_scale_bias)

        def _pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
            return torch_npu.npu_quantize(
                weight.to(torch.float32),
                torch.tensor([1.0]).npu(),
                None,
                torch.quint4x2,
                -1,
                False,
            )

        # Accuracy problem in nz format
        # layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        # layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
        layer.w13_weight.data = _pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = _pack_to_int32(layer.w2_weight.data)
