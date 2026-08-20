#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme

# 310P GE retile of FRACTAL_NZ W8A8-Dynamic weights during torch.compile
# launches QuantBatchMatmulV3_NZ_NZ kernel 21 (hash 5247287448945562503).
# Eager NZ works; compiled Qwen3.5-2B TP2 does not (fused qkv KV shard N=256
# and MLP). Linear layers keep ND [N, K] and dequant to fp16 once at load.
# MoE experts still use grouped-matmul NZ and keep ND [E, N, K] for
# npu_quant_grouped_matmul_dequant (WeightNZ 3D parameters lose FRACTAL_NZ
# under GE and fail tiling).
_MIN_NZ_QUANT_MATMUL_N = 512


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod310(AscendMoEScheme):
    """310P-only FusedMoE method for Ascend W8A8_DYNAMIC.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8

    def __init__(self):
        self.ep_group = get_ep_group()
        vllm_config = get_current_vllm_config()
        self.in_dtype = vllm_config.model_config.dtype

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        # Fused gate_up_proj (column parallel)
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        # down_proj (row parallel)
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            ),
        )
        return final_hidden_states

    def process_weights_after_loading(self, layer):
        # Keep ModelSlim [E, N, K] as ND. FRACTAL_NZ is applied inside the
        # grouped-matmul path at runtime so torch.compile/GE cannot strip
        # format-29 from 3D Parameters (Qwen3-30B-A3B-W8A8 MRv2).
        layer.w13_weight.data = layer.w13_weight.data.contiguous()
        layer.w2_weight.data = layer.w2_weight.data.contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)


@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 dynamic linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    act_quant_type: torch.dtype = torch.int8

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # Always ND [N, K] + precomputed fp16 dequant. torch.compile/GE cannot
        # safely run 310P QuantBatchMatmulV3_NZ_NZ on these dynamic-quant linears.
        bias_term = bias if (tp_rank is None or tp_rank == 0) else None
        weight_fp = layer.weight_fp
        if weight_fp.dtype != x.dtype:
            weight_fp = weight_fp.to(x.dtype)
        return torch.nn.functional.linear(x, weight_fp, bias_term)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        # Dequant once at load: scales are static and redoing int8->fp16 * scale
        # on every forward is pure hot-path overhead.
        params_dtype = getattr(layer, "params_dtype", torch.float16)
        dtype = params_dtype if isinstance(params_dtype, torch.dtype) else torch.float16
        scale = layer.weight_scale.data.to(dtype).view(-1, 1)
        layer.weight_fp = torch.nn.Parameter(layer.weight.data.to(dtype) * scale, requires_grad=False)
