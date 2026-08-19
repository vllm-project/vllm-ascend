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
from vllm_ascend.utils import maybe_trans_nz

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme

# 310P GE retile of FRACTAL_NZ W8A8-Dynamic weights during torch.compile
# launches QuantBatchMatmulV3_NZ_NZ kernel 21 (hash 5247287448945562503).
# Eager NZ works; compiled Qwen3.5-2B TP2 does not (fused qkv KV shard N=256
# and MLP). Linear layers keep ND [N, K] and dequant to fp16. MoE experts
# still use grouped-matmul NZ.
_MIN_NZ_QUANT_MATMUL_N = 512


def _needs_fp16_quant_matmul_fallback(_layer: torch.nn.Module, _weight: torch.Tensor) -> bool:
    """310P W8A8-Dynamic linear always uses ND + fp16 dequant.

    GE retile of FRACTAL_NZ weights during torch.compile launches
    ``QuantBatchMatmulV3_NZ_NZ`` kernel 21 (hash 5247287448945562503). Eager NZ
    works; compiled 2B TP2 does not, including fused qkv and MLP. Keep this
    helper so tests can assert the fallback policy.
    """
    return True


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
        # The grouped matmul consumes [E, K, N]. ModelSlim checkpoints store
        # expert weights as [E, N, K], so move the output dimension last before
        # converting to FRACTAL_NZ.
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data.transpose(1, 2).contiguous())
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data.transpose(1, 2).contiguous())
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
        # Always ND [N, K] + fp16 dequant. torch.compile/GE cannot safely run
        # 310P QuantBatchMatmulV3_NZ_NZ on these dynamic-quant linears.
        scale = layer.weight_scale.data if hasattr(layer.weight_scale, "data") else layer.weight_scale
        weight_fp = layer.weight.data.to(x.dtype) * scale.to(x.dtype).view(-1, 1)
        bias_term = bias if (tp_rank is None or tp_rank == 0) else None
        return torch.nn.functional.linear(x, weight_fp, bias_term)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.contiguous()
        layer._310p_w8a8_dynamic_fp16_fallback = _needs_fp16_quant_matmul_fallback(layer, layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
