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
"""Ascend W4A16_MXFP4 linear and fused MoE quantization methods."""

from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import dispose_tensor

from ..base import AscendLinearScheme, AscendMoEScheme, QuantType
from ..registry import register_scheme


# Unpack the weights to FP4 and return them in float32 format
def unpack_uint8_to_fp4_return_float32(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = packed // 16
    # The high 4 bits and low 4 bits are arranged alternately, with the low 4 bits in front.
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    # A 4-digit integer is mapped to mxfp4 based on its value.
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    return fp4_values[unpacked.to(torch.long)]


@register_scheme("W4A16_MXFP4", "linear")
class AscendW4A16MXFP4LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A16_MXFP4 (weight-only MXFP4 quantization).

    The checkpoint stores packed FP4 weights (uint8, two E2M1 values per
    byte, low nibble first) and per-group E8M0 scales (uint8, exponent + 127
    biased). Weight processing mirrors the fused MoE scheme: unpack to
    float, transpose, cast to the int4pack-friendly format, and pack with
    npu_convert_weight_to_int4pack; scales stay uint8 E8M0 and are consumed
    by npu_weight_quant_batchmatmul as antiquant scales.
    """

    # Row-parallel shards may start halfway through a global MX group.
    supports_unaligned_tp_groups = True

    def __init__(self) -> None:
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, cdiv(input_size, self.group_size), dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # Pad the activation at the same K positions where the weight was
        # padded to restore the global MX group phase; zero entries do not
        # change the GEMM result.
        prefix_padding, suffix_padding = vars(layer).get("mxfp4_tp_padding", (0, 0))
        if prefix_padding or suffix_padding:
            x = F.pad(x, (prefix_padding, suffix_padding), mode="constant", value=0)
        output = torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight,
            antiquant_scale=layer.weight_scale,
            bias=bias,
            antiquant_group_size=self.group_size,
        )
        return output

    def process_weights_after_loading(self, layer):
        # Row parallelism may split K in the middle of a globally quantized MX
        # group. Prefix padding restores that group's original phase; suffix
        # padding completes the final group. Packed FP4 stores two logical K
        # values per byte, so logical padding is halved when applied to the
        # packed weight tensor.
        prefix_padding = suffix_padding = 0
        if isinstance(layer, RowParallelLinear):
            global_start = layer.tp_rank * layer.input_size_per_partition
            prefix_padding = global_start % self.group_size
            suffix_padding = -(prefix_padding + layer.input_size_per_partition) % self.group_size
            if prefix_padding % 2 != 0 or suffix_padding % 2 != 0:
                raise ValueError("MXFP4 packed weight padding must be divisible by 2")
            if prefix_padding or suffix_padding:
                layer.weight.data = F.pad(
                    layer.weight.data,
                    (prefix_padding // 2, suffix_padding // 2),
                    mode="constant",
                    value=0,
                )
        layer.mxfp4_tp_padding = (prefix_padding, suffix_padding)

        layer.weight.data = unpack_uint8_to_fp4_return_float32(layer.weight.data)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29, customize_dtype=torch.bfloat16)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.weight.data).contiguous()

        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()


@register_scheme("W4A16_MXFP4", "moe")
class AscendW4A16MXFP4FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16_MXFP4."""

    supports_eplb = False
    quant_type: QuantType = QuantType.W4A16MXFP
    act_quant_type: torch.dtype | None = None
    # Like W4A16, W4A16MXFP keeps activations unquantized and has no fused
    # gmm1+act+quant kernel: the action method runs gmm1 -> activation -> gmm2.
    fused_activations = frozenset()

    def __init__(self) -> None:
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // 2,
            dtype=torch.uint8,
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // 2,
            dtype=torch.uint8,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
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
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=None,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=None,
                mxfp_use_bf16=(x.dtype == torch.bfloat16),
            ),
            quant_method=self,
        )

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = unpack_uint8_to_fp4_return_float32(layer.w13_weight.data)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29, customize_dtype=torch.bfloat16)
        layer.w13_weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.w13_weight.data).contiguous()

        layer.w2_weight.data = unpack_uint8_to_fp4_return_float32(layer.w2_weight.data)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29, customize_dtype=torch.bfloat16)
        layer.w2_weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.w2_weight.data).contiguous()

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        layer = mlp_compute_input.layer
        assert layer is not None
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[mlp_compute_input.hidden_states],
            weight=[layer.w13_weight],
            antiquant_scale=[layer.w13_weight_scale],
            group_list=cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 0),
            split_item=3,
            group_type=0,
            output_dtype=mlp_compute_input.hidden_states.dtype,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        # A16 activation: no (re)quantization.
        return hidden_states, None

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        assert layer is not None
        input_dtype = mlp_compute_input.hidden_states.dtype
        use_bf16 = input_dtype == torch.bfloat16
        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            antiquant_scale=[layer.w2_weight_scale],
            split_item=3,
            group_type=0,
            group_list_type=mlp_compute_input.group_list_type,
            group_list=mlp_compute_input.group_list,
            output_dtype=output_dtype,
        )[0]
