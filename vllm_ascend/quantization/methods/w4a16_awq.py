#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendMoEScheme, QuantType
from .registry import register_scheme

if TYPE_CHECKING:
    from vllm_ascend.quantization.awq_config import AWQConfig

# Bit shift pattern for unpacking 4-bit values from int32, see
# https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack_qzero_from_int32(
    weight: torch.Tensor,
    param_dtype: torch.dtype,
    pack_factor: int = 8,
    is_moe_layer: bool = False,
) -> torch.Tensor:
    """Unpack and convert AWQ zero-points (qzeros) from int32 to target dtype.

    :param weight: Packed int32 tensor containing zero-points
    :param param_dtype: Target dtype (e.g., bfloat16)
    :param pack_factor: Number of 4-bit values per int32 (default: 8)
    :param is_moe_layer: Whether this is for MoE layer (default: False)

    :return: Unpacked and converted zero-points tensor
    """
    weight_list = []

    for i in range(pack_factor):
        shift_num = REVERSE_AWQ_PACK_ORDER[i] * 4
        weight_list.append((weight.reshape(-1, 1) >> shift_num) & 0xF)

    if is_moe_layer:
        weight = torch.cat(weight_list, dim=-1).reshape(weight.shape[0], weight.shape[1], -1)
    else:
        weight = torch.cat(weight_list, dim=-1).reshape(weight.shape[0], -1)

    # Convert unsigned int4 [0,15] to signed int4 [-8,7]
    weight = -(weight - 8)
    return weight.to(param_dtype).contiguous()


def _unpack_weight_from_int32(
    weight: torch.Tensor,
    pack_factor: int = 8,
) -> torch.Tensor:
    """Unpack and convert AWQ weights (qweight) from int32 to NPU format.

    :param weight: Packed int32 tensor containing quantized weights
    :param pack_factor: Number of 4-bit values per int32 (default: 8)

    :return: Unpacked and NPU-formatted weight tensor
    """
    weight_tmp = torch.zeros_like(weight)
    for i in range(pack_factor):
        shift_num = REVERSE_AWQ_PACK_ORDER[i] * 4
        weight_tmp.bitwise_or_(((weight >> shift_num) * (2 ** (4 * i))) & (0xF << (4 * i)))
    weight_tmp.bitwise_xor_(0x88888888)
    return weight_tmp.contiguous()


class AscendW4A16AWQLinearMethod(AWQLinearMethod):
    """Linear method for Ascend W4A16 AWQ quantization."""

    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config
        self.pack_factor = self.quant_config.pack_factor
        self.group_size = self.quant_config.group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.qzeros.data,
                param_dtype=layer.scales.data.dtype,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )
        layer.qweight = torch.nn.Parameter(
            _unpack_weight_from_int32(weight=layer.qweight.data, pack_factor=self.pack_factor), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=layer.scales,
            antiquant_offset=layer.qzeros,
            antiquant_group_size=self.group_size,
            bias=bias,
        )
        out_shape = x.shape[:-1] + (qweight.shape[-1] * self.pack_factor,)
        return out.reshape(out_shape)


@register_scheme("W4A16_AWQ", "moe")
class AscendW4A16AWQFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16 AWQ quantization."""

    quant_type: QuantType = QuantType.W4A16_AWQ
    weight_attrs: dict = {"is_transposed": True}

    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config
        self.pack_factor = self.quant_config.pack_factor
        self.group_size = self.quant_config.group_size
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `pack_factor` {self.pack_factor}"
        )
        assert hidden_sizes % self.pack_factor == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"
        )

        param_dict = {}
        param_dict["w13_qweight"] = torch.empty(
            num_experts,
            hidden_sizes,
            2 * intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32,
        )
        param_dict["w2_qweight"] = torch.empty(
            num_experts,
            intermediate_size_per_partition,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `group_size` {self.group_size}"
        )
        assert hidden_sizes % self.group_size == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"
        )

        param_dict = {}
        num_groups_w13 = hidden_sizes // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size
        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        param_dict["w13_scales"] = torch.empty(
            num_experts,
            num_groups_w13,
            intermediate_size_per_partition * 2,
            dtype=params_dtype,
        )
        param_dict["w2_scales"] = torch.empty(num_experts, num_groups_w2, hidden_sizes, dtype=params_dtype)
        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        param_dict["w13_qzeros"] = torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32,
        )
        param_dict["w2_qzeros"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32,
        )
        return param_dict

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.w13_qzeros.data,
                param_dtype=layer.w13_scales.data.dtype,
                pack_factor=self.pack_factor,
                is_moe_layer=True,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        w13_qweight = (
            torch.nn.Parameter(
                _unpack_weight_from_int32(
                    weight=layer.w13_qweight.data,
                    pack_factor=self.pack_factor,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter("w13_qweight", w13_qweight)

        w2_qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.w2_qzeros.data,
                param_dtype=layer.w2_scales.data.dtype,
                pack_factor=self.pack_factor,
                is_moe_layer=True,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        w2_qweight = torch.nn.Parameter(
            _unpack_weight_from_int32(
                weight=layer.w2_qweight.data,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)

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
        global_redundant_expert_num=0,
        pertoken_scale: torch.Tensor | None = None,
        activation: MoEActivation = MoEActivation.SILU,
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert activation == MoEActivation.SILU, "Only SiLU activation is supported."

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_qweight,
                w2=layer.w2_qweight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=layer.w13_scales,
                w2_scale=layer.w2_scales,
                w1_offset=layer.w13_qzeros,
                w2_offset=layer.w2_qzeros,
            )
        )
