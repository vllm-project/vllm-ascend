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

from typing import Any, Callable, Dict, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts


def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param shape: Original shape to restore, defaults to None
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >>
                                                  (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >>
                                                  (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)

    return unpacked_weight


def pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """
    Packs quantized weights into int32 format for storage.

    :param weight: The 3D tensor to pack, must be int8 or int32 dtype
    :return: Packed tensor with int32 dtype optimized for storage
    """
    assert weight.dim(
    ) == 3, f"Expecting `weight.dim()` is 3 ([e, n, k] or [e, k, n]) but got {weight.dim()}."
    assert weight.dtype in [
        torch.int8, torch.int32
    ], f"Expecting `weight.dtype` is torch.int8 or torch.int32 bug got {weight.dtype}."

    if weight.dtype == torch.int32:
        assert weight.shape[
            -1] % 8 == 0, "the last dim of weight needs to be divided by 8."
        packed_weight = torch_npu.npu_convert_weight_to_int4pack(
            weight.flatten(0, 1))
        packed_weight = packed_weight.view(weight.shape[0], weight.shape[1],
                                           -1)
    else:
        assert weight.shape[
            -1] % 4 == 0, "the last dim of weight needs to be divided by 4."
        packed_weight = weight.view(torch.int32).contiguous()

    return packed_weight


class AscendW4A16FusedMoEMethod:
    """FusedMoe method for Ascend W4A16.
    """

    def __init__(self) -> None:
        self.transpose_weight = True
        self.num_bits = 4  # dtype = torch.int4
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get(
            "group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        assert intermediate_size_per_partition % self.pack_factor == 0, f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} can be divided by `pack_factor` {self.pack_factor}"
        assert hidden_sizes % self.pack_factor == 0, f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"

        param_dict = {}

        param_dict["w13_weight_packed"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32)
        param_dict["w2_weight_packed"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32)

        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} can be divided by `group_size` {self.group_size}"
        assert hidden_sizes % self.group_size == 0, f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"

        param_dict = {}

        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w13_weight_shape"] = torch.empty(num_experts,
                                                     2,
                                                     dtype=torch.int32)
        param_dict["w2_weight_shape"] = torch.empty(num_experts,
                                                    2,
                                                    dtype=torch.int32)
        param_dict["w13_weight_offset"] = torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w2_weight_offset"] = torch.zeros(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.bfloat16)

        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"

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
            global_num_experts=global_num_experts)

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
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
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.transpose_weight:
            w13_shape = layer.w13_weight_packed.data.shape
            w2_shape = layer.w2_weight_packed.data.shape
            unpacked_w13_weight = (unpack_from_int32(
                layer.w13_weight_packed.data.flatten(0, 1),
                torch.Size([
                    w13_shape[0] * w13_shape[1],
                    w13_shape[2] * self.pack_factor
                ]),
                self.num_bits,
            ).view(w13_shape[0], w13_shape[1],
                   -1).transpose(1, 2).contiguous().int())
            unpacked_w2_weight = (unpack_from_int32(
                layer.w2_weight_packed.data.flatten(0, 1),
                torch.Size([
                    w2_shape[0] * w2_shape[1], w2_shape[2] * self.pack_factor
                ]),
                self.num_bits,
            ).view(w2_shape[0], w2_shape[1],
                   -1).transpose(1, 2).contiguous().int())
            layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)
            layer.w2_weight_packed.data = pack_to_int32(unpacked_w2_weight)

            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(
                1, 2).contiguous()
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(
                1, 2).contiguous()

            layer.w13_weight_offset.data = layer.w13_weight_offset.data.transpose(
                1, 2).contiguous()
            layer.w2_weight_offset.data = layer.w2_weight_offset.data.transpose(
                1, 2).contiguous()
