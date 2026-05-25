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
"""Ascend W4A16 quantization helpers and fused MoE method."""

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """Unpack sub-byte quantized weights from signed int32 storage.

    Each int32 element stores ``32 // num_bits`` quantized values. The helper
    extracts values from low bits to high bits, crops padding back to ``shape``,
    then subtracts the symmetric zero-point offset so the result is signed int8.

    Example:
        With ``num_bits=4``, ``pack_factor=8`` and ``mask=0xF``. For a packed
        value ``0x76543210``, element ``i`` is recovered by
        ``(packed >> (4 * i)) & 0xF``. This yields unsigned values
        ``[0, 1, 2, 3, 4, 5, 6, 7]``; subtracting offset ``8`` maps them to
        signed int4 values ``[-8, -7, -6, -5, -4, -3, -2, -1]``.

    Args:
        weight: Packed int32 tensor containing quantized values.
        shape: Original unpacked shape. Extra values introduced by packing are
            cropped to this shape.
        num_bits: Number of bits per quantized value. It must be positive, no
            larger than 8, and divide 32 exactly.
        packed_dim: Dimension along which values were packed. Only 0 and 1 are
            supported.

    Returns:
        Tensor with ``shape`` and int8 dtype.

    Raises:
        AssertionError: If the dtype, bit width, or packed dimension is invalid.
    """
    # TODO 在docstring中是否可以加个例子，说明一下流程mask以及bit如何移动
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits > 0, f"Expecting `num_bits` should be positive but got {num_bits}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."
    assert 32 % num_bits == 0, f"Expecting `num_bits` {num_bits} to divide 32 exactly."
    assert packed_dim in [0, 1], f"Expecting `packed_dim` is 0 or 1 but got {packed_dim}."

    pack_factor = 32 // num_bits  # TODO 这里新增一个assert，如果32没法被num_bits整除时，报错
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)

    return unpacked_weight


def pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """Pack a 3D MoE weight tensor into int32 storage for W4A16 kernels.

    The expected logical shape is either ``[e, n, k]`` or ``[e, k, n]``:
    ``e`` is the number of experts, ``n`` is the expert output/intermediate
    channel dimension, and ``k`` is the expert input/hidden channel dimension.

    Int32 input contains one unpacked int4 value per element and is converted by
    ``torch_npu.npu_convert_weight_to_int4pack`` into the device int4pack
    layout. Int8 input is already byte-packed with two int4 values per byte, so
    four int8 bytes can be reinterpreted as one int32 word.

    Args:
        weight: A 3D int8 or int32 tensor.

    Returns:
        A contiguous int32 tensor in packed representation.

    Raises:
        AssertionError: If the rank, dtype, or packed dimension alignment is
            invalid.
    """
    assert weight.dim() == 3, (
        "Expecting `weight.dim()` is 3 ([expert, output_channel, input_channel] or "
        "[expert, input_channel, output_channel]) but got "
        f"{weight.dim()}."
    )  # TODO 这里没有对e n 和 k这三个变量做解释，建议写全称
    assert weight.dtype in [torch.int8, torch.int32], (
        f"Expecting `weight.dtype` is torch.int8 or torch.int32 but got {weight.dtype}."  # TODO 这里but评错了，`bug` -> `but`
    )

    if weight.dtype == torch.int32:
        # 原因：int32路径表示尚未打包的int4值，NPU int4pack每个int32承载8个4-bit值。
        assert weight.shape[-1] % 8 == 0, "the last dim of weight needs to be divided by 8."  # TODO 为什么这里时一定要被8整除
        packed_weight = torch_npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
        packed_weight = packed_weight.view(weight.shape[0], weight.shape[1], -1)
    else:
        # 原因：int8路径已经是两个int4值共用1个byte，重新解释成int32时需要4个byte一组。
        assert weight.shape[-1] % 4 == 0, "the last dim of weight needs to be divided by 4."  # TODO 为什么这里时一定要被4整除
        # 原因：这里不做数值转换，只把连续的4个int8 byte重解释为1个int32存储单元。
        packed_weight = weight.contiguous().view(torch.int32).contiguous()  # TODO 为什么这里直接view就行了？

    return packed_weight


@register_scheme("W4A16", "moe")
class AscendW4A16FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16."""

    quant_type: QuantType = QuantType.W4A16

    def __init__(self) -> None:
        self.num_bits = 4  # dtype = torch.int4
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
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

        param_dict["w13_weight_packed"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.pack_factor, dtype=torch.int32
        )
        param_dict["w2_weight_packed"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.pack_factor, dtype=torch.int32
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

        # TODO 为啥这里一定是bfloat16？
        # 原因：Ascend W4A16 fused MoE反量化接口按bfloat16读取scale/offset，保持bf16也避免运行期转换。
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w13_weight_shape"] = torch.empty(num_experts, 2, dtype=torch.int32)
        param_dict["w2_weight_shape"] = torch.empty(num_experts, 2, dtype=torch.int32)
        param_dict["w13_weight_offset"] = torch.zeros(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w2_weight_offset"] = torch.zeros(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.bfloat16
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
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight_packed,
                w2=layer.w2_weight_packed,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_offset=layer.w13_weight_offset,
                w2_offset=layer.w2_weight_offset,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_shape = layer.w13_weight_packed.data.shape
        w2_shape = layer.w2_weight_packed.data.shape
        # TODO 为啥unpack了还需要pack回去？
        # 原因：checkpoint的int32 bit-pack布局和NPU fused MoE要求的int4pack布局不同，需要先还原再按目标布局重打包。
        unpacked_w13_weight = (
            unpack_from_int32(
                layer.w13_weight_packed.data.flatten(0, 1),
                torch.Size([w13_shape[0] * w13_shape[1], w13_shape[2] * self.pack_factor]),
                self.num_bits,
            )
            .view(w13_shape[0], w13_shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            unpack_from_int32(
                layer.w2_weight_packed.data.flatten(0, 1),
                torch.Size([w2_shape[0] * w2_shape[1], w2_shape[2] * self.pack_factor]),
                self.num_bits,
            )
            .view(w2_shape[0], w2_shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)
        layer.w2_weight_packed.data = pack_to_int32(unpacked_w2_weight)

        # TODO 为啥需要transpose？
        # 原因：权重已从[E, N, K]转为[E, K, N]供kernel访问，scale/offset也必须转到相同轴顺序。
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()

        layer.w13_weight_offset.data = layer.w13_weight_offset.data.transpose(1, 2).contiguous()
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.transpose(1, 2).contiguous()
