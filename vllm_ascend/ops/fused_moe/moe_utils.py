# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from importlib import import_module

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from torch.nn.functional import pad

from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op

COMM_STREAM = None

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def load_cann_mega_moe_ops():
    ops_module = import_module("cann_ops_transformer.ops")
    get_symm_buffer_for_mega_moe = ops_module.get_symm_buffer_for_mega_moe
    mega_moe = ops_module.mega_moe
    return get_symm_buffer_for_mega_moe, mega_moe


def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    # Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
    # The current custom op package still requires explicit INT4 for W4A8
    # packed weights; otherwise it derives W4A8's packed N as an INT8 N and
    # rejects weight2.
    #
    # dispatch_quant_out_dtype: the doc types this as torch.dtype (torch.int8 /
    # torch.float8_e4m3fn). We pass the ACL enum ints (258 / 24) because W8A8
    # was validated end-to-end this way in PD; switching W4A8 to torch.int8 did
    # NOT fix the W4A8 accuracy issue and slowed graph capture (see bug_a3.md),
    # so keep the working values until the W4A8 accuracy root cause is found on
    # the operator side.
    if quant_type == QuantType.W8A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8)
    if quant_type == QuantType.W4A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4)
    raise RuntimeError(
        "MegaMoe integration supports W8A8/W4A8 INT on A2/A3 and MXFP on FP8-capable "
        "MegaMoe platforms. "
        f"Unsupported quant type: {quant_type}."
    )


def zero_experts_compute(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales = torch.where(zero_expert_mask, 0.0, zero_expert_scales)

        hidden_states = hidden_states.unsqueeze(1)
        zero_expert_scales = zero_expert_scales.unsqueeze(2)
        result = hidden_states * zero_expert_scales
        result = result.sum(dim=1)

    normal_expert_mask = expert_indices >= num_experts
    expert_indices = torch.where(normal_expert_mask, 0, expert_indices)
    expert_scales = torch.where(normal_expert_mask, 0.0, expert_scales)

    return expert_indices, expert_scales, result


def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)

    return int(num_experts - global_redundant_expert_num - num_shared_experts)


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation=None):
    return (
        fusion
        and dynamic_eplb
        and getattr(activation, "value", activation) != "swigluoai_uninterleave"
        and enable_custom_op()
    )


def _gmm_swiglu_quant_fusion_enabled(use_mxfp_quant, fusion, dynamic_eplb, activation=None):
    return (use_mxfp_quant or (fusion and not dynamic_eplb)) and (
        getattr(activation, "value", activation) != "swigluoai_uninterleave"
    )


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def _prepare_dequant_swiglu_weight_scale(
    w1_scale: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Prepare w1_scale for the swigluoai_uninterleave dequant fused op."""
    if isinstance(w1_scale, list):
        if len(w1_scale) == 1:
            weight_scale = w1_scale[0]
        else:
            weight_scale = torch.stack([scale.reshape(-1) for scale in w1_scale], dim=0)
    else:
        weight_scale = w1_scale
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.to(torch.float32)
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.reshape(1, -1)
    return weight_scale


def _prepare_swigluoai_grouped_matmul_scales(
    weight_scale: list[torch.Tensor] | torch.Tensor, output_dtype: torch.dtype
) -> list[torch.Tensor]:
    scales = weight_scale if isinstance(weight_scale, list) else [weight_scale]
    return [scale.to(output_dtype) if scale.dtype != output_dtype else scale for scale in scales]
