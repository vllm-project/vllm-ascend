# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch


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

        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output,
                                                 input_.contiguous(),
                                                 group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        output_tensor_list = list(
            torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def _gather_along_last_dim(input_, group):
    """Gather tensors and concatenate along the last dimension."""

    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size,
                         dtype=input_.dtype,
                         device=torch.npu.current_device())
    torch.distributed.all_gather_into_tensor(output,
                                             input_.contiguous(),
                                             group=group)
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()

    return output


def _reduce_scatter_along_first_dim(input_,
                                    group,
                                    input_split_sizes=None,
                                    use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        torch.distributed.reduce_scatter_tensor(output,
                                                input_.contiguous(),
                                                group=group)
    else:
        rank = torch.distributed.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(output,
                                         input_tensor_list,
                                         group=group)
    return output


def _reduce_scatter_along_last_dim(input_, group):
    """Reduce-scatter tensors on the last dimension."""
    world_size = torch.distributed.get_world_size(group)
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(input_,
                                split_size_or_sections=input_.shape[-1] //
                                world_size,
                                dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor,
                                             group).reshape(target_shape)
    return output


def all_gather_last_dim_from_tensor_parallel_region(input_, group):
    """Wrapper for autograd function: forward: AG, backward RS <last dim>"""
    return _gather_along_last_dim(input_, group)


def reduce_scatter_to_sequence_parallel_region(input_,
                                               group,
                                               input_split_sizes=None):
    """Wrapper for autograd function: forward: RS, backward AG <first dim>"""
    return _reduce_scatter_along_first_dim(input_, group, input_split_sizes)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group):
    """Wrapper for autograd function: forward: RS, backward AG: AG <last dim>"""
    return _reduce_scatter_along_last_dim(input_, group)


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def all_to_all(group, input, output_split_sizes=None, input_split_sizes=None):
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input

    input = input.contiguous()
    if output_split_sizes is None:
        # Equal split (all2all)
        output = torch.empty_like(input)
    else:
        # Unequal split (all2all-v)
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:]),
            dtype=input.dtype,
            device=torch.npu.current_device(),
        )
    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def all_to_all_sp2hp(input_, group):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the sequence
            dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    if group is None:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    tp_group = group
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(input_,
                                split_size_or_sections=input_.shape[-1] //
                                world_size,
                                dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(tp_group, concat_tensor)
    return output


def all_to_all_hp2sp(input_, group):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the hidden
            dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    if group is None:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_ = input_.reshape(-1, input_.shape[-1])
    tp_group = group
    input_exchanged = all_to_all(tp_group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped,
        split_size_or_sections=input_reshaped.shape[0] // world_size,
        dim=0)
    output = torch.cat(split_tensors, dim=-1)
    return output
