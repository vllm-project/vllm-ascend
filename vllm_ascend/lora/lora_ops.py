#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch


def _print_op_args(op_name: str, **kwargs) -> None:
    formatted_args = []
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            formatted_args.append(
                f"{name}=(shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device})"
            )
        else:
            formatted_args.append(f"{name}={value!r}")
    print(
        f"[vllm_ascend.lora.lora_ops] {op_name}: " + ", ".join(formatted_args),
        flush=True,
    )


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    _print_op_args(
        "torch.ops._C_ascend.bgmv_shrink",
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        lora_indices_tensor=lora_indices_tensor,
        output_tensor=output_tensor,
        scaling=scaling,
    )
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    _print_op_args(
        "torch.ops._C_ascend.bgmv_expand",
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        lora_indices_tensor=lora_indices_tensor,
        output_tensor=output_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
    )
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        slice_offset,
        slice_size,
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    _print_op_args(
        "torch.ops._C_ascend.bgmv_expand",
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        lora_indices_tensor=lora_indices_tensor,
        output_tensor=output_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
    )
    return torch.ops._C_ascend.bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    _print_op_args(
        "torch.ops._C_ascend.sgmv_shrink",
        inputs=inputs,
        lora_a_weights=lora_a_weights,
        lora_indices_tensor=lora_indices_tensor,
        seq_len_tensor=seq_len_tensor,
        output_tensor=output_tensor,
        scaling=scaling,
    )
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    _print_op_args(
        "torch.ops._C_ascend.sgmv_expand",
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        lora_indices_tensor=lora_indices_tensor,
        seq_len_tensor=seq_len_tensor,
        output_tensor=output_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
    )
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        slice_offset,
        slice_size,
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    _print_op_args(
        "torch.ops._C_ascend.sgmv_expand",
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        lora_indices_tensor=lora_indices_tensor,
        seq_len_tensor=seq_len_tensor,
        output_tensor=output_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
    )
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
