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


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    # The Ascend C++ kernel (csrc/kernels/bgmv_shrink.cpp) always writes the
    # output in float32 (Y_T = float), regardless of output_tensor's dtype, and
    # the C++ binding does not check output_tensor.dtype. Feeding a non-float32
    # output (e.g. bf16, which vllm's torch ops freely accept) makes the kernel
    # write 4 bytes per element into a 2-byte buffer and silently corrupt memory.
    # Honor the contract: accumulate into a float32 tensor and cast back when the
    # caller's output is not float32. In production the LoRA shrink buffer is
    # already float32 (see punica_npu.py), so this branch is a no-op there.
    if output_tensor.dtype != torch.float32:
        out = output_tensor.to(torch.float32)
        torch.ops._C_ascend.bgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            out,
            scaling,
        )
        output_tensor.copy_(out)
        return output_tensor
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
    # The Ascend C++ kernel (csrc/kernels/bgmv_expand.cpp) always reads the input
    # in float32 (X_T = float), regardless of inputs' dtype, and the C++ binding
    # does not check inputs.dtype. Feeding a non-float32 input (e.g. bf16) makes
    # the kernel read 4 bytes per element from a 2-byte buffer, producing garbage.
    # Honor the contract by casting the input to float32. In production the input
    # is the float32 shrink buffer, so this is a no-op there.
    if inputs.dtype != torch.float32:
        inputs = inputs.to(torch.float32)
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
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
    # See bgmv_expand: the kernel reads the input as float32.
    if inputs.dtype != torch.float32:
        inputs = inputs.to(torch.float32)
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
    # See bgmv_shrink: the kernel writes the output as float32.
    if output_tensor.dtype != torch.float32:
        out = output_tensor.to(torch.float32)
        torch.ops._C_ascend.sgmv_shrink(inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, out, scaling)
        output_tensor.copy_(out)
        return output_tensor
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
    # See bgmv_expand: the kernel reads the input as float32.
    if inputs.dtype != torch.float32:
        inputs = inputs.to(torch.float32)
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
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
    # See bgmv_expand: the kernel reads the input as float32.
    if inputs.dtype != torch.float32:
        inputs = inputs.to(torch.float32)
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
