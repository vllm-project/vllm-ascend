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


@torch.library.custom_op("vllm_ascend::lora_bmm_expand_slice", mutates_args={"output_tensor"})
def _bmm_expand_slice_op(
    output_tensor: torch.Tensor,
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> None:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D LoRA shrink input, got shape {tuple(inputs.shape)}")
    if output_tensor.ndim != 2:
        raise ValueError(f"Expected 2D LoRA output, got shape {tuple(output_tensor.shape)}")
    if lora_b_weights.ndim != 4 or lora_b_weights.shape[1] != 1:
        raise ValueError(f"Expected LoRA-B shape [slots, 1, output_size, rank], got {tuple(lora_b_weights.shape)}")
    if lora_indices_tensor.ndim != 1 or lora_indices_tensor.shape[0] != inputs.shape[0]:
        raise ValueError(
            "LoRA indices and shrink input must have the same row count, "
            f"got indices={tuple(lora_indices_tensor.shape)} and inputs={tuple(inputs.shape)}"
        )
    if output_tensor.shape[0] != inputs.shape[0]:
        raise ValueError(
            "LoRA output and shrink input must have the same row count, "
            f"got output={tuple(output_tensor.shape)} and inputs={tuple(inputs.shape)}"
        )
    if inputs.shape[-1] != lora_b_weights.shape[-1]:
        raise ValueError(
            "LoRA shrink rank must match LoRA-B input rank, "
            f"got input rank={inputs.shape[-1]} and LoRA-B rank={lora_b_weights.shape[-1]}"
        )
    if lora_b_weights.shape[-2] != slice_size:
        raise ValueError(
            "LoRA-B output size must match the destination slice, "
            f"got LoRA-B output={lora_b_weights.shape[-2]} and slice={slice_size}"
        )
    if slice_offset < 0 or slice_offset + slice_size > output_tensor.shape[-1]:
        raise ValueError(
            "LoRA destination slice is outside the output tensor, "
            f"got offset={slice_offset}, size={slice_size}, output={output_tensor.shape[-1]}"
        )

    safe_indices = lora_indices_tensor.clamp(min=0).to(torch.long)
    gathered_weights = lora_b_weights[safe_indices, 0].to(output_tensor.dtype)
    if inputs.shape[0] == 0 or gathered_weights.shape[1] == 0:
        return

    delta = torch.bmm(gathered_weights, inputs.to(output_tensor.dtype).unsqueeze(-1)).squeeze(-1)
    delta = torch.where(
        (lora_indices_tensor >= 0).unsqueeze(-1),
        delta,
        torch.zeros_like(delta),
    )
    output_slice = output_tensor.narrow(1, slice_offset, slice_size)
    if add_inputs:
        output_slice.add_(delta)
    else:
        output_slice.copy_(delta)


@_bmm_expand_slice_op.register_fake
def _bmm_expand_slice_fake(
    output_tensor: torch.Tensor,
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> None:
    return None


def bmm_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
) -> None:
    _bmm_expand_slice_op(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
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
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
