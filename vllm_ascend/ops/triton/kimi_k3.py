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
# Adapted from https://github.com/sgl-project/sglang/pull/32544.

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_tokens", "num_residuals"])
def _attention_residual_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_weight_ptr,
    projection_weight_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    num_residuals,
    eps: tl.constexpr,
    num_cores: tl.constexpr,
    residual_block_size: tl.constexpr,
):
    tokens_per_core = (num_tokens - 1) // num_cores + 1
    core_idx = tl.program_id(0)
    token_start = core_idx * tokens_per_core
    if token_start >= num_tokens:
        return
    token_end = tl.minimum(token_start + tokens_per_core, num_tokens)

    hidden_offsets = tl.arange(0, hidden_size)
    residual_offsets = tl.arange(0, residual_block_size)

    norm_weight = tl.load(norm_weight_ptr + hidden_offsets).to(tl.float32)
    projection_weight = tl.load(projection_weight_ptr + hidden_offsets).to(tl.float32)
    score_weight = norm_weight * projection_weight
    block_stride = num_residuals * hidden_size

    for token_idx in range(token_start, token_end):
        scores = tl.full([residual_block_size], -float("inf"), dtype=tl.float32)
        for residual_idx in range(num_residuals + 1):
            if residual_idx < num_residuals:
                value = tl.load(
                    block_residual_ptr + token_idx * block_stride + residual_idx * hidden_size + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_sum_ptr + token_idx * hidden_size + hidden_offsets).to(tl.float32)
            inverse_rms = tl.rsqrt(tl.sum(value * value) / hidden_size + eps)
            scores = tl.where(
                residual_offsets == residual_idx,
                tl.sum(value * inverse_rms * score_weight),
                scores,
            )

        max_score = tl.max(scores)
        probabilities = tl.exp(scores - max_score)
        probabilities /= tl.sum(probabilities)

        output = tl.zeros([hidden_size], dtype=tl.float32)
        for residual_idx in range(num_residuals + 1):
            if residual_idx < num_residuals:
                value = tl.load(
                    block_residual_ptr + token_idx * block_stride + residual_idx * hidden_size + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_sum_ptr + token_idx * hidden_size + hidden_offsets).to(tl.float32)
            probability = tl.sum(tl.where(residual_offsets == residual_idx, probabilities, 0.0))
            output += probability * value

        tl.store(
            output_ptr + token_idx * hidden_size + hidden_offsets,
            output.to(output_ptr.dtype.element_ty),
        )


def fused_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Kimi K3's learned mixture over residual block starts."""
    num_tokens, hidden_size = prefix_sum.shape
    num_residuals = block_residual.shape[1]
    output = torch.empty_like(prefix_sum)
    residual_block_size = triton.next_power_of_2(num_residuals + 1)
    num_cores = get_vectorcore_num()

    _attention_residual_kernel[(num_cores,)](
        block_residual,
        prefix_sum,
        norm_weight,
        projection_weight,
        output,
        num_tokens,
        hidden_size,
        num_residuals,
        eps,
        num_cores,
        residual_block_size,
        multibuffer=True,
    )
    return output
