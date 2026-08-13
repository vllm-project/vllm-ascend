# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.utils import enable_categorical_sample_op, vllm_version_is


@triton.jit(do_not_specialize=["logits_stride", "vocab_size"])
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """
    Args:
        logits: Tensor of shape (num_tokens, vocab_size) containing the logits.
        expanded_idx_mapping: Tensor containing the mapping from token index
            to request index of tensor temperature.
        temperature: Tensor containing the temperature value for each request.
    """
    num_tokens, vocab_size = logits.shape
    # BLOCK_SIZE keeps BF16/FP16 logits and the FP32 working vector within
    # the Ascend A2/A3 UB limit of 1572864 bits (192 KB):
    #   32768 * (16 + 32) bits = 1572864 bits
    # The previous value 44032 overflowed UB when vLLM v0.26+ stopped
    # upcasting logits to FP32 upstream and the kernel started receiving
    # BF16/FP16 logits (44032 * 48 bits = 2113536 bits > 1572864).
    if vllm_version_is("0.26.0"):
        BLOCK_SIZE = 32768
    else:
        BLOCK_SIZE = 44032
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    if not enable_categorical_sample_op():
        raise RuntimeError("The categorical sampling custom operator is unavailable")

    # Match the vLLM v0.26 wrapper contract: only metadata that the upstream
    # implementation normalizes is made contiguous here. Logits, temperature,
    # seed, and the cache retain their original layouts.
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if output_processed_logits_col is not None:
        output_processed_logits_col = output_processed_logits_col.contiguous()

    sampled_token_ids, _ = torch.ops._C_ascend.npu_categorical_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        False,
        apply_temperature,
        output_processed_logits,
        output_processed_logits_col,
        use_fp64,
    )
    return sampled_token_ids
