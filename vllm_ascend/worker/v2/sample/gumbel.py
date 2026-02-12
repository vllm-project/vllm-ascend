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
#

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        # NOTE(Ronald1995): change pos's dtype to tl.int32, because triton-ascend's
        # compiler doesn't support unint64 of pos arg.
        pos = tl.load(pos_ptr + batch_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)

        # Generate gumbel noise.
        # NOTE(Ronald1995): r is tl.float64 in vllm, change it to tl.float32,
        # or triton-ascend's compiler will raise error.
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        gumbel_noise = gumbel_noise.to(tl.float32)

        # Apply temperature.
        if APPLY_TEMPERATURE:
            # NOTE(woosuk): Match the behavior of _temperature_kernel.
            # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
            logits = logits / temp

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + batch_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + batch_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_reqs, vocab_size]
    idx_mapping: torch.Tensor,  # [num_reqs]
    temperature: torch.Tensor,  # [num_reqs]
    seed: torch.Tensor,  # [num_reqs]
    pos: torch.Tensor,  # [num_reqs]
    apply_temperature: bool,
) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    # TODO(Ronald1995): Optimize the performance of the kernel in npu.
    _gumbel_sample_kernel[(num_reqs, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
