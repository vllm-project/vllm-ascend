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
"""
Fused Rejection Sampler - Memory-Optimized Implementation

This module implements the complete fused rejection sampling algorithm from
upstream vLLM MRV2, which avoids the O(N*V) FP32 softmax tensor.

Key innovations:
1. Block-level LSE computation instead of full softmax
2. Log-space arithmetic for numerical stability
3. Fused rejection + resample in a single pass

The block statistics use O(N * num_vocab_blocks) memory instead of an O(N * V)
FP32 probability tensor. Re-sampling uses one O(batch_size * V) random tensor,
matching the existing rejection sampler's per-request random draw.

Reference: vllm/v1/worker/gpu/spec_decode/probabilistic_rejection_sampler_utils.py
"""

import torch
from vllm.triton_utils import tl, triton

# Block size optimized for NPU UB (196KB), verified across vocab sizes
VOCAB_BLOCK_SIZE = 8192


@triton.jit
def _compute_block_max_and_sumexp(logits):
    """Compute block-level max and sumexp for LSE reconstruction."""
    block_max = tl.max(logits, axis=0)
    block_sumexp = tl.where(
        block_max > float("-inf"),
        tl.sum(tl.exp(logits - block_max)),
        0.0,
    )
    return block_max, block_sumexp


@triton.jit
def _compute_global_lse(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """Reconstruct global LSE from block-level statistics."""
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_block_max_and_sumexp_kernel(
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    target_logits_ptr,
    target_logits_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute block-level statistics for both target and draft logits.

    For each block of logits, compute:
    - block_max = max(logits[block])
    - block_sumexp = sum(exp(logits[block] - block_max))

    For greedy sampling (temp=0), only compute target argmax.
    """
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr + logit_idx * target_local_argmax_stride + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        draft_max, draft_sumexp = _compute_block_max_and_sumexp(draft_logits)
        tl.store(
            draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
            draft_max,
        )
        tl.store(
            draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
            draft_sumexp,
        )

        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_block_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx,
            target_sumexp,
        )


@triton.jit
def _gumbel_block_argmax(
    logits,
    block,
    mask,
    uniform_ptr,
    uniform_stride,
):
    """Gumbel-max sampling for resample using pre-generated uniform values."""
    u = tl.load(uniform_ptr + uniform_stride + block, mask=mask, other=1.0)
    u = tl.maximum(u, 1e-10)
    gumbel = -tl.log(-tl.log(u))

    perturbed = logits + gumbel
    value = tl.max(perturbed, axis=0)
    idx = tl.argmax(perturbed, axis=0)

    return value, idx


@triton.jit
def _probabilistic_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    rejected_steps_ptr,
    target_rejected_logsumexp_ptr,
    draft_rejected_logsumexp_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    bonus_token_ids_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    uniform_probs_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """
    Perform fused rejection sampling using block-level LSE.

    For each request:
    1. Iterate through draft tokens
    2. Compute target_prob = exp(logit - LSE) on-the-fly
    3. Compare with draft_prob using log-space arithmetic
    4. Track first rejection position
    """
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True

    for i in range(num_tokens):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx)

            if temp == 0.0:
                target_blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
                target_blocks_mask = target_blocks < vocab_num_blocks
                target_local_max = tl.load(
                    target_local_max_ptr + logit_idx * target_local_max_stride + target_blocks,
                    mask=target_blocks_mask,
                    other=float("-inf"),
                )
                max_target_block_idx = tl.argmax(target_local_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_target_block_idx
                )
                accepted &= target_argmax == draft_sampled
                tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            else:
                target_logit = tl.load(target_logits_ptr + logit_idx * target_logits_stride + draft_sampled).to(
                    tl.float32
                )
                draft_logit = tl.load(
                    draft_logits_ptr + req_state_idx * draft_logits_stride_0 + i * draft_logits_stride_1 + draft_sampled
                ).to(tl.float32)

                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                draft_lse = _compute_global_lse(
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )

                target_log_prob = target_logit - target_lse
                draft_log_prob = draft_logit - draft_lse

                u = tl.load(uniform_probs_ptr + logit_idx)
                u = tl.maximum(u, 1e-10)

                accepted &= target_log_prob > tl.log(u) + draft_log_prob
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

            rejected_step += accepted

    if accepted:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(sampled_ptr + req_idx * sampled_stride + num_tokens, bonus_token_id)

    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _resample_kernel(
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    resampled_local_max_ptr,
    resampled_local_max_stride,
    target_logits_ptr,
    target_logits_stride,
    target_rejected_logsumexp_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_rejected_logsumexp_ptr,
    rejected_step_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    uniform_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Resample rejected/bonus tokens from residual distribution.

    Residual distribution: max(p(x) - q(x), 0)
    Computed in log-space: log(max(exp(log_p) - exp(log_q), 0))
                         = log_p + log(max(1 - exp(log_q - log_p), 0))
    """
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    if resample_idx >= end_idx - start_idx:
        return

    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1

    if temp == 0.0 and not is_bonus:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    if is_bonus:
        residual_logits = tl.load(
            target_logits_ptr + resample_token_idx * target_logits_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        target_logits = tl.load(
            target_logits_ptr + resample_token_idx * target_logits_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        draft_logits = tl.load(
            draft_logits_ptr + req_state_idx * draft_logits_stride_0 + resample_idx * draft_logits_stride_1 + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        draft_log_probs = draft_logits - draft_lse

        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tl.log(1 - ratio),
            float("-inf"),
        ).to(tl.float32)

    value, idx = _gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        uniform_ptr,
        req_idx * vocab_size,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    """Insert resampled tokens into the output."""
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    if resample_token_idx >= end_idx:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    if temp == 0.0 and resample_token_idx < end_idx - 1:
        return

    blocks = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    blocks_mask = blocks < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled_token = tl.load(resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + max_block_idx)
    tl.store(sampled_ptr + req_idx * sampled_stride + num_sampled, resampled_token)


def fused_probabilistic_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    uniform_probs: torch.Tensor,
    uniform_resample: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fused probabilistic rejection sampling.

    This function implements the complete rejection sampling algorithm without
    materializing the O(N*V) softmax tensor.

    Args:
        target_logits: Target model logits [num_logits, V]
        draft_logits: Draft model logits [max_num_reqs, num_speculative_steps, V]
        draft_sampled: Draft token IDs [num_logits]
        bonus_token_ids: Bonus token IDs [num_reqs, 1] or [num_reqs]
        cu_num_logits: Cumulative logits count [num_reqs + 1]
        pos: Position indices [num_logits]
        idx_mapping: Request index mapping [num_reqs]
        expanded_idx_mapping: Expanded request index mapping [num_logits]
        expanded_local_pos: Expanded local position [num_logits]
        temperature: Temperature values [max_num_reqs]
        uniform_probs: Uniform random values for rejection [num_logits]
        uniform_resample: Uniform random values for resample [num_reqs, vocab_size]
        num_speculative_steps: Number of speculative steps

    Returns:
        sampled: Sampled token IDs [num_reqs, num_speculative_steps + 1]
        num_sampled: Number of sampled tokens per request [num_reqs]
    """
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    assert uniform_resample.shape == (num_reqs, vocab_size)
    bonus_token_ids = bonus_token_ids.view(-1)

    VOCAB_BLOCK_SIZE_LOCAL = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE_LOCAL)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

    target_local_argmax = target_logits.new_empty((num_logits, vocab_num_blocks), dtype=torch.int64)
    target_local_max = target_logits.new_empty((num_logits, vocab_num_blocks), dtype=torch.float32)
    target_local_sumexp = target_logits.new_empty((num_logits, vocab_num_blocks), dtype=torch.float32)
    draft_local_max = target_logits.new_empty((num_logits, vocab_num_blocks), dtype=torch.float32)
    draft_local_sumexp = target_logits.new_empty((num_logits, vocab_num_blocks), dtype=torch.float32)

    _compute_block_max_and_sumexp_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE_LOCAL,
    )

    sampled = draft_sampled.new_empty((num_reqs, num_speculative_steps + 1), dtype=torch.int64)
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)

    _probabilistic_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        bonus_token_ids,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        uniform_probs,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
    )

    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty((num_reqs, resample_num_blocks), dtype=torch.int64)
    resampled_local_max = target_logits.new_empty((num_reqs, resample_num_blocks), dtype=torch.float32)

    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        uniform_resample,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
    )

    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )

    return sampled, num_sampled


def _expand_draft_probs_to_logits(
    draft_probs: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    max_spec_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(num_draft_tokens)
    vocab_size = draft_probs.shape[-1]
    device = draft_probs.device
    draft_logits = draft_probs.new_full((batch_size, max_spec_len, vocab_size), float("-inf"))

    num_draft_tokens_tensor = torch.tensor(num_draft_tokens, dtype=cu_num_draft_tokens.dtype, device=device)
    req_indices = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int64, device=device),
        num_draft_tokens_tensor.to(torch.int64),
    )
    cu_start = torch.cat([cu_num_draft_tokens.new_zeros(1), cu_num_draft_tokens[:-1]])
    local_pos = torch.arange(draft_probs.shape[0], dtype=torch.int64, device=device)
    local_pos = local_pos - cu_start[req_indices].to(torch.int64)
    safe_draft_probs = draft_probs.clamp_min(torch.finfo(draft_probs.dtype).tiny)
    draft_logits[req_indices, local_pos] = safe_draft_probs.log()
    return draft_logits, req_indices, local_pos


def generate_uniform_resample(
    batch_size: int,
    vocab_size: int,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    uniform_resample = torch.rand((batch_size, vocab_size), dtype=torch.float32, device=device)
    for req_idx, generator in generators.items():
        uniform_resample[req_idx].uniform_(generator=generator)
    return uniform_resample


def fused_rejection_sample_from_probs(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    uniform_resample: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    cu_num_logits = torch.cat([cu_num_draft_tokens.new_zeros(1), cu_num_draft_tokens])
    draft_logits, expanded_idx_mapping, expanded_local_pos = _expand_draft_probs_to_logits(
        draft_probs,
        num_draft_tokens,
        cu_num_draft_tokens,
        max_spec_len,
    )
    idx_mapping = torch.arange(batch_size, dtype=torch.int64, device=target_logits.device)
    output_token_ids, _ = fused_probabilistic_rejection_sample(
        target_logits,
        draft_logits,
        draft_token_ids,
        bonus_token_ids,
        cu_num_logits,
        torch.empty(0, dtype=torch.int64, device=target_logits.device),
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        uniform_probs,
        uniform_resample,
        max_spec_len,
    )
    return output_token_ids.to(torch.int32)
