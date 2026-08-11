# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
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
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp as _compute_global_lse,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _insert_resampled_kernel,
)

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

_ASCEND_TRITON_GRID_LIMIT = 65_535


def _iter_grid_row_ranges(num_rows: int, num_columns: int):
    """Split a 2D Triton launch so each grid has at most 65535 programs."""
    if num_columns > _ASCEND_TRITON_GRID_LIMIT:
        raise ValueError(f"Triton grid column count {num_columns} exceeds the Ascend limit")
    rows_per_launch = _ASCEND_TRITON_GRID_LIMIT // num_columns
    for row_start in range(0, num_rows, rows_per_launch):
        yield row_start, min(row_start + rows_per_launch, num_rows)


@triton.jit
def _npu_rand32(seed, offset):
    # Triton Ascend does not support the uint64 offset arithmetic used by the
    # CUDA implementation. Positions fit in int32 in the model-runner hot path.
    offset = offset.to(tl.int32)
    uniform = tl.rand(seed, offset).to(tl.float32)
    # Smallest positive value produced by Triton's FP32 RNG. Rejection sampling
    # takes log(uniform), so zero must be excluded.
    return tl.maximum(uniform, 4.6566127342e-10)


@triton.jit
def _npu_gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp

    if processed_logits_ptr is not None:
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr + req_state_idx * processed_logits_stride + col * vocab_size + block,
            logits,
            mask=mask,
        )

    logits = logits.to(tl.float32)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req_state_idx)
        # NPU: cast pos to int32 to avoid uint64 in philox (NPU umulhi only
        # supports int32/uint32). Position values fit in int32 in practice.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)
        # NPU: use tl.rand (float32) instead of tl_rand64 (float64 not supported)
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if is_bonus:
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
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
    else:
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    value, idx = _npu_gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,
        0,
        None,
        vocab_size,
        APPLY_TEMPERATURE=False,
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
def _materialize_residual_logits_kernel(
    # [num_reqs, V]
    residual_logits_ptr,
    residual_logits_stride,
    # [num_reqs]
    resample_mapping_ptr,
    # [num_reqs]
    resample_pos_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    needs_resample = temp != 0.0 or is_bonus
    if block_idx == 0:
        tl.store(resample_mapping_ptr + req_idx, tl.where(needs_resample, req_state_idx, -1))
        tl.store(resample_pos_ptr + req_idx, tl.load(pos_ptr + resample_token_idx))
    if not needs_resample:
        return

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if is_bonus:
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
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
            target_log_probs + tl.log(1.0 - ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(block != rejected_draft_token, target_logits, float("-inf")).to(tl.float32)

    tl.store(
        residual_logits_ptr + req_idx * residual_logits_stride + block,
        residual_logits,
        mask=mask,
    )


@triton.jit
def _insert_resampled_tokens_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs]
    resampled_token_ids_ptr,
    # [num_reqs]
    resample_mapping_ptr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)
    if tl.load(resample_mapping_ptr + req_idx) >= 0:
        token_id = tl.load(resampled_token_ids_ptr + req_idx)
        tl.store(sampled_ptr + req_idx * sampled_stride + num_sampled, token_id)


@triton.jit
def _probabilistic_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1)
            if temp == 0.0:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
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
                is_valid_draft = draft_sampled >= 0
                draft_sampled = tl.maximum(draft_sampled, 0)
                target_logit = tl.load(target_logits_ptr + logit_idx * target_logits_stride + draft_sampled).to(
                    tl.float32
                )
                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                target_log_prob = target_logit - target_lse
                pos = tl.load(pos_ptr + logit_idx)
                u = _npu_rand32(seed, pos)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + i * draft_logits_stride_1
                        + draft_sampled
                    ).to(tl.float32)
                    draft_lse = _compute_global_lse(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        logit_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    draft_log_prob = draft_logit - draft_lse
                else:
                    # One-hot draft: q(draft_token) = 1, log_q = 0.
                    draft_log_prob = 0
                # Probability ratio test: p(x) > u * q(x)
                # Equivalent log form: log_p(x) > log(u) + log_q(x)
                accepted &= target_log_prob > tl.log(u) + draft_log_prob
                accepted &= is_valid_draft
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    # TODO: refactor speculative decoding functionality in a future PR.
    # `use_block_verification` is accepted but not yet implemented on NPU;
    # wire it up when the block verification path is supported.
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_block_verification:
        logger.warning_once(
            "Block verification is not supported on NPU Model Runner V2 yet; "
            "falling back to standard token-by-token verification."
        )
    if synthetic_conditional_rates is not None:
        # Synthetic rejection sampling needs tl_rand64, which NPU Triton does
        # not support. The greedy fallback below would silently use u=0.0 and
        # produce wrong acceptance — refuse loudly instead.
        raise NotImplementedError(
            "Synthetic rejection sampling is not supported on NPU yet; use rejection_sample_method='standard'."
        )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    has_draft_logits = draft_logits is not None

    if draft_logits is None:
        # When draft_logits is None, create a dummy tensor so that Triton
        # kernel signatures receive valid pointers/strides. The kernels
        # will never read from it when HAS_DRAFT_LOGITS=False.
        draft_logits = target_logits.new_empty(1, 1, 1)

    # Compute the block-level logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
    target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    for logit_start, logit_end in _iter_grid_row_ranges(num_logits, vocab_num_blocks):
        launch_num_logits = logit_end - logit_start
        _compute_block_stats_kernel[(launch_num_logits, vocab_num_blocks)](
            target_local_argmax[logit_start:logit_end],
            target_local_argmax.stride(0),
            target_local_max[logit_start:logit_end],
            target_local_max.stride(0),
            target_local_sumexp[logit_start:logit_end],
            target_local_sumexp.stride(0),
            draft_local_max[logit_start:logit_end],
            draft_local_max.stride(0),
            draft_local_sumexp[logit_start:logit_end],
            draft_local_sumexp.stride(0),
            target_logits[logit_start:logit_end],
            target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            expanded_idx_mapping[logit_start:logit_end],
            expanded_local_pos[logit_start:logit_end],
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=VOCAB_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=has_draft_logits,
        )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
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
        seed,
        pos,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        num_warps=1,
    )

    if use_fp64:
        # The AscendC categorical sampler owns the high-precision RNG and
        # categorical selection. Materialize only the rejected/bonus residual
        # distributions here; greedy non-bonus rows are marked as padding and
        # skipped by the categorical operator.
        RESIDUAL_BLOCK_SIZE = 1024
        residual_num_blocks = triton.cdiv(vocab_size, RESIDUAL_BLOCK_SIZE)
        residual_logits = target_logits.new_empty((num_reqs, vocab_size), dtype=torch.float32)
        resample_mapping = expanded_idx_mapping.new_empty(num_reqs)
        resample_pos = pos.new_empty(num_reqs)
        for req_start, req_end in _iter_grid_row_ranges(num_reqs, residual_num_blocks):
            launch_num_reqs = req_end - req_start
            _materialize_residual_logits_kernel[(launch_num_reqs, residual_num_blocks)](
                residual_logits[req_start:req_end],
                residual_logits.stride(0),
                resample_mapping[req_start:req_end],
                resample_pos[req_start:req_end],
                target_logits,
                target_logits.stride(0),
                target_rejected_logsumexp[req_start:req_end],
                draft_logits,
                draft_logits.stride(0),
                draft_logits.stride(1),
                draft_rejected_logsumexp[req_start:req_end],
                num_sampled[req_start:req_end],
                cu_num_logits[req_start : req_end + 1],
                expanded_idx_mapping,
                draft_sampled,
                temperature,
                pos,
                vocab_size,
                BLOCK_SIZE=RESIDUAL_BLOCK_SIZE,
                HAS_DRAFT_LOGITS=has_draft_logits,
            )
        resampled_token_ids = gumbel_sample(
            residual_logits,
            resample_mapping,
            temperature,
            seed,
            resample_pos,
            apply_temperature=False,
            use_fp64=True,
        )
        _insert_resampled_tokens_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            resampled_token_ids,
            resample_mapping,
        )
        return sampled, num_sampled

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.int64)
    # NPU does not support float64; use float32 for resampled_local_max.
    resampled_local_max = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.float32)
    for req_start, req_end in _iter_grid_row_ranges(num_reqs, resample_num_blocks):
        launch_num_reqs = req_end - req_start
        _resample_kernel[(launch_num_reqs, resample_num_blocks)](
            resampled_local_argmax[req_start:req_end],
            resampled_local_argmax.stride(0),
            resampled_local_max[req_start:req_end],
            resampled_local_max.stride(0),
            target_logits,
            target_logits.stride(0),
            target_rejected_logsumexp[req_start:req_end],
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            draft_rejected_logsumexp[req_start:req_end],
            num_sampled[req_start:req_end],
            cu_num_logits[req_start : req_end + 1],
            expanded_idx_mapping,
            draft_sampled,
            temperature,
            seed,
            pos,
            vocab_size,
            BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
            HAS_DRAFT_LOGITS=has_draft_logits,
        )

    # Insert the resampled tokens into the output sampled.
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
