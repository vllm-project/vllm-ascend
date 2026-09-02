# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Categorical resampling for speculative decoding on Ascend NPU."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_RESAMPLE_BLOCK_SIZE = 1024


def _get_vectorcore_num() -> int:
    try:
        return int(get_vectorcore_num())
    except AssertionError:
        init_device_properties_triton()
        return int(get_vectorcore_num())


@triton.jit
def _resample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    local_mass_ptr,
    local_mass_stride,
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
    draft_sampled_ptr,
    temp_ptr,
    num_reqs,
    num_blocks,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    """Compute one probability-mass statistic per request/vocabulary block."""
    worker_id = tl.program_id(0)
    num_workers = tl.num_programs(0)
    total_tasks = num_reqs * num_blocks
    tasks_per_worker = total_tasks // num_workers
    extra_tasks = total_tasks % num_workers
    task_start = worker_id * tasks_per_worker + tl.minimum(worker_id, extra_tasks)
    task_count = tasks_per_worker + (worker_id < extra_tasks)
    block_lanes = tl.arange(0, BLOCK_SIZE)

    for task_idx in tl.range(task_start, task_start + task_count):
        req_idx = task_idx // num_blocks
        block_idx = task_idx - req_idx * num_blocks
        resample_idx = tl.load(rejected_step_ptr + req_idx)
        start_idx = tl.load(cu_num_logits_ptr + req_idx)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
        resample_token_idx = start_idx + resample_idx
        req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)
        temperature = tl.load(temp_ptr + req_state_idx).to(tl.float32)
        is_bonus = resample_token_idx == end_idx - 1
        needs_resample = (temperature != 0.0) | is_bonus

        if needs_resample:
            vocab_offsets = block_idx * BLOCK_SIZE + block_lanes
            vocab_mask = vocab_offsets < vocab_size
            target_block_logits = tl.load(
                target_logits_ptr + resample_token_idx * target_logits_stride + vocab_offsets,
                mask=vocab_mask,
                other=float("-inf"),
            ).to(tl.float32)

            if is_bonus:
                block_max, block_argmax = tl.max(target_block_logits, axis=0, return_indices=True)
                has_mass = block_max > float("-inf")
                safe_block_max = tl.where(has_mass, block_max, 0.0)
                block_sumexp = tl.where(has_mass, tl.sum(tl.exp(target_block_logits - safe_block_max), axis=0), 0.0)
                tl.store(
                    local_argmax_ptr + req_idx * local_argmax_stride + block_idx, block_idx * BLOCK_SIZE + block_argmax
                )
                tl.store(local_max_ptr + req_idx * local_max_stride + block_idx, block_max)
                tl.store(local_mass_ptr + req_idx * local_mass_stride + block_idx, block_sumexp)
            else:
                target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
                target_prob = tl.exp(target_block_logits - target_lse)

                if HAS_DRAFT_LOGITS:
                    draft_block_logits = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + resample_idx * draft_logits_stride_1
                        + vocab_offsets,
                        mask=vocab_mask,
                        other=float("-inf"),
                    ).to(tl.float32)
                    draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
                    draft_prob = tl.exp(draft_block_logits - draft_lse)
                    token_mass = tl.maximum(target_prob - draft_prob, 0.0)
                else:
                    rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
                    token_mass = tl.where(vocab_offsets != rejected_draft_token, target_prob, 0.0)

                token_mass = tl.where(vocab_mask, token_mass, 0.0)
                tl.store(local_mass_ptr + req_idx * local_mass_stride + block_idx, tl.sum(token_mass, axis=0))


@triton.jit
def _categorical_finalize_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    local_mass_ptr,
    local_mass_stride,
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
    draft_sampled_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    """Select the final token using one global categorical threshold per request."""
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temperature = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temperature == 0.0
    is_bonus = resample_token_idx == end_idx - 1
    is_greedy_bonus = is_greedy & is_bonus
    is_random = ~is_greedy
    is_random_bonus = is_random & is_bonus
    is_random_residual = is_random & (~is_bonus)

    block_ids = tl.arange(0, PADDED_NUM_BLOCKS)
    valid_block_mask = block_ids < num_blocks

    # Greedy bonus: select the global target argmax from per-block maxima.
    greedy_block_max = tl.load(
        local_max_ptr + req_idx * local_max_stride + block_ids,
        mask=valid_block_mask & is_greedy_bonus,
        other=float("-inf"),
    ).to(tl.float32)
    greedy_block_max = tl.where(greedy_block_max != greedy_block_max, float("-inf"), greedy_block_max)
    greedy_selected_block = tl.argmax(greedy_block_max, axis=0)
    greedy_sampled_token = tl.load(
        local_argmax_ptr + req_idx * local_argmax_stride + greedy_selected_block,
        mask=is_greedy_bonus,
        other=0,
    )

    # One random value defines one point on the whole vocabulary-mass interval.
    seed = tl.load(seed_ptr + req_state_idx)
    position = tl.load(pos_ptr + resample_token_idx).to(tl.int32)
    uniform = tl.max(tl.rand(tl.randint(seed, position), tl.arange(0, 1)).to(tl.float32), axis=0)

    stored_block_mass = tl.load(
        local_mass_ptr + req_idx * local_mass_stride + block_ids,
        mask=valid_block_mask & is_random,
        other=0.0,
    ).to(tl.float32)

    # Bonus blocks use different local maxima, so convert their masses to one common scale.
    bonus_block_max = tl.load(
        local_max_ptr + req_idx * local_max_stride + block_ids,
        mask=valid_block_mask & is_random_bonus,
        other=float("-inf"),
    ).to(tl.float32)
    bonus_global_max = tl.max(bonus_block_max, axis=0)
    safe_bonus_global_max = tl.where(bonus_global_max > float("-inf"), bonus_global_max, 0.0)
    bonus_block_mass = stored_block_mass * tl.exp(bonus_block_max - safe_bonus_global_max)
    block_mass = tl.where(is_bonus, bonus_block_mass, stored_block_mass)
    block_mass = tl.where(valid_block_mask & is_random, block_mass, 0.0)

    total_mass = tl.sum(block_mass, axis=0)
    has_total_mass = total_mass > 0.0
    global_threshold = uniform * total_mass
    block_prefix = tl.cumsum(block_mass, axis=0)
    candidate_blocks = tl.where(
        (block_prefix > global_threshold) & valid_block_mask & has_total_mass,
        block_ids,
        PADDED_NUM_BLOCKS,
    )
    selected_block = tl.minimum(tl.min(candidate_blocks, axis=0), num_blocks - 1)
    block_prefix_before = tl.sum(tl.where(valid_block_mask & (block_ids < selected_block), block_mass, 0.0), axis=0)
    remaining_threshold = global_threshold - block_prefix_before

    # Rebuild token masses only for the selected block.
    block_offsets = tl.arange(0, BLOCK_SIZE)
    token_ids = selected_block * BLOCK_SIZE + block_offsets
    valid_token_mask = token_ids < vocab_size
    active_token_mask = valid_token_mask & is_random & has_total_mass
    target_block_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + token_ids,
        mask=active_token_mask,
        other=float("-inf"),
    ).to(tl.float32)

    selected_block_max = tl.load(
        local_max_ptr + req_idx * local_max_stride + selected_block,
        mask=is_random_bonus & has_total_mass,
        other=0.0,
    ).to(tl.float32)
    safe_bonus_logits = tl.where(
        is_random_bonus & has_total_mass & valid_token_mask, target_block_logits, float("-inf")
    )
    selected_block_scale = tl.exp(selected_block_max - safe_bonus_global_max)
    bonus_token_mass = tl.exp(safe_bonus_logits - selected_block_max) * selected_block_scale

    target_lse = tl.load(
        target_rejected_logsumexp_ptr + req_idx, mask=is_random_residual & has_total_mass, other=0.0
    ).to(tl.float32)
    residual_target_logits = tl.where(
        is_random_residual & has_total_mass & valid_token_mask,
        target_block_logits,
        float("-inf"),
    )
    target_prob = tl.exp(residual_target_logits - target_lse)

    if HAS_DRAFT_LOGITS:
        draft_block_logits = tl.load(
            draft_logits_ptr + req_state_idx * draft_logits_stride_0 + resample_idx * draft_logits_stride_1 + token_ids,
            mask=valid_token_mask & is_random_residual & has_total_mass,
            other=float("-inf"),
        ).to(tl.float32)
        draft_lse = tl.load(
            draft_rejected_logsumexp_ptr + req_idx, mask=is_random_residual & has_total_mass, other=0.0
        ).to(tl.float32)
        draft_prob = tl.exp(draft_block_logits - draft_lse)
        residual_token_mass = tl.maximum(target_prob - draft_prob, 0.0)
    else:
        rejected_draft_token = tl.load(
            draft_sampled_ptr + resample_token_idx + 1, mask=is_random_residual & has_total_mass, other=-1
        )
        residual_token_mass = tl.where(token_ids != rejected_draft_token, target_prob, 0.0)

    token_mass = tl.where(is_bonus, bonus_token_mass, residual_token_mass)
    token_mass = tl.where(active_token_mask, token_mass, 0.0)
    selected_block_mass = tl.sum(token_mass, axis=0)
    has_selected_block_mass = selected_block_mass > 0.0
    token_prefix = tl.cumsum(token_mass, axis=0)
    candidate_offsets = tl.where(
        (token_prefix > remaining_threshold) & valid_token_mask & has_selected_block_mass,
        block_offsets,
        BLOCK_SIZE,
    )
    selected_offset = tl.min(candidate_offsets, axis=0)
    fallback_offset = tl.max(tl.where(valid_token_mask & (token_mass > 0.0), block_offsets, 0), axis=0)
    selected_offset = tl.where(selected_offset < BLOCK_SIZE, selected_offset, fallback_offset)

    categorical_token = selected_block * BLOCK_SIZE + selected_offset
    zero_mass_fallback = tl.where(has_total_mass, selected_block * BLOCK_SIZE, 0)
    categorical_token = tl.where(has_selected_block_mass, categorical_token, zero_mass_fallback)
    sampled_token = tl.where(is_greedy_bonus, greedy_sampled_token, categorical_token)

    # Greedy rejection already wrote the target argmax in the verification kernel.
    write_resampled_token = (~is_greedy) | is_bonus
    tl.store(sampled_ptr + req_idx * sampled_stride + resample_idx, sampled_token, mask=write_resampled_token)
    tl.store(num_sampled_ptr + req_idx, resample_idx + 1)


def resample(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    target_logits: torch.Tensor,
    target_rejected_logsumexp: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_rejected_logsumexp: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    has_draft_logits: bool | None = None,
) -> None:
    """Resample the first rejected token or the bonus token in place.

    ``sampled`` and ``num_sampled`` use the same names and ownership as the
    surrounding ``rejection_sample`` path. Random rejection/bonus tokens are
    written to ``sampled[req_idx, num_sampled[req_idx]]`` and ``num_sampled`` is
    advanced by one. Greedy non-bonus rows preserve the target argmax already
    written by the verification kernel and only advance ``num_sampled``.

    ``draft_logits=None`` selects the one-hot draft path. Callers that already
    replaced ``None`` with a dummy tensor can pass ``has_draft_logits=False``
    explicitly to preserve the same semantics.
    """
    if target_logits.ndim != 2 or target_logits.dtype != torch.float32 or target_logits.stride(-1) != 1:
        raise ValueError("target_logits must be contiguous-vocab FP32 [num_logits, vocab_size]")
    if sampled.ndim != 2 or sampled.dtype != torch.int64:
        raise ValueError("sampled must be int64 [num_reqs, num_speculative_steps + 1]")
    if num_sampled.ndim != 1 or num_sampled.dtype != torch.int32:
        raise ValueError("num_sampled must be int32 [num_reqs]")

    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = target_logits.shape[1]
    if sampled.shape[0] != num_reqs or num_sampled.shape[0] != num_reqs:
        raise ValueError("sampled, num_sampled, and cu_num_logits disagree on num_reqs")

    if has_draft_logits is None:
        has_draft_logits = draft_logits is not None
    if has_draft_logits:
        if draft_logits is None:
            raise ValueError("draft_logits cannot be None when has_draft_logits=True")
        if (
            draft_logits.ndim != 3
            or draft_logits.dtype != torch.float32
            or draft_logits.shape[-1] != vocab_size
            or draft_logits.stride(-1) != 1
        ):
            raise ValueError(
                "draft_logits must be contiguous-vocab FP32 [max_num_reqs, num_speculative_steps, vocab_size]"
            )
    elif draft_logits is None:
        draft_logits = target_logits.new_empty(1, 1, 1)

    num_blocks = triton.cdiv(vocab_size, _RESAMPLE_BLOCK_SIZE)
    local_argmax = torch.empty((num_reqs, num_blocks), dtype=torch.int64, device=target_logits.device)
    local_max = torch.empty((num_reqs, num_blocks), dtype=torch.float32, device=target_logits.device)
    local_mass = torch.empty((num_reqs, num_blocks), dtype=torch.float32, device=target_logits.device)
    num_workers = min(_get_vectorcore_num(), num_reqs * num_blocks)

    _resample_kernel[(num_workers,)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        local_mass,
        local_mass.stride(0),
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
        draft_sampled,
        temperature,
        num_reqs,
        num_blocks,
        vocab_size,
        BLOCK_SIZE=_RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        has_auto_blockify_blacklist_op=True,
    )

    _categorical_finalize_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        local_mass,
        local_mass.stride(0),
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
        draft_sampled,
        temperature,
        seed,
        pos,
        vocab_size,
        num_blocks,
        BLOCK_SIZE=_RESAMPLE_BLOCK_SIZE,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        HAS_DRAFT_LOGITS=has_draft_logits,
    )
