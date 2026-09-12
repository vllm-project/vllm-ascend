# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Categorical resampling for speculative decoding on Ascend NPU."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_RESAMPLE_BLOCK_SIZE = 1024
# Offset salt keeping the resample threshold's noise stream disjoint from the
# verification kernel's acceptance draws (keyed by the raw position) and from
# the draft sampling salt (upstream #54282 uses 1 << 30). Positions never
# approach either constant in practice.
_RESAMPLE_NOISE_SALT = tl.constexpr(1 << 29)


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
    # [num_logits]
    cumulative_log_p_ptr,
    num_reqs,
    num_blocks,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
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
        # Cast to int64 before pointer arithmetic: triton-ascend computes
        # offsets in int32, which overflows when num_logits exceeds INT32_MAX
        # (upstream vllm #46560).
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
        resample_token_idx = start_idx + resample_idx
        req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)
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
                rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
                is_valid_rejected_draft = rejected_draft_token >= 0
                target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
                target_prob = tl.exp(target_block_logits - target_lse)

                if not is_valid_rejected_draft:
                    # -1 placeholder draft token: verification stopped at the
                    # placeholder, so the residual is the full target
                    # distribution (no draft subtraction; the draft logits at
                    # a placeholder step are stale).
                    token_mass = target_prob
                elif HAS_DRAFT_LOGITS:
                    # draft_logits is stored pre-temperature, so apply scale
                    # first (matches the block-stats kernel and the upstream
                    # rejection helpers, which all divide by temp).
                    draft_block_logits = (
                        tl.load(
                            draft_logits_ptr
                            + req_state_idx * draft_logits_stride_0
                            + resample_idx * draft_logits_stride_1
                            + vocab_offsets,
                            mask=vocab_mask,
                            other=float("-inf"),
                        ).to(tl.float32)
                        / temperature
                    )
                    draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
                    draft_prob = tl.exp(draft_block_logits - draft_lse)
                    if USE_BLOCK_VERIFICATION:
                        # Block verification (Sun et al., 2024,
                        # https://arxiv.org/abs/2403.10444): the residual is
                        #   max(p_tau * p(x) - q(x), 0) / Z,
                        # where p_tau is the joint ratio of the accepted
                        # prefix. Scale the target probabilities by p_tau
                        # before subtracting the draft distribution.
                        # cumulative_log_p[start + i] = log(p_{i+1}), so the
                        # ratio after tau = resample_idx accepted tokens
                        # lives at resample_token_idx - 1. p_0 = 1 (nothing
                        # accepted), so skip the load when resample_idx == 0.
                        if resample_idx > 0:
                            log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(tl.float32)
                            target_prob = target_prob * tl.exp(log_p_tau)
                    # NPU: upstream #46665 computes this residual in log space
                    # with tldevice.log1p(-ratio); that extern is unavailable
                    # on triton-ascend, and this kernel works in mass space.
                    # The subtraction is still exact where it matters: by the
                    # Sterbenz lemma, target_prob - draft_prob is exactly
                    # representable in fp32 when the two probabilities are
                    # within a factor of 2, so no catastrophic cancellation
                    # occurs when the draft closely matches the target.
                    token_mass = tl.maximum(target_prob - draft_prob, 0.0)
                else:
                    # One-hot draft. NOTE: during block verification the
                    # residual becomes p_tau * p(x) / Z for x != draft token,
                    # so the constant p_tau cancels under normalization and
                    # does not need to be applied.
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
    # [num_logits]
    cumulative_log_p_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    """Select the final token using one global categorical threshold per request."""
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    # Cast to int64 before pointer arithmetic: triton-ascend computes
    # offsets in int32, which overflows when num_logits exceeds INT32_MAX
    # (upstream vllm #46560).
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

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
    # NPU: salt the position so this draw is independent of the acceptance
    # threshold u, which the verification kernel derives from the same
    # (seed, pos) pair at the rejected row. Reusing that value conditions the
    # residual sample on the rejection -- there u only spans (h, 1) of the
    # unit interval, so the threshold u * Z can never land in the first h * Z
    # of the residual CDF and the output marginal drifts off the target.
    seed = tl.load(seed_ptr + req_state_idx)
    position = tl.load(pos_ptr + resample_token_idx).to(tl.int32) + _RESAMPLE_NOISE_SALT
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

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1, mask=is_random_residual & has_total_mass, other=-1
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    if HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first
        # (matches the block-stats kernel and the upstream rejection helpers,
        # which all divide by temp). Greedy lanes hold no draft mass anyway;
        # dividing by 1.0 there keeps the arithmetic finite.
        draft_block_logits = tl.load(
            draft_logits_ptr + req_state_idx * draft_logits_stride_0 + resample_idx * draft_logits_stride_1 + token_ids,
            mask=valid_token_mask & is_random_residual & has_total_mass,
            other=float("-inf"),
        ).to(tl.float32)
        draft_block_logits = draft_block_logits / tl.where(temperature != 0.0, temperature, 1.0)
        draft_lse = tl.load(
            draft_rejected_logsumexp_ptr + req_idx, mask=is_random_residual & has_total_mass, other=0.0
        ).to(tl.float32)
        draft_prob = tl.exp(draft_block_logits - draft_lse)
        # Block verification (Sun et al., 2024): the residual is
        #   max(p_tau * p(x) - q(x), 0) / Z.
        # Scale the target probabilities by the accepted prefix's joint
        # ratio p_tau, mirroring _resample_kernel so the block selection
        # and the within-block token selection stay consistent.
        # cumulative_log_p[start + i] = log(p_{i+1}), so the ratio after
        # tau = resample_idx accepted tokens lives at
        # resample_token_idx - 1. p_0 = 1 (nothing accepted), so skip
        # the load when resample_idx == 0. A -1 placeholder draft token
        # means verification stopped at the placeholder; that residual is
        # the full target distribution (the draft logits at a placeholder
        # step are stale) and must stay unscaled to match _resample_kernel's
        # block masses on the same path.
        if USE_BLOCK_VERIFICATION and (is_valid_rejected_draft and resample_idx > 0):
            log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(tl.float32)
            target_prob = target_prob * tl.exp(log_p_tau)
        residual_token_mass = tl.where(
            is_valid_rejected_draft,
            tl.maximum(target_prob - draft_prob, 0.0),
            target_prob,
        )
    else:
        # One-hot draft. NOTE: during block verification the residual becomes
        # p_tau * p(x) / Z for x != draft token, so the constant p_tau cancels
        # under normalization and does not need to be applied.
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
    cumulative_log_p: torch.Tensor | None = None,
    use_block_verification: bool = False,
) -> None:
    """Resample the first rejected token or the bonus token in place.

    ``sampled`` and ``num_sampled`` use the same names and ownership as the
    surrounding ``rejection_sample`` path. Random rejection/bonus tokens are
    written to ``sampled[req_idx, num_sampled[req_idx]]`` and ``num_sampled`` is
    advanced by one. Greedy non-bonus rows preserve the target argmax already
    written by the verification kernel and only advance ``num_sampled``.

    ``draft_logits`` contains raw cached logits; draft logsumexp statistics
    must include temperature scaling, as in the verification path.

    ``target_logits`` and full ``draft_logits`` support fp16, bf16, and fp32;
    probability-mass arithmetic is fp32. Their vocabulary dimension must be
    contiguous. ``draft_logits=None`` selects the one-hot draft path. Callers
    that already replaced ``None`` with a dummy tensor can pass
    ``has_draft_logits=False`` explicitly to preserve the same semantics.

    ``use_block_verification=True`` (Sun et al., 2024) resamples the rejected
    token from ``max(p_tau * p(x) - q(x), 0) / Z`` instead of the token-wise
    residual, where ``p_tau`` is the joint ratio of the accepted prefix read
    from ``cumulative_log_p`` (``cumulative_log_p[start + i] = log(p_{i+1})``,
    as produced by the upstream ``_compute_cumulative_log_p_kernel``).
    """
    num_reqs = cu_num_logits.shape[0] - 1
    if num_reqs == 0:
        return

    vocab_size = target_logits.shape[1]
    if vocab_size == 0:
        raise ValueError("vocab_size must be greater than 0")
    if target_logits.stride(-1) != 1:
        raise ValueError("target_logits vocabulary dimension must be contiguous")

    if has_draft_logits is None:
        has_draft_logits = draft_logits is not None
    if has_draft_logits:
        if draft_logits is None:
            raise ValueError("draft_logits cannot be None when has_draft_logits=True")
        if draft_logits.stride(-1) != 1:
            raise ValueError("draft_logits vocabulary dimension must be contiguous")
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding. Clamp so the
        # kernels only read the draft logits within their valid range; the
        # target padding columns are never sampled anyway because their
        # logits are dominated by real tokens.
        vocab_size = min(vocab_size, draft_logits.size(-1))
    elif draft_logits is None:
        draft_logits = target_logits.new_empty(1, 1, 1)

    if use_block_verification and cumulative_log_p is None:
        raise ValueError("cumulative_log_p cannot be None when use_block_verification=True")
    if cumulative_log_p is None:
        # Dummy tensor so the kernel signature receives a valid pointer; it is
        # never read when USE_BLOCK_VERIFICATION=False.
        cumulative_log_p = target_logits.new_empty(1, dtype=torch.float32)

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
        cumulative_log_p,
        num_reqs,
        num_blocks,
        vocab_size,
        BLOCK_SIZE=_RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_BLOCK_VERIFICATION=use_block_verification,
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
        cumulative_log_p,
        vocab_size,
        num_blocks,
        BLOCK_SIZE=_RESAMPLE_BLOCK_SIZE,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_BLOCK_VERIFICATION=use_block_verification,
    )
