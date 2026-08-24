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
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
    _compute_global_logprobs_and_logsumexp,
    _compute_global_residual_mass,
    _compute_global_target_argmax,
    _compute_local_residual_mass_kernel,
    _insert_resampled_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp as _compute_global_lse,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
)


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
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
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
    # [num_logits]
    cumulative_log_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    # NPU: use `== 0` instead of Python `not` on the scalar tensor so the
    # mask stays a plain tensor expression.
    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=is_bonus == 0,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus or not is_valid_rejected_draft:
        # Bonus token (no rejections) or -1 placeholder token. In either case,
        # directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logits = (
            tl.load(
                draft_logits_ptr + req_state_idx * draft_logits_stride_0 + resample_idx * draft_logits_stride_1 + block,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            # Block residual is:
            #   max(p_tau * M_b(x) - M_s(x), 0) / Z.
            # Scale the target logprobs by log(p_tau). p_0 = 1, so skip
            # shifting when nothing was accepted (tau == 0).
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(tl.float32)
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual:
        #   r(x) = max(p(x) - q(x), 0)
        # Gumbel sampling needs logits, so we compute it in log space:
        #   log(r(x)) = log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        #   log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        # NOTE: During block verification, the residual becomes:
        #   0                   if x == rejected_draft_token
        #   p_tau * M_b(x) / Z  otherwise
        # Therefore p_tau is a constant that cancels under normalization,
        # and does not need to be applied.
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
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        # Avoid possible OOB ptr access.
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft

        if verifying:
            # Draw the acceptance threshold u ~ U(0, 1). Upstream uses
            # tl_rand32(seed, pos, includes_zero=False); NPU Triton lacks
            # scalar tl.rand, so generate u from a 1-element block + reduction
            # and clamp away from 0 so that tl.log(u) stays finite.
            # NPU: cast pos to int32 so philox uses the 32-bit path.
            # uint64 umulhi is not supported by the Ascend vector core.
            # Position values fit in int32 in practice.
            u_pos = tl.load(pos_ptr + logit_idx).to(tl.int32)
            u_seed = tl.randint(seed, u_pos)
            u = tl.max(tl.rand(u_seed, tl.arange(0, 1)).to(tl.float32), axis=0)
            u = tl.maximum(u, 4.6566127342e-10)
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    # Synthetic acceptance: accept IFF u ~ U(0, 1) < rate.
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                # -1 placeholder draft tokens can never be accepted.
                accepted &= is_valid_draft
                verifying = accepted
                accepted_length += accepted.to(tl.int64)
                # Keep the accepted draft token; store the target argmax
                # upon rejection so resampling can be skipped.
                token = tl.where(accepted, draft_sampled, target_argmax)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, token)
            elif USE_BLOCK_VERIFICATION:
                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
                prefix_joint_ratio = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32))
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                target_log_prob, draft_log_prob, target_lse, draft_lse = _compute_global_logprobs_and_logsumexp(
                    draft_sampled,
                    True,  # mask
                    logit_idx,
                    req_state_idx,
                    i,
                    temp,
                    target_logits_ptr,
                    target_logits_stride,
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    draft_logits_ptr,
                    draft_logits_stride_0,
                    draft_logits_stride_1,
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    HAS_DRAFT_LOGITS,
                )
                if SYNTHETIC_MODE:
                    # Synthetic acceptance: accept IFF u ~ U(0, 1) < rate.
                    # The logprob/LSE values above are still needed to
                    # resample the rejected token.
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_log_prob > tl.log(u) + draft_log_prob
                # -1 placeholder draft tokens can never be accepted.
                accepted &= is_valid_draft
                verifying = accepted
                accepted_length += accepted.to(tl.int64)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    tl.store(rejected_steps_ptr + req_idx, accepted_length.to(tl.int32))
    # NPU: parenthesize the tensor-side condition. A flat `constexpr and
    # tensor and tensor` chain fails to compile on triton-ascend.
    if USE_BLOCK_VERIFICATION and (not is_greedy and accepted_length < num_draft_tokens):
        # Compute the target and draft log exponential sums for the
        # rejected token.
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_lse(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_lse(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
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
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fp64:
        raise NotImplementedError("FP64 rejection sampling is not supported on NPU.")

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
    _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
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
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio
        # after the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
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
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later
        # reduced to the total by _compute_global_residual_mass. Only
        # launched for full draft logits distributions. One-hot drafts use a
        # closed-form residual mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits.stride(0),
                draft_logits.stride(1),
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_speculative_steps,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

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
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.int64)
    # NPU does not support float64; use float32 for resampled_local_max.
    resampled_local_max = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.float32)
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
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_BLOCK_VERIFICATION=use_block_verification,
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
