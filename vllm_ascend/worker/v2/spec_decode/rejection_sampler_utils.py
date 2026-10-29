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
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import _get_num_sampled_and_rejected_kernel
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
    _compute_global_residual_mass,
    _compute_local_residual_mass_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp as _compute_global_lse,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as _upstream_block_stats_kernel,
)

from vllm_ascend import envs
from vllm_ascend.ops.triton.v2.spec_decode.resample import resample

logger = init_logger(__name__)

# Workspaces for the block-level logits statistics and the rejected-token
# normalizers. They are keyed by the launch geometry and reused across decode
# steps: every cell is written by the kernels before it is read, the kernels
# are enqueued on the sampler stream, and no workspace is handed back to
# callers, so reuse cannot race with an in-flight step.
_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}
_DUMMY_DRAFT_LOGITS_CACHE: dict[tuple, torch.Tensor] = {}


def _get_workspace(
    target_logits: torch.Tensor,
    num_logits: int,
    num_reqs: int,
    vocab_num_blocks: int,
) -> tuple[torch.Tensor, ...]:
    key = (target_logits.device, num_logits, num_reqs, vocab_num_blocks)
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = (
            target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64),
            target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32),
            target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32),
            target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32),
            target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32),
            target_logits.new_empty(num_reqs, dtype=torch.float32),
            target_logits.new_empty(num_reqs, dtype=torch.float32),
        )
        _WORKSPACE_CACHE[key] = workspace
    return workspace


def _get_dummy_draft_logits(target_logits: torch.Tensor) -> torch.Tensor:
    """Placeholder pointer for the one-hot draft path; never read."""
    key = (target_logits.device, target_logits.dtype)
    dummy = _DUMMY_DRAFT_LOGITS_CACHE.get(key)
    if dummy is None:
        dummy = target_logits.new_empty(1, 1, 1)
        _DUMMY_DRAFT_LOGITS_CACHE[key] = dummy
    return dummy


@triton.jit(do_not_specialize=["n_reqs", "n_logits", "total_logits"])
def _pack_rejection_sample_metadata_kernel(
    # [n_reqs + 1] -> [MAX_REQS + 1]
    cu_src_ptr,
    cu_dst_ptr,
    # [n_reqs] -> [MAX_REQS]
    idx_src_ptr,
    idx_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    exp_idx_src_ptr,
    exp_idx_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    elp_src_ptr,
    elp_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    pos_src_ptr,
    pos_dst_ptr,
    # [n_logits] -> [MAX_LOGITS + 1]
    draft_sampled_src_ptr,
    draft_sampled_dst_ptr,
    # [MAX_REQS] -> [MAX_REQS]
    temp_src_ptr,
    temp_dst_ptr,
    # [MAX_REQS] -> [MAX_REQS]
    seed_src_ptr,
    seed_dst_ptr,
    n_reqs,
    n_logits,
    total_logits,
    MAX_REQS: tl.constexpr,
    MAX_LOGITS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPEC_STEPS: tl.constexpr,
):
    """Copy the small rejection-sampling inputs into the static graph buffers.

    One launch replaces eight ``copy_``/``fill_`` dispatches. Each program
    handles one segment; lanes past the live region receive the padding value
    that turns padded requests into no-ops for the captured kernels.
    """
    seg = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    # Each segment keeps a dedicated result variable: triton-ascend requires
    # variables assigned in every branch of an if/elif chain to share one
    # type, and the segments mix int32/int64/fp32.
    if seg == 0:
        # Padded requests repeat the total, so cu_num_logits[r] == cu_num_logits[r + 1]
        # and their token loops never execute.
        cu_vals = tl.load(cu_src_ptr + lanes, mask=lanes <= n_reqs, other=total_logits)
        tl.store(cu_dst_ptr + lanes, cu_vals, mask=lanes < MAX_REQS + 1)
    elif seg == 1:
        idx_vals = tl.load(idx_src_ptr + lanes, mask=lanes < n_reqs, other=0)
        tl.store(idx_dst_ptr + lanes, idx_vals, mask=lanes < MAX_REQS)
    elif seg == 2:
        exp_idx_vals = tl.load(exp_idx_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(exp_idx_dst_ptr + lanes, exp_idx_vals, mask=lanes < MAX_LOGITS)
    elif seg == 3:
        # Padded logits rows claim the bonus step so the stats kernel skips them.
        elp_vals = tl.load(elp_src_ptr + lanes, mask=lanes < n_logits, other=NUM_SPEC_STEPS)
        tl.store(elp_dst_ptr + lanes, elp_vals, mask=lanes < MAX_LOGITS)
    elif seg == 4:
        pos_vals = tl.load(pos_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(pos_dst_ptr + lanes, pos_vals, mask=lanes < MAX_LOGITS)
    elif seg == 5:
        # One extra slot: the finalize kernel reads draft_sampled[total_logits + 1]
        # for padded requests (their bonus test is false).
        ds_vals = tl.load(draft_sampled_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(draft_sampled_dst_ptr + lanes, ds_vals, mask=lanes < MAX_LOGITS + 1)
    elif seg == 6:
        temp_vals = tl.load(temp_src_ptr + lanes, mask=lanes < MAX_REQS, other=0.0)
        tl.store(temp_dst_ptr + lanes, temp_vals, mask=lanes < MAX_REQS)
    else:
        seed_vals = tl.load(seed_src_ptr + lanes, mask=lanes < MAX_REQS, other=0)
        tl.store(seed_dst_ptr + lanes, seed_vals, mask=lanes < MAX_REQS)


@triton.jit
def _compute_block_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    # The vocabulary and the speculative depth are fixed for a given model and
    # draft configuration. Declaring them constexpr lets triton-ascend resolve
    # the launch-time tiling at compile time instead of per launch.
    vocab_size: tl.constexpr,
    num_speculative_steps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    _upstream_block_stats_kernel(
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
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_DRAFT_LOGITS=HAS_DRAFT_LOGITS,
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
    # Fixed for a given model: constexpr so the tiling is resolved at compile
    # time rather than per launch.
    vocab_num_blocks: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # Cast to int64 before pointer arithmetic: triton-ascend computes
    # offsets in int32, which overflows when num_logits exceeds INT32_MAX
    # (upstream vllm #46560).
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)  # noqa: F841
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = tl.full((), 0, tl.int64)
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
                if SYNTHETIC_MODE:
                    # Synthetic acceptance: accept IFF u ~ U(0, 1) < rate.
                    # Upstream uses tl_rand32(seed, pos, includes_zero=False);
                    # NPU Triton lacks scalar tl.rand, so reuse the 1-element
                    # block + reduction pattern from the non-greedy path below.
                    u_pos = tl.load(pos_ptr + logit_idx).to(tl.int32)
                    u_seed = tl.randint(seed, u_pos)
                    u = tl.max(tl.rand(u_seed, tl.arange(0, 1)).to(tl.float32), axis=0)
                    u = tl.maximum(u, 4.6566127342e-10)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                    # -1 placeholder draft tokens can never be accepted.
                    accepted &= draft_sampled >= 0
                    # Keep the accepted draft token; store the target argmax
                    # upon rejection so resampling can be skipped.
                    token = tl.where(accepted, draft_sampled.to(tl.int64), target_argmax)
                    tl.store(sampled_ptr + req_idx * sampled_stride + i, token)
                else:
                    accepted &= target_argmax == draft_sampled
                    tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            elif USE_BLOCK_VERIFICATION:
                accepted = draft_sampled >= 0
                if accepted:
                    u_pos = tl.load(pos_ptr + logit_idx).to(tl.int32)
                    u_seed = tl.randint(seed, u_pos)
                    u = tl.max(tl.rand(u_seed, tl.arange(0, 1)).to(tl.float32), axis=0)
                    u = tl.maximum(u, 4.6566127342e-10)
                    prefix_joint_ratio = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32))
                    next_draft_token = tl.load(
                        draft_sampled_ptr + logit_idx + 2,
                        mask=i < num_tokens - 2,
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
                    rejected_step = tl.where(u <= h, i + 1, rejected_step)
                    tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                # -1 is used for placeholder draft token ids that should be
                # rejected.
                is_valid_draft = draft_sampled >= 0
                # Avoid possible OOB ptr access.
                draft_sampled = tl.maximum(0, draft_sampled)
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
                # Draw the acceptance threshold u ~ U(0, 1). Upstream uses
                # tl_rand64/tl_rand32; NPU Triton lacks float64 tl_rand64 and
                # scalar tl.rand, so generate u from a 1-element block and
                # clamp away from 0 so that tl.log(u) stays finite.
                # NPU: cast pos to int32 so philox uses the 32-bit path.
                # uint64 umulhi is not supported by the Ascend vector core.
                # Position values fit in int32 in practice.
                u_pos = tl.load(pos_ptr + logit_idx).to(tl.int32)
                u_seed = tl.randint(seed, u_pos)
                u = tl.max(tl.rand(u_seed, tl.arange(0, 1)).to(tl.float32), axis=0)
                u = tl.maximum(u, 4.6566127342e-10)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + i * draft_logits_stride_1
                        + draft_sampled
                    ).to(tl.float32)
                    # Match the temperature-scaled block statistics.
                    draft_logit = draft_logit / temp
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
                if SYNTHETIC_MODE:
                    # Synthetic acceptance: accept IFF u ~ U(0, 1) < rate.
                    # The logprob/LSE values above are still needed to
                    # resample the rejected token.
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted &= target_log_prob > tl.log(u) + draft_log_prob
                # -1 placeholder draft tokens can never be accepted.
                accepted &= is_valid_draft
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            if not USE_BLOCK_VERIFICATION or temp == 0.0:
                rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    if USE_BLOCK_VERIFICATION and (temp != 0.0 and rejected_step < num_tokens - 1):
        rejected_idx = start_idx + rejected_step
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


def _rejection_sample_impl(
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
    # Optional output buffers for the ACL graph path; allocated when None.
    sampled: torch.Tensor | None = None,
    num_sampled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fp64:
        raise NotImplementedError("FP64 rejection sampling is not supported on NPU.")
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    has_draft_logits = draft_logits is not None

    if draft_logits is None:
        # When draft_logits is None, pass a cached dummy tensor so that Triton
        # kernel signatures receive valid pointers/strides. The kernels
        # will never read from it when HAS_DRAFT_LOGITS=False.
        draft_logits = _get_dummy_draft_logits(target_logits)
    else:
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the block-level logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    (
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_local_max,
        draft_local_sumexp,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
    ) = _get_workspace(target_logits, num_logits, num_reqs, vocab_num_blocks)
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
        vocab_size=vocab_size,
        num_speculative_steps=num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        # TODO: Remove this workaround after the Triton Ascend AutoBlockify
        # bug for max-with-index reductions is fixed.
        has_auto_blockify_blacklist_op=True,
    )

    if use_block_verification:
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
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
            if num_logits > 1:
                _compute_local_residual_mass_kernel[(num_logits - 1, vocab_num_blocks)](
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
    if sampled is None:
        sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
    if num_sampled is None:
        num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
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
        vocab_num_blocks=vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    resample(
        sampled,
        num_sampled,
        target_logits,
        target_rejected_logsumexp,
        draft_logits,
        draft_rejected_logsumexp,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        has_draft_logits=has_draft_logits,
        cumulative_log_p=cumulative_log_p,
        use_block_verification=use_block_verification,
    )
    return sampled, num_sampled


class _RejectionSamplerAclGraph:
    """Replay the four rejection-sampling Triton kernels from one ACL graph.

    The eager path pays four Triton dispatches (~1.2ms host time per decode
    step) for ~170us of device work. This manager records the kernels once
    against persistent static buffers and replays them per step:

    - Before each replay the live inputs are copied into the static buffers:
      the two large logits buffers via ``copy_`` and all small index tensors
      via one metadata-pack kernel launch. All copies are stream-ordered, so
      no synchronization is needed before the replay.
    - Requests and logits rows are padded to ``max_num_reqs`` /
      ``max_num_reqs * (num_speculative_steps + 1)`` so one graph serves every
      batch. Padding values make padded rows no-ops: ``cu_num_logits`` repeats
      the total (empty token range), ``expanded_local_pos`` claims the bonus
      step (stats kernel skips the row) and the padded outputs are ignored by
      callers, which only read the live ``num_reqs`` prefix.
    - The static output buffers are consumed by the caller in stream order
      (num_sampled reduction, logprob kernels, async D2H copies); the next
      replay overwrites them only after those stream-ordered consumers.

    Capture happens lazily on the first eligible call: the buffers are filled
    with that call's live data, the impl runs once eagerly to compile the four
    kernels with their final launch keys, then the capture records the same
    launches. Any capture failure permanently disables the graph and falls
    back to the eager path.

    Lifetime: one instance per worker process, single-threaded; the static
    buffers (~50MB for logits) live for the process duration.
    """

    def __init__(self) -> None:
        self._graph = None
        self._disabled = False
        self._pack_block = 0
        self.device = None
        self.max_reqs = 0
        self.max_logits = 0
        self.vocab = 0
        self.num_spec_steps = 0
        self.has_draft = False
        self.target_buf: torch.Tensor | None = None
        self.draft_buf: torch.Tensor | None = None
        self.cu_buf: torch.Tensor | None = None
        self.idx_buf: torch.Tensor | None = None
        self.exp_idx_buf: torch.Tensor | None = None
        self.elp_buf: torch.Tensor | None = None
        self.pos_buf: torch.Tensor | None = None
        self.ds_buf: torch.Tensor | None = None
        self.temp_buf: torch.Tensor | None = None
        self.seed_buf: torch.Tensor | None = None
        self.sampled_buf: torch.Tensor | None = None
        self.num_sampled_buf: torch.Tensor | None = None

    def _is_stream_capturing(self) -> bool:
        try:
            return torch.npu.is_current_stream_capturing()
        except (AttributeError, RuntimeError):
            # Missing torch_npu or an uninitialized device: not capturing.
            return False

    def _sampler_aclgraph_enabled(self) -> bool:
        try:
            return bool(envs.VLLM_ASCEND_SAMPLER_ACLGRAPH)
        except AttributeError:
            # A stale vllm_ascend.envs without the registration must disable
            # the graph instead of crashing the worker at the first step.
            logger.warning_once(
                "vllm_ascend.envs has no VLLM_ASCEND_SAMPLER_ACLGRAPH entry; sampler ACL graph stays disabled."
            )
            return False

    def can_use(
        self,
        *,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        cu_num_logits: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
        synthetic_conditional_rates: torch.Tensor | None,
        use_fp64: bool,
        use_block_verification: bool,
    ) -> bool:
        if self._disabled or not self._sampler_aclgraph_enabled():
            return False
        # Only the plain probabilistic path is captured; the other modes keep
        # their eager implementations.
        if use_fp64 or use_block_verification or synthetic_conditional_rates is not None:
            return False
        if cu_num_logits.shape[0] - 1 < 1 or self._is_stream_capturing():
            return False
        if self._graph is None:
            # Fresh capture: buffers are sized from the request-state width,
            # so any batch that fits can be captured and padded.
            max_logits = temperature.shape[0] * (num_speculative_steps + 1)
            return target_logits.shape[0] <= max_logits
        return (
            num_speculative_steps == self.num_spec_steps
            and target_logits.device == self.device
            and target_logits.dtype == self.target_buf.dtype
            and target_logits.shape[1] == self.vocab
            and (draft_logits is not None) == self.has_draft
            and cu_num_logits.shape[0] - 1 <= self.max_reqs
            and target_logits.shape[0] <= self.max_logits
            and temperature.shape[0] == self.max_reqs
            and seed.shape[0] == self.max_reqs
            and (draft_logits is None or draft_logits.shape == self.draft_buf.shape)
        )

    def _fill_static_inputs(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
    ) -> None:
        n_logits = target_logits.shape[0]
        self.target_buf[:n_logits].copy_(target_logits, non_blocking=True)
        if self.has_draft:
            self.draft_buf.copy_(draft_logits, non_blocking=True)
        _pack_rejection_sample_metadata_kernel[(8,)](
            cu_num_logits,
            self.cu_buf,
            idx_mapping,
            self.idx_buf,
            expanded_idx_mapping,
            self.exp_idx_buf,
            expanded_local_pos,
            self.elp_buf,
            pos,
            self.pos_buf,
            draft_sampled,
            self.ds_buf,
            temperature,
            self.temp_buf,
            seed,
            self.seed_buf,
            cu_num_logits.shape[0] - 1,
            n_logits,
            n_logits,
            MAX_REQS=self.max_reqs,
            MAX_LOGITS=self.max_logits,
            BLOCK=self._pack_block,
            NUM_SPEC_STEPS=self.num_spec_steps,
        )

    def _capture(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        self.device = target_logits.device
        self.max_reqs = temperature.shape[0]
        self.num_spec_steps = num_speculative_steps
        self.max_logits = self.max_reqs * (num_speculative_steps + 1)
        self.vocab = target_logits.shape[1]
        self.has_draft = draft_logits is not None
        self._pack_block = triton.next_power_of_2(max(self.max_logits + 1, self.max_reqs + 1))
        static_args: tuple[torch.Tensor | None, ...]
        try:
            self.target_buf = torch.zeros((self.max_logits, self.vocab), dtype=target_logits.dtype, device=self.device)
            self.cu_buf = torch.zeros(self.max_reqs + 1, dtype=cu_num_logits.dtype, device=self.device)
            self.idx_buf = torch.zeros(self.max_reqs, dtype=idx_mapping.dtype, device=self.device)
            self.exp_idx_buf = torch.zeros(self.max_logits, dtype=expanded_idx_mapping.dtype, device=self.device)
            self.elp_buf = torch.zeros(self.max_logits, dtype=expanded_local_pos.dtype, device=self.device)
            self.pos_buf = torch.zeros(self.max_logits, dtype=pos.dtype, device=self.device)
            self.ds_buf = torch.zeros(self.max_logits + 1, dtype=draft_sampled.dtype, device=self.device)
            self.temp_buf = torch.zeros(self.max_reqs, dtype=temperature.dtype, device=self.device)
            self.seed_buf = torch.zeros(self.max_reqs, dtype=seed.dtype, device=self.device)
            if self.has_draft:
                self.draft_buf = torch.zeros(draft_logits.shape, dtype=draft_logits.dtype, device=self.device)
            self.sampled_buf = torch.empty(
                self.max_reqs, num_speculative_steps + 1, dtype=torch.int64, device=self.device
            )
            self.num_sampled_buf = torch.empty(self.max_reqs, dtype=torch.int32, device=self.device)

            static_args = (
                self.target_buf,
                self.draft_buf,
                self.ds_buf,
                self.cu_buf,
                self.pos_buf,
                self.idx_buf,
                self.exp_idx_buf,
                self.elp_buf,
                self.temp_buf,
                self.seed_buf,
                num_speculative_steps,
            )
            self._fill_static_inputs(
                target_logits,
                draft_logits,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                seed,
            )
            # Execute once outside capture so the four kernels are compiled
            # with their final launch keys (pointers, strides, constexprs)
            # before being recorded.
            _rejection_sample_impl(*static_args, sampled=self.sampled_buf, num_sampled=self.num_sampled_buf)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            # NOTE: super_kernel scope/optimize is intentionally NOT applied to
            # this graph. The optimization targets the aclnn-op model graphs;
            # wrapping raw Triton launches in it hangs on 910B at capture.
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
                _rejection_sample_impl(*static_args, sampled=self.sampled_buf, num_sampled=self.num_sampled_buf)
            self._graph = graph
            logger.info_once(
                "Captured rejection sampler ACL graph (num_reqs=%d, num_logits=%d, vocab=%d, draft_logits=%s).",
                self.max_reqs,
                self.max_logits,
                self.vocab,
                self.has_draft,
            )
        except Exception as exc:
            self._disabled = True
            logger.warning("Rejection sampler ACL graph capture failed; using eager rejection sampling: %s", exc)

    def run(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._graph is None:
            self._capture(
                target_logits,
                draft_logits,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                seed,
                num_speculative_steps,
            )
            if self._disabled:
                # Capture failed on this very call; produce the result eagerly.
                return _rejection_sample_impl(
                    target_logits,
                    draft_logits,
                    draft_sampled,
                    cu_num_logits,
                    pos,
                    idx_mapping,
                    expanded_idx_mapping,
                    expanded_local_pos,
                    temperature,
                    seed,
                    num_speculative_steps,
                )
            # The eager warm-up run already filled the static outputs.
            num_reqs = cu_num_logits.shape[0] - 1
            return self.sampled_buf[:num_reqs], self.num_sampled_buf[:num_reqs]

        self._fill_static_inputs(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
        )
        self._graph.replay()
        logger.info_once("Rejection sampler ACL graph replay engaged.")
        num_reqs = cu_num_logits.shape[0] - 1
        return self.sampled_buf[:num_reqs], self.num_sampled_buf[:num_reqs]


_SAMPLER_ACLGRAPH = _RejectionSamplerAclGraph()


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
    if _SAMPLER_ACLGRAPH.can_use(
        target_logits=target_logits,
        draft_logits=draft_logits,
        cu_num_logits=cu_num_logits,
        temperature=temperature,
        seed=seed,
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=synthetic_conditional_rates,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
    ):
        return _SAMPLER_ACLGRAPH.run(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            num_speculative_steps,
        )
    return _rejection_sample_impl(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        synthetic_conditional_rates,
        use_fp64,
        use_block_verification,
    )


class _FastKernelLaunch:
    """Replay the concrete ``CompiledKernel.run`` call of a Triton kernel.

    The triton-ascend launcher costs ~200us of host time per launch even for
    pointer-only kernels, and after the sampler graph replay the device queue
    is empty, so that cost turns directly into a device bubble before
    ``get_num_sampled_and_rejected``. This helper records the argument tuple
    that ``JITFunction`` passes to ``CompiledKernel.run`` on a real launch and
    replays it with only the grid, the stream and the tensor arguments
    substituted, skipping argument binding and specialization-key work.

    The record is taken empirically from a live launch, so the launcher
    signature is whatever the installed Triton actually uses. Every fast call
    validates pointer alignment and dtypes against the recorded launch (the
    binary was specialized on them); any deviation, probe failure or exception
    permanently falls back to the standard ``kernel[grid](...)`` dispatch.
    """

    def __init__(self, jit_fn) -> None:
        self._jit_fn = jit_fn
        self._learned = None
        self._disabled = False

    def launch(self, grid_0: int, *tensor_args: torch.Tensor) -> None:
        if self._disabled:
            self._jit_fn[(grid_0,)](*tensor_args)
            return
        state = self._learned
        if state is None:
            self._jit_fn[(grid_0,)](*tensor_args)
            self._learn(grid_0, tensor_args)
            return
        compiled, base, tensor_slots, stream_slot, device, capture_aligned, capture_dtypes = state
        try:
            for tensor, aligned, dtype in zip(tensor_args, capture_aligned, capture_dtypes):
                if aligned and tensor.data_ptr() % 16 != 0:
                    raise RuntimeError("pointer alignment differs from the recorded launch")
                if tensor.dtype != dtype:
                    raise RuntimeError("argument dtype differs from the recorded launch")
            args = list(base)
            args[0], args[1], args[2] = grid_0, 1, 1
            args[stream_slot] = self._current_stream(device)
            for slot, tensor in zip(tensor_slots, tensor_args):
                args[slot] = tensor
            compiled.run(*args)
        except Exception as exc:
            self._disable(exc)
            self._jit_fn[(grid_0,)](*tensor_args)

    def _learn(self, grid_0: int, tensor_args: tuple[torch.Tensor, ...]) -> None:
        try:
            compiled = self._probe_compiled()
            if compiled is None:
                raise RuntimeError("compiled-kernel cache probe failed")
            base_holder: dict[str, tuple] = {}
            orig_run = compiled.run
            had_instance_attr = "run" in vars(compiled)

            def recorder(*args, **kwargs):
                base_holder["args"] = args
                base_holder["kwargs"] = kwargs
                return orig_run(*args, **kwargs)

            compiled.run = recorder
            try:
                self._jit_fn[(grid_0,)](*tensor_args)
            finally:
                if had_instance_attr:
                    compiled.run = orig_run
                else:
                    compiled.__dict__.pop("run", None)

            if base_holder.get("kwargs"):
                raise RuntimeError("launcher received keyword arguments")
            base = base_holder.get("args")
            if not base or len(base) < len(tensor_args) + 4 or tuple(base[:3]) != (grid_0, 1, 1):
                raise RuntimeError("unexpected launcher argument layout")
            if not isinstance(base[3], int) or base[3] != self._current_stream(torch.npu.current_device()):
                raise RuntimeError("unexpected launcher stream layout")
            tensor_slots = []
            for tensor in tensor_args:
                slots = [i for i, value in enumerate(base) if value is tensor]
                if len(slots) != 1:
                    raise RuntimeError("tensor argument not found exactly once in the launcher call")
                tensor_slots.append(slots[0])
            if len(set(tensor_slots)) != len(tensor_args) or min(tensor_slots) <= 3:
                raise RuntimeError("unexpected tensor slot layout")
            self._learned = (
                compiled,
                base,
                tuple(tensor_slots),
                3,
                torch.npu.current_device(),
                tuple(tensor.data_ptr() % 16 == 0 for tensor in tensor_args),
                tuple(tensor.dtype for tensor in tensor_args),
            )
            logger.info_once("Fast triton launch learned for %s.", getattr(self._jit_fn, "__name__", "kernel"))
        except Exception as exc:
            self._disable(exc)

    def _probe_compiled(self):
        jit = self._jit_fn
        device = torch.npu.current_device()
        caches = getattr(jit, "device_caches", None)
        if caches is not None:
            entry = caches.get(device)
            if entry is None and len(caches) == 1:
                entry = next(iter(caches.values()))
            kernel_cache = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(kernel_cache, dict) and len(kernel_cache) == 1:
                return next(iter(kernel_cache.values()))
        cache = getattr(jit, "cache", None)
        if isinstance(cache, dict):
            entry = cache.get(device)
            if isinstance(entry, dict) and len(entry) == 1:
                return next(iter(entry.values()))
        return None

    @staticmethod
    def _current_stream(device) -> int:
        from triton.runtime import driver

        return driver.active.get_current_stream(device)

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        logger.warning(
            "Fast triton launch for %s disabled (%s); using standard dispatch.",
            getattr(self._jit_fn, "__name__", "kernel"),
            exc,
        )


# Persistent ``num_rejected`` buffers keyed by (device, num_reqs); upstream
# consumes them via a stream-ordered async D2H before the next step's kernel
# rewrites them, the same contract as the graph's static output buffers.
_NUM_REJECTED_BUF_CACHE: dict[tuple, torch.Tensor] = {}

_GET_NUM_FAST = _FastKernelLaunch(_get_num_sampled_and_rejected_kernel)


def get_num_sampled_and_rejected(
    # [num_reqs], mutated in place (chunked-prefill rows are zeroed)
    num_sampled: torch.Tensor,
    # [max_num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    if num_reqs < 1:
        return num_sampled, torch.empty_like(num_sampled)
    key = (num_sampled.device, num_reqs)
    num_rejected = _NUM_REJECTED_BUF_CACHE.get(key)
    if num_rejected is None:
        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=num_sampled.device)
        _NUM_REJECTED_BUF_CACHE[key] = num_rejected
    _GET_NUM_FAST.launch(num_reqs, num_sampled, num_rejected, seq_lens, cu_num_logits, idx_mapping, prefill_len)
    return num_sampled, num_rejected
