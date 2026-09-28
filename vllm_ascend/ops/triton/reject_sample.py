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

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num


def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


@triton.jit(do_not_specialize=["vec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    bonus_token_id = tl.load(bonus_token_ids_ptr + offset, mask)

    if SYNTHETIC_MODE:
        # Synthetic: accept the draft token with prob conditional_rates[0],
        # regardless of target match. Accepted => emit draft token (pos 0) and
        # the bonus token (pos 1); rejected => emit target_argmax (pos 0).
        uniform_prob = tl.load(uniform_probs_ptr + offset, mask)
        # spec_len == 1 => only position 0.
        rate = tl.load(synthetic_conditional_rates_ptr + 0)
        accepted = (uniform_prob < rate) & (draft_token_id >= 0) & mask
        # Cast both arms to int32: draft_token_id is int32, target_argmax_id is
        # int64 (from argmax); tl.where requires matching dtypes.
        token_id = tl.where(accepted, draft_token_id.to(tl.int32), target_argmax_id.to(tl.int32))
        tl.store(output_token_ids_ptr + offset * 2, token_id, mask)
        accept_mask = accepted
    else:
        tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)
        accept_mask = (draft_token_id == target_argmax_id) & mask
    tl.store(output_token_ids_ptr + offset * 2 + 1, bonus_token_id, accept_mask)


class RejectionGreedySpecLen1Kernel(VllmTritonJitKernel["RejectionGreedySpecLen1Kernel.CompileKey"]):
    kernel = rejection_greedy_sample_spec_len_1_triton

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        synthetic_mode: bool

    def dispatch(self, *, block_size: int, synthetic_mode: bool) -> CompileKey:
        return self.CompileKey(block_size=block_size, synthetic_mode=synthetic_mode)

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        # num_draft_tokens is all ones only in the spec-len-1 path. For larger
        # max_spec_len the runtime dispatcher selects the general greedy kernel.
        if context.max_spec_len != 1:
            return []
        rows = [
            dict(block_size=block_size, synthetic_mode=context.synthetic_mode) for block_size in context.block_sizes
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, 2)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            target_argmax=TritonWarmupTensor(torch.int64, shape=(1,)),
            bonus_token_ids=TritonWarmupTensor(torch.int64, shape=(1,)),
            vec_len=1,
            uniform_probs=(TritonWarmupTensor(torch.float32, shape=(1,)) if compile_key.synthetic_mode else None),
            synthetic_conditional_rates=(
                TritonWarmupTensor(torch.float32, shape=(1,)) if compile_key.synthetic_mode else None
            ),
            block_size=compile_key.block_size,
            synthetic_mode=compile_key.synthetic_mode,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        uniform_probs: torch.Tensor | None,
        synthetic_conditional_rates: torch.Tensor | None,
        *,
        vec_len: int,
        block_size: int,
        synthetic_mode: bool,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(SYNTHETIC_MODE=synthetic_mode, BLOCK_SIZE=block_size)


_REJECTION_GREEDY_SPEC_LEN_1_KERNEL = RejectionGreedySpecLen1Kernel()


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1, bonus_token_id)


@triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    if is_greedy_ptr is None:
        is_greedy_mask = mask
    else:
        is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
        is_greedy_mask = mask & (is_greedy != 0)

    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offset == 0 must
    # be masked out at the load. other=0 keeps num_draft_tokens deterministic
    # (0) for masked lanes, which the per-position loop below relies on to
    # skip them.
    start_idx = tl.load(cu_num_draft_tokens_ptr + offset - 1, mask=is_greedy_mask & (offset > 0), other=0)
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask, other=0)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_draft_tokens, (pos,))
        rejected = False
        start_idx1 = get_element(start_idx, (pos,))
        is_greedy_mask1 = get_element(is_greedy_mask, (pos,))
        position = block_idx * BLOCK_SIZE + pos
        for i in range(num_tokens1):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx1 + i)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx1 + i)
                if SYNTHETIC_MODE:
                    # Synthetic: accept draft token i with prob
                    # conditional_rates[i], independent of target match. Store
                    # each arm separately (draft on accept, target_argmax on
                    # reject) so the int32 (draft) / int64 (target_argmax)
                    # dtype mismatch is handled by the store's implicit cast --
                    # no ternary, no explicit cast (matches the random kernel's
                    # synthetic branch).
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx1 + i)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = (uniform_prob < rate) & (draft_token_id >= 0)
                    if accepted:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            draft_token_id,
                        )
                    else:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            target_argmax_id,
                        )
                        rejected = True
                else:
                    tl.store(
                        output_token_ids_ptr + position * (max_spec_len + 1) + i,
                        target_argmax_id,
                    )
                    if draft_token_id != target_argmax_id:
                        # Reject.
                        rejected = True

        if not rejected and is_greedy_mask1:
            bonus_renew(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
                max_spec_len,
                num_tokens1,
            )


class RejectionGreedyKernel(VllmTritonJitKernel["RejectionGreedyKernel.CompileKey"]):
    kernel = rejection_greedy_sample_triton

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        synthetic_mode: bool
        has_is_greedy: bool

    def dispatch(self, *, block_size: int, synthetic_mode: bool, has_is_greedy: bool) -> CompileKey:
        return self.CompileKey(block_size=block_size, synthetic_mode=synthetic_mode, has_is_greedy=has_is_greedy)

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        # With spec_len == 1, is_greedy=None is handled by the spec-len-1
        # kernel. Only the explicit is_greedy tensor path reaches this kernel.
        # For larger spec lengths, both paths use the general greedy kernel.
        has_is_greedy_values = (True,) if context.max_spec_len == 1 else (False, True)
        rows = [
            dict(
                block_size=block_size,
                synthetic_mode=context.synthetic_mode,
                has_is_greedy=has_is_greedy,
            )
            for block_size in context.block_sizes
            for has_is_greedy in has_is_greedy_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, MAX_SPEC_LEN + 1)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            target_argmax=TritonWarmupTensor(torch.int64, shape=(1,)),
            bonus_token_ids=TritonWarmupTensor(torch.int64, shape=(1,)),
            is_greedy=(TritonWarmupTensor(torch.bool, shape=(1,)) if compile_key.has_is_greedy else None),
            vec_len=1,
            max_spec_len=1,
            uniform_probs=(TritonWarmupTensor(torch.float32, shape=(1,)) if compile_key.synthetic_mode else None),
            synthetic_conditional_rates=(
                TritonWarmupTensor(torch.float32, shape=(1,)) if compile_key.synthetic_mode else None
            ),
            block_size=compile_key.block_size,
            synthetic_mode=compile_key.synthetic_mode,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        is_greedy: torch.Tensor | None,
        uniform_probs: torch.Tensor | None,
        synthetic_conditional_rates: torch.Tensor | None,
        *,
        vec_len: int,
        max_spec_len: int,
        block_size: int,
        synthetic_mode: bool,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(SYNTHETIC_MODE=synthetic_mode, BLOCK_SIZE=block_size)


_REJECTION_GREEDY_KERNEL = RejectionGreedyKernel()


@triton.jit(
    do_not_specialize=[
        "max_spec_len",
        "vec_len",
    ]
)
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size] or [num_tokens, selected_vocab_size] if ENABLE_REDUCE_SAMPLING
    target_indices_ptr,  # [num_tokens, selected_vocab_size] global vocab indices, only used if ENABLE_REDUCE_SAMPLING
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,  # vocab_size or selected_vocab_size if ENABLE_REDUCE_SAMPLING
    global_vocab_size,  # global vocab size for draft_probs indexing (only used if ENABLE_REDUCE_SAMPLING)
    vec_len,
    ori_target_probs_ptr,  # [num_tokens, ori_vocab_size] original probs for entropy
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,  # Whether using reduce sampling
    SYNTHETIC_MODE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
    POSTERIOR_THRESHOLD: tl.constexpr = 0.95,
    POSTERIOR_ALPHA: tl.constexpr = 0.4,
    SUB_BLOCK: tl.constexpr = 4096,
    EPSILON: tl.constexpr = 1e-10,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offsets == 0 must
    # be masked out at the load.
    start_idxs = tl.load(cu_num_draft_tokens_ptr + offsets - 1, mask=not_greedy_mask & (offsets > 0), other=0)
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask, other=0)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                if not rejected:
                    if SYNTHETIC_MODE:
                        # Synthetic: accept draft token with prob
                        # conditional_rates[pos], bypassing target/draft prob
                        # comparison. Output may be incorrect - benchmarking only.
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                        uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                        rate = tl.load(synthetic_conditional_rates_ptr + pos)
                        accepted = (uniform_prob < rate) & (draft_token_id >= 0)
                        if accepted:
                            tl.store(
                                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                                draft_token_id,
                            )
                        else:
                            rejected = True
                            recovered = tl.load(recovered_token_ids_ptr + token_idx)
                            tl.store(
                                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                                recovered,
                            )
                    elif ENABLE_REDUCE_SAMPLING:
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                        if draft_token_id == -1:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        else:
                            target_prob = 0.0
                            found = False

                            for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
                                if not found:
                                    vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
                                    vocab_mask = vocab_offsets < vocab_size

                                    candidate_indices = tl.load(
                                        target_indices_ptr + token_idx * vocab_size + vocab_offsets,
                                        mask=vocab_mask,
                                        other=-1,
                                    )

                                    match_mask = candidate_indices == draft_token_id

                                    candidate_probs = tl.load(
                                        target_probs_ptr + token_idx * vocab_size + vocab_offsets,
                                        mask=vocab_mask,
                                        other=0.0,
                                    )

                                    current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)
                                    if current_match_prob > 0.0:
                                        target_prob = current_match_prob
                                        found = True

                            if NO_DRAFT_PROBS:
                                draft_prob = 1
                            else:
                                draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)

                            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

                            # Acceptance condition
                            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                                # Accept
                                token_id = draft_token_id
                            else:
                                # Reject - use recovered token
                                rejected = True
                                token_id = tl.load(recovered_token_ids_ptr + token_idx)

                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
                    else:
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                        if draft_token_id == -1:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        else:
                            target_prob = tl.load(target_probs_ptr + token_idx * global_vocab_size + draft_token_id)
                            if NO_DRAFT_PROBS:
                                draft_prob = 1
                            else:
                                draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)
                            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

                            if ENTROPY_VERIFY:
                                loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
                                entropy = 0.0
                                for loop_i in range(loop):
                                    vocab_start = loop_i * SUB_BLOCK
                                    vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                                    vocab_mask = vocab_offset < vocab_size
                                    if NO_ORI_TARGET_PROBS:
                                        probs = tl.load(
                                            target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                            vocab_mask,
                                            other=0,
                                        )
                                    else:
                                        probs = tl.load(
                                            ori_target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                            vocab_mask,
                                            other=0,
                                        )
                                    log_probs = tl.log(probs + EPSILON)
                                    entropy_contrib = -probs * log_probs
                                    entropy += tl.sum(entropy_contrib)

                                exp_neg_entropy = tl.exp(-entropy * POSTERIOR_ALPHA)
                                threshold_by_entropy = exp_neg_entropy
                                threshold = tl.minimum(threshold_by_entropy, POSTERIOR_THRESHOLD)
                                _uniform_prob = threshold * uniform_prob
                            else:
                                _uniform_prob = uniform_prob
                            # NOTE(woosuk): While the draft probability should never be 0,
                            # we check it to avoid NaNs. If it happens to be 0, we reject.
                            if draft_prob > 0 and target_prob / draft_prob >= _uniform_prob:
                                # Accept.
                                token_id = draft_token_id
                            else:
                                # Reject. Use recovered token.
                                rejected = True
                                token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if not rejected:
                # If all tokens are accepted, append the bonus token.
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )


class RejectionRandomSampleKernel(VllmTritonJitKernel["RejectionRandomSampleKernel.CompileKey"]):
    kernel = rejection_random_sample_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        no_ori_target_probs: bool
        no_draft_probs: bool
        enable_reduce_sampling: bool
        synthetic_mode: bool
        entropy_verify: bool
        vocab_block_size: int
        posterior_threshold: float
        posterior_alpha: float
        sub_block: int
        epsilon: float

    def dispatch(
        self,
        *,
        block_size: int,
        no_ori_target_probs: bool,
        no_draft_probs: bool,
        enable_reduce_sampling: bool,
        synthetic_mode: bool,
        entropy_verify: bool,
        vocab_block_size: int,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
    ) -> CompileKey:
        return self.CompileKey(
            block_size=block_size,
            no_ori_target_probs=no_ori_target_probs,
            no_draft_probs=no_draft_probs,
            enable_reduce_sampling=enable_reduce_sampling,
            synthetic_mode=synthetic_mode,
            entropy_verify=entropy_verify,
            vocab_block_size=vocab_block_size,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            sub_block=sub_block,
            epsilon=epsilon,
        )

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        no_ori = (False, True) if context.entropy_verify else (True,)
        rows = [
            dict(
                block_size=block_size,
                no_ori_target_probs=no_ori_value,
                no_draft_probs=no_draft,
                enable_reduce_sampling=context.enable_reduce_sampling,
                synthetic_mode=context.synthetic_mode,
                entropy_verify=context.entropy_verify,
                vocab_block_size=context.vocab_block_size,
                posterior_threshold=context.posterior_threshold,
                posterior_alpha=context.posterior_alpha,
                sub_block=context.sub_block,
                epsilon=context.epsilon,
            )
            for block_size in context.block_sizes
            for no_ori_value in no_ori
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, MAX_SPEC_LEN + 1)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            target_indices=(
                TritonWarmupTensor(torch.int32, shape=(1, 1)) if compile_key.enable_reduce_sampling else None
            ),
            bonus_token_ids=TritonWarmupTensor(torch.int64, shape=(1,)),
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            uniform_probs=TritonWarmupTensor(torch.float32, shape=(1,)),
            is_greedy=TritonWarmupTensor(torch.bool, shape=(1,)),
            max_spec_len=1,
            vocab_size=1,
            global_vocab_size=1,
            vec_len=1,
            ori_target_probs=(
                None if compile_key.no_ori_target_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))
            ),
            synthetic_conditional_rates=(
                TritonWarmupTensor(torch.float32, shape=(1,)) if compile_key.synthetic_mode else None
            ),
            block_size=compile_key.block_size,
            no_ori_target_probs=compile_key.no_ori_target_probs,
            no_draft_probs=compile_key.no_draft_probs,
            enable_reduce_sampling=compile_key.enable_reduce_sampling,
            synthetic_mode=compile_key.synthetic_mode,
            entropy_verify=compile_key.entropy_verify,
            vocab_block_size=compile_key.vocab_block_size,
            posterior_threshold=compile_key.posterior_threshold,
            posterior_alpha=compile_key.posterior_alpha,
            sub_block=compile_key.sub_block,
            epsilon=compile_key.epsilon,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        target_indices,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        ori_target_probs,
        synthetic_conditional_rates,
        *,
        max_spec_len,
        vocab_size,
        global_vocab_size,
        vec_len,
        block_size,
        no_ori_target_probs,
        no_draft_probs,
        enable_reduce_sampling,
        synthetic_mode,
        entropy_verify,
        vocab_block_size,
        posterior_threshold,
        posterior_alpha,
        sub_block,
        epsilon,
        grid_size,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            NO_ORI_TARGET_PROBS=no_ori_target_probs,
            NO_DRAFT_PROBS=no_draft_probs,
            ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
            SYNTHETIC_MODE=synthetic_mode,
            ENTROPY_VERIFY=entropy_verify,
            BLOCK_SIZE=block_size,
            VOCAB_BLOCK_SIZE=vocab_block_size,
            POSTERIOR_THRESHOLD=posterior_threshold,
            POSTERIOR_ALPHA=posterior_alpha,
            SUB_BLOCK=sub_block,
            EPSILON=epsilon,
        )


_REJECTION_RANDOM_SAMPLE_KERNEL = RejectionRandomSampleKernel()


@triton.jit(do_not_specialize=["replace_from", "replace_to", "vec_len"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    vec_len,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offset = req_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vec_len

    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offset == 0 must
    # be masked out at the load. other=0 keeps num_tokens deterministic (0)
    # for out-of-range lanes, which the store mask below relies on.
    start_idx = tl.load(cu_num_tokens_ptr + offset - 1, mask=len_mask & (offset > 0), other=0)
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask, other=0)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_tokens, (i,))
        start_idx1 = get_element(start_idx, (i,))
        src_val1 = get_element(src_val, (i,))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1, src_val1, mask=offset1 < num_tokens1)


class ExpandKernel(VllmTritonJitKernel["ExpandKernel.CompileKey"]):
    kernel = expand_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        max_num_tokens: int
        dtype: torch.dtype

    def dispatch(self, *, block_size: int, max_num_tokens: int, dtype: torch.dtype) -> CompileKey:
        return self.CompileKey(block_size=block_size, max_num_tokens=max_num_tokens, dtype=dtype)

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        rows = [
            dict(block_size=block_size, max_num_tokens=MAX_SPEC_LEN, dtype=dtype)
            for block_size in context.block_sizes
            for dtype in (torch.int32, torch.float32)
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            expanded_x=TritonWarmupTensor(compile_key.dtype, shape=(1,)),
            x=TritonWarmupTensor(compile_key.dtype, shape=(1,)),
            cu_num_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            replace_from=-1,
            replace_to=0,
            vec_len=1,
            max_num_tokens=compile_key.max_num_tokens,
            block_size=compile_key.block_size,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        expanded_x: torch.Tensor,
        x: torch.Tensor,
        cu_num_tokens: torch.Tensor,
        *,
        replace_from: int,
        replace_to: int,
        vec_len: int,
        max_num_tokens: int,
        block_size: int,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            output_ptr=expanded_x,
            input_ptr=x,
            MAX_NUM_TOKENS=max_num_tokens,
            BLOCK_SIZE=block_size,
        )


_EXPAND_KERNEL = ExpandKernel()


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    target_indices_ptr,
    q_ptr,
    vocab_size,
    global_vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)

    # Compute token index. Clamp the previous-request index instead of
    # relying on tl.where: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), and a 0-d masked load is
    # not reliably supported on triton-ascend. The clamped load address always
    # stays inside the buffer; tl.where merely discards the value for
    # req_idx == 0.
    prev_req_idx = tl.maximum(req_idx - 1, 0)
    start_idx = tl.where(req_idx == 0, 0, tl.load(cu_num_draft_tokens_ptr + prev_req_idx))
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if ENABLE_REDUCE_SAMPLING:
        C = vocab_size
        n_loop = tl.cdiv(C, VOCAB_BLOCK_SIZE)

        global_max_p = tl.full((), -float("inf"), tl.float32)
        global_recovered_id = tl.full((), -1, tl.int64)
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int64)

        for li in range(n_loop):
            c_start = li * VOCAB_BLOCK_SIZE
            offs = c_start + tl.arange(0, VOCAB_BLOCK_SIZE)
            mask = offs < C

            # Load target prob and global index
            tprob = tl.load(target_probs_ptr + token_idx * C + offs, mask=mask, other=0.0).to(tl.float32)

            gidx = tl.load(target_indices_ptr + token_idx * C + offs, mask=mask, other=0).to(tl.int64)

            if NO_DRAFT_PROBS:
                is_draft = (gidx == draft_token_id) & mask
                prob = tl.where(is_draft, 0.0, tprob)
            else:
                valid = (gidx >= 0) & (gidx < global_vocab_size) & mask
                dprob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + gidx, mask=valid, other=0.0).to(
                    tl.float32
                )
                prob = tl.maximum(tprob - dprob, 0.0)

            qv = tl.load(q_ptr + req_idx * C + offs, mask=mask, other=1.0).to(tl.float32)

            bad_q = (qv <= 0) | (qv != qv) | (qv == float("inf")) | (qv == -float("inf"))
            score = tl.where(bad_q, float("-inf"), prob / qv)
            score = tl.where(mask, score, float("-inf"))

            block_best_score = tl.max(score, axis=0)
            block_best_idx = tl.argmax(score, axis=0).to(tl.int64)
            block_best_global_id = tl.load(target_indices_ptr + token_idx * C + (c_start + block_best_idx)).to(tl.int64)

            better = block_best_score > global_max_p
            global_max_p = tl.where(better, block_best_score, global_max_p)
            global_recovered_id = tl.where(better, block_best_global_id, global_recovered_id)

        tl.store(output_token_ids_ptr + token_idx, global_recovered_id)
    else:
        vocab_size = global_vocab_size
        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_p = -1.0
        if NO_DRAFT_PROBS:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                prob = tl.where(vocab_offset == draft_token_id, 0.0, prob)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
                )
                new_p = prob / q
                recovered_id = tl.argmax(new_p, axis=-1)
                max_p = get_element(new_p, (recovered_id,))
                if max_p > global_max_p:
                    global_max_p = max_p
                    global_recovered_id = vocab_start + recovered_id
        else:
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                prob = tl.maximum(target_prob - draft_prob, 0)
                # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
                # `tl.argmax` will select the maximum value.

                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
                )
                new_p = prob / q
                recovered_id = tl.argmax(new_p, axis=-1)
                max_p = get_element(new_p, (recovered_id,))
                if max_p > global_max_p:
                    global_max_p = max_p
                    global_recovered_id = vocab_start + recovered_id

        tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)


class SampleRecoveredTokensKernel(VllmTritonJitKernel["SampleRecoveredTokensKernel.CompileKey"]):
    kernel = sample_recovered_tokens_kernel

    @dataclass(frozen=True)
    class CompileKey:
        no_draft_probs: bool
        enable_reduce_sampling: bool
        vocab_block_size: int
        sub_block: int

    def dispatch(
        self, *, no_draft_probs: bool, enable_reduce_sampling: bool, vocab_block_size: int, sub_block: int
    ) -> CompileKey:
        return self.CompileKey(
            no_draft_probs=no_draft_probs,
            enable_reduce_sampling=enable_reduce_sampling,
            vocab_block_size=vocab_block_size,
            sub_block=sub_block,
        )

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        rows = [
            dict(
                no_draft_probs=no_draft,
                enable_reduce_sampling=context.enable_reduce_sampling,
                vocab_block_size=context.vocab_block_size,
                sub_block=context.sub_block,
            )
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            target_indices=(
                TritonWarmupTensor(torch.int32, shape=(1, 1)) if compile_key.enable_reduce_sampling else None
            ),
            q=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            vocab_size=1,
            global_vocab_size=1,
            no_draft_probs=compile_key.no_draft_probs,
            enable_reduce_sampling=compile_key.enable_reduce_sampling,
            vocab_block_size=compile_key.vocab_block_size,
            sub_block=compile_key.sub_block,
            batch_size=1,
            max_spec_len=1,
        )

    @kernel_launcher
    def __call__(
        self,
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        target_indices,
        q,
        *,
        vocab_size,
        global_vocab_size,
        no_draft_probs,
        enable_reduce_sampling,
        vocab_block_size,
        sub_block,
        batch_size,
        max_spec_len,
    ) -> LaunchSpec:
        return (batch_size, max_spec_len), dict(
            output_token_ids_ptr=recovered_token_ids,
            NO_DRAFT_PROBS=no_draft_probs,
            ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
            VOCAB_BLOCK_SIZE=vocab_block_size,
            SUB_BLOCK=sub_block,
            multibuffer=False,
        )


_SAMPLE_RECOVERED_TOKENS_KERNEL = SampleRecoveredTokensKernel()


def rejection_greedy_sample_with_triton(
    output_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    grid,
    block_size,
    uniform_probs=None,
    synthetic_conditional_rates=None,
    synthetic_mode=False,
):
    vec_len = output_token_ids.shape[0]

    if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and is_greedy is None:
        _REJECTION_GREEDY_SPEC_LEN_1_KERNEL(
            output_token_ids=output_token_ids,
            draft_token_ids=draft_token_ids,
            target_argmax=target_argmax,
            bonus_token_ids=bonus_token_ids,
            vec_len=vec_len,
            uniform_probs=uniform_probs,
            synthetic_conditional_rates=synthetic_conditional_rates,
            block_size=block_size,
            synthetic_mode=synthetic_mode,
            grid_size=grid,
        )
    else:
        _REJECTION_GREEDY_KERNEL(
            output_token_ids=output_token_ids,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_token_ids=draft_token_ids,
            target_argmax=target_argmax,
            bonus_token_ids=bonus_token_ids,
            is_greedy=is_greedy,
            vec_len=vec_len,
            max_spec_len=max_spec_len,
            uniform_probs=uniform_probs,
            synthetic_conditional_rates=synthetic_conditional_rates,
            block_size=block_size,
            synthetic_mode=synthetic_mode,
            grid_size=grid,
        )


def expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens):
    vec_len = batch_size
    grid, block_size = cal_grid_and_block_size(batch_size)

    _EXPAND_KERNEL(
        expanded_x=expanded_x,
        x=x,
        cu_num_tokens=cu_num_tokens,
        replace_from=replace_from,
        replace_to=replace_to,
        vec_len=vec_len,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        grid_size=grid,
    )


@triton.jit(
    do_not_specialize=[
        "max_spec_len",
        "vec_len",
    ]
)
def rejection_random_sample_block_verify_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size] or [num_tokens, selected_vocab_size] if ENABLE_REDUCE_SAMPLING
    target_indices_ptr,  # [num_tokens, selected_vocab_size] global vocab indices, only used if ENABLE_REDUCE_SAMPLING
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,  # vocab_size or selected_vocab_size if ENABLE_REDUCE_SAMPLING
    global_vocab_size,  # global vocab size for draft_probs indexing (only used if ENABLE_REDUCE_SAMPLING)
    vec_len,
    ori_target_probs_ptr,  # [num_tokens, ori_vocab_size] original probs for entropy
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,  # Whether using reduce_sampling
    BLOCK_SIZE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
    POSTERIOR_THRESHOLD: tl.constexpr = 0.95,
    POSTERIOR_ALPHA: tl.constexpr = 0.4,
    SUB_BLOCK: tl.constexpr = 4096,
    EPSILON: tl.constexpr = 1e-10,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    prev_mask = not_greedy_mask & (offsets > 0)
    prev_end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets - 1, prev_mask, other=0)
    start_idxs = tl.where(offsets == 0, 0, prev_end_idxs)
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs

    if ENABLE_REDUCE_SAMPLING:
        for req_i in range(BLOCK_SIZE):
            not_greedy = get_element(not_greedy_mask, (req_i,))
            if not_greedy:
                pi = 1.0
                uniform_prob = 1.0
                last_accepted_token_pos = -1
                start_idx = get_element(start_idxs, (req_i,))
                req_idx = block_idx * BLOCK_SIZE + req_i
                num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

                for pos in range(num_draft_tokens):
                    token_idx = start_idx + pos
                    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                    if draft_token_id == -1:
                        pi = 0.0
                    else:
                        target_prob = 0.0
                        found = False

                        for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
                            if not found:
                                vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
                                vocab_mask = vocab_offsets < vocab_size

                                candidate_indices = tl.load(
                                    target_indices_ptr + token_idx * vocab_size + vocab_offsets,
                                    mask=vocab_mask,
                                    other=-1,
                                )

                                match_mask = candidate_indices == draft_token_id

                                candidate_probs = tl.load(
                                    target_probs_ptr + token_idx * vocab_size + vocab_offsets,
                                    mask=vocab_mask,
                                    other=0.0,
                                )

                                current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)

                                if current_match_prob > 0.0:
                                    target_prob = current_match_prob
                                    found = True

                        tmp_uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                        uniform_prob = uniform_prob * tmp_uniform_prob

                        if NO_DRAFT_PROBS:
                            draft_prob = 1.0
                        else:
                            draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)

                        pi = min(pi * target_prob / draft_prob, 1.0)
                        if draft_prob > 0 and pi >= uniform_prob:
                            last_accepted_token_pos = pos

                # Store accepted tokens
                if last_accepted_token_pos > -1:
                    for pos in range(last_accepted_token_pos + 1):
                        token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

                # Store recovered or bonus token
                if last_accepted_token_pos + 1 < num_draft_tokens:
                    # Rejected - store recovered token
                    recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                        recovered_token_id,
                    )
                else:
                    # All accepted - store bonus token
                    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)
    else:
        for req_i in range(BLOCK_SIZE):
            not_greedy = get_element(not_greedy_mask, (req_i,))
            if not_greedy:
                pi = 1.0
                uniform_prob = 1.0
                last_accepted_token_pos = -1
                start_idx = get_element(start_idxs, (req_i,))
                req_idx = block_idx * BLOCK_SIZE + req_i
                num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

                for pos in range(num_draft_tokens):
                    token_idx = start_idx + pos
                    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                    if draft_token_id == -1:
                        pi = 0.0
                    else:
                        target_prob = tl.load(target_probs_ptr + token_idx * vocab_size + draft_token_id)

                        tmp_uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                        uniform_prob = uniform_prob * tmp_uniform_prob

                        if NO_DRAFT_PROBS:
                            draft_prob = 1.0
                        else:
                            vocab_for_draft = global_vocab_size if ENABLE_REDUCE_SAMPLING else vocab_size
                            draft_prob = tl.load(draft_probs_ptr + token_idx * vocab_for_draft + draft_token_id)

                        if ENTROPY_VERIFY:
                            loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
                            entropy = 0.0
                            for loop_i in range(loop):
                                vocab_start = loop_i * SUB_BLOCK
                                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                                vocab_mask = vocab_offset < vocab_size
                                if NO_ORI_TARGET_PROBS:
                                    probs = tl.load(
                                        target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                        vocab_mask,
                                        other=0,
                                    )
                                else:
                                    probs = tl.load(
                                        ori_target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                        vocab_mask,
                                        other=0,
                                    )
                                log_probs = tl.log(probs + EPSILON)
                                entropy_contrib = -probs * log_probs
                                entropy += tl.sum(entropy_contrib)

                            exp_neg_entropy = tl.exp(-entropy * POSTERIOR_ALPHA)
                            threshold_by_entropy = exp_neg_entropy
                            threshold = tl.minimum(threshold_by_entropy, POSTERIOR_THRESHOLD)
                            _uniform_prob = threshold * uniform_prob
                        else:
                            _uniform_prob = uniform_prob

                        pi = min(pi * target_prob / draft_prob, 1.0)
                        if draft_prob > 0 and pi >= _uniform_prob:
                            last_accepted_token_pos = pos

                # Store accepted tokens
                if last_accepted_token_pos > -1:
                    for pos in range(last_accepted_token_pos + 1):
                        token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

                # Store recovered or bonus token
                if last_accepted_token_pos + 1 < num_draft_tokens:
                    # Rejected - store recovered token
                    recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                        recovered_token_id,
                    )
                else:
                    # All accepted - store bonus token
                    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)


class RejectionRandomSampleBlockVerifyKernel(VllmTritonJitKernel["RejectionRandomSampleBlockVerifyKernel.CompileKey"]):
    kernel = rejection_random_sample_block_verify_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        no_ori_target_probs: bool
        no_draft_probs: bool
        enable_reduce_sampling: bool
        entropy_verify: bool
        vocab_block_size: int
        posterior_threshold: float
        posterior_alpha: float
        sub_block: int
        epsilon: float

    def dispatch(
        self,
        *,
        block_size: int,
        no_ori_target_probs: bool,
        no_draft_probs: bool,
        enable_reduce_sampling: bool,
        entropy_verify: bool,
        vocab_block_size: int,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
    ) -> CompileKey:
        return self.CompileKey(
            block_size=block_size,
            no_ori_target_probs=no_ori_target_probs,
            no_draft_probs=no_draft_probs,
            enable_reduce_sampling=enable_reduce_sampling,
            entropy_verify=entropy_verify,
            vocab_block_size=vocab_block_size,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            sub_block=sub_block,
            epsilon=epsilon,
        )

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        no_ori = (False, True) if context.entropy_verify else (True,)
        rows = [
            dict(
                block_size=block_size,
                no_ori_target_probs=no_ori_value,
                no_draft_probs=no_draft,
                enable_reduce_sampling=context.enable_reduce_sampling,
                entropy_verify=context.entropy_verify,
                vocab_block_size=context.vocab_block_size,
                posterior_threshold=context.posterior_threshold,
                posterior_alpha=context.posterior_alpha,
                sub_block=context.sub_block,
                epsilon=context.epsilon,
            )
            for block_size in context.block_sizes
            for no_ori_value in no_ori
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, MAX_SPEC_LEN + 1)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            target_indices=(
                TritonWarmupTensor(torch.int32, shape=(1, 1)) if compile_key.enable_reduce_sampling else None
            ),
            bonus_token_ids=TritonWarmupTensor(torch.int64, shape=(1,)),
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            uniform_probs=TritonWarmupTensor(torch.float32, shape=(1,)),
            is_greedy=TritonWarmupTensor(torch.bool, shape=(1,)),
            max_spec_len=1,
            vocab_size=1,
            global_vocab_size=1,
            vec_len=1,
            ori_target_probs=(
                None if compile_key.no_ori_target_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))
            ),
            block_size=compile_key.block_size,
            no_ori_target_probs=compile_key.no_ori_target_probs,
            no_draft_probs=compile_key.no_draft_probs,
            enable_reduce_sampling=compile_key.enable_reduce_sampling,
            entropy_verify=compile_key.entropy_verify,
            vocab_block_size=compile_key.vocab_block_size,
            posterior_threshold=compile_key.posterior_threshold,
            posterior_alpha=compile_key.posterior_alpha,
            sub_block=compile_key.sub_block,
            epsilon=compile_key.epsilon,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        target_indices,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        ori_target_probs,
        *,
        max_spec_len,
        vocab_size,
        global_vocab_size,
        vec_len,
        block_size,
        no_ori_target_probs,
        no_draft_probs,
        enable_reduce_sampling,
        entropy_verify,
        vocab_block_size,
        posterior_threshold,
        posterior_alpha,
        sub_block,
        epsilon,
        grid_size,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            NO_ORI_TARGET_PROBS=no_ori_target_probs,
            NO_DRAFT_PROBS=no_draft_probs,
            ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
            ENTROPY_VERIFY=entropy_verify,
            BLOCK_SIZE=block_size,
            VOCAB_BLOCK_SIZE=vocab_block_size,
            POSTERIOR_THRESHOLD=posterior_threshold,
            POSTERIOR_ALPHA=posterior_alpha,
            SUB_BLOCK=sub_block,
            EPSILON=epsilon,
        )


_REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL = RejectionRandomSampleBlockVerifyKernel()
