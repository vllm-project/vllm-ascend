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
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
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
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    global_vocab_size,
    vec_len,
    ori_target_probs_ptr,  # [num_tokens, ori_vocab_size] original probs for entropy
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
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


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    q_ptr,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
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


class RejectionGreedySpecLen1Kernel(AscendTritonJitKernel["RejectionGreedySpecLen1Kernel.CompileKey"]):
    kernel = rejection_greedy_sample_spec_len_1_triton

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        synthetic_mode: bool

    def dispatch(self, *, block_size: int, synthetic_mode: bool) -> "RejectionGreedySpecLen1Kernel.CompileKey":
        return self.CompileKey(block_size=block_size, synthetic_mode=synthetic_mode)

    def get_warmup_keys(self, context: Any) -> list["RejectionGreedySpecLen1Kernel.CompileKey"]:
        # The spec-len-1 kernel only runs when every request drafts exactly
        # one token and the batch is all-greedy (is_greedy=None); for larger
        # max_spec_len the general greedy kernel handles both paths.
        if context.max_spec_len != 1:
            return []
        rows = [
            dict(block_size=block_size, synthetic_mode=context.synthetic_mode) for block_size in context.block_sizes
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "RejectionGreedySpecLen1Kernel.CompileKey") -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, 2)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            target_argmax=TritonWarmupTensor(torch.int64, shape=(1,)),
            bonus_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
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


class RejectionGreedyKernel(AscendTritonJitKernel["RejectionGreedyKernel.CompileKey"]):
    kernel = rejection_greedy_sample_triton

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        synthetic_mode: bool
        has_is_greedy: bool

    def dispatch(
        self,
        *,
        block_size: int,
        synthetic_mode: bool,
        has_is_greedy: bool,
    ) -> "RejectionGreedyKernel.CompileKey":
        return self.CompileKey(
            block_size=block_size,
            synthetic_mode=synthetic_mode,
            has_is_greedy=has_is_greedy,
        )

    def get_warmup_keys(self, context: Any) -> list["RejectionGreedyKernel.CompileKey"]:
        # With max_spec_len == 1 the is_greedy=None case is routed to the
        # spec-len-1 kernel, so only the explicit tensor path reaches here;
        # for larger spec lengths both paths use this kernel.
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

    def warmup_inputs(self, compile_key: "RejectionGreedyKernel.CompileKey") -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, 2)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            target_argmax=TritonWarmupTensor(torch.int64, shape=(1,)),
            bonus_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
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
            uniform_probs=uniform_probs,
            synthetic_conditional_rates=synthetic_conditional_rates,
            vec_len=vec_len,
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
            uniform_probs=uniform_probs,
            synthetic_conditional_rates=synthetic_conditional_rates,
            vec_len=vec_len,
            max_spec_len=max_spec_len,
            block_size=block_size,
            synthetic_mode=synthetic_mode,
            grid_size=grid,
        )


class ExpandKernel(AscendTritonJitKernel["ExpandKernel.CompileKey"]):
    kernel = expand_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        max_num_tokens: int
        dtype: torch.dtype

    def dispatch(
        self,
        *,
        block_size: int,
        max_num_tokens: int,
        dtype: torch.dtype,
    ) -> "ExpandKernel.CompileKey":
        return self.CompileKey(block_size=block_size, max_num_tokens=max_num_tokens, dtype=dtype)

    def get_warmup_keys(self, context: Any) -> list["ExpandKernel.CompileKey"]:
        # Runtime expand dtypes: temperature/top_p float32, top_k int32,
        # repeat/expansion indices int64. MAX_NUM_TOKENS is pinned to the
        # upstream MAX_SPEC_LEN constant by every caller.
        from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN

        rows = [
            dict(block_size=block_size, max_num_tokens=MAX_SPEC_LEN, dtype=dtype)
            for block_size in context.block_sizes
            for dtype in (torch.int32, torch.int64, torch.float32)
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "ExpandKernel.CompileKey") -> dict[str, Any]:
        # Pointer keys follow the kernel parameters (output_ptr/input_ptr);
        # the runtime launcher keeps its expanded_x/x argument names.
        return dict(
            output=TritonWarmupTensor(compile_key.dtype, shape=(1,)),
            input=TritonWarmupTensor(compile_key.dtype, shape=(1,)),
            cu_num_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            replace_from=0,
            replace_to=0,
            vec_len=1,
            max_num_tokens=compile_key.max_num_tokens,
            block_size=compile_key.block_size,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
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
            MAX_NUM_TOKENS=max_num_tokens,  # To avoid recompilation.
            BLOCK_SIZE=block_size,
        )


_EXPAND_KERNEL = ExpandKernel()


def expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens):
    vec_len = batch_size
    grid, block_size = cal_grid_and_block_size(batch_size)

    _EXPAND_KERNEL(
        output=expanded_x,
        input=x,
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
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    global_vocab_size,
    vec_len,
    ori_target_probs_ptr,  # [num_tokens, ori_vocab_size] original probs for entropy
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
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
                        vocab_for_draft = vocab_size
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


class SampleRecoveredTokensKernel(AscendTritonJitKernel["SampleRecoveredTokensKernel.CompileKey"]):
    kernel = sample_recovered_tokens_kernel

    @dataclass(frozen=True)
    class CompileKey:
        no_draft_probs: bool
        sub_block: int
        # Representative of vocab_size's Triton specialization class.
        vocab_size: int

    def dispatch(
        self,
        *,
        no_draft_probs: bool,
        sub_block: int,
        vocab_size: int,
    ) -> "SampleRecoveredTokensKernel.CompileKey":
        return self.CompileKey(
            no_draft_probs=no_draft_probs,
            sub_block=sub_block,
            vocab_size=vocab_size,
        )

    def get_warmup_keys(self, context: Any) -> list["SampleRecoveredTokensKernel.CompileKey"]:
        # The 2D grid is (batch_size, max_spec_len) with no BLOCK_SIZE
        # constexpr, so one key per NO_DRAFT_PROBS value covers every batch.
        rows = [
            dict(no_draft_probs=no_draft, sub_block=context.sub_block, vocab_size=context.vocab_size_rep)
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "SampleRecoveredTokensKernel.CompileKey") -> dict[str, Any]:
        return dict(
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            q=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            vocab_size=compile_key.vocab_size,
            no_draft_probs=compile_key.no_draft_probs,
            sub_block=compile_key.sub_block,
            # Any positive value compiles the same binary; the grid only
            # affects launch. Keep the cheapest (1, 1) warmup launch.
            max_spec_len=1,
        )

    @kernel_launcher
    def __call__(
        self,
        recovered_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        q: torch.Tensor,
        *,
        vocab_size: int,
        no_draft_probs: bool,
        sub_block: int,
        max_spec_len: int,
    ) -> LaunchSpec:
        # 2D grid (batch_size, max_spec_len): axis 1 must cover every draft
        # position. q holds one row per request, so deriving axis 1 from
        # q.shape[0] under-covers whenever batch_size < max_spec_len and
        # leaves recovered slots unwritten (garbage token ids then feed
        # out-of-bounds gathers downstream). A (1, 1) grid still compiles
        # the kernel; the real grid only affects launch, not the JIT key.
        return (cu_num_draft_tokens.shape[0], max(1, max_spec_len)), dict(
            output_token_ids_ptr=recovered_token_ids,
            NO_DRAFT_PROBS=no_draft_probs,
            SUB_BLOCK=sub_block,
            multibuffer=False,
        )


_SAMPLE_RECOVERED_TOKENS_KERNEL = SampleRecoveredTokensKernel()


class RejectionRandomSampleKernel(AscendTritonJitKernel["RejectionRandomSampleKernel.CompileKey"]):
    kernel = rejection_random_sample_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        no_ori_target_probs: bool
        no_draft_probs: bool
        synthetic_mode: bool
        entropy_verify: bool
        # Representatives of the vocab scalars' Triton specialization classes.
        vocab_size: int
        global_vocab_size: int
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
        synthetic_mode: bool,
        entropy_verify: bool,
        vocab_size: int,
        global_vocab_size: int,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
    ) -> "RejectionRandomSampleKernel.CompileKey":
        return self.CompileKey(
            block_size=block_size,
            no_ori_target_probs=no_ori_target_probs,
            no_draft_probs=no_draft_probs,
            synthetic_mode=synthetic_mode,
            entropy_verify=entropy_verify,
            vocab_size=vocab_size,
            global_vocab_size=global_vocab_size,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            sub_block=sub_block,
            epsilon=epsilon,
        )

    def get_warmup_keys(self, context: Any) -> list["RejectionRandomSampleKernel.CompileKey"]:
        # ``ori_target_probs`` is materialized exactly when entropy verify is
        # on, so NO_ORI_TARGET_PROBS is not an independent axis.
        no_ori_values = (not context.entropy_verify,)
        rows = [
            dict(
                block_size=block_size,
                no_ori_target_probs=no_ori,
                no_draft_probs=no_draft,
                synthetic_mode=context.synthetic_mode,
                entropy_verify=context.entropy_verify,
                vocab_size=context.vocab_size_rep,
                global_vocab_size=(context.vocab_size_rep if no_draft else context.global_vocab_size_rep),
                posterior_threshold=context.posterior_threshold,
                posterior_alpha=context.posterior_alpha,
                sub_block=context.sub_block,
                epsilon=context.epsilon,
            )
            for block_size in context.block_sizes
            for no_ori in no_ori_values
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "RejectionRandomSampleKernel.CompileKey") -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, 2)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            bonus_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            uniform_probs=TritonWarmupTensor(torch.float32, shape=(1,)),
            is_greedy=TritonWarmupTensor(torch.bool, shape=(1,)),
            max_spec_len=1,
            vocab_size=compile_key.vocab_size,
            global_vocab_size=compile_key.global_vocab_size,
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
            synthetic_mode=compile_key.synthetic_mode,
            entropy_verify=compile_key.entropy_verify,
            posterior_threshold=compile_key.posterior_threshold,
            posterior_alpha=compile_key.posterior_alpha,
            sub_block=compile_key.sub_block,
            epsilon=compile_key.epsilon,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        recovered_token_ids: torch.Tensor,
        uniform_probs: torch.Tensor,
        is_greedy: torch.Tensor,
        ori_target_probs: torch.Tensor | None,
        synthetic_conditional_rates: torch.Tensor | None,
        *,
        max_spec_len: int,
        vocab_size: int,
        global_vocab_size: int,
        vec_len: int,
        block_size: int,
        no_ori_target_probs: bool,
        no_draft_probs: bool,
        synthetic_mode: bool,
        entropy_verify: bool,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            NO_ORI_TARGET_PROBS=no_ori_target_probs,
            NO_DRAFT_PROBS=no_draft_probs,
            SYNTHETIC_MODE=synthetic_mode,
            ENTROPY_VERIFY=entropy_verify,
            BLOCK_SIZE=block_size,
            POSTERIOR_THRESHOLD=posterior_threshold,
            POSTERIOR_ALPHA=posterior_alpha,
            SUB_BLOCK=sub_block,
            EPSILON=epsilon,
        )


_REJECTION_RANDOM_SAMPLE_KERNEL = RejectionRandomSampleKernel()


class RejectionRandomSampleBlockVerifyKernel(
    AscendTritonJitKernel["RejectionRandomSampleBlockVerifyKernel.CompileKey"]
):
    kernel = rejection_random_sample_block_verify_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        no_ori_target_probs: bool
        no_draft_probs: bool
        entropy_verify: bool
        # Representatives of the vocab scalars' Triton specialization classes.
        vocab_size: int
        global_vocab_size: int
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
        entropy_verify: bool,
        vocab_size: int,
        global_vocab_size: int,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
    ) -> "RejectionRandomSampleBlockVerifyKernel.CompileKey":
        return self.CompileKey(
            block_size=block_size,
            no_ori_target_probs=no_ori_target_probs,
            no_draft_probs=no_draft_probs,
            entropy_verify=entropy_verify,
            vocab_size=vocab_size,
            global_vocab_size=global_vocab_size,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            sub_block=sub_block,
            epsilon=epsilon,
        )

    def get_warmup_keys(self, context: Any) -> list["RejectionRandomSampleBlockVerifyKernel.CompileKey"]:
        # Block verify needs the config flag and max_spec_len >= 3 and is
        # incompatible with synthetic mode (rejection_sample raises).
        if not context.block_verify:
            return []
        no_ori_values = (not context.entropy_verify,)
        rows = [
            dict(
                block_size=block_size,
                no_ori_target_probs=no_ori,
                no_draft_probs=no_draft,
                entropy_verify=context.entropy_verify,
                vocab_size=context.vocab_size_rep,
                global_vocab_size=(context.vocab_size_rep if no_draft else context.global_vocab_size_rep),
                posterior_threshold=context.posterior_threshold,
                posterior_alpha=context.posterior_alpha,
                sub_block=context.sub_block,
                epsilon=context.epsilon,
            )
            for block_size in context.block_sizes
            for no_ori in no_ori_values
            for no_draft in context.no_draft_probs_values
        ]
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "RejectionRandomSampleBlockVerifyKernel.CompileKey") -> dict[str, Any]:
        return dict(
            output_token_ids=TritonWarmupTensor(torch.int32, shape=(1, 2)),
            cu_num_draft_tokens=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            draft_probs=(None if compile_key.no_draft_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))),
            target_probs=TritonWarmupTensor(torch.float32, shape=(1, 1)),
            bonus_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            recovered_token_ids=TritonWarmupTensor(torch.int32, shape=(1,)),
            uniform_probs=TritonWarmupTensor(torch.float32, shape=(1,)),
            is_greedy=TritonWarmupTensor(torch.bool, shape=(1,)),
            max_spec_len=1,
            vocab_size=compile_key.vocab_size,
            global_vocab_size=compile_key.global_vocab_size,
            vec_len=1,
            ori_target_probs=(
                None if compile_key.no_ori_target_probs else TritonWarmupTensor(torch.float32, shape=(1, 1))
            ),
            block_size=compile_key.block_size,
            no_ori_target_probs=compile_key.no_ori_target_probs,
            no_draft_probs=compile_key.no_draft_probs,
            entropy_verify=compile_key.entropy_verify,
            posterior_threshold=compile_key.posterior_threshold,
            posterior_alpha=compile_key.posterior_alpha,
            sub_block=compile_key.sub_block,
            epsilon=compile_key.epsilon,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        output_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        recovered_token_ids: torch.Tensor,
        uniform_probs: torch.Tensor,
        is_greedy: torch.Tensor,
        ori_target_probs: torch.Tensor | None,
        *,
        max_spec_len: int,
        vocab_size: int,
        global_vocab_size: int,
        vec_len: int,
        block_size: int,
        no_ori_target_probs: bool,
        no_draft_probs: bool,
        entropy_verify: bool,
        posterior_threshold: float,
        posterior_alpha: float,
        sub_block: int,
        epsilon: float,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            NO_ORI_TARGET_PROBS=no_ori_target_probs,
            NO_DRAFT_PROBS=no_draft_probs,
            ENTROPY_VERIFY=entropy_verify,
            BLOCK_SIZE=block_size,
            POSTERIOR_THRESHOLD=posterior_threshold,
            POSTERIOR_ALPHA=posterior_alpha,
            SUB_BLOCK=sub_block,
            EPSILON=epsilon,
        )


_REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL = RejectionRandomSampleBlockVerifyKernel()


@dataclass(frozen=True)
class RejectionWarmupContext:
    """Everything the rejection-sampler warmup key spaces derive from."""

    max_spec_len: int
    # One representative batch size per distinct BLOCK_SIZE from
    # ``cal_grid_and_block_size`` over the runtime batch range.
    block_sizes: tuple[int, ...]
    # ``NO_DRAFT_PROBS`` values reachable at runtime (see
    # ``_collect_no_draft_probs_values`` in the former hand-written warmup):
    # ngram never provides draft probs; model-based drafts only when PP > 1.
    no_draft_probs_values: tuple[bool, ...]
    synthetic_mode: bool
    entropy_verify: bool
    block_verify: bool
    posterior_threshold: float
    posterior_alpha: float
    sub_block: int
    epsilon: float
    # Triton specialization representatives of the runtime vocab scalars.
    vocab_size_rep: int
    global_vocab_size_rep: int


def collect_rejection_warmup_block_sizes(max_num_reqs: int) -> list[int]:
    """Return one representative batch_size per distinct ``BLOCK_SIZE``.

    Rejection / expand / greedy kernels specialize on ``BLOCK_SIZE`` from
    ``cal_grid_and_block_size``, not on the raw batch size.
    """
    if max_num_reqs <= 0:
        return []

    block_size_to_batch: dict[int, int] = {}
    for batch_size in range(1, max_num_reqs + 1):
        _, block_size = cal_grid_and_block_size(batch_size)
        if block_size not in block_size_to_batch:
            block_size_to_batch[block_size] = batch_size

    # Ensure the largest batch is included for the top BLOCK_SIZE bucket.
    _, max_block_size = cal_grid_and_block_size(max_num_reqs)
    block_size_to_batch[max_block_size] = max_num_reqs
    return sorted(block_size_to_batch.values())


def _is_ngram_spec_method(spec_config: Any) -> bool:
    method = getattr(spec_config, "method", None)
    if method in ("ngram", "ngram_gpu"):
        return True
    use_ngram_gpu = getattr(spec_config, "use_ngram_gpu", None)
    return bool(callable(use_ngram_gpu) and use_ngram_gpu())


def build_rejection_warmup_context(vllm_config: Any) -> RejectionWarmupContext | None:
    """Derive the rejection-sampler warmup key space from one vllm_config."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_ascend.ascend_config import get_ascend_config

    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return None
    max_spec_len = spec_config.num_speculative_tokens
    if max_spec_len <= 0:
        return None

    rejection_config = get_ascend_config().rejection_sampler_config
    if _is_ngram_spec_method(spec_config) or vllm_config.parallel_config.pipeline_parallel_size <= 1:
        no_draft_probs_values: tuple[bool, ...] = (True,)
    else:
        no_draft_probs_values = (False, True)

    vocab_size = max(1, vllm_config.model_config.get_vocab_size() // get_tensor_model_parallel_world_size())
    return RejectionWarmupContext(
        max_spec_len=max_spec_len,
        block_sizes=tuple(collect_rejection_warmup_block_sizes(vllm_config.scheduler_config.max_num_seqs)),
        no_draft_probs_values=no_draft_probs_values,
        synthetic_mode=(spec_config.rejection_sample_method == "synthetic"),
        entropy_verify=bool(rejection_config.enable_entropy_verify),
        # Match rejection_sample: block verify needs the config flag and
        # max_spec_len >= 3.
        block_verify=(max_spec_len >= 3 and bool(rejection_config.enable_block_verify)),
        posterior_threshold=float(rejection_config.posterior_threshold),
        posterior_alpha=float(rejection_config.posterior_alpha),
        sub_block=4 * 1024,
        epsilon=1e-10,
        vocab_size_rep=triton_scalar_specialization_rep(vocab_size),
        global_vocab_size_rep=triton_scalar_specialization_rep(vllm_config.model_config.get_vocab_size()),
    )


def register_rejection_warmup(vllm_config: Any) -> bool:
    """Register every v1 rejection-sampler Triton kernel with the active
    registry. No-op outside a ``JitWarmupRegistry.activate()`` window."""
    context = build_rejection_warmup_context(vllm_config)
    if context is None:
        return False
    _REJECTION_GREEDY_SPEC_LEN_1_KERNEL.register_warmup(context)
    _REJECTION_GREEDY_KERNEL.register_warmup(context)
    _EXPAND_KERNEL.register_warmup(context)
    _SAMPLE_RECOVERED_TOKENS_KERNEL.register_warmup(context)
    _REJECTION_RANDOM_SAMPLE_KERNEL.register_warmup(context)
    _REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL.register_warmup(context)
    return True
