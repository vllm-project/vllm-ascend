# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# Adapted from vLLM's spec_decode/dflash2/speculator.py.

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _cache_candidates_kernel(
    logits_ptr,
    cached_ids_ptr,
    candidates_ptr,
    realized_ptr,
    mapping_ptr,
    stride_request,
    stride_step,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CLEAR_ONLY: tl.constexpr,
):
    flat = tl.program_id(0)
    state = tl.load(mapping_ptr + flat).to(tl.int64)
    step = flat % NUM_STEPS
    offsets = tl.arange(0, BLOCK_K)
    valid = (state >= 0) & (offsets < TOP_K)
    cache_base = (state * NUM_STEPS + step) * TOP_K
    logits_base = logits_ptr + state * stride_request + step * stride_step
    if CLEAR_ONLY:
        old_ids = tl.load(cached_ids_ptr + cache_base + offsets, mask=valid, other=0)
        tl.store(logits_base + old_ids, -float("inf"), mask=valid)
    else:
        ids = tl.load(candidates_ptr + flat * TOP_K + offsets, mask=valid, other=0)
        scores = tl.load(realized_ptr + flat * TOP_K + offsets, mask=valid, other=0)
        tl.store(logits_base + ids, scores, mask=valid)
        tl.store(cached_ids_ptr + cache_base + offsets, ids, mask=valid)


def cache_draft_logits(
    candidate_ids: torch.Tensor,
    num_sample: int,
    idx_mapping: torch.Tensor,
    realized_scores: torch.Tensor,
    cached_candidate_ids: torch.Tensor,
    draft_logits: torch.Tensor,
) -> None:
    num_steps, top_k = candidate_ids.shape[1:]
    block_k = triton.next_power_of_2(top_k)
    # Separate launches order the scatters even when old/new IDs overlap.
    for clear_only in (True, False):
        _cache_candidates_kernel[(num_sample,)](
            draft_logits,
            cached_candidate_ids,
            candidate_ids,
            realized_scores,
            idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            NUM_STEPS=num_steps,
            TOP_K=top_k,
            BLOCK_K=block_k,
            CLEAR_ONLY=clear_only,
            num_warps=1,
        )
