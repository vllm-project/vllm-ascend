# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
"""Triton kernels for the Model Runner V2 thinking EOS mask.

Device-side implementation of the mask carried by PR #20811 (see
``vllm_ascend/patch/worker/patch_thinking_eos_mask.py`` and upstream
vllm#55420 for the rationale): while a request is inside its think block,
the configured EOS token ids are masked to ``-inf`` on every logits row of
the request. The kernelized design mirrors the upstream thinking-budget
path (``vllm.v1.worker.gpu.sample.thinking_budget``): a per-request
marker-cache kernel scans the committed token history incrementally, and a
per-row kernel applies the mask from the cached marker positions, so the
Python-side per-request D2H scan is gone.

Semantics match the V1 holder (``vllm_ascend/sample/thinking_eos_mask.py``):
a request is in think iff the last reasoning-start marker occurs after every
natural reasoning-end *or* implicit-end (e.g. tool-call start) marker in the
committed history. In-flight draft tokens are deliberately ignored: masking
every row of an in-think request also covers the spec-decode verify and
bonus rows, so a mid-think EOS can neither be sampled nor drafted-and-
accepted.
"""

import torch
from vllm.triton_utils import tl, triton

_COLD_SCAN_BLOCK = 1024


@triton.jit
def _update_thinking_mask_cache_kernel(
    req_ids_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    implicit_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    IMPLICIT_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if (scan_pos == 0 and last_start < 0) and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the
        # first block with a marker; only the relative order of the two
        # positions found matters below.
        block_hi = total_len
        while (block_hi > 0 and last_start < 0) and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            # A position counts as an end when it starts the natural end
            # sequence or matches any single-token implicit end marker.
            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)
            for k in tl.static_range(0, IMPLICIT_END_LEN):
                expected = tl.load(implicit_end_token_ids_ptr + k)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs,
                    mask=offs < block_hi,
                    other=-1,
                )
                end_match = end_match | (actual == expected)

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            block_hi = block_lo
    else:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j)
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            end_match = False
            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j)
                    end_match = end_match & (actual == expected)
            for k in tl.static_range(0, IMPLICIT_END_LEN):
                expected = tl.load(implicit_end_token_ids_ptr + k)
                actual = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + i)
                end_match = end_match | (actual == expected)
            if end_match:
                last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _thinking_eos_mask_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    eos_mask_token_ids_ptr,
    EOS_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    if last_start < 0:
        return
    last_end = tl.load(cached_last_end_ptr + req_state_idx)
    if last_start <= last_end:
        return

    for j in tl.static_range(0, EOS_LEN):
        eos_id = tl.load(eos_mask_token_ids_ptr + j)
        tl.store(logits_ptr + token_idx * logits_stride + eos_id, float("-inf"))


def apply_thinking_eos_mask(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    implicit_end_token_ids: torch.Tensor,
    eos_mask_token_ids: torch.Tensor,
) -> None:
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    implicit_end_len = implicit_end_token_ids.shape[0]
    eos_len = eos_mask_token_ids.shape[0]

    _update_thinking_mask_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        implicit_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        IMPLICIT_END_LEN=implicit_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _thinking_eos_mask_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        cached_last_start,
        cached_last_end,
        eos_mask_token_ids,
        EOS_LEN=eos_len,
    )
