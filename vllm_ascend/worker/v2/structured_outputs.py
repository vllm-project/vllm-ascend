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
#
# NPU-compatible structured output bitmask kernel.
#
# Upstream cannot be used directly on Ascend NPU: `BLOCK_SIZE=8192` overflows
# UB, while a smaller `BLOCK_SIZE` makes the grid unstable. We therefore keep
# `BLOCK_SIZE=8192` and split each block with `BLOCK_SIZE_SUB=1024`.
#
#

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

# Packed bitmask uses one int32 word per 32 vocab tokens.
BITMASK_BITS_PER_WORD = 32
# Number of vocab tokens processed per kernel grid step along the vocab axis.
# Fixed at 8192 for the Ascend NPU: smaller blocks make the grid unstable,
# while the kernel internally tiles each block by BLOCK_SIZE_SUB to fit UB.
GRAMMAR_BITMASK_BLOCK_SIZE = 8192


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
# Ascend NPU bitmask kernel (BLOCK_SIZE_SUB tiling)
# TODO: Optimize the kernel performance with NPU profiling data.
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_SIZE_SUB: tl.constexpr = 1024
    bitmask_idx = tl.program_id(0)
    block_id = tl.program_id(1)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Sub-block tiling loop: process BLOCK_SIZE_SUB tokens per iteration
    for sub_offset in tl.range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        global_token_offset = block_id * BLOCK_SIZE + sub_offset
        bitmask_word_start = global_token_offset // 32
        bitmask_offset = bitmask_word_start + tl.arange(0, BLOCK_SIZE_SUB // 32)
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=bitmask_offset < bitmask_stride,
            other=0,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE_SUB)

        # Apply: set blocked positions to -inf
        block_offset = global_token_offset + tl.arange(0, BLOCK_SIZE_SUB)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=bitmask & (block_offset < vocab_size),
        )


def _launch_grammar_bitmask_kernel(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    logits_indices: torch.Tensor,
) -> None:
    """Launch the NPU grammar-bitmask kernel in-place on ``logits``.

    Args:
        logits: ``[num_logits, vocab_size]`` row-major logits, modified
            in-place. Kept in its native dtype (no float upcast needed).
        bitmask: packed ``int32`` mask of shape
            ``[num_masks, cdiv(vocab_size, 32)]``. Row ``i`` is the mask for
            the logits row ``logits_indices[i]``; a ``0`` bit blocks a token.
        logits_indices: ``int32`` tensor of shape ``[num_masks]`` mapping each
            bitmask row to its target row in ``logits``.
    """
    vocab_size = logits.shape[-1]
    num_masks = logits_indices.shape[0]
    grid = (num_masks, triton.cdiv(vocab_size, GRAMMAR_BITMASK_BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=GRAMMAR_BITMASK_BLOCK_SIZE,
    )


def build_grammar_bitmask_indices(
    req_ids: list[str],
    structured_output_request_ids: list[str],
    scheduled_spec_decode_tokens: dict,
) -> tuple[list[int], list[int]]:
    """Compute how compacted bitmask rows map onto batch-aligned logits rows.

    Structured-output requests are a subset of the batch in a different order,
    and each occupies ``1 + num_spec_tokens`` consecutive logits rows. The
    packed bitmask produced by xgrammar is compacted (one row per masked logits
    position, in ``structured_output_request_ids`` order with speculative rows
    expanded).

    Returns ``(bitmask_rows, out_indices)`` of equal length, where
    ``bitmask_rows[i]`` selects a row from the source bitmask and
    ``out_indices[i]`` is the logits row that row targets. Rows for
    structured-output requests not present in the current batch are skipped from
    both lists, so they stay aligned (bitmask row i <-> out_indices[i]).
    """
    # Map each structured-output request to its first logits row, accounting for
    # speculative-decode rows that shift later requests' offsets.
    struct_out_req_ids = set(structured_output_request_ids)
    struct_out_req_first_logit: dict[str, int] = {}
    cumulative_offset = 0
    for batch_index, req_id in enumerate(req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(scheduled_spec_decode_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_first_logit[req_id] = logit_index

    # Walk the compacted bitmask rows in their native order. ``cumulative_index``
    # tracks the position in the source bitmask.
    bitmask_rows: list[int] = []
    out_indices: list[int] = []
    cumulative_index = 0
    for req_id in structured_output_request_ids:
        num_logit_rows = 1 + len(scheduled_spec_decode_tokens.get(req_id, ()))
        logit_idx = struct_out_req_first_logit.get(req_id)
        if logit_idx is not None:
            for i in range(num_logit_rows):
                bitmask_rows.append(cumulative_index + i)
                out_indices.append(logit_idx + i)
        cumulative_index += num_logit_rows

    return bitmask_rows, out_indices


def apply_grammar_bitmask(
    logits: torch.Tensor,
    scheduler_output: "SchedulerOutput",
    grammar_output: "GrammarOutput",
    input_batch: "InputBatch",
) -> None:
    """Apply structured-output bitmasks to ``logits`` in-place on the device.

    This mirrors the device branch of
    ``vllm.v1.structured_output.utils.apply_grammar_bitmask`` but launches the
    Ascend NPU Triton kernel directly, avoiding the device->CPU->device
    round-trip used by the historical v1 path. ``logits`` are masked in their
    native dtype (no float upcast).
    """
    bitmask_rows, out_indices = build_grammar_bitmask_indices(
        input_batch.req_ids,
        grammar_output.structured_output_request_ids,
        scheduler_output.scheduled_spec_decode_tokens,
    )
    if not out_indices:
        return

    # Gather only the applied rows so bitmask row i aligns with out_indices[i].
    # The common case (all structured-output requests present and contiguous)
    # gathers every row, which numpy returns as a cheap copy.
    selected_bitmask = grammar_output.grammar_bitmask[bitmask_rows]
    bitmask = torch.as_tensor(selected_bitmask).to(logits.device, non_blocking=True)
    logits_indices = torch.tensor(out_indices, dtype=torch.int32, device=logits.device)
    _launch_grammar_bitmask_kernel(logits, bitmask, logits_indices)


def warmup_grammar_bitmask_kernel(device: torch.device, vocab_size: int, logits_dtype: torch.dtype) -> None:
    """Trigger the one-time Triton JIT compile of the grammar-bitmask kernel.

    The first launch of the kernel pays a ~5s JIT cost. Calling this once at
    startup (e.g. during graph capture) keeps that cost off the first guided
    decode step. The dummy launch masks no tokens (all-ones bitmask), so it is
    a no-op on the throwaway logits buffer.

    ``logits_dtype`` MUST match the dtype of the real sampler logits (e.g. bf16).
    Triton specializes/caches the compiled kernel on the logits pointer's
    element dtype, so warming a different dtype (e.g. float32) leaves the real
    dtype's cache entry cold and the first guided step pays the JIT a second
    time.
    """
    logits = torch.zeros((1, vocab_size), dtype=logits_dtype, device=device)
    num_words = (vocab_size + BITMASK_BITS_PER_WORD - 1) // BITMASK_BITS_PER_WORD
    # -1 (all bits set) => every token allowed => no stores happen.
    bitmask = torch.full((1, num_words), -1, dtype=torch.int32, device=device)
    logits_indices = torch.zeros(1, dtype=torch.int32, device=device)
    _launch_grammar_bitmask_kernel(logits, bitmask, logits_indices)
