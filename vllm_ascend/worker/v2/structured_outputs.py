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
# NPU-compatible structured output bitmask kernel and apply method.
#
# Upstream cannot be used directly on Ascend NPU: `BLOCK_SIZE=8192` overflows
# UB, while a smaller `BLOCK_SIZE` makes the grid unstable. We therefore keep
# `BLOCK_SIZE=8192` and split each block with `BLOCK_SIZE_SUB=1024`.
#
#


import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

_NPU_BLOCK_SIZE = 8192
_NPU_BLOCK_SIZE_SUB = 1024


def apply_grammar_bitmask(self, logits, input_batch, grammar_req_ids, grammar_bitmask):
    """NPU-compatible apply_grammar_bitmask for StructuredOutputsWorker.

    Differences from the upstream implementation:
      - Uses BLOCK_SIZE_SUB tiling (gather_kernel pattern) to avoid UB
        overflow while keeping BLOCK_SIZE=8192 for a small grid.

    """
    if not grammar_req_ids:
        return

    # Asynchronously copy the bitmask to NPU.
    with torch.npu.stream(self.copy_stream):
        bitmask = async_copy_to_gpu(
            grammar_bitmask,
            out=self.grammar_bitmask[: grammar_bitmask.shape[0]],
        )

    # Construct bitmask -> logits mapping
    mapping: list[int] = []
    req_ids = input_batch.req_ids
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        logits_start_idx = cu_num_logits[req_idx]
        logits_end_idx = cu_num_logits[req_idx + 1]
        mapping.extend(range(logits_start_idx, logits_end_idx))

    num_masks = bitmask.shape[0]
    assert num_masks == len(mapping), f"num_masks={num_masks} != len(mapping)={len(mapping)}"

    # Asynchronously copy the mapping to NPU.
    with torch.npu.stream(self.copy_stream):
        logits_indices_cpu = torch.tensor(mapping, dtype=torch.int32, device="cpu", pin_memory=True)
        logits_indices = self.logits_indices[: len(mapping)].copy_(logits_indices_cpu, non_blocking=True)
    # ensure copies finish before kernel launch
    current_stream = torch.npu.current_stream()
    current_stream.wait_stream(self.copy_stream)

    vocab_size = logits.shape[-1]
    BLOCK_SIZE = _NPU_BLOCK_SIZE
    BLOCK_SIZE_SUB = _NPU_BLOCK_SIZE_SUB
    grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
    )

    # Ensure the copy stream waits for the device tensors to finish being used
    # before it reuses or deallocates them
    self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
# Ascend NPU bitmask kernel (BLOCK_SIZE_SUB tiling)
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
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
        # Unpack: each int32 word → 32 bool values (0 = blocked)
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE_SUB)

        # Apply: set blocked positions to -inf
        block_offset = global_token_offset + tl.arange(0, BLOCK_SIZE_SUB)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=bitmask & (block_offset < vocab_size),
        )
