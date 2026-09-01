# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Triton-Ascend ScatterNdUpdateV2 (drops the AscendC op entirely).

Replaces `torch.ops._C_ascend.npu_scatter_nd_update_v2` on every call the triton
kernel can correctly execute, with the same in-place semantics:
    var[indices[i,0], indices[i,1], ..., indices[i,K-1], :, ...] = updates[i]

Why drop AscendC: its no-sort kernel partitions the output *element range* across
vector cores and scans all indices per core. When the slot mapping clusters (real
DeepSeek V4 SVF/DSA decode: a few sequences -> a few hot block ranges) only the
cores owning those ranges do work and the rest stall -> up to ~48x imbalance
(msprof captures show ~1 ms for a 8160-token / 512-dim write that the triton path
finishes in ~95 us). The triton kernel issues one program per token and lets the
vector cores schedule them freely, so cost stays flat across index distributions.

Coverage (aligned to the AscendC op's capability, except int64 indices where
triton-ascend i64 arithmetic lowers to scalar and hurts the kernel):
  * var rank >= 1 with trailing block contiguous in row-major;
  * any K = indices.shape[-1] in [1, 4];
  * var/updates dtype in {bf16, fp16, fp32, int8, int16, int32, int64, bool};
  * linear offset must fit int32 (the kernel computes offsets in i32; larger
    cache + int64 indices is the one case that purposefully falls back).

Everything outside this envelope (int64 indices for huge caches, K > 4, or a
non-contiguous trailing block) falls back to PyTorch eager advanced-indexing
(`var[tuple(indices.unbind(-1))] = updates`); the AscendC op is no longer used.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# Dtypes the triton kernel scatters verbatim (no cast/quant). Verified bit-exact
# on Ascend910_9382 / triton 3.2.0 for {bf16, fp16, fp32, int8, int16, int32,
# int64, bool}; the quant itself (for the int8 indexer cache) happens upstream
# in `npu_dynamic_quant`, this op is pure scatter.
_TRITON_SCATTER_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)

# Unified Buffer is 192 KB on Ascend 910B/C (DAV_2201). The kernel materialises a
# single [BLOCK_D] tile per program (BLOCK_D = next_pow2(D)); cap it at half the
# UB to leave room for indices/flags in the double-buffered queue. Larger rows
# would need an intra-row tile loop and are out of scope; they fall back to
# PyTorch eager (still correct, just not vector-core-fast).
_UB_BUDGET_BYTES = 192 * 1024 // 2

# i32 carries the linear offset inside the kernel; jumbo caches with int64
# indices are intentionally excluded (i64 arithmetic on Ascend lowers to scalar
# and would defeat the optimization). The PyTorch eager fallback handles them.
_MAX_LINEAR_OFFSET = (1 << 31) - 1

# The kernel unrolls strides over [0, K) with an if/elif on the constexpr loop
# index; 4 leading-dim strides covers every KV-cache / indexer / SWA / compressor
# shape seen in DeepSeek V4 SVF/DSA. Higher K falls back to PyTorch eager.
_MAX_INDEX_DEPTH = 4


@triton.jit
def _scatter_nd_update_kernel(
    updates_ptr,
    indices_ptr,
    var_ptr,
    num_tokens,
    s0,
    s1,
    s2,
    s3,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per (`indices[i,:]`) update row. Each program:
    #   1) loads the K leading-dim index entries for this token;
    #   2) folds them with their (i32) strides into one element offset `lin`;
    #   3) loads D contiguous elements from `updates[i]`;
    #   4) stores them as D contiguous elements at `var[lin:lin+D]`.
    # Roughly num_tokens programs are scheduled freely across the 48 vector
    # cores, so index clustering cannot starve cores the way AscendC's
    # range-partition scheme does. Values are copied bit-for-bit.
    pid = tl.program_id(0)
    lin = 0
    # tl.static_range + if/elif chain is lowered at compile time (one branch
    # per used K); the unused stride slots are passed as 0 from the host.
    for d in tl.static_range(K):
        idx = tl.load(indices_ptr + pid * K + d)
        if d == 0:
            lin = lin + idx * s0
        elif d == 1:
            lin = lin + idx * s1
        elif d == 2:
            lin = lin + idx * s2
        else:
            lin = lin + idx * s3
    off = tl.arange(0, BLOCK_D)
    mask = off < D
    val = tl.load(updates_ptr + pid * D + off, mask=mask)
    tl.store(var_ptr + lin + off, val, mask=mask)


def _trailing_block_is_contiguous(var: torch.Tensor, k: int) -> bool:
    """`var[indices[i,:]]` must span `prod(var.shape[k:])` contiguous elements.

    Required because the kernel writes one run of D = prod(var.shape[k:]) bf16
    starting at the computed element offset.
    """
    # Every dim from k+1 .. ndim-1 must be row-major contiguous: stride(i) =
    # stride(i+1) * shape(i+1), and the last dim stride == 1.
    expected = 1
    for i in range(var.dim() - 1, k - 1, -1):
        if var.stride(i) != expected:
            return False
        expected *= var.shape[i]
    return True


def can_use_triton_scatter(var: torch.Tensor, indices: torch.Tensor) -> bool:
    """Return True iff the triton kernel can correctly execute this call.

    Pure feasibility check (no per-shape performance heuristic): a `True` result
    means the same in-place result as the AscendC op; `False` means the call's
    shape/dtype/layout is outside what the kernel handles, so the dispatcher
    falls back to PyTorch eager.
    """
    # indices: 2D [num_tokens, K], int32. (int64 indices are excluded because
    # the kernel keeps offsets in i32; PyTorch eager handles that case.)
    if indices.dim() != 2 or indices.shape[-1] < 1:
        return False
    k = indices.shape[-1]
    if k > _MAX_INDEX_DEPTH:
        return False
    if indices.dtype != torch.int32:
        return False
    # var must have at least K leading dims to be indexed, plus a trailing
    # block (the leading-K dims could also be the whole tensor when D == 1).
    if var.dim() < k:
        return False
    # var.shape[:K] is the index-indexed region; we do not bound-check index
    # values (callers guarantee in-range); only the linear-offset range.
    if var.dtype not in _TRITON_SCATTER_DTYPES:
        return False
    if not _trailing_block_is_contiguous(var, k):
        return False
    # Linear offset (over the leading K dims) must fit i32; same intent as the
    # AscendC IsLinearIndex guard before it picks the int32 vs int64 path.
    max_lin = 0
    for i in range(k):
        max_lin = max_lin + (var.shape[i] - 1) * var.stride(i)
    if max_lin > _MAX_LINEAR_OFFSET:
        return False
    # Row payload D = prod(var.shape[k:]) and its pow2-padded BLOCK_D must each
    # fit the UB budget. The kernel issues one [BLOCK_D] load + one [BLOCK_D]
    # store per program; if D itself would overflow UB we would need an
    # intra-row tile loop, which is out of scope -> fall back to PyTorch.
    d = 1
    for i in range(k, var.dim()):
        d *= int(var.shape[i])
    block_d = 1
    while block_d < d:
        block_d <<= 1
    if block_d * var.element_size() > _UB_BUDGET_BYTES:
        return False
    # num_tokens must be representable as a 1D triton grid ( Triton-Ascend grid
    # upper bound is 65535 along each dim); huge batches fall back too.
    if indices.shape[0] > 65535:
        return False
    return True


def triton_scatter_nd_update(
    var: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """In-place scatter `var[indices[i,0..K-1], ...]` = updates[i] via triton.

    Launches the kernel unconditionally; the caller (`device_op._scatter_nd_update`)
    gates with `can_use_triton_scatter` first and holds the PyTorch-eager fallback
    itself, so this path is only reached for shapes the kernel can execute. No
    env-var opt-in and no per-shape branching here; the AscendC
    `npu_scatter_nd_update_v2` op is no longer used by this dispatcher.
    """
    num_tokens = indices.shape[0]
    k = indices.shape[-1]
    d = 1
    for i in range(k, var.dim()):
        d *= int(var.shape[i])  # elements in the trailing (row-major) block
    # tl.arange needs a power-of-two length; pad up and mask the tail.
    block_d = 1
    while block_d < d:
        block_d <<= 1
    strides = [var.stride(i) for i in range(k)]
    strides += [0] * (_MAX_INDEX_DEPTH - k)
    _scatter_nd_update_kernel[(num_tokens,)](
        updates, indices, var, num_tokens,
        strides[0], strides[1], strides[2], strides[3],
        K=k, D=d, BLOCK_D=block_d,
    )
