#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Triton-Ascend implementation of AscendSFADCPImpl._remap_sparse_indices.

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.triton_utils import get_element


@triton.jit(do_not_specialize=["topk_count", "dcp_size", "interleave_size", "dcp_rank"])
def remap_sparse_indices_fused_kernel(
    indices_ptr,  # [rows, topk_count] int32, replicated-view topk indices
    chunk_out_ptr,  # [rows * num_chunks, BLOCK] int32 (output)
    chunk_count_ptr,  # [rows, num_chunks] int32 (output)
    out_ptr,  # [rows, topk_count] int32 (output): tail pre-filled with -1 here
    topk_count,
    dcp_size,
    interleave_size,
    dcp_rank,
    INTERLEAVE_ONE: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Fused remap + per-chunk compaction + output tail pre-fill.

    One program per (chunk, row). The vectorized remap computes the
    DCP-local index and the owner-validity mask for this chunk; then the
    chunk's valid entries are compacted to the front of its ``chunk_out``
    slot in order via scalar ``get_element`` operations, the chunk's valid
    count is written to ``chunk_count``, and the chunk's region of ``out``
    is pre-filled with -1 (the gather kernel later overwrites the compacted
    front).
    """
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < topk_count

    idx = tl.load(indices_ptr + row * topk_count + offsets, mask=in_bounds, other=-1)
    # Integer remap math: matches the fp32 torch fallback bit-exactly
    # (indices are far below 2^24), and the int32/fp32 variants measured
    # identically on NPU.
    block_idx = idx // interleave_size
    owner = block_idx - (block_idx // dcp_size) * dcp_size
    valid = (idx >= 0) & (owner == dcp_rank)

    if INTERLEAVE_ONE:
        remapped = idx // dcp_size
    else:
        local_offsets = idx - block_idx * interleave_size
        remapped = (idx // (dcp_size * interleave_size)) * interleave_size + local_offsets

    valid_i32 = valid.to(tl.int32)
    chunk_out = chunk_out_ptr + (row * NUM_CHUNKS + chunk) * BLOCK
    # Pre-fill this chunk's output region with -1 first; the serial stores
    # below overwrite the compacted front in program order.
    tl.store(out_ptr + row * topk_count + offsets, -1, mask=in_bounds)
    cnt = 0
    for i in range(BLOCK):
        if get_element(valid_i32, (i,)) == 1:
            tl.store(chunk_out + cnt, get_element(remapped, (i,)))
            cnt += 1
    tl.store(chunk_count_ptr + row * NUM_CHUNKS + chunk, cnt)


@triton.jit(do_not_specialize=["topk_count"])
def remap_sparse_indices_compact_gather_kernel(
    chunk_out_ptr,  # [rows * num_chunks, BLOCK] int32
    chunk_count_ptr,  # [rows, num_chunks] int32
    out_ptr,  # [rows, topk_count] int32 (output): tail pre-filled with -1
    topk_count,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Gather chunks to the row front in chunk order.

    One program per row. The per-chunk copy is vectorized (masked load +
    masked contiguous store); serial iteration count is num_chunks only.
    """
    row = tl.program_id(0)
    write_pos = 0
    for chunk in range(NUM_CHUNKS):
        cnt = tl.load(chunk_count_ptr + row * NUM_CHUNKS + chunk)
        offs = tl.arange(0, BLOCK)
        m = offs < cnt
        x = tl.load(chunk_out_ptr + (row * NUM_CHUNKS + chunk) * BLOCK + offs, mask=m, other=0)
        tl.store(out_ptr + row * topk_count + write_pos + offs, x, mask=m)
        write_pos += cnt


def remap_sparse_indices_triton(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    *,
    out: torch.Tensor | None = None,
    chunk_out: torch.Tensor | None = None,
    chunk_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton replacement for the torch implementation in
    ``AscendSFADCPImpl._remap_sparse_indices``.

    Args:
        topk_indices: [..., topk_count] int32 replicated-view topk indices.
        dcp_size: Number of DCP ranks.
        dcp_rank: This rank's index in the DCP group.
        interleave_size: KV cache interleave size of the replicated view.

    Returns:
        Tensor with the same shape and dtype as ``topk_indices``: indices
        owned by this rank remapped to DCP-local KV positions and compacted
        to the front of the last dim in top-k order, padded with -1. Callers
        may provide all three output/scratch tensors to keep their addresses
        stable across graph capture and replay.

    Design notes (verified on triton-ascend 3.2.0 / CANN 9.1):
    - Do not use a ``tl.cumsum`` prefix-sum compaction: on a2 it produces
      nondeterministic garbage values, and on a3 it traps the vector core
      under repeated launches (single-shot results are correct). ``tl.cumsum``
      alone carries a large per-program fixed cost that makes row-parallel
      layouts very slow.
    - Do not fill the output tail inside the gather kernel after its copy
      loop with a scalar-comparison mask: it writes wrong values. The tail
      is pre-filled in the fused kernel with a plain ``in_bounds`` mask
      instead, so this path has only two device ops (fused + gather).
    """
    orig_dtype = topk_indices.dtype
    orig_shape = topk_indices.shape
    if not topk_indices.is_contiguous():
        topk_indices = topk_indices.contiguous()
    indices = topk_indices if topk_indices.dtype == torch.int32 else topk_indices.to(torch.int32)
    topk_count = indices.shape[-1]
    rows = indices.numel() // topk_count
    if rows == 0 or topk_count == 0:
        return topk_indices
    # The torch implementation operates per-row on the last dim, so arbitrary
    # leading dims (e.g. [dcp_size, 1, topk_count] from the DCP all_gather) can
    # be flattened to a 2D view and restored afterwards.
    indices = indices.view(rows, topk_count)

    block = min(128, next_power_of_2(topk_count))
    num_chunks = triton.cdiv(topk_count, block)
    expected_chunk_out_shape = (rows * num_chunks, block)
    expected_chunk_count_shape = (rows, num_chunks)
    if out is None:
        out_2d = torch.empty_like(indices)
    else:
        if orig_dtype != torch.int32:
            raise ValueError("a caller-provided remap output requires int32 TopK indices")
        if out.shape != orig_shape or out.dtype != torch.int32 or out.device != indices.device:
            raise ValueError(
                "remap output must match the input shape/device and use int32, "
                f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}."
            )
        if not out.is_contiguous():
            raise ValueError("remap output must be contiguous")
        out_2d = out.view(rows, topk_count)
    if chunk_out is None:
        chunk_out = torch.empty(expected_chunk_out_shape, dtype=indices.dtype, device=indices.device)
    elif (
        chunk_out.shape != expected_chunk_out_shape
        or chunk_out.dtype != torch.int32
        or chunk_out.device != indices.device
        or not chunk_out.is_contiguous()
    ):
        raise ValueError(
            "remap chunk_out has an incompatible shape, dtype, device or stride: "
            f"expected={expected_chunk_out_shape}, actual={tuple(chunk_out.shape)}."
        )
    if chunk_count is None:
        chunk_count = torch.empty(expected_chunk_count_shape, dtype=torch.int32, device=indices.device)
    elif (
        chunk_count.shape != expected_chunk_count_shape
        or chunk_count.dtype != torch.int32
        or chunk_count.device != indices.device
        or not chunk_count.is_contiguous()
    ):
        raise ValueError(
            "remap chunk_count has an incompatible shape, dtype, device or stride: "
            f"expected={expected_chunk_count_shape}, actual={tuple(chunk_count.shape)}."
        )

    remap_sparse_indices_fused_kernel[(num_chunks, rows)](
        indices,
        chunk_out,
        chunk_count,
        out_2d,
        topk_count,
        dcp_size,
        interleave_size,
        dcp_rank,
        INTERLEAVE_ONE=interleave_size == 1,
        BLOCK=block,
        NUM_CHUNKS=num_chunks,
        multibuffer=False,
    )
    remap_sparse_indices_compact_gather_kernel[(rows,)](
        chunk_out,
        chunk_count,
        out_2d,
        topk_count,
        BLOCK=block,
        NUM_CHUNKS=num_chunks,
        multibuffer=False,
    )
    remapped = out_2d.view(orig_shape)
    return remapped if orig_dtype == torch.int32 else remapped.to(orig_dtype)


@triton.jit
def _mask_invalid_sparse_attention_rows_kernel(
    indices_ptr,
    output_ptr,
    lse_ptr,
    indices_stride_token,
    output_stride_token,
    output_stride_head,
    lse_stride_token,
    lse_stride_head,
    num_heads,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row - token * num_heads
    first_local_index = tl.load(indices_ptr + token * indices_stride_token)
    invalid = first_local_index < 0
    offsets = tl.arange(0, BLOCK_D)
    tl.store(
        output_ptr + token * output_stride_token + head * output_stride_head + offsets,
        0.0,
        mask=invalid & (offsets < head_dim),
    )
    tl.store(
        lse_ptr + token * lse_stride_token + head * lse_stride_head,
        -float("inf"),
        mask=invalid,
    )


def mask_invalid_sparse_attention_rows(
    local_topk_indices: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
) -> None:
    """Make a rank with no selected C4 rows an exact empty contribution.

    ``remap_sparse_indices_triton`` compacts every valid local index to the
    front, so the first entry is negative exactly when this rank owns none of
    the global TopK for the token. The sparse-attention kernel uses a finite
    disabled-sink sentinel for that case; replace it on device with the
    distributed-softmax identity: zero output and negative-infinite LSE.
    """
    if output.ndim != 3 or softmax_lse.ndim != 3 or local_topk_indices.ndim != 3:
        raise ValueError("C4 partial output, LSE and TopK indices must all use TND-shaped 3D tensors")
    num_tokens, num_heads, head_dim = output.shape
    if local_topk_indices.shape[0] != num_tokens or local_topk_indices.shape[1] != 1:
        raise ValueError("C4 local TopK indices must have shape [tokens, 1, topk]")
    if softmax_lse.shape != (num_tokens, num_heads, 1):
        raise ValueError(
            "C4 partial LSE must have shape [tokens, heads, 1], "
            f"got {tuple(softmax_lse.shape)}."
        )
    if num_tokens == 0:
        return
    _mask_invalid_sparse_attention_rows_kernel[(num_tokens * num_heads,)](
        local_topk_indices,
        output,
        softmax_lse,
        local_topk_indices.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        num_heads,
        head_dim,
        BLOCK_D=next_power_of_2(head_dim),
        multibuffer=False,
    )

