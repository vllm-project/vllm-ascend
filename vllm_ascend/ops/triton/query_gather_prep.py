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
# Fused head-major assembly of the DCP query-gather input.

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "qn_stride_t",
        "qn_stride_h",
        "qp_stride_t",
        "qp_stride_h",
    ]
)
def _q_gather_prep_head_major_kernel(
    qn_ptr,  # [T, H, nope_dim] ql_nope (any non-negative strides)
    qp_ptr,  # [T, H, rope_dim] q_pe (any non-negative strides)
    out_ptr,  # [H, T, nope_dim + rope_dim] contiguous, head-major gather input
    num_tokens,
    num_heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    total_dim: tl.constexpr,
    qn_stride_t,
    qn_stride_h,
    qn_stride_d: tl.constexpr,
    qp_stride_t,
    qp_stride_h,
    qp_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Assemble the fused query directly in head-major layout.

    Grid-stride over the ``num_heads * num_tokens`` output rows: each program
    processes a subset of the (head, token) rows. Writes the fused row
    ``[nope | rope]`` into ``out[h, t, :]`` (contiguous), producing exactly
    the buffer that ``all_gather_into_tensor`` needs for the native-DCP head
    gather. This replaces the torch ``cat -> permute -> contiguous`` chain
    with a single kernel.

    Keep each tile inside one head for affine source/destination strides.
    Masks protect tail rows/features; clamping their addresses would force
    element-wise movement instead of DMA. Empty fragments are omitted.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    token_tiles = tl.cdiv(num_tokens, BLOCK_ROWS)
    total_tasks = token_tiles * num_heads

    for task in range(pid, total_tasks, num_programs):
        head_idx = (task // token_tiles).to(tl.int64)
        token_idx = (task % token_tiles) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = token_idx < num_tokens
        token_idx = token_idx.to(tl.int64)
        dst_base = (head_idx * num_tokens + token_idx) * total_dim
        src_base_n = token_idx * qn_stride_t + head_idx * qn_stride_h
        src_base_p = token_idx * qp_stride_t + head_idx * qp_stride_h

        if nope_dim > 0:
            offs_n = tl.arange(0, BLOCK_N)[None, :]
            n_mask = row_mask & (offs_n < nope_dim)
            qn = tl.load(qn_ptr + src_base_n + offs_n * qn_stride_d, mask=n_mask, other=0)
            tl.store(out_ptr + dst_base + offs_n, qn, mask=n_mask)
        if rope_dim > 0:
            offs_p = tl.arange(0, BLOCK_R)[None, :]
            p_mask = row_mask & (offs_p < rope_dim)
            qp = tl.load(qp_ptr + src_base_p + offs_p * qp_stride_d, mask=p_mask, other=0)
            tl.store(out_ptr + dst_base + nope_dim + offs_p, qp, mask=p_mask)


def prep_query_head_major(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
) -> torch.Tensor:
    """Return the fused query as a contiguous [H, T, nope+rope] tensor.

    Args:
        ql_nope: [T, H, nope_dim] partial query (rope-excluded).
        q_pe: [T, H, rope_dim] rope part.

    Returns:
        [H, T, nope_dim + rope_dim] contiguous tensor, in the layout
        ``all_gather_into_tensor`` produces for a head gather (each rank's
        head chunk is contiguous along dim 0). Empty inputs return an empty
        output without launching a kernel. Zero-width fragments and strided
        views are supported; kernel failures propagate to the caller.

    Raises:
        RuntimeError: on genuinely invalid inputs (mismatched (T, H) or
        dtype).
    """
    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise RuntimeError(
            f"prep_query_head_major requires matching (T, H), got {tuple(ql_nope.shape)} and {tuple(q_pe.shape)}"
        )
    if ql_nope.dtype != q_pe.dtype or ql_nope.device != q_pe.device:
        raise RuntimeError("prep_query_head_major requires ql_nope and q_pe to share a dtype and device")

    num_tokens, num_heads, nope_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    total_dim = nope_dim + rope_dim
    out = torch.empty(
        (num_heads, num_tokens, total_dim),
        dtype=ql_nope.dtype,
        device=ql_nope.device,
    )
    if out.numel() == 0:
        return out
    init_device_properties_triton()
    num_vectorcore = get_vectorcore_num()
    block_rows = 8 if num_tokens * num_heads >= 8 * num_vectorcore else 1
    grid = (min(triton.cdiv(num_tokens, block_rows) * num_heads, num_vectorcore),)

    _q_gather_prep_head_major_kernel[grid](
        ql_nope,
        q_pe,
        out,
        num_tokens,
        num_heads,
        nope_dim,
        rope_dim,
        total_dim,
        ql_nope.stride(0),
        ql_nope.stride(1),
        ql_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        BLOCK_N=next_power_of_2(max(nope_dim, 1)),
        BLOCK_R=next_power_of_2(max(rope_dim, 1)),
        BLOCK_ROWS=block_rows,
        multibuffer=False,
    )

    return out
