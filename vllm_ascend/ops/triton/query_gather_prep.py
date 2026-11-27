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


@triton.jit(do_not_specialize=["num_groups"])
def _prepare_sfa_pair_kernel(
    nope,
    rope,
    out_n,
    out_r,
    num_groups,
    ROWS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    N_STRIDE_G: tl.constexpr,
    N_STRIDE_ROW: tl.constexpr,
    N_STRIDE_D: tl.constexpr,
    R_STRIDE_G: tl.constexpr,
    R_STRIDE_ROW: tl.constexpr,
    R_STRIDE_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row_tiles = tl.cdiv(OUT_ROWS, BLOCK_ROWS)
    for task in range(tl.program_id(0), num_groups * row_tiles, tl.num_programs(0)):
        group = (task // row_tiles).to(tl.int64)
        row = ((task % row_tiles) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]).to(tl.int64)
        out_row = group * OUT_ROWS + row
        if DN > 0:
            dim_n = tl.arange(0, BLOCK_N)[None, :]
            values_n = tl.load(
                nope + group * N_STRIDE_G + row * N_STRIDE_ROW + dim_n * N_STRIDE_D,
                mask=(row < ROWS) & (dim_n < DN),
                other=0,
            )
            tl.store(out_n + out_row * DN + dim_n, values_n, mask=(row < OUT_ROWS) & (dim_n < DN))
        if DR > 0:
            dim_r = tl.arange(0, BLOCK_R)[None, :]
            values_r = tl.load(
                rope + group * R_STRIDE_G + row * R_STRIDE_ROW + dim_r * R_STRIDE_D,
                mask=(row < ROWS) & (dim_r < DR),
                other=0,
            )
            tl.store(out_r + out_row * DR + dim_r, values_r, mask=(row < OUT_ROWS) & (dim_r < DR))


def _prepare_sfa_pair(nope: torch.Tensor, rope: torch.Tensor, output_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    # One group is a token for queries or a physical cache page for KV.
    groups, rows, dim_n = nope.shape
    dim_r = rope.shape[2]
    out_n = nope.new_empty((groups, output_rows, dim_n))
    out_r = rope.new_empty((groups, output_rows, dim_r))
    if groups and output_rows and (dim_n or dim_r):
        init_device_properties_triton()
        block_rows = min(8, next_power_of_2(output_rows))
        grid = (min(groups * triton.cdiv(output_rows, block_rows), get_vectorcore_num()),)
        _prepare_sfa_pair_kernel[grid](
            nope,
            rope,
            out_n,
            out_r,
            groups,
            rows,
            output_rows,
            dim_n,
            dim_r,
            *nope.stride(),
            *rope.stride(),
            block_rows,
            next_power_of_2(max(dim_n, 1)),
            next_power_of_2(max(dim_r, 1)),
            multibuffer=False,
        )
    return out_n, out_r


def prepare_sfa_query_pair(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, output_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy two BF16 query views into independent contiguous TND tensors, padding extra heads with zeros."""
    if ql_nope.ndim != 3 or q_pe.ndim != 3 or ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("SFA queries require matching token and head dimensions in TND layout")
    if ql_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16 or ql_nope.device != q_pe.device:
        raise ValueError("SFA queries must be BF16 tensors on the same device")

    if output_heads < ql_nope.shape[1]:
        raise ValueError("output_heads must cover every input query head")
    return _prepare_sfa_pair(ql_nope, q_pe, output_heads)


@triton.jit(do_not_specialize=["num_pages"])
def _copy_sfa_kv_pages_kernel(
    latent,
    rope,
    out_latent,
    out_rope,
    num_pages,
    STRIDE_N: tl.constexpr,
    STRIDE_R: tl.constexpr,
    TILES_PER_PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    for tile in range(tl.program_id(0), num_pages * TILES_PER_PAGE, tl.num_programs(0)):
        out_tile = tl.cast(tile, tl.int64)
        page = out_tile // TILES_PER_PAGE
        within_page = out_tile % TILES_PER_PAGE
        n = tl.arange(0, BLOCK_N)
        r = tl.arange(0, BLOCK_R)
        values_n = tl.load(latent + page * STRIDE_N + within_page * BLOCK_N + n)
        tl.store(out_latent + out_tile * BLOCK_N + n, values_n)
        values_r = tl.load(rope + page * STRIDE_R + within_page * BLOCK_R + r)
        tl.store(out_rope + out_tile * BLOCK_R + r, values_r)


def prepare_sfa_kv_pair(kv: torch.Tensor, key_rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact single-head PA_BSND latent and RoPE planes while preserving physical page IDs."""
    if kv.ndim != 4 or key_rope.ndim != 4 or kv.shape[:3] != key_rope.shape[:3] or kv.shape[2] != 1:
        raise ValueError("SFA KV planes require matching [pages, block_size, 1, dim] layouts")
    if kv.dtype != torch.bfloat16 or key_rope.dtype != torch.bfloat16 or kv.device != key_rope.device:
        raise ValueError("SFA KV planes must be BF16 tensors on the same device")
    if kv.is_contiguous() or key_rope.is_contiguous():
        return kv.contiguous(), key_rope.contiguous()
    pages, rows, _, dim_n = kv.shape
    dim_r = key_rope.shape[-1]
    # The common PA_BSND planes have independent page pitches but flat inner rows.
    # Whole power-of-two tiles lower to unmasked copies, up to 32 KiB per plane.
    tile_rows = min(32, rows & -rows)  # Largest power of two dividing rows, capped at 32.
    if (
        tile_rows >= 8
        and 0 < dim_n <= 512
        and dim_n == next_power_of_2(dim_n)
        and 0 < dim_r <= 512
        and dim_r == next_power_of_2(dim_r)
        and (kv.stride(1), kv.stride(3)) == (dim_n, 1)
        and (key_rope.stride(1), key_rope.stride(3)) == (dim_r, 1)
    ):
        latent = kv.new_empty(kv.shape)
        rope = key_rope.new_empty(key_rope.shape)
        init_device_properties_triton()
        tiles_per_page = rows // tile_rows
        grid = (min(pages * tiles_per_page, get_vectorcore_num()),)
        _copy_sfa_kv_pages_kernel[grid](
            kv,
            key_rope,
            latent,
            rope,
            pages,
            kv.stride(0),
            key_rope.stride(0),
            tiles_per_page,
            tile_rows * dim_n,
            tile_rows * dim_r,
            multibuffer=False,
        )
        return latent, rope
    latent, rope = _prepare_sfa_pair(kv.squeeze(2), key_rope.squeeze(2), kv.shape[1])
    return latent.unsqueeze(2), rope.unsqueeze(2)
