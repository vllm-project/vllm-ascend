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
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_FP8_E4M3_MAX = 448.0


def _fp8_rope_head_block(n_heads: int, cap: int = 16) -> int:
    """Size the FP8 RoPE head tile to the actual MiniMax-M3 head count."""
    if n_heads <= 0:
        return 1
    return min(triton.next_power_of_2(n_heads), cap)


@triton.jit
def _store_fp8_e4m3(dst_ptr, offsets, vals_fp32, mask, FP8_MAX: tl.constexpr):
    vals = tl.minimum(tl.maximum(vals_fp32, -FP8_MAX), FP8_MAX)
    tl.store(dst_ptr + offsets, vals.to(dst_ptr.dtype.element_ty), mask=mask)


def _get_block_size_head(n_q_head: int, n_kv_head: int, is_neox_style: bool) -> int:
    max_block_size = 64 if is_neox_style else 32
    actual_head_size = triton.next_power_of_2(max(n_q_head, n_kv_head))
    return min(max_block_size, actual_head_size)


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    USE_COS_SIN: tl.constexpr,
):
    """
    This triton kernel applies rotary embedding on q and k.
    It supports rope_dim != head_dim scenario.
    It supports both neox style and non-neox style rope computation.
    q/k head dimensions are tiled with BLOCK_SIZE_HEAD to avoid UB overflow.

    Input tensor layout assumptions:

    q size: (num_tokens, num_q_heads, head_dim)
    q stride: (num_q_heads * head_dim, head_dim, 1)
    k size: (num_tokens, num_kv_heads, head_dim)
    k stride: (num_kv_heads * head_dim, head_dim, 1)
    cos/sin size: (num_tokens, rope_dim/2)
    cos/sin stride: (rope_dim/2, 1)

    Different compute pattern of IS_NEOX_STYLE:

    if IS_NEOX_STYLE:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if IS_NEOX_STYLE:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)
    """
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        q_row_start_ptr = q_ptr + row_idx * q_row_stride
        k_row_start_ptr = k_ptr + row_idx * k_row_stride

        # ####################################################################
        # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
        # m of this program instance
        # ####################################################################
        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        sin_offsets = tl.arange(pad_rope_dim // 2, pad_rope_dim)
        cos_mask = cos_offsets < (rope_dim // 2)
        if USE_COS_SIN:
            pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
            cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(cos_start_ptr + sin_offsets, mask=cos_mask, other=0).to(tl.float32)
        else:
            cos_start_ptr = cos_ptr + row_idx * cos_row_stride
            sin_start_ptr = sin_ptr + row_idx * sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        # ####################################################################
        # Tile over q heads in chunks of BLOCK_SIZE_HEAD
        # ####################################################################
        for q_head_base in tl.range(0, n_qh, BLOCK_SIZE_HEAD):
            q_tile_start_ptr = q_row_start_ptr + q_head_base * hd
            q_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_q_offsets = q_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_q_mask = ((q_head_base + q_heads)[:, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                q_tile_1 = tl.load(q_tile_start_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
                    sin_row.dtype
                )
                second_half_q_offsets = first_half_q_offsets + (rope_dim // 2)
                second_q_mask = first_q_mask
                q_tile_2 = tl.load(q_tile_start_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
                    sin_row.dtype
                )
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                tl.store(q_tile_start_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                tl.store(q_tile_start_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
            else:
                pair_offsets = (
                    q_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((q_head_base + q_heads)[:, None, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                q_tile = tl.load(q_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                q_tile_1, q_tile_2 = tl.split(q_tile)
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                q_tile_out = tl.join(new_q_tile_1, new_q_tile_2)
                tl.store(q_tile_start_ptr + pair_offsets, q_tile_out, mask=pair_mask)

        # ####################################################################
        # Tile over k heads in chunks of BLOCK_SIZE_HEAD
        # ####################################################################
        for k_head_base in tl.range(0, n_kh, BLOCK_SIZE_HEAD):
            k_tile_start_ptr = k_row_start_ptr + k_head_base * hd
            k_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_k_offsets = k_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_k_mask = ((k_head_base + k_heads)[:, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                k_tile_1 = tl.load(k_tile_start_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
                    sin_row.dtype
                )
                second_half_k_offsets = first_half_k_offsets + (rope_dim // 2)
                second_k_mask = first_k_mask
                k_tile_2 = tl.load(k_tile_start_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
                    sin_row.dtype
                )
                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                tl.store(k_tile_start_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                tl.store(k_tile_start_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
            else:
                pair_offsets = (
                    k_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((k_head_base + k_heads)[:, None, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                k_tile = tl.load(k_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                k_tile_1, k_tile_2 = tl.split(k_tile)

                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                k_tile_out = tl.join(new_k_tile_1, new_k_tile_2)
                tl.store(k_tile_start_ptr + pair_offsets, k_tile_out, mask=pair_mask)


@triton.jit
def _triton_rope_siso(
    qk_ptr,
    qk_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_h: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_h: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    USE_COS_SIN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        qk_start_ptr = qk_ptr + row_idx * qk_row_stride

        # ####################################################################
        # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
        # m of this program instance
        # ####################################################################
        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        sin_offsets = tl.arange(pad_rope_dim // 2, pad_rope_dim)
        cos_mask = cos_offsets < (rope_dim // 2)
        if USE_COS_SIN:
            pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
            cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(cos_start_ptr + sin_offsets, mask=cos_mask, other=0).to(tl.float32)
        else:
            cos_start_ptr = cos_ptr + row_idx * cos_row_stride
            sin_start_ptr = sin_ptr + row_idx * sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        # ####################################################################
        # Load the left and right half of q and k for the current
        # program instance (i.e. for the current token) separately
        # ####################################################################
        # left half of the head
        if IS_NEOX_STYLE:
            first_half_offsets = tl.arange(0, pad_n_h)[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
        else:
            first_half_offsets = tl.arange(0, pad_n_h)[:, None] * hd + (2 * tl.arange(0, pad_rope_dim // 2)[None, :])

        first_mask = (tl.arange(0, pad_n_h)[:, None] < n_h) & (
            tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
        )
        qk_tile_1 = tl.load(qk_start_ptr + first_half_offsets, mask=first_mask, other=0).to(sin_row.dtype)

        # right half of the head
        if IS_NEOX_STYLE:
            second_half_offsets = first_half_offsets + (rope_dim // 2)
        else:
            second_half_offsets = first_half_offsets + 1
        second_mask = first_mask
        qk_tile_2 = tl.load(qk_start_ptr + second_half_offsets, mask=second_mask, other=0).to(sin_row.dtype)

        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_qk_tile_1 = qk_tile_1 * cos_row - qk_tile_2 * sin_row
        tl.store(qk_start_ptr + first_half_offsets, new_qk_tile_1, mask=first_mask)

        new_qk_tile_2 = qk_tile_2 * cos_row + qk_tile_1 * sin_row
        tl.store(qk_start_ptr + second_half_offsets, new_qk_tile_2, mask=second_mask)


@triton.jit
def _triton_rope_fp8(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    q_out_ptr,
    q_out_row_stride,
    k_out_ptr,
    k_out_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_half: tl.constexpr,
    pad_pass: tl.constexpr,
    pass_dim: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """NeoX RoPE with a direct fixed-scale E4M3 output store."""
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)
    half: tl.constexpr = rope_dim // 2
    cos_offsets = tl.arange(0, pad_half)
    cos_mask = cos_offsets < half

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
        cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
        sin_row = tl.load(cos_start_ptr + half + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        q_row = q_ptr + row_idx * q_row_stride
        q_out_row = q_out_ptr + row_idx * q_out_row_stride
        for q_base in tl.range(0, n_qh, BLOCK_QH):
            q_heads = tl.arange(0, BLOCK_QH)
            head_ok = (q_base + q_heads)[:, None] < n_qh
            base = q_row + q_base * hd
            out_base = q_out_row + q_base * hd

            half_offs = tl.arange(0, pad_half)
            half_mask = head_ok & (half_offs[None, :] < half)
            x1 = tl.load(
                base + q_heads[:, None] * hd + half_offs[None, :],
                mask=half_mask,
                other=0,
            ).to(tl.float32)
            x2 = tl.load(
                base + q_heads[:, None] * hd + half + half_offs[None, :],
                mask=half_mask,
                other=0,
            ).to(tl.float32)
            _store_fp8_e4m3(
                out_base,
                q_heads[:, None] * hd + half_offs[None, :],
                x1 * cos_row - x2 * sin_row,
                half_mask,
                FP8_MAX,
            )
            _store_fp8_e4m3(
                out_base,
                q_heads[:, None] * hd + half + half_offs[None, :],
                x2 * cos_row + x1 * sin_row,
                half_mask,
                FP8_MAX,
            )
            if pass_dim > 0:
                pass_offs = tl.arange(0, pad_pass)
                pass_mask = head_ok & (pass_offs[None, :] < pass_dim)
                pass_vals = tl.load(
                    base + q_heads[:, None] * hd + rope_dim + pass_offs[None, :],
                    mask=pass_mask,
                    other=0,
                ).to(tl.float32)
                _store_fp8_e4m3(
                    out_base,
                    q_heads[:, None] * hd + rope_dim + pass_offs[None, :],
                    pass_vals,
                    pass_mask,
                    FP8_MAX,
                )

        k_row = k_ptr + row_idx * k_row_stride
        k_out_row = k_out_ptr + row_idx * k_out_row_stride
        for k_base in tl.range(0, n_kh, BLOCK_KH):
            k_heads = tl.arange(0, BLOCK_KH)
            head_ok = (k_base + k_heads)[:, None] < n_kh
            base = k_row + k_base * hd
            out_base = k_out_row + k_base * hd

            half_offs = tl.arange(0, pad_half)
            half_mask = head_ok & (half_offs[None, :] < half)
            x1 = tl.load(
                base + k_heads[:, None] * hd + half_offs[None, :],
                mask=half_mask,
                other=0,
            ).to(tl.float32)
            x2 = tl.load(
                base + k_heads[:, None] * hd + half + half_offs[None, :],
                mask=half_mask,
                other=0,
            ).to(tl.float32)
            _store_fp8_e4m3(
                out_base,
                k_heads[:, None] * hd + half_offs[None, :],
                x1 * cos_row - x2 * sin_row,
                half_mask,
                FP8_MAX,
            )
            _store_fp8_e4m3(
                out_base,
                k_heads[:, None] * hd + half + half_offs[None, :],
                x2 * cos_row + x1 * sin_row,
                half_mask,
                FP8_MAX,
            )
            if pass_dim > 0:
                pass_offs = tl.arange(0, pad_pass)
                pass_mask = head_ok & (pass_offs[None, :] < pass_dim)
                pass_vals = tl.load(
                    base + k_heads[:, None] * hd + rope_dim + pass_offs[None, :],
                    mask=pass_mask,
                    other=0,
                ).to(tl.float32)
                _store_fp8_e4m3(
                    out_base,
                    k_heads[:, None] * hd + rope_dim + pass_offs[None, :],
                    pass_vals,
                    pass_mask,
                    FP8_MAX,
                )


def rope_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor = None,
    sin: torch.Tensor = None,
    cos_sin_cache: torch.Tensor = None,
    positions: torch.Tensor = None,
    rope_dim: int = -1,
    is_neox_style: bool = True,
    out_dtype: torch.dtype | None = None,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE and optionally write fixed-scale E4M3 outputs."""
    if out_dtype is not None:
        if out_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"Unsupported RoPE output dtype: {out_dtype}")
        if cos_sin_cache is None or positions is None:
            raise ValueError("FP8 RoPE output requires cos_sin_cache and positions")
        if rope_dim == -1:
            raise ValueError("FP8 RoPE output requires rope_dim")
        return _rope_forward_triton_fp8(
            q,
            k,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            rope_dim=rope_dim,
            is_neox_style=is_neox_style,
            q_out=q_out,
            k_out=k_out,
        )
    if q_out is not None or k_out is not None:
        raise ValueError("q_out and k_out require an FP8 output dtype")

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    num_tokens, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[1]
    BLOCK_SIZE_HEAD = _get_block_size_head(n_q_head, n_kv_head, is_neox_style)
    num_vectorcore = get_vectorcore_num()
    n_row = min(num_tokens, num_vectorcore)

    if cos_sin_cache is not None and positions is not None:
        assert positions.shape[0] == num_tokens
        assert rope_dim <= head_dim
        pad_rope_dim = triton.next_power_of_2(rope_dim)
        _triton_rope[(n_row,)](
            q,
            q.stride(0),
            k,
            k.stride(0),
            None,
            None,
            None,
            None,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            positions,
            num_tokens,
            n_q_head,
            n_kv_head,
            head_dim,
            rope_dim,
            pad_rope_dim,
            BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
            IS_NEOX_STYLE=is_neox_style,
            USE_COS_SIN=True,
        )
    elif cos is not None and sin is not None:
        assert cos.shape[0] == num_tokens and sin.shape[0] == num_tokens
        cos = cos.view(num_tokens, -1)
        sin = sin.view(num_tokens, -1)
        if rope_dim == -1:
            # If rope_dim is not specified, we assume that input cos/sin is not
            # duplicated to rope_dim, which means rope_dim == cos.shape[-1] * 2
            rope_dim = cos.shape[-1] * 2
        assert rope_dim <= head_dim
        pad_rope_dim = triton.next_power_of_2(rope_dim)
        _triton_rope[(n_row,)](
            q,
            q.stride(0),
            k,
            k.stride(0),
            cos,
            cos.stride(0),
            sin,
            sin.stride(0),
            None,
            None,
            None,
            num_tokens,
            n_q_head,
            n_kv_head,
            head_dim,
            rope_dim,
            pad_rope_dim,
            BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
            IS_NEOX_STYLE=is_neox_style,
            USE_COS_SIN=False,
        )
    else:
        raise ValueError(
            "Currently, rope_forward_triton supports passing:\n"
            "1. positions and original cos_sin_cache.\n"
            "2. cos and sin which are already selected by positions\n"
            "Please check whether you call rope_forward_triton correctly."
        )
    return q, k


def rope_forward_triton_siso(
    qk: torch.Tensor,
    cos: torch.Tensor = None,
    sin: torch.Tensor = None,
    cos_sin_cache: torch.Tensor = None,
    positions: torch.Tensor = None,
    rope_dim: int = -1,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not qk.is_contiguous():
        qk = qk.contiguous()

    num_tokens, n_head, head_dim = qk.shape
    assert rope_dim <= head_dim
    pad_rope_dim = triton.next_power_of_2(rope_dim)
    pad_n_head = triton.next_power_of_2(n_head)
    BLOCK_SIZE = pad_n_head
    num_vectorcore = get_vectorcore_num()
    n_row = min(num_tokens, num_vectorcore)

    if cos_sin_cache is not None and positions is not None:
        assert positions.shape[0] == num_tokens
        _triton_rope_siso[(n_row,)](
            qk,
            qk.stride(0),
            None,
            None,
            None,
            None,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            positions,
            num_tokens,
            n_head,
            head_dim,
            rope_dim,
            pad_n_head,
            pad_rope_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_NEOX_STYLE=is_neox_style,
            USE_COS_SIN=True,
        )
    elif cos is not None and sin is not None:
        assert cos.shape[0] == num_tokens and sin.shape[0] == num_tokens
        cos = cos.view(num_tokens, -1)
        sin = sin.view(num_tokens, -1)
        if rope_dim == -1:
            # If rope_dim is not specified, we assume that input cos/sin is not
            # duplicated to rope_dim, which means rope_dim == cos.shape[-1] * 2
            rope_dim = cos.shape[-1] * 2
        _triton_rope_siso[(n_row,)](
            qk,
            qk.stride(0),
            cos,
            cos.stride(0),
            sin,
            sin.stride(0),
            None,
            None,
            None,
            num_tokens,
            n_head,
            head_dim,
            rope_dim,
            pad_n_head,
            pad_rope_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_NEOX_STYLE=is_neox_style,
            USE_COS_SIN=False,
        )
    else:
        raise ValueError(
            "Currently, rope_forward_triton supports passing:\n"
            "1. positions and original cos_sin_cache.\n"
            "2. cos and sin which are already selected by positions\n"
            "Please check whether you call rope_forward_triton correctly."
        )
    return qk


def _rope_forward_triton_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool = True,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NeoX RoPE and write fixed-scale float8_e4m3fn outputs."""
    q = q.contiguous()
    k = k.contiguous()
    num_tokens, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[1]
    if k.shape[0] != num_tokens or k.shape[2] != head_dim:
        raise ValueError("q and k must share token and head dimensions")

    fp8_dtype = torch.float8_e4m3fn
    q_out = torch.empty_like(q, dtype=fp8_dtype) if q_out is None else q_out
    k_out = torch.empty_like(k, dtype=fp8_dtype) if k_out is None else k_out
    if q_out.dtype != fp8_dtype or k_out.dtype != fp8_dtype:
        raise TypeError("q_out and k_out must use torch.float8_e4m3fn")
    if not q_out.is_contiguous() or not k_out.is_contiguous():
        raise ValueError("q_out and k_out must be contiguous")
    if not is_neox_style:
        raise NotImplementedError("FP8 RoPE currently supports NeoX style only")
    if positions.shape[0] != num_tokens:
        raise ValueError("positions and q must contain the same number of tokens")
    if rope_dim > head_dim or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even and no larger than head_dim")

    pass_dim = head_dim - rope_dim
    pad_half = triton.next_power_of_2(rope_dim // 2)
    pad_pass = triton.next_power_of_2(pass_dim) if pass_dim > 0 else 1
    vector_cores = get_vectorcore_num()
    grid = min(num_tokens, max(vector_cores * 8, 256))
    _triton_rope_fp8[(grid,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        q_out,
        q_out.stride(0),
        k_out,
        k_out.stride(0),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        positions,
        num_tokens,
        n_q_head,
        n_kv_head,
        head_dim,
        rope_dim,
        pad_half,
        pad_pass,
        pass_dim,
        BLOCK_QH=_fp8_rope_head_block(n_q_head),
        BLOCK_KH=_fp8_rope_head_block(n_kv_head),
        FP8_MAX=_FP8_E4M3_MAX,
    )
    return q_out, k_out
