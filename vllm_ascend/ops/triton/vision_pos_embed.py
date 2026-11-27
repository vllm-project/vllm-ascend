# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Qwen learned visual position interpolation (upstream vLLM #37948)."""

import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON, tl, triton

# Bound vector working storage independently of the tower's hidden dimension.
_BLOCK_SIZE = 1024
_ND_FORMAT = 2


if HAS_TRITON:

    @triton.jit(do_not_specialize=["h", "w", "total_elements", "h_scale", "w_scale"])
    def _vision_pos_embed_kernel(
        embed_ptr,
        output_ptr,
        h,
        w,
        total_elements,
        h_scale,
        w_scale,
        NUM_GRID: tl.constexpr,
        MERGE: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        spatial = (offsets // HIDDEN) % (h * w)
        channel = offsets % HIDDEN
        block = spatial // (MERGE * MERGE)
        local = spatial % (MERGE * MERGE)
        row = block // (w // MERGE) * MERGE + local // MERGE
        col = block % (w // MERGE) * MERGE + local % MERGE
        # Like linspace, compute the second half from the endpoint. Clamp both
        # neighbors so floating-point rounding cannot read past the table.
        y = tl.where(row < h // 2, row * h_scale, NUM_GRID - 1 - (h - 1 - row) * h_scale)
        x = tl.where(col < w // 2, col * w_scale, NUM_GRID - 1 - (w - 1 - col) * w_scale)
        y = tl.where(h == 1, 0.0, tl.minimum(tl.maximum(y, 0.0), NUM_GRID - 1))
        x = tl.where(w == 1, 0.0, tl.minimum(tl.maximum(x, 0.0), NUM_GRID - 1))
        y0 = y.to(tl.int32)
        x0 = x.to(tl.int32)
        y1 = tl.minimum(y0 + 1, NUM_GRID - 1)
        x1 = tl.minimum(x0 + 1, NUM_GRID - 1)
        dy = y - y0
        dx = x - x0
        w11 = dy * dx
        w10 = dy - w11
        w01 = dx - w11
        w00 = 1.0 - dy - w01
        dtype = output_ptr.dtype.element_ty
        e00 = tl.load(embed_ptr + (y0 * NUM_GRID + x0) * HIDDEN + channel, mask, other=0)
        e01 = tl.load(embed_ptr + (y0 * NUM_GRID + x1) * HIDDEN + channel, mask, other=0)
        e10 = tl.load(embed_ptr + (y1 * NUM_GRID + x0) * HIDDEN + channel, mask, other=0)
        e11 = tl.load(embed_ptr + (y1 * NUM_GRID + x1) * HIDDEN + channel, mask, other=0)
        # Native rounds each in-place product to weight dtype before sum,
        # whose accumulator is FP32 for FP16/BF16. Do not fuse those roundings.
        p00 = (e00.to(tl.float32) * w00.to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        p01 = (e01.to(tl.float32) * w01.to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        p10 = (e10.to(tl.float32) * w10.to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        p11 = (e11.to(tl.float32) * w11.to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        tl.store(output_ptr + offsets, p00 + p01 + p10 + p11, mask)


def supports_vision_pos_embed(weight: torch.Tensor, dtype: torch.dtype) -> bool:
    """Layout/dtype gate; the caller additionally excludes 310P devices."""
    return (
        HAS_TRITON
        and weight.device.type == "npu"
        and weight.ndim == 2
        and weight.is_contiguous()
        and torch_npu.get_npu_format(weight) == _ND_FORMAT
        and weight.shape[1] > 0
        and weight.dtype == dtype
        and dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def vision_pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return contiguous [t*h*w, D] in merge-block order without mutation.

    Only contiguous NPU tables with matching input/output floating dtype are
    supported. Unsupported model inputs must use upstream's native fallback.
    """
    if not supports_vision_pos_embed(embed_weight, dtype):
        raise ValueError("vision position interpolation requires a contiguous NPU table and matching floating dtype")
    if min(t, h, w, num_grid_per_side, m_size) <= 0 or h % m_size or w % m_size:
        raise ValueError("positive grid dimensions and spatial merge divisibility are required")
    if embed_weight.shape[0] != num_grid_per_side**2:
        raise ValueError("position table must contain num_grid_per_side**2 rows")
    hidden = embed_weight.shape[1]
    total = t * h * w * hidden
    # Kernel offsets use int32; do not silently wrap a large allocation.
    if max(total, embed_weight.numel()) > 2**31 - _BLOCK_SIZE:
        raise ValueError("position interpolation exceeds int32 indexing")
    output = torch.empty((t * h * w, hidden), device=embed_weight.device, dtype=dtype)
    _vision_pos_embed_kernel[(triton.cdiv(total, _BLOCK_SIZE),)](
        embed_weight,
        output,
        h,
        w,
        total,
        (num_grid_per_side - 1) / (h - 1) if h > 1 else 0.0,
        (num_grid_per_side - 1) / (w - 1) if w > 1 else 0.0,
        num_grid_per_side,
        m_size,
        hidden,
        _BLOCK_SIZE,
        enable_fp_fusion=False,
    )
    return output
