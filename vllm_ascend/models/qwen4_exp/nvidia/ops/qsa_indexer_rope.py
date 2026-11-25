# SPDX-License-Identifier: Apache-2.0
"""QSA-indexer MRoPE helpers that preserve the materialized Norm boundary."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _qsa_merge_mrope_cache_kernel(
    positions_ptr,
    cache_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    stride_axis,
    stride_token,
    stride_cache,
    BLOCK_T: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    pair = tl.arange(0, 32)
    valid = token < num_tokens

    pos_t = tl.load(positions_ptr + token * stride_token, mask=valid)
    pos_h = tl.load(
        positions_ptr + stride_axis + token * stride_token,
        mask=valid,
    )
    pos_w = tl.load(
        positions_ptr + 2 * stride_axis + token * stride_token,
        mask=valid,
    )
    h_axis = ((pair % 3) == 1)[None, :]
    w_axis = (((pair % 3) == 2) & (pair < 30))[None, :]
    selected_position = tl.where(
        h_axis,
        pos_h[:, None],
        tl.where(w_axis, pos_w[:, None], pos_t[:, None]),
    )

    mask = valid[:, None]
    cos = tl.load(
        cache_ptr + selected_position * stride_cache + pair[None, :],
        mask=mask,
    )
    sin = tl.load(
        cache_ptr + selected_position * stride_cache + 32 + pair[None, :],
        mask=mask,
    )
    output_offset = token[:, None] * 32 + pair[None, :]
    tl.store(cos_ptr + output_offset, cos, mask=mask)
    tl.store(sin_ptr + output_offset, sin, mask=mask)


def qsa_merge_mrope_cos_sin(
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge Qwen MRoPE cache rows for an already-normalized BF16 tensor."""
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("QSA MRoPE positions must have shape [3, num_tokens]")
    if cache.ndim != 2 or cache.shape[1] != 64:
        raise ValueError("QSA MRoPE cache must have shape [max_position, 64]")

    num_tokens = positions.shape[1]
    cos = torch.empty((num_tokens, 32), dtype=cache.dtype, device=cache.device)
    sin = torch.empty_like(cos)
    block_t = 16
    _qsa_merge_mrope_cache_kernel[(triton.cdiv(num_tokens, block_t),)](
        positions,
        cache,
        cos,
        sin,
        num_tokens,
        positions.stride(0),
        positions.stride(1),
        cache.stride(0),
        BLOCK_T=block_t,
    )
    return cos, sin


__all__ = ["qsa_merge_mrope_cos_sin"]
