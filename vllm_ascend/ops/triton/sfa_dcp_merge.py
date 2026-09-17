# SPDX-License-Identifier: Apache-2.0
"""FP32 local merge for the small-batch DCP exchange."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cann import extension

    get_element = extension.get_element
except ImportError:  # CPU-only test environments retain exact torch semantics.
    triton = None
    tl = None
    get_element = None


_SUPPORTED_RANKS = (2, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def torch_merge(out_recv: torch.Tensor, lse_recv: torch.Tensor, token_dim: int) -> torch.Tensor:
    """Reference merge with upstream invalid-rank masking semantics."""
    if out_recv.ndim != 4 or lse_recv.ndim != 3 or out_recv.shape[:3] != lse_recv.shape:
        raise RuntimeError(
            "DCP output merge expects matching rank/token/head dimensions, "
            f"got {tuple(out_recv.shape)} and {tuple(lse_recv.shape)}."
        )
    if token_dim not in (1, 2):
        raise RuntimeError(f"DCP output merge token_dim must be 1 or 2, got {token_dim}.")
    finite = torch.isfinite(lse_recv)
    lse = lse_recv.masked_fill(~finite, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(lse, dim=0), nan=0.0)
    values = out_recv.to(lse.dtype).masked_fill(~finite.unsqueeze(-1), 0.0)
    output = (values * weights.unsqueeze(-1)).sum(dim=0)
    return output.movedim(token_dim - 1, 0).contiguous()


if triton is not None:

    @triton.jit
    def _fused_merge_kernel(
        out_ptr,
        lse_ptr,
        result_ptr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        lse_s0: tl.constexpr,
        lse_s1: tl.constexpr,
        lse_s2: tl.constexpr,
        result_s0: tl.constexpr,
        result_s1: tl.constexpr,
        result_s2: tl.constexpr,
        rows: tl.constexpr,
        head_count: tl.constexpr,
        head_dim: tl.constexpr,
        total_tiles: tl.constexpr,
        num_programs: tl.constexpr,
        ranks: tl.constexpr,
        native_layout: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rank_ids = tl.arange(0, ranks)
        for tile in tl.range(pid, total_tiles, num_programs):
            row = tile // tl.cdiv(head_dim, block_d)
            dim = (tile % tl.cdiv(head_dim, block_d)) * block_d + tl.arange(0, block_d)
            mask = dim < head_dim
            token = row // head_count
            head = row % head_count
            if native_layout:
                lse_offsets = rank_ids * lse_s0 + head * lse_s1 + token * lse_s2
            else:
                lse_offsets = rank_ids * lse_s0 + token * lse_s1 + head * lse_s2
            raw_lse = tl.load(lse_ptr + lse_offsets).to(tl.float32)
            finite = (raw_lse == raw_lse) & (raw_lse > float("-inf")) & (raw_lse < float("inf"))
            safe_lse = tl.where(finite, raw_lse, float("-inf"))
            max_lse = tl.max(safe_lse, axis=0)
            shifted = tl.where(finite, safe_lse - max_lse, 0.0)
            exponent = tl.where(finite, tl.exp(shifted), 0.0)
            denominator = tl.sum(exponent, axis=0)
            safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
            weights = exponent / safe_denominator
            result = tl.zeros((block_d,), dtype=tl.float32)
            for rank in tl.static_range(0, ranks):
                weight = get_element(weights, (rank,))
                if native_layout:
                    out_offsets = rank * out_s0 + head * out_s1 + token * out_s2 + dim * out_s3
                else:
                    out_offsets = rank * out_s0 + token * out_s1 + head * out_s2 + dim * out_s3
                values = tl.load(out_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
                # Match upstream: invalid-rank NaN/Inf outputs must not leak
                # through a zero LSE weight (NaN * 0 is still NaN).
                values = tl.where(get_element(finite, (rank,)), values, 0.0)
                result += values * weight
            tl.store(result_ptr + token * result_s0 + head * result_s1 + dim * result_s2, result, mask=mask)


def _fast_path(out_recv: torch.Tensor, lse_recv: torch.Tensor, token_dim: int) -> bool:
    if triton is None or get_element is None or out_recv.device.type != "npu" or out_recv.device != lse_recv.device:
        return False
    if out_recv.ndim != 4 or lse_recv.ndim != 3 or 0 in out_recv.shape or 0 in lse_recv.shape:
        return False
    if out_recv.shape[0] not in _SUPPORTED_RANKS or out_recv.dtype not in _SUPPORTED_DTYPES:
        return False
    if lse_recv.dtype != torch.float32 or token_dim not in (1, 2):
        return False
    return not (
        out_recv.shape[:3] != lse_recv.shape
        or any(stride < 0 for stride in out_recv.stride())
        or any(stride < 0 for stride in lse_recv.stride())
        or out_recv.stride(-1) != 1
        or lse_recv.stride(-1) != 1
    )


def fused_merge(out_recv: torch.Tensor, lse_recv: torch.Tensor, token_dim: int) -> torch.Tensor:
    """Fuse only the post-All2All local math; unsupported inputs use torch."""
    if not _fast_path(out_recv, lse_recv, token_dim):
        return torch_merge(out_recv, lse_recv, token_dim)
    ranks = out_recv.shape[0]
    if token_dim == 2:
        _, heads, tokens, head_dim = out_recv.shape
        native_layout = True
    else:
        _, tokens, heads, head_dim = out_recv.shape
        native_layout = False
    result = torch.empty((tokens, heads, head_dim), dtype=torch.float32, device=out_recv.device)
    # Rank-wise accumulation avoids the failed R×D materialization. D512 now
    # holds one FP32 accumulator plus one rank vector; retry verifies compiler UB.
    block_d = min(triton.next_power_of_2(head_dim), 512)
    tiles = tokens * heads * math.ceil(head_dim / block_d)
    properties = triton.runtime.driver.active.utils.get_device_properties(out_recv.device.index)
    programs = min(tiles, properties["num_vectorcore"])
    _fused_merge_kernel[(programs,)](
        out_recv,
        lse_recv,
        result,
        *out_recv.stride(),
        *lse_recv.stride(),
        *result.stride(),
        tokens * heads,
        heads,
        head_dim,
        total_tiles=tiles,
        num_programs=programs,
        ranks=ranks,
        native_layout=native_layout,
        block_d=block_d,
    )
    return result
