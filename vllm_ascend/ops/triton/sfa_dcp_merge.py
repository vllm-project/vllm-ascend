# SPDX-License-Identifier: Apache-2.0
"""FP32 local merge for the small-batch DCP exchange."""

from __future__ import annotations

import math

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    from triton.language.extra.cann import extension  # type: ignore[import-untyped]

    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

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

    @triton.jit(
        do_not_specialize=[
            "lse_s1",
            "head_count",
            "total_tiles",
            "num_programs",
        ]
    )
    def _fused_merge_kernel(
        out_ptr,
        lse_ptr,
        result_ptr,
        out_s0,
        out_s1,
        out_s2,
        LSE_RANK_STRIDE: tl.constexpr,
        lse_s1,
        head_count,
        head_dim,
        total_tiles,
        num_programs,
        RANKS: tl.constexpr,
        NATIVE_LAYOUT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rank_ids = tl.arange(0, RANKS)
        for tile in tl.range(pid, total_tiles, num_programs):
            row = tile // tl.cdiv(head_dim, BLOCK_D)
            dim = (tile % tl.cdiv(head_dim, BLOCK_D)) * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = dim < head_dim
            token = row // head_count
            head = row % head_count
            if NATIVE_LAYOUT:
                lse_offsets = rank_ids * LSE_RANK_STRIDE + head * lse_s1 + token
            else:
                lse_offsets = rank_ids * LSE_RANK_STRIDE + token * lse_s1 + head
            raw_lse = tl.load(lse_ptr + lse_offsets).to(tl.float32)
            finite = (raw_lse == raw_lse) & (raw_lse > float("-inf")) & (raw_lse < float("inf"))
            safe_lse = tl.where(finite, raw_lse, float("-inf"))
            max_lse = tl.max(safe_lse, axis=0)
            # An all-invalid row must not evaluate -inf - (-inf), even in a
            # branch later discarded by tl.where.
            safe_max = tl.where(max_lse > float("-inf"), max_lse, 0.0)
            shifted = tl.where(finite, safe_lse - safe_max, 0.0)
            exponent = tl.where(finite, tl.exp(shifted), 0.0)
            denominator = tl.sum(exponent, axis=0)
            safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
            weights = exponent / safe_denominator
            result = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for rank in tl.static_range(0, RANKS):
                weight = get_element(weights, (rank,))
                if NATIVE_LAYOUT:
                    out_offsets = rank * out_s0 + head * out_s1 + token * out_s2 + dim
                else:
                    out_offsets = rank * out_s0 + token * out_s1 + head * out_s2 + dim
                values = tl.load(out_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
                # Match upstream: invalid-rank NaN/Inf outputs must not leak
                # through a zero LSE weight (NaN * 0 is still NaN).
                values = tl.where(get_element(finite, (rank,)), values, 0.0)
                result += values * weight
            tl.store(result_ptr + (token * head_count + head) * head_dim + dim, result, mask=mask)


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
    init_device_properties_triton()
    programs = min(tiles, get_vectorcore_num())
    _fused_merge_kernel[(programs,)](
        out_recv,
        lse_recv,
        result,
        # Eligibility guarantees unit innermost strides. The result is always
        # allocated contiguous, so those strides need no kernel arguments.
        *out_recv.stride()[:3],
        *lse_recv.stride()[:2],
        heads,
        head_dim,
        total_tiles=tiles,
        num_programs=programs,
        RANKS=ranks,
        NATIVE_LAYOUT=native_layout,
        BLOCK_D=block_d,
    )
    return result
