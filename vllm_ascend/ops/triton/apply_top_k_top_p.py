# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""apply_top_k_top_p: Triton implementation of top-k/top-p filtering
(+ optional fused softmax), semantically identical to CANN
``torch_npu.npu_top_k_top_p``:

    sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)
    topKValue[b]  = sortedValue[b][V - k[b]]
    sortedValue   = where(sortedValue < topKValue, -inf, sortedValue)
    probsSum      = cumsum(softmax(sortedValue, dim=-1), dim=-1)
    topPMask[b][v]= probsSum[b][v] <= 1 - p[b];  topPMask[b][-1] = false
    sortedValue   = where(topPMask, -inf, sortedValue)
    out[b][sortedIndices[b][v]] = sortedValue[b][v]

Non-fused entry points return original-order masked logits (kept entries =
original logits, filtered entries = -inf), bit-exact with
``torch_npu.npu_top_k_top_p``. Fused entry points run the final softmax in
the kernel and return fp32 probabilities (masked entries = 0, each row sums
to 1).

Notes:
  - k / p accept per-request [B] tensors directly (vllm SamplingMetadata
    shape): no ``.item()`` CPU sync and no need for uniform sampling params.
  - Filtered positions are pre-filled on the host; the kernel only scatters
    the surviving elements.
  - The survivor set is suffix-monotone, i.e. kept(v) <=> v >= m for a
    scalar boundary m. The scatter mask MUST stay the comparison form
    ``offs >= m``: a cumsum-derived boolean vector used directly as a
    scatter mask miscompiles on triton-ascend 3.2. Do not change it back.
  - k[b] <= 0 clamps to V-1 (keep only the max); k[b] >= V (vllm's disabled
    convention) means no filtering.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_ub_size_bytes,
    get_vectorcore_num,
    init_device_properties_triton,
)

__all__ = [
    "apply_top_k_top_p",
    "apply_top_k_top_p_with_sorted",
    "fused_topk_topp_softmax",
    "fused_topk_topp_softmax_with_sorted",
]

# UB-driven tile planning. No fixed tile fits all Ascend chips (UB is 192 KB
# on 910B/910C, 248 KB on 950), so block sizes are derived from the runtime
# UB size (shared helper in triton_utils) using a conservative per-element
# footprint of the simultaneously live BLOCK-shaped tensors: ~36 B/elem for
# single-pass, ~48 B/elem for tiled. Up to 90% of the UB is usable; sizes
# are floored to the largest power of 2 in [16, 16384] and cached.
_UB_USABLE_FRACTION = 0.9
_SINGLE_PASS_BYTES_PER_ELEM = 36
_TILED_BYTES_PER_ELEM = 48
_BLOCK_FLOOR = 16
_BLOCK_CEIL = 16384

_tile_config_cache: tuple[int, int] | None = None


def _largest_pow2_le(value: int) -> int:
    block = _BLOCK_FLOOR
    while block * 2 <= value and block < _BLOCK_CEIL:
        block *= 2
    return block


def _get_tile_config() -> tuple[int, int]:
    """Return (single_pass_block, tiled_block) for the current device."""
    global _tile_config_cache
    if _tile_config_cache is None:
        init_device_properties_triton()
        ub_budget = int(get_ub_size_bytes() * _UB_USABLE_FRACTION)
        single_pass_block = _largest_pow2_le(
            max(_BLOCK_FLOOR, ub_budget // _SINGLE_PASS_BYTES_PER_ELEM)
        )
        tiled_block = _largest_pow2_le(
            max(_BLOCK_FLOOR, ub_budget // _TILED_BYTES_PER_ELEM)
        )
        _tile_config_cache = (single_pass_block, tiled_block)
    return _tile_config_cache


@triton.jit
def _apply_topk_topp_single_pass_kernel(
    sv_ptr,  # sortedValue [B, V], ascending
    si_ptr,  # sortedIndices [B, V], int32/int64
    k_ptr,  # k [B], int32
    p_ptr,  # p [B], fp32
    out_ptr,  # out [B, V], pre-filled -inf (or 0 with FUSED_SOFTMAX)
    B,
    V,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,  # int32 row offsets when B*V < 2^31
    FUSED_SOFTMAX: tl.constexpr,  # scatter probs vs raw logits
):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, B)
    offs = tl.arange(0, BLOCK)

    for row in range(row_start, row_end):
        mask = offs < V
        if USE_I32:
            row_base = row * V
        else:
            row_base = row.to(tl.int64) * V  # type: ignore[attr-defined]

        # topKValue[b] = sortedValue[b][V - k[b]], index clamped to [0, V-1]
        k_b = tl.load(k_ptr + row)
        k_idx = tl.minimum(tl.maximum(V - k_b, 0), V - 1)
        topk_value = tl.load(sv_ptr + row_base + k_idx).to(tl.float32)

        s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        s_max = tl.load(sv_ptr + row_base + V - 1).to(tl.float32)  # ascending => max at end

        # softmax (fp32) over the top-k survivors; only used for the top-p boundary
        topk_mask = s < topk_value
        e = tl.where(mask & ~topk_mask, tl.exp(s - s_max), 0.0)
        probs = e / tl.sum(e, axis=0)

        # survivor set is a suffix: kept(v) <=> v >= m (comparison mask on purpose)
        p_b = tl.load(p_ptr + row)
        cum = tl.cumsum(probs, axis=0)
        m_p = tl.min(tl.where(mask & (cum > 1.0 - p_b), offs, V), axis=0)
        m_k = tl.min(tl.where(mask & ~topk_mask, offs, V), axis=0)
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        kept = (offs >= m) & mask
        idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
        if FUSED_SOFTMAX:
            # denominator = exp sum over the kept entries
            e_kept = tl.sum(tl.where(kept, e, 0.0), axis=0)
            val = tl.where(kept, e / e_kept, 0.0)
        else:
            val = s
        tl.store(out_ptr + row_base + idx, val, mask=kept)


@triton.jit
def _apply_topk_topp_tiled_kernel(
    sv_ptr,
    si_ptr,
    k_ptr,
    p_ptr,
    out_ptr,
    B,
    V,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
    FUSED_SOFTMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, B)
    offs_base = tl.arange(0, BLOCK)

    for row in range(row_start, row_end):
        if USE_I32:
            row_base = row * V
        else:
            row_base = row.to(tl.int64) * V  # type: ignore[attr-defined]

        k_b = tl.load(k_ptr + row)
        p_b = tl.load(p_ptr + row)
        thr_p = 1.0 - p_b
        k_idx = tl.minimum(tl.maximum(V - k_b, 0), V - 1)
        topk_value = tl.load(sv_ptr + row_base + k_idx).to(tl.float32)
        s_max = tl.load(sv_ptr + row_base + V - 1).to(tl.float32)

        # pass 1: softmax denominator over top-k survivors + survivor lower bound m_k
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        m_k = V
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            alive = mask & (s >= topk_value)
            acc += tl.where(alive, tl.exp(s - s_max), 0.0)
            m_k = tl.minimum(m_k, tl.min(tl.where(alive, offs, V), axis=0))
        E = tl.sum(acc, axis=0)

        # pass 2: m_p = first index whose cumsum exceeds 1-p (V if none).
        # Branchless on purpose: 0-D tensor control flow inside the loop is
        # not portable across Triton backends.
        running = 0.0
        m_p = V
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            probs = tl.where(mask & (s >= topk_value), tl.exp(s - s_max), 0.0) / E
            tile_sum = tl.sum(probs, axis=0)
            c = running + tl.cumsum(probs, axis=0)
            hit = tl.min(tl.where(c > thr_p, offs_base, BLOCK), axis=0)
            m_p = tl.minimum(m_p, tl.where(hit < BLOCK, t + hit, V))
            running += tile_sum

        # survivor set is a suffix: kept(v) <=> v >= m
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        # pass 3: static loop start (0) + kept mask on purpose — avoids tensor
        # floor-division and a runtime range start (fragile on some backends).
        if FUSED_SOFTMAX:
            # pass 3a: fused-softmax denominator over the kept suffix
            e_acc = tl.zeros([BLOCK], dtype=tl.float32)
            for t in range(0, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                e_acc += tl.where(kept, tl.exp(s - s_max), 0.0)
            e_inv = 1.0 / tl.sum(e_acc, axis=0)

            # pass 3b: scatter probabilities
            for t in range(0, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
                tl.store(out_ptr + row_base + idx, tl.exp(s - s_max) * e_inv, mask=kept)
        else:
            for t in range(0, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=0.0).to(tl.float32)
                idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
                tl.store(out_ptr + row_base + idx, s, mask=kept)


def _validate_and_prepare(sorted_values: torch.Tensor, sorted_indices: torch.Tensor):
    """Validate and normalize the sorted inputs; returns (sv, si, B, V)."""
    if sorted_values.dim() != 2:
        raise ValueError(f"sorted_values must be 2D [B, V], got {sorted_values.dim()}D")
    if sorted_values.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"sorted_values only supports float32/bfloat16/float16, got {sorted_values.dtype}")
    if sorted_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"sorted_indices only supports int32/int64, got {sorted_indices.dtype}")
    if sorted_values.shape != sorted_indices.shape:
        raise ValueError(f"shape mismatch: {sorted_values.shape} vs {sorted_indices.shape}")
    if sorted_values.numel() == 0:
        raise ValueError("input tensor must not be empty")
    if sorted_values.device != sorted_indices.device:
        raise ValueError(f"device mismatch: {sorted_values.device} vs {sorted_indices.device}")
    if sorted_values.device.type != "npu":
        raise ValueError(f"input must be on npu, got {sorted_values.device}")

    sv = sorted_values if sorted_values.is_contiguous() else sorted_values.contiguous()
    si = sorted_indices if sorted_indices.is_contiguous() else sorted_indices.contiguous()
    B, V = sv.shape
    if V >= 2**31:
        raise ValueError(f"vocab size {V} too large (>= 2^31)")
    return sv, si, B, V


def _normalize_k(k, B: int, V: int, device) -> torch.Tensor:
    """Normalize k to a [B] int32 device tensor; None -> V (top-k disabled)."""
    if k is None:
        return torch.full((B,), V, dtype=torch.int32, device=device)
    if isinstance(k, bool) or not isinstance(k, (int, float, torch.Tensor)):
        raise TypeError(f"k must be None or int or [B] tensor, got {type(k).__name__}")
    if isinstance(k, float):
        if not k.is_integer():
            raise ValueError(f"k must be an integer, got {k}")
        k = int(k)
    if isinstance(k, int):
        return torch.full((B,), k, dtype=torch.int32, device=device)
    if k.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"k tensor only supports int32/int64, got {k.dtype}")
    if k.numel() == 1:
        k = k.reshape(1).expand(B)
    if k.shape != (B,):
        raise ValueError(f"k tensor must have shape [{B}], got {tuple(k.shape)}")
    return k.to(device=device, dtype=torch.int32).contiguous()


def _normalize_p(p, B: int, device) -> torch.Tensor:
    """Normalize p to a [B] fp32 device tensor; None -> 1.0 (top-p disabled)."""
    if p is None:
        return torch.ones((B,), dtype=torch.float32, device=device)
    if isinstance(p, bool) or not isinstance(p, (int, float, torch.Tensor)):
        raise TypeError(f"p must be None or float or [B] tensor, got {type(p).__name__}")
    if isinstance(p, (int, float)):
        return torch.full((B,), float(p), dtype=torch.float32, device=device)
    if not p.is_floating_point():
        raise ValueError(f"p tensor must be floating point, got {p.dtype}")
    if p.numel() == 1:
        p = p.reshape(1).expand(B)
    if p.shape != (B,):
        raise ValueError(f"p tensor must have shape [{B}], got {tuple(p.shape)}")
    return p.to(device=device, dtype=torch.float32).contiguous()


def _launch(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k,
    p,
    fused_softmax: bool,
) -> torch.Tensor:
    """Validate inputs, normalize k/p, pre-fill out, and launch the kernel."""
    sv, si, B, V = _validate_and_prepare(sorted_values, sorted_indices)
    k_t = _normalize_k(k, B, V, sv.device)
    p_t = _normalize_p(p, B, sv.device)

    if fused_softmax:
        # fp32 probs, masked positions pre-filled 0
        out = torch.zeros((B, V), device=sv.device, dtype=torch.float32)
    else:
        # kernel only scatters survivors; everything else stays -inf
        out = torch.full_like(sv, float("-inf"))

    single_pass_block, tiled_block = _get_tile_config()
    core_num = get_vectorcore_num()
    if core_num >= B:
        grid, rows_per_prog = (B,), 1
    else:
        grid, rows_per_prog = (core_num,), triton.cdiv(B, core_num)

    use_i32 = B * V < 2**31
    # si is not cast on the host (saves a full-vocab copy); index values stay < V < 2^31.

    if V <= single_pass_block:
        block = max(triton.next_power_of_2(V), _BLOCK_FLOOR)
        _apply_topk_topp_single_pass_kernel[grid](
            sv,
            si,
            k_t,
            p_t,
            out,
            B,
            V,
            rows_per_prog,
            BLOCK=block,
            USE_I32=use_i32,
            FUSED_SOFTMAX=fused_softmax,
        )
    else:
        _apply_topk_topp_tiled_kernel[grid](
            sv,
            si,
            k_t,
            p_t,
            out,
            B,
            V,
            rows_per_prog,
            BLOCK=tiled_block,
            USE_I32=use_i32,
            FUSED_SOFTMAX=fused_softmax,
        )
    return out


def _sort_ascending(logits: torch.Tensor):
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, V], got {logits.dim()}D")
    if logits.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits.device}")
    return torch.sort(logits, dim=-1, descending=False, stable=True)


def apply_top_k_top_p_with_sorted(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """Top-k/top-p filtering on ascending sorted inputs.

    Args:
        sorted_values/sorted_indices: [B, V] outputs of ``torch.sort``.
        k: keep count, scalar or [B] tensor; k[b] >= V or None disables.
        p: threshold in [0, 1], scalar or [B] tensor; None disables.

    Returns original-order masked logits (kept = original, filtered = -inf).
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=False)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """Ascending stable sort + top-k/top-p filtering. k/p as in
    :func:`apply_top_k_top_p_with_sorted`; returns original-order masked
    logits (kept = original, filtered = -inf).
    """
    sorted_values, sorted_indices = _sort_ascending(logits)
    return apply_top_k_top_p_with_sorted(sorted_values, sorted_indices, k, p)


def fused_topk_topp_softmax_with_sorted(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """Top-k/top-p on sorted inputs with the final softmax fused in the
    kernel. k/p as in :func:`apply_top_k_top_p_with_sorted`.

    Returns [B, V] fp32 probabilities (masked = 0, each row sums to 1),
    equivalent to ``...npu_top_k_top_p(...).softmax(dim=-1, dtype=fp32)``.
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=True)


def fused_topk_topp_softmax(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """Sort + top-k/top-p + fused softmax. Same contract as
    ``torch_npu.npu_top_k_top_p(...).softmax(dim=-1, dtype=fp32)``; returns
    [B, V] fp32 probabilities (masked = 0, each row sums to 1).
    """
    sorted_values, sorted_indices = _sort_ascending(logits)
    return fused_topk_topp_softmax_with_sorted(sorted_values, sorted_indices, k, p)
