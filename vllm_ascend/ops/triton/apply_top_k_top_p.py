# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""apply_top_k_top_p: Triton implementation of top-k/top-p filtering
(+ optional fused softmax).

Semantics are strictly identical to CANN ``torch_npu.npu_top_k_top_p``
(ascending stable sort -> k-th largest threshold -> entries strictly below
the threshold set to -inf -> softmax for probabilities -> accumulate from
the smallest probability -> filter where cumsum <= 1-p with the last entry
always kept -> scatter back to the original order via the sort indices):

    sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)
    topKValue[b]  = sortedValue[b][V - k[b]]
    topKMask      = sortedValue < topKValue
    sortedValue   = where(topKMask, -inf, sortedValue)
    probsSum      = cumsum(softmax(sortedValue, dim=-1), dim=-1)
    topPMask[b][v]= probsSum[b][v] <= 1 - p[b];  topPMask[b][-1] = false
    sortedValue   = where(topPMask, -inf, sortedValue)
    out[b][sortedIndices[b][v]] = sortedValue[b][v]

Two output contracts (same pair of kernels, compile-time ``FUSED_SOFTMAX``
switch):

- :func:`apply_top_k_top_p` / :func:`apply_top_k_top_p_with_sorted`
  return original-order masked logits (kept entries = original logits,
  filtered entries = -inf), bit-exact with ``torch_npu.npu_top_k_top_p``
  (verified with ``torch.equal`` on randomized cases).
- :func:`fused_topk_topp_softmax` / :func:`fused_topk_topp_softmax_with_sorted`
  fuse the final softmax inside the kernel and return fp32 probabilities
  directly (masked entries = 0, each row sums to 1), equivalent to the
  production chain ``npu_top_k_top_p + softmax(dim=-1, dtype=float32)``.

Design notes:

  - k / p accept per-request [B] tensors directly (the native shape from
    vllm SamplingMetadata) — no ``k[0].item()``-style CPU sync, and the
    batch does not need uniform sampling parameters.
  - Filtered positions are pre-filled in one contiguous block via
    ``torch.full``/``torch.zeros``; the kernel only scatters the surviving
    elements (~ the kept count), avoiding V random stores per row.

Implementation notes:

  - kept set = ``s >= topKValue AND (probsSum > 1-p OR v == V-1)``. Both
    conditions are suffix-monotone, so the kept set is exactly ``v >= m``
    for a scalar boundary m. The scatter mask uses the comparison form
    ``offs >= m`` — a cumsum-derived boolean vector used directly as a
    scatter mask miscompiles on triton-ascend 3.2 (observed: the store
    collapses to a few elements). Do not change this back.
  - Two paths: ``V <= 4096`` loads the row in a single block;
    ``V > 4096`` goes tiled (denominator + top-k boundary -> locate the
    top-p boundary -> process only the surviving suffix).
  - k[b] <= 0 clamps the index to V-1 (keep only the max); k[b] >= V
    (vllm's disabled convention) clamps to 0 (no filtering).
"""

import torch
from vllm.triton_utils import tl, triton

__all__ = [
    "apply_top_k_top_p",
    "apply_top_k_top_p_with_sorted",
    "fused_topk_topp_softmax",
    "fused_topk_topp_softmax_with_sorted",
]

# Single-pass path upper bound: fp32 intermediates at BLOCK=8192 exceed the
# 192KB UB.
_SINGLE_PASS_MAX_V = 4096
# Tiled path tile size (one fp32 tile is ~8KB, well within the UB budget).
_TILED_BLOCK = 2048
_MIN_BLOCK = 16
_NUM_VECTORCORE = -1


def _get_npu_vectorcore_num() -> int:
    """Vector core count of the current device (40 on 910B3)."""
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE <= 0:
        device = torch.npu.current_device()
        _NUM_VECTORCORE = int(triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"])
    return _NUM_VECTORCORE


@triton.jit
def _apply_topk_topp_single_pass_kernel(
    sv_ptr,  # *sortedValue [B, V], ascending
    si_ptr,  # *sortedIndices [B, V], int32/int64 -> original vocab index
    k_ptr,  # *k [B], int32
    p_ptr,  # *p [B], fp32
    out_ptr,  # *out [B, V], original order, pre-filled with -inf (0 if FUSED_SOFTMAX)
    B,
    V,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,  # int32 row base when B*V < 2^31
    FUSED_SOFTMAX: tl.constexpr,  # True: scatter probs exp/E_kept; False: scatter raw logits
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

        # topKValue[b] = sortedValue[b][V - k[b]] (index clamped to [0, V-1])
        k_b = tl.load(k_ptr + row)
        k_idx = tl.minimum(tl.maximum(V - k_b, 0), V - 1)
        topk_value = tl.load(sv_ptr + row_base + k_idx).to(tl.float32)

        s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        # With ascending order the global max sits at the end; use it as the
        # softmax numerical-stability term.
        s_max = tl.load(sv_ptr + row_base + V - 1).to(tl.float32)

        # softmax (fp32) after topKMask set entries to -inf; the probs are
        # only used to derive the top-p threshold
        topk_mask = s < topk_value
        e = tl.where(mask & ~topk_mask, tl.exp(s - s_max), 0.0)
        probs = e / tl.sum(e, axis=0)

        # kept(v) = ~topKMask(v) AND (probsSum(v) > 1-p[b] OR v == V-1).
        # Both conditions are suffix-monotone, i.e. kept(v) <=> v >= m; use a
        # scalar boundary + comparison mask (same structure as the tiled
        # kernel) to avoid the triton-ascend miscompilation when a
        # cumsum-derived boolean vector is used directly as a scatter mask.
        p_b = tl.load(p_ptr + row)
        cum = tl.cumsum(probs, axis=0)
        m_p = tl.min(tl.where(mask & (cum > 1.0 - p_b), offs, V), axis=0)
        m_k = tl.min(tl.where(mask & ~topk_mask, offs, V), axis=0)
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        # Only scatter the surviving elements; the rest stays as pre-filled
        # (-inf or 0)
        kept = (offs >= m) & mask
        idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
        if FUSED_SOFTMAX:
            # Fused final softmax: denominator = exp sum over the kept set
            # (masked positions are pre-filled with 0)
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

        # pass 1: softmax denominator E = sum(exp(s - s_max)) over the top-k
        # survivors, plus the top-k survivor lower bound m_k (ascending order
        # + strictly-below-threshold filtering => the survivor set is a
        # suffix)
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

        # pass 2: m_p = first index whose ascending cumsum exceeds 1-p[b]
        # (V if there is none)
        running = 0.0
        m_p = V
        for t in range(0, V, BLOCK):
            if m_p == V:
                offs = t + offs_base
                mask = offs < V
                s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
                probs = tl.where(mask & (s >= topk_value), tl.exp(s - s_max), 0.0) / E
                tile_sum = tl.sum(probs, axis=0)
                if running + tile_sum > thr_p:
                    c = running + tl.cumsum(probs, axis=0)
                    hit = tl.min(tl.where(c > thr_p, offs_base, BLOCK), axis=0)
                    m_p = tl.where(hit < BLOCK, t + hit, V)
                running += tile_sum

        # kept(v) = [s>=topk_value] AND [probsSum > 1-p OR v == V-1]; both
        # conditions are suffix-monotone, i.e. kept(v) <=> v >= m
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        # pass 3: only process the surviving suffix; the rest stays as
        # pre-filled (-inf or 0)
        t_start = (m // BLOCK) * BLOCK

        if FUSED_SOFTMAX:
            # pass 3a: fused-softmax denominator = exp sum over the kept
            # suffix
            e_acc = tl.zeros([BLOCK], dtype=tl.float32)
            for t in range(t_start, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                e_acc += tl.where(kept, tl.exp(s - s_max), 0.0)
            e_inv = 1.0 / tl.sum(e_acc, axis=0)

            # pass 3b: scatter probabilities exp(s - s_max) / E_kept
            for t in range(t_start, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
                tl.store(out_ptr + row_base + idx, tl.exp(s - s_max) * e_inv, mask=kept)
        else:
            # scatter the raw logits (masked logits, CANN semantics)
            for t in range(t_start, V, BLOCK):
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
    """Normalize k to a [B] int32 device tensor; None -> V (top-k disabled).

    A [B] int32 tensor from vllm is a no-op (no extra op is launched).
    """
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
    """Normalize p to a [B] fp32 device tensor; None -> 1.0 (top-p disabled).

    A [B] fp32 tensor from vllm is a no-op (no extra op is launched).
    """
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
        # probability output is always fp32 (matches softmax(dtype=fp32));
        # masked positions are pre-filled with 0
        out = torch.zeros((B, V), device=sv.device, dtype=torch.float32)
    else:
        # filtered positions are pre-filled with -inf in one contiguous
        # block write; the kernel only scatters the surviving elements
        out = torch.full_like(sv, float("-inf"))

    core_num = _get_npu_vectorcore_num()
    if core_num >= B:
        grid, rows_per_prog = (B,), 1
    else:
        grid, rows_per_prog = (core_num,), triton.cdiv(B, core_num)

    use_i32 = B * V < 2**31
    # si is not cast to int32 on the host (saves a full-vocab copy); after
    # the in-kernel load the pointer arithmetic widens it automatically
    # (int64 index values < V < 2^31, so the value range is safe).

    if V <= _SINGLE_PASS_MAX_V:
        block = max(triton.next_power_of_2(V), _MIN_BLOCK)
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
            BLOCK=_TILED_BLOCK,
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
    """Top-k/top-p filtering on ascending sorted inputs; returns
    original-order masked logits.

    Args:
        sorted_values: [B, V] ascending sorted logits (values from
            torch.sort).
        sorted_indices: [B, V] matching sort indices (int32/int64).
        k: top-k keep count, scalar or [B] tensor; k[b] >= V disables
            filtering, None disables it.
        p: top-p threshold, scalar or [B] tensor in [0, 1]; None disables
            it.
    Returns:
        [B, V] original-order masked logits: kept entries hold the original
        logits, filtered entries are -inf.
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=False)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """End-to-end entry: ascending stable sort + top-k/top-p filtering;
    returns original-order masked logits.

    Args:
        logits: [B, V] raw logits (float32/bfloat16/float16).
        k / p: same as :func:`apply_top_k_top_p_with_sorted`.
    Returns:
        [B, V] original-order masked logits: kept entries hold the original
        logits, filtered entries are -inf.
    """
    sorted_values, sorted_indices = _sort_ascending(logits)
    return apply_top_k_top_p_with_sorted(sorted_values, sorted_indices, k, p)


def fused_topk_topp_softmax_with_sorted(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """Sorted inputs + top-k/top-p filtering + fused softmax; returns the
    original-order probability distribution.

    Equivalent to ``apply_top_k_top_p_with_sorted(...).softmax(dim=-1,
    dtype=float32)`` (masked -inf -> probability 0), but the softmax runs
    inside the kernel: surviving elements are scattered as
    exp(s - s_max) / E_kept and the rest stays pre-filled with 0.

    Returns:
        [B, V] fp32 probability distribution: masked entries = 0, each row
        sums to 1.
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=True)


def fused_topk_topp_softmax(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """End-to-end entry: ascending stable sort + top-k/top-p filtering +
    fused softmax.

    Same contract as the production chain
    ``torch_npu.npu_top_k_top_p(logits, k, p).softmax(dim=-1, dtype=float32)``.

    Args:
        logits: [B, V] raw logits (float32/bfloat16/float16).
        k / p: same as :func:`apply_top_k_top_p_with_sorted`.
    Returns:
        [B, V] fp32 probability distribution: masked entries = 0, each row
        sums to 1.
    """
    sorted_values, sorted_indices = _sort_ascending(logits)
    return fused_topk_topp_softmax_with_sorted(sorted_values, sorted_indices, k, p)
