# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""apply_top_k_top_p: Triton top-k/top-p filtering of LM-head logits.

Two top-p semantics (``top_p_mode``):

- ``full_vocab`` (default): cutoff from the full-vocabulary softmax before
  top-k masking; fp32 bit-exact with the legacy 0.29/0.32 reference.
- ``topk_renorm``: top-k first, cutoff after re-normalizing over survivors;
  bit-exact with CANN ``torch_npu.npu_top_k_top_p``.

    sortedValue, sortedIndices = sort(logits, descending=false, stable=true)
    topKValue[b] = sortedValue[b][V - k[b]]
    probsSum = cumsum(softmax(...), -1)
    topPMask = probsSum <= 1 - p;  topPMask[:, -1] = false
    out[sortedIndices] = where(topPMask, -inf, sortedValue)

All entries return original-order masked logits.

Dispatch (full_vocab, k < V): ``torch.topk`` gives the k-th value, then two
linear passes compute the denominator and write the mask, skipping the [B, V]
sort. One row is split across ``C = max(1, cores // B)`` programs; pass 1
stores per-chunk partials and pass 2 reduces them in chunk order, so the
accumulation order matches the sort path. Everything else (k >= V,
``topk_renorm``) uses sort + the tiled kernel.

Notes:
- k / p are per-request [B] tensors: no ``.item()`` CPU sync.
- full_vocab keeps ``s >= cutoff`` (ties preserved); topk_renorm keeps
  ``offs >= m``. The value form is required because a cumsum-derived boolean
  scatter mask miscompiles on triton-ascend 3.2.
- k <= 0 clamps to V-1; k >= V disables top-k.
"""

from typing import Literal

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc
from vllm_ascend.ops.triton.triton_utils import (
    get_ub_size_bytes,
    get_vectorcore_num,
    init_device_properties_triton,
)

__all__ = [
    "apply_top_k_top_p",
    "is_triton_apply_topk_topp_supported",
    "is_triton_topk_topp_profitable",
]

TopPMode = Literal["full_vocab", "topk_renorm"]
_VALID_TOPP_MODES = ("full_vocab", "topk_renorm")

# Tile size is derived from the runtime UB (192 KB on A2/A3, 248 KB on A5)
# using the live BLOCK-shaped footprint (52 B/elem), 90% usable, floored to a
# power of two in [16, 16384] and cached per device.
_UB_USABLE_FRACTION = 0.9
_TILED_BYTES_PER_ELEM = 52
_BLOCK_FLOOR = 16
_BLOCK_CEIL = 16384

# Measured A5 crossover for the topk_renorm fallback (full API vs PyTorch):
# below these row counts the PyTorch path wins because the kernel cannot split
# one vocab row. Re-calibrate on A2/A3 before relaxing the thresholds.
_GATE_V_LARGE = 65536
_GATE_B_MIN_LARGE_V = 16
_GATE_B_MIN_SMALL_V = 8

_tiled_block_cache: dict[int, int] = {}

# No-sort fast path: the full-vocab cutoff follows from the k-th value and the
# survivor mass, so topk + two linear scans replace the [B, V] sort.
_NO_SORT_TOPK_CAP = 256


def _largest_pow2_le(value: int) -> int:
    block = _BLOCK_FLOOR
    while block * 2 <= value and block < _BLOCK_CEIL:
        block *= 2
    return block


def _get_tiled_block(device: int | None = None) -> int:
    """Return the UB-derived tile block for the (current) device."""
    if device is None:
        device = torch.npu.current_device()
    if device not in _tiled_block_cache:
        init_device_properties_triton()
        ub_budget = int(get_ub_size_bytes() * _UB_USABLE_FRACTION)
        _tiled_block_cache[device] = _largest_pow2_le(max(_BLOCK_FLOOR, ub_budget // _TILED_BYTES_PER_ELEM))
    return _tiled_block_cache[device]


def _validate_top_p_mode(top_p_mode: str) -> None:
    if top_p_mode not in _VALID_TOPP_MODES:
        raise ValueError(f"top_p_mode must be one of {_VALID_TOPP_MODES}, got {top_p_mode!r}")


def is_triton_apply_topk_topp_supported() -> bool:
    """Hardware gate: triton-ascend does not support Atlas 310P.

    Returns False (instead of raising) on devices that cannot be resolved,
    so callers can fall back to the PyTorch path on unexpected hardware.
    """
    try:
        import torch_npu

        soc_version = int(torch_npu.npu.get_soc_version())
        return device_type_from_runtime_soc(soc_version) is not AscendDeviceType._310P
    except Exception:
        return False


def is_triton_topk_topp_profitable(B: int, V: int, *, disable_gate: bool = False) -> bool:
    """Batch-size gate for sampler wiring (direct op callers are not gated).

    Below the measured crossover the PyTorch path wins because the kernel
    cannot split one vocab row. ``disable_gate`` is the benchmark escape
    hatch (``additional_config.triton_topk_topp_disable_gate``); keep it
    False in production.
    """
    if disable_gate:
        return True
    b_min = _GATE_B_MIN_LARGE_V if V >= _GATE_V_LARGE else _GATE_B_MIN_SMALL_V
    return b_min <= B


@triton.jit(do_not_specialize=["B", "V", "rows_per_prog"])
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
    TOPP_FULL_VOCAB: tl.constexpr,
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
        # All--inf rows would compute exp(-inf - -inf) = NaN; clamp the
        # subtraction base. NaN rows deliberately stay NaN (fail loud).
        dead = s_max == -float("inf")
        nan_row = s_max != s_max
        s_max_c = tl.where(dead, 0.0, s_max)

        # pass 1: full-vocab denominator (acc_all) + top-k survivor mass/bound.
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc_all = tl.zeros([BLOCK], dtype=tl.float32)
        m_k = V
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            e = tl.exp(s - s_max_c)
            alive = mask & (s >= topk_value)
            acc += tl.where(alive, e, 0.0)
            acc_all += tl.where(mask, e, 0.0)
            m_k = tl.minimum(m_k, tl.min(tl.where(alive, offs, V), axis=0))
        E = tl.sum(acc, axis=0)
        E_all = tl.sum(acc_all, axis=0)

        # pass 2: m_p = first index whose cumsum exceeds 1-p (V if none).
        # Branchless: 0-D tensor control flow is not portable across backends.
        running = 0.0
        m_p = V
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            e = tl.exp(s - s_max_c)
            if TOPP_FULL_VOCAB:
                probs = tl.where(mask, e, 0.0) / E_all
            else:
                probs = tl.where(mask & (s >= topk_value), e, 0.0) / E
            tile_sum = tl.sum(probs, axis=0)
            c = running + tl.cumsum(probs, axis=0)
            hit = tl.min(tl.where(c > thr_p, offs_base, BLOCK), axis=0)
            m_p = tl.minimum(m_p, tl.where(hit < BLOCK, t + hit, V))
            running += tile_sum

        # Cutoff: full_vocab keeps s >= cutoff (ties); topk_renorm keeps offs >= m.
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)
        cutoff = tl.load(sv_ptr + row_base + m).to(tl.float32)

        # pass 3: static range + kept mask (avoids tensor floor-div and runtime
        # range starts). Dead rows write nothing; NaN rows keep one NaN.
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            if TOPP_FULL_VOCAB:
                kept = (s >= cutoff) & mask
                kept = kept | (nan_row & (offs == V - 1))
            else:
                kept = (offs >= m) & mask
            kept = kept & ~dead
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
        raise ValueError(f"k tensor must be shape [{B}], got {tuple(k.shape)}")
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
        raise ValueError(f"p tensor must be shape [{B}], got {tuple(p.shape)}")
    return p.to(device=device, dtype=torch.float32).contiguous()


def _launch(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k,
    p,
    top_p_mode: TopPMode = "full_vocab",
) -> torch.Tensor:
    """Validate inputs, normalize k/p, pre-fill out, and launch the kernel."""
    _validate_top_p_mode(top_p_mode)
    sv, si, B, V = _validate_and_prepare(sorted_values, sorted_indices)
    k_t = _normalize_k(k, B, V, sv.device)
    p_t = _normalize_p(p, B, sv.device)

    # kernel only scatters survivors; everything else stays -inf
    out = torch.full_like(sv, float("-inf"))

    # Unified tiled kernel for every V; shrink the block for small rows so
    # a 1k vocab does not pay for a 4096-element tile.
    block = min(_get_tiled_block(sv.device.index), max(triton.next_power_of_2(V), _BLOCK_FLOOR))
    core_num = get_vectorcore_num()
    if core_num >= B:
        grid, rows_per_prog = (B,), 1
    else:
        grid, rows_per_prog = (core_num,), triton.cdiv(B, core_num)

    use_i32 = B * V < 2**31
    # si is not cast on the host (saves a full-vocab copy); index values stay < V < 2^31.

    _apply_topk_topp_tiled_kernel[grid](
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
        TOPP_FULL_VOCAB=top_p_mode == "full_vocab",
    )
    return out


@triton.jit(do_not_specialize=["B", "V", "C", "T"])
def _apply_topk_topp_split_pass1_kernel(
    logits_ptr,
    topk_ptr,
    k_ptr,
    partial_ptr,
    nan_ptr,
    B,
    V,
    C,
    T,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
):
    """Per-(row, chunk) partial sums over T consecutive BLOCK tiles."""
    pid = tl.program_id(0)
    row = pid // C
    chunk = pid % C
    offs_base = tl.arange(0, BLOCK)

    if USE_I32:
        row_base = row * V
    else:
        row_base = row.to(tl.int64) * V  # type: ignore[attr-defined]

    maxv = tl.load(topk_ptr + row * TOPK).to(tl.float32)
    dead = maxv == -float("inf")
    s_max_c = tl.where(dead, 0.0, maxv)

    e = tl.zeros([BLOCK], dtype=tl.float32)
    last_nan = -1
    for i in range(0, T):
        t = chunk * T + i
        offs = t * BLOCK + offs_base
        mask = offs < V
        s = tl.load(logits_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        e += tl.where(mask, tl.exp(s - s_max_c), 0.0)
        last_nan = tl.maximum(last_nan, tl.max(tl.where(s != s, offs, -1), axis=0))

    tl.store(partial_ptr + (row * C + chunk) * BLOCK + offs_base, e)
    tl.store(nan_ptr + row * C + chunk, last_nan)


@triton.jit(do_not_specialize=["B", "V", "C", "T"])
def _apply_topk_topp_split_pass2_kernel(
    logits_ptr,
    topk_ptr,
    k_ptr,
    p_ptr,
    partial_ptr,
    nan_ptr,
    out_ptr,
    B,
    V,
    C,
    T,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
):
    """Reduce the C partials per row and write the masked chunk.

    The chunk-order reduction matches the sort-path accumulation order.
    """
    pid = tl.program_id(0)
    row = pid // C
    chunk = pid % C
    offs_base = tl.arange(0, BLOCK)

    if USE_I32:
        row_base = row * V
    else:
        row_base = row.to(tl.int64) * V  # type: ignore[attr-defined]

    k_b = tl.minimum(tl.maximum(tl.load(k_ptr + row), 1), V - 1)
    p_b = tl.load(p_ptr + row)
    maxv = tl.load(topk_ptr + row * TOPK).to(tl.float32)
    kth = tl.load(topk_ptr + row * TOPK + k_b - 1).to(tl.float32)
    dead = maxv == -float("inf")
    s_max_c = tl.where(dead, 0.0, maxv)

    # chunk-order reduction keeps the sort-path accumulation order
    e_all = tl.zeros([BLOCK], dtype=tl.float32)
    last_nan = -1
    for j in range(0, C):
        e_all += tl.load(partial_ptr + (row * C + j) * BLOCK + offs_base)
        last_nan = tl.maximum(last_nan, tl.load(nan_ptr + row * C + j))
    E_all = tl.sum(e_all, axis=0)
    nan_row = last_nan >= 0

    # the top-p cutoff can only be stricter than top-k inside the candidates
    j_base = tl.arange(0, TOPK)
    cand = tl.where(
        j_base < k_b,
        tl.exp(tl.load(topk_ptr + row * TOPK + j_base).to(tl.float32) - s_max_c),
        0.0,
    )
    cum = tl.cumsum(cand, axis=0)
    hit = tl.min(tl.where(cum > p_b * E_all, j_base, TOPK), axis=0)
    cand_cutoff = tl.load(topk_ptr + row * TOPK + tl.minimum(hit, TOPK - 1)).to(tl.float32)
    cutoff = tl.where(p_b >= 1.0, kth, tl.where(hit < k_b, cand_cutoff, kth))

    for i in range(0, T):
        t = chunk * T + i
        offs = t * BLOCK + offs_base
        mask = offs < V
        s = tl.load(logits_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        kept = tl.where(nan_row, offs == last_nan, s >= cutoff) & mask
        kept = kept & ~dead
        val = tl.where(kept, s, -float("inf"))
        tl.store(out_ptr + row_base + offs, val, mask=mask)


def _validate_logits(logits: torch.Tensor):
    """Validate a raw [B, V] logits tensor; returns (contiguous, B, V)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, V], got {logits.dim()}D")
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"logits only supports float32/bfloat16/float16, got {logits.dtype}")
    if logits.numel() == 0:
        raise ValueError("input tensor must not be empty")
    if logits.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits.device}")
    sv = logits if logits.is_contiguous() else logits.contiguous()
    B, V = sv.shape
    if V >= 2**31:
        raise ValueError(f"vocab size {V} too large (>= 2^31)")
    return sv, B, V


def _try_apply_top_k_top_p_no_sort(logits: torch.Tensor, k, p, k_max: int | None = None) -> torch.Tensor | None:
    """Masked logits without the full-vocab sort, or None to fall back.

    Only full_vocab with active top-k is handled; k >= V needs the exact
    sorted cutoff. ``k_max`` avoids the host sync when the caller knows it
    (the V2 sampler reads it from the CPU mirror of the sampling states).
    """
    # k >= V or a non-integer scalar needs the sorted path.
    if k is None or isinstance(k, (bool, float)):
        return None

    sv, B, V = _validate_logits(logits)
    if isinstance(k, int):
        k_t = _normalize_k(k, B, V, sv.device)
        k_max = k
    else:
        k_t = _normalize_k(k, B, V, sv.device)
        if k_max is None:
            # direct callers: one host sync
            k_max = int(k_t.max())

    if k_max >= V or k_max > _NO_SORT_TOPK_CAP:
        return None
    width = triton.next_power_of_2(max(1, k_max))
    if width > V:
        return None

    p_t = _normalize_p(p, B, sv.device)
    topk_vals = torch.topk(sv, width, dim=-1, sorted=True).values
    # pass 2 writes every position, so no -inf prefill is required.
    out = torch.empty_like(sv)

    block = min(_get_tiled_block(sv.device.index), max(triton.next_power_of_2(V), _BLOCK_FLOOR))
    core_num = get_vectorcore_num()
    use_i32 = B * V < 2**31

    # C = max(1, cores // B); C == 1 still uses the split pair (one
    # implementation).
    tiles = triton.cdiv(V, block)
    chunks = min(tiles, max(1, core_num // B))
    tiles_per_chunk = triton.cdiv(tiles, chunks)
    partial = torch.empty((B, chunks, block), dtype=torch.float32, device=sv.device)
    nan_part = torch.empty((B, chunks), dtype=torch.int32, device=sv.device)
    split_grid = (B * chunks,)
    _apply_topk_topp_split_pass1_kernel[split_grid](
        sv,
        topk_vals,
        k_t,
        partial,
        nan_part,
        B,
        V,
        chunks,
        tiles_per_chunk,
        TOPK=width,
        BLOCK=block,
        USE_I32=use_i32,
    )
    _apply_topk_topp_split_pass2_kernel[split_grid](
        sv,
        topk_vals,
        k_t,
        p_t,
        partial,
        nan_part,
        out,
        B,
        V,
        chunks,
        tiles_per_chunk,
        TOPK=width,
        BLOCK=block,
        USE_I32=use_i32,
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
    *,
    top_p_mode: TopPMode = "full_vocab",
) -> torch.Tensor:
    """Top-k/top-p filtering on ascending sorted inputs.

    Args:
        sorted_values/sorted_indices: [B, V] outputs of ``torch.sort``.
        k: keep count, scalar or [B] tensor; k[b] >= V or None disables.
        p: threshold in [0, 1], scalar or [B] tensor; None disables.
        top_p_mode: ``"full_vocab"`` (default, matches the legacy full-vocabulary
            reference) or ``"topk_renorm"`` (matches CANN ``npu_top_k_top_p``).

    Returns original-order masked logits (kept = original, filtered = -inf).
    """
    return _launch(sorted_values, sorted_indices, k, p, top_p_mode)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k=None,
    p=None,
    *,
    top_p_mode: TopPMode = "full_vocab",
    k_max: int | None = None,
    no_sort: bool = True,
) -> torch.Tensor:
    """Top-k/top-p filtering without sorted inputs.

    full_vocab takes the no-sort path when k is active and within the cap;
    otherwise falls back to sort + ``apply_top_k_top_p_with_sorted``.
    ``k_max`` avoids the host sync; ``no_sort=False`` forces the sort path
    (``additional_config.triton_topk_topp_disable_no_sort``).
    """
    _validate_top_p_mode(top_p_mode)
    if top_p_mode == "full_vocab" and no_sort:
        out = _try_apply_top_k_top_p_no_sort(logits, k, p, k_max)
        if out is not None:
            return out
    sorted_values, sorted_indices = _sort_ascending(logits)
    return apply_top_k_top_p_with_sorted(sorted_values, sorted_indices, k, p, top_p_mode=top_p_mode)
