# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""apply_top_k_top_p: Triton top-k/top-p filtering of LM-head logits.

Two top-p semantics (``top_p_mode``):

- ``full_vocab``: cutoff from the full-vocabulary softmax before top-k
  masking; fp32 bit-exact with the legacy 0.29/0.32 reference. Kept as an
  explicit legacy compatibility mode.
- ``topk_renorm`` (default): top-k first, cutoff after re-normalizing over
  survivors; bit-exact with CANN ``torch_npu.npu_top_k_top_p`` and upstream
  vLLM.

    logits_sort, logits_idx = sort(logits, descending=false, stable=true)
    top_k_cutoff = logits_sort[V - k]
    probs_sum = cumsum(softmax(...), -1)
    top_p_mask = probs_sum <= 1 - p;  top_p_mask[:, -1] = false
    masked_logits[logits_idx] = where(top_p_mask, -inf, logits_sort)

All entries return original-order masked logits.

Dispatch (full_vocab, k < V): ``torch.topk`` gives the k-th value, then two
linear passes compute the denominator and write the mask, skipping the [B, V]
sort. One row is split across ``C = max(1, cores // B)`` programs; pass 1
stores per-chunk partials and pass 2 reduces them in chunk order, so the
accumulation order matches the sort path. Everything else (k >= V,
``topk_renorm``) uses sort + the tiled kernel.

Notes:
- k / p are per-request [B] tensors: no ``.item()`` CPU sync.
- full_vocab keeps ``logits_sort >= cutoff`` (ties preserved); topk_renorm
  keeps ``offs >= mask_boundary``. The value form is required because a
  cumsum-derived boolean scatter mask miscompiles on triton-ascend 3.2.
- k <= 0 clamps to V-1; k >= V disables top-k.
- The default mode is ``topk_renorm`` (CANN/upstream semantics). Pass
  ``top_p_mode="full_vocab"`` explicitly only for legacy compatibility.
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


@triton.jit(do_not_specialize=["batch_size", "vocab_size", "rows_per_prog"])
def _apply_topk_topp_tiled_kernel(
    logits_sort_ptr,
    logits_idx_ptr,
    k_ptr,
    p_ptr,
    out_ptr,
    batch_size,
    vocab_size,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
    TOPP_FULL_VOCAB: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, batch_size)
    offs_base = tl.arange(0, BLOCK)

    for row in range(row_start, row_end):
        if USE_I32:
            row_base = row * vocab_size
        else:
            row_base = row.to(tl.int64) * vocab_size  # type: ignore[attr-defined]

        k = tl.load(k_ptr + row)
        p = tl.load(p_ptr + row)
        top_p_threshold = 1.0 - p
        top_k_index = tl.minimum(tl.maximum(vocab_size - k, 0), vocab_size - 1)
        top_k_cutoff = tl.load(logits_sort_ptr + row_base + top_k_index).to(tl.float32)
        max_logits = tl.load(logits_sort_ptr + row_base + vocab_size - 1).to(tl.float32)
        # All--inf rows would compute exp(-inf - -inf) = NaN; clamp the
        # subtraction base. NaN rows deliberately stay NaN (fail loud).
        all_neg_inf_row = max_logits == -float("inf")
        nan_row = max_logits != max_logits
        max_logits_safe = tl.where(all_neg_inf_row, 0.0, max_logits)

        # pass 1: full-vocab denominator (sum_exp_all) + top-k survivor mass/bound.
        sum_exp_topk = tl.zeros([BLOCK], dtype=tl.float32)
        sum_exp_all = tl.zeros([BLOCK], dtype=tl.float32)
        top_k_boundary = vocab_size
        for t in range(0, vocab_size, BLOCK):
            offs = t + offs_base
            mask = offs < vocab_size
            logits_sort = tl.load(logits_sort_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            exp_logits = tl.exp(logits_sort - max_logits_safe)
            top_k_mask = mask & (logits_sort >= top_k_cutoff)
            sum_exp_topk += tl.where(top_k_mask, exp_logits, 0.0)
            sum_exp_all += tl.where(mask, exp_logits, 0.0)
            top_k_boundary = tl.minimum(top_k_boundary, tl.min(tl.where(top_k_mask, offs, vocab_size), axis=0))
        sum_exp_topk_total = tl.sum(sum_exp_topk, axis=0)
        sum_exp_all_total = tl.sum(sum_exp_all, axis=0)

        # pass 2: top_p_boundary = first index whose cumsum exceeds 1-p
        # (vocab_size if none). Branchless: 0-D tensor control flow is not
        # portable across backends.
        running_sum = 0.0
        top_p_boundary = vocab_size
        for t in range(0, vocab_size, BLOCK):
            offs = t + offs_base
            mask = offs < vocab_size
            logits_sort = tl.load(logits_sort_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            exp_logits = tl.exp(logits_sort - max_logits_safe)
            if TOPP_FULL_VOCAB:
                probs_sort = tl.where(mask, exp_logits, 0.0) / sum_exp_all_total
            else:
                probs_sort = tl.where(mask & (logits_sort >= top_k_cutoff), exp_logits, 0.0) / sum_exp_topk_total
            probs_tile_sum = tl.sum(probs_sort, axis=0)
            probs_sum = running_sum + tl.cumsum(probs_sort, axis=0)
            top_p_hit = tl.min(tl.where(probs_sum > top_p_threshold, offs_base, BLOCK), axis=0)
            top_p_boundary = tl.minimum(top_p_boundary, tl.where(top_p_hit < BLOCK, t + top_p_hit, vocab_size))
            running_sum += probs_tile_sum

        # Cutoff: full_vocab keeps logits_sort >= cutoff_value (ties);
        # topk_renorm keeps offs >= mask_boundary.
        mask_boundary = tl.minimum(tl.maximum(top_p_boundary, top_k_boundary), vocab_size - 1)
        cutoff_value = tl.load(logits_sort_ptr + row_base + mask_boundary).to(tl.float32)

        # pass 3: static range + keep_mask (avoids tensor floor-div and runtime
        # range starts). keep_mask is the union of the top-k mask
        # (offs >= top_k_boundary) and the top-p mask (offs >= top_p_boundary);
        # both are prefixes in ascending order. Dead rows write nothing; NaN
        # rows keep one NaN.
        for t in range(0, vocab_size, BLOCK):
            offs = t + offs_base
            mask = offs < vocab_size
            logits_sort = tl.load(logits_sort_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            if TOPP_FULL_VOCAB:
                keep_mask = (logits_sort >= cutoff_value) & mask
                keep_mask = keep_mask | (nan_row & (offs == vocab_size - 1))
            else:
                keep_mask = (offs >= mask_boundary) & mask
            keep_mask = keep_mask & ~all_neg_inf_row
            logits_idx = tl.load(logits_idx_ptr + row_base + offs, mask=keep_mask, other=0)
            tl.store(out_ptr + row_base + logits_idx, logits_sort, mask=keep_mask)


def _validate_and_prepare(logits_sort: torch.Tensor, logits_idx: torch.Tensor):
    """Validate and normalize the sorted inputs; returns (logits_sort, logits_idx, batch_size, vocab_size)."""
    if logits_sort.dim() != 2:
        raise ValueError(f"logits_sort must be 2D [B, V], got {logits_sort.dim()}D")
    if logits_sort.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"logits_sort only supports float32/bfloat16/float16, got {logits_sort.dtype}")
    if logits_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"logits_idx only supports int32/int64, got {logits_idx.dtype}")
    if logits_sort.shape != logits_idx.shape:
        raise ValueError(f"shape mismatch: {logits_sort.shape} vs {logits_idx.shape}")
    if logits_sort.numel() == 0:
        raise ValueError("input tensor must not be empty")
    if logits_sort.device != logits_idx.device:
        raise ValueError(f"device mismatch: {logits_sort.device} vs {logits_idx.device}")
    if logits_sort.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits_sort.device}")

    logits_sort = logits_sort if logits_sort.is_contiguous() else logits_sort.contiguous()
    logits_idx = logits_idx if logits_idx.is_contiguous() else logits_idx.contiguous()
    batch_size, vocab_size = logits_sort.shape
    if vocab_size >= 2**31:
        raise ValueError(f"vocab size {vocab_size} too large (>= 2^31)")
    return logits_sort, logits_idx, batch_size, vocab_size


def _normalize_k(k, batch_size: int, vocab_size: int, device) -> torch.Tensor:
    """Normalize k to a [B] int32 device tensor; None -> V (top-k disabled)."""
    if k is None:
        return torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
    if isinstance(k, bool) or not isinstance(k, (int, float, torch.Tensor)):
        raise TypeError(f"k must be None or int or [B] tensor, got {type(k).__name__}")
    if isinstance(k, float):
        if not k.is_integer():
            raise ValueError(f"k must be an integer, got {k}")
        k = int(k)
    if isinstance(k, int):
        return torch.full((batch_size,), k, dtype=torch.int32, device=device)
    if k.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"k tensor only supports int32/int64, got {k.dtype}")
    if k.numel() == 1:
        k = k.reshape(1).expand(batch_size)
    if k.shape != (batch_size,):
        raise ValueError(f"k tensor must be shape [{batch_size}], got {tuple(k.shape)}")
    return k.to(device=device, dtype=torch.int32).contiguous()


def _normalize_p(p, batch_size: int, device) -> torch.Tensor:
    """Normalize p to a [B] fp32 device tensor; None -> 1.0 (top-p disabled)."""
    if p is None:
        return torch.ones((batch_size,), dtype=torch.float32, device=device)
    if isinstance(p, bool) or not isinstance(p, (int, float, torch.Tensor)):
        raise TypeError(f"p must be None or float or [B] tensor, got {type(p).__name__}")
    if isinstance(p, (int, float)):
        return torch.full((batch_size,), float(p), dtype=torch.float32, device=device)
    if not p.is_floating_point():
        raise ValueError(f"p tensor must be floating point, got {p.dtype}")
    if p.numel() == 1:
        p = p.reshape(1).expand(batch_size)
    if p.shape != (batch_size,):
        raise ValueError(f"p tensor must be shape [{batch_size}], got {tuple(p.shape)}")
    return p.to(device=device, dtype=torch.float32).contiguous()


def _launch(
    logits_sort: torch.Tensor,
    logits_idx: torch.Tensor,
    k,
    p,
    top_p_mode: TopPMode = "topk_renorm",
) -> torch.Tensor:
    """Validate inputs, normalize k/p, pre-fill out, and launch the kernel."""
    _validate_top_p_mode(top_p_mode)
    logits_sort, logits_idx, batch_size, vocab_size = _validate_and_prepare(logits_sort, logits_idx)
    k_ptr = _normalize_k(k, batch_size, vocab_size, logits_sort.device)
    p_ptr = _normalize_p(p, batch_size, logits_sort.device)

    # kernel only scatters survivors; everything else stays -inf
    masked_logits = torch.full_like(logits_sort, float("-inf"))

    # Unified tiled kernel for every V; shrink the block for small rows so
    # a 1k vocab does not pay for a 4096-element tile.
    block = min(_get_tiled_block(logits_sort.device.index), max(triton.next_power_of_2(vocab_size), _BLOCK_FLOOR))
    core_num = get_vectorcore_num()
    if core_num >= batch_size:
        grid, rows_per_prog = (batch_size,), 1
    else:
        grid, rows_per_prog = (core_num,), triton.cdiv(batch_size, core_num)

    use_i32 = batch_size * vocab_size < 2**31
    # logits_idx is not cast on the host (saves a full-vocab copy); index values stay < V < 2^31.

    _apply_topk_topp_tiled_kernel[grid](
        logits_sort,
        logits_idx,
        k_ptr,
        p_ptr,
        masked_logits,
        batch_size,
        vocab_size,
        rows_per_prog,
        BLOCK=block,
        USE_I32=use_i32,
        TOPP_FULL_VOCAB=top_p_mode == "full_vocab",
    )
    return masked_logits


@triton.jit(do_not_specialize=["batch_size", "vocab_size", "num_chunks", "tiles_per_chunk"])
def _apply_topk_topp_split_pass1_kernel(
    logits_ptr,
    top_k_values_ptr,
    k_ptr,
    partial_sum_ptr,
    nan_ptr,
    batch_size,
    vocab_size,
    num_chunks,
    tiles_per_chunk,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
):
    """Per-(row, chunk) partial sums over tiles_per_chunk consecutive BLOCK tiles."""
    pid = tl.program_id(0)
    row = pid // num_chunks
    chunk = pid % num_chunks
    offs_base = tl.arange(0, BLOCK)

    if USE_I32:
        row_base = row * vocab_size
    else:
        row_base = row.to(tl.int64) * vocab_size  # type: ignore[attr-defined]

    max_logits = tl.load(top_k_values_ptr + row * TOPK).to(tl.float32)
    all_neg_inf_row = max_logits == -float("inf")
    max_logits_safe = tl.where(all_neg_inf_row, 0.0, max_logits)

    sum_exp = tl.zeros([BLOCK], dtype=tl.float32)
    last_nan = -1
    for i in range(0, tiles_per_chunk):
        t = chunk * tiles_per_chunk + i
        offs = t * BLOCK + offs_base
        mask = offs < vocab_size
        logits_sort = tl.load(logits_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        sum_exp += tl.where(mask, tl.exp(logits_sort - max_logits_safe), 0.0)
        last_nan = tl.maximum(last_nan, tl.max(tl.where(logits_sort != logits_sort, offs, -1), axis=0))

    tl.store(partial_sum_ptr + (row * num_chunks + chunk) * BLOCK + offs_base, sum_exp)
    tl.store(nan_ptr + row * num_chunks + chunk, last_nan)


@triton.jit(do_not_specialize=["batch_size", "vocab_size", "num_chunks", "tiles_per_chunk"])
def _apply_topk_topp_split_pass2_kernel(
    logits_ptr,
    top_k_values_ptr,
    k_ptr,
    p_ptr,
    partial_sum_ptr,
    nan_ptr,
    out_ptr,
    batch_size,
    vocab_size,
    num_chunks,
    tiles_per_chunk,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
):
    """Reduce the num_chunks partials per row and write the masked chunk.

    The chunk-order reduction matches the sort-path accumulation order.
    """
    pid = tl.program_id(0)
    row = pid // num_chunks
    chunk = pid % num_chunks
    offs_base = tl.arange(0, BLOCK)

    if USE_I32:
        row_base = row * vocab_size
    else:
        row_base = row.to(tl.int64) * vocab_size  # type: ignore[attr-defined]

    k = tl.minimum(tl.maximum(tl.load(k_ptr + row), 1), vocab_size - 1)
    p = tl.load(p_ptr + row)
    max_logits = tl.load(top_k_values_ptr + row * TOPK).to(tl.float32)
    top_k_cutoff = tl.load(top_k_values_ptr + row * TOPK + k - 1).to(tl.float32)
    all_neg_inf_row = max_logits == -float("inf")
    max_logits_safe = tl.where(all_neg_inf_row, 0.0, max_logits)

    # chunk-order reduction keeps the sort-path accumulation order
    sum_exp_all = tl.zeros([BLOCK], dtype=tl.float32)
    last_nan = -1
    for j in range(0, num_chunks):
        sum_exp_all += tl.load(partial_sum_ptr + (row * num_chunks + j) * BLOCK + offs_base)
        last_nan = tl.maximum(last_nan, tl.load(nan_ptr + row * num_chunks + j))
    sum_exp_all_total = tl.sum(sum_exp_all, axis=0)
    nan_row = last_nan >= 0

    # the top-p cutoff can only be stricter than top-k inside the candidates
    j_base = tl.arange(0, TOPK)
    probs_sort = tl.where(
        j_base < k,
        tl.exp(tl.load(top_k_values_ptr + row * TOPK + j_base).to(tl.float32) - max_logits_safe),
        0.0,
    )
    probs_sum = tl.cumsum(probs_sort, axis=0)
    top_p_hit = tl.min(tl.where(probs_sum > p * sum_exp_all_total, j_base, TOPK), axis=0)
    top_p_cutoff = tl.load(top_k_values_ptr + row * TOPK + tl.minimum(top_p_hit, TOPK - 1)).to(tl.float32)
    cutoff_value = tl.where(p >= 1.0, top_k_cutoff, tl.where(top_p_hit < k, top_p_cutoff, top_k_cutoff))

    for i in range(0, tiles_per_chunk):
        t = chunk * tiles_per_chunk + i
        offs = t * BLOCK + offs_base
        mask = offs < vocab_size
        logits_sort = tl.load(logits_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        keep_mask = tl.where(nan_row, offs == last_nan, logits_sort >= cutoff_value) & mask
        keep_mask = keep_mask & ~all_neg_inf_row
        masked_logits = tl.where(keep_mask, logits_sort, -float("inf"))
        tl.store(out_ptr + row_base + offs, masked_logits, mask=mask)


def _validate_logits(logits: torch.Tensor):
    """Validate a raw [B, V] logits tensor; returns (contiguous, batch_size, vocab_size)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, V], got {logits.dim()}D")
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"logits only supports float32/bfloat16/float16, got {logits.dtype}")
    if logits.numel() == 0:
        raise ValueError("input tensor must not be empty")
    if logits.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits.device}")
    logits = logits if logits.is_contiguous() else logits.contiguous()
    batch_size, vocab_size = logits.shape
    if vocab_size >= 2**31:
        raise ValueError(f"vocab size {vocab_size} too large (>= 2^31)")
    return logits, batch_size, vocab_size


def _try_apply_top_k_top_p_no_sort(logits: torch.Tensor, k, p, k_max: int | None = None) -> torch.Tensor | None:
    """Masked logits without the full-vocab sort, or None to fall back.

    Only full_vocab with active top-k is handled; k >= V needs the exact
    sorted cutoff. ``k_max`` avoids the host sync when the caller knows it
    (the V2 sampler reads it from the CPU mirror of the sampling states).
    """
    # k >= V or a non-integer scalar needs the sorted path.
    if k is None or isinstance(k, (bool, float)):
        return None

    logits, batch_size, vocab_size = _validate_logits(logits)
    if isinstance(k, int):
        k_ptr = _normalize_k(k, batch_size, vocab_size, logits.device)
        k_max = k
    else:
        k_ptr = _normalize_k(k, batch_size, vocab_size, logits.device)
        if k_max is None:
            # direct callers: one host sync
            k_max = int(k_ptr.max())

    if k_max >= vocab_size or k_max > _NO_SORT_TOPK_CAP:
        return None
    width = triton.next_power_of_2(max(1, k_max))
    if width > vocab_size:
        return None

    p_ptr = _normalize_p(p, batch_size, logits.device)
    top_k_values = torch.topk(logits, width, dim=-1, sorted=True).values
    # pass 2 writes every position, so no -inf prefill is required.
    masked_logits = torch.empty_like(logits)

    block = min(_get_tiled_block(logits.device.index), max(triton.next_power_of_2(vocab_size), _BLOCK_FLOOR))
    core_num = get_vectorcore_num()
    use_i32 = batch_size * vocab_size < 2**31

    # num_chunks = max(1, cores // B); num_chunks == 1 still uses the split
    # pair (one implementation).
    tiles = triton.cdiv(vocab_size, block)
    num_chunks = min(tiles, max(1, core_num // batch_size))
    tiles_per_chunk = triton.cdiv(tiles, num_chunks)
    partial_sum = torch.empty((batch_size, num_chunks, block), dtype=torch.float32, device=logits.device)
    nan_part = torch.empty((batch_size, num_chunks), dtype=torch.int32, device=logits.device)
    split_grid = (batch_size * num_chunks,)
    _apply_topk_topp_split_pass1_kernel[split_grid](
        logits,
        top_k_values,
        k_ptr,
        partial_sum,
        nan_part,
        batch_size,
        vocab_size,
        num_chunks,
        tiles_per_chunk,
        TOPK=width,
        BLOCK=block,
        USE_I32=use_i32,
    )
    _apply_topk_topp_split_pass2_kernel[split_grid](
        logits,
        top_k_values,
        k_ptr,
        p_ptr,
        partial_sum,
        nan_part,
        masked_logits,
        batch_size,
        vocab_size,
        num_chunks,
        tiles_per_chunk,
        TOPK=width,
        BLOCK=block,
        USE_I32=use_i32,
    )
    return masked_logits


def _sort_ascending(logits: torch.Tensor):
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, V], got {logits.dim()}D")
    if logits.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits.device}")
    return torch.sort(logits, dim=-1, descending=False, stable=True)


def apply_top_k_top_p_with_sorted(
    logits_sort: torch.Tensor,
    logits_idx: torch.Tensor,
    k=None,
    p=None,
    *,
    top_p_mode: TopPMode = "topk_renorm",
) -> torch.Tensor:
    """Top-k/top-p filtering on ascending sorted inputs.

    Args:
        logits_sort/logits_idx: [B, V] outputs of ``torch.sort``.
        k: keep count, scalar or [B] tensor; k[b] >= V or None disables.
        p: threshold in [0, 1], scalar or [B] tensor; None disables.
        top_p_mode: ``"topk_renorm"`` (default, matches CANN/upstream vLLM)
            or ``"full_vocab"`` (legacy vllm-ascend compatibility, explicit).

    Returns original-order masked logits (kept = original, filtered = -inf).
    """
    return _launch(logits_sort, logits_idx, k, p, top_p_mode)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k=None,
    p=None,
    *,
    top_p_mode: TopPMode = "topk_renorm",
    k_max: int | None = None,
    no_sort: bool = True,
) -> torch.Tensor:
    """Top-k/top-p filtering without sorted inputs.

    The default ``topk_renorm`` matches CANN and the current upstream PyTorch
    fallback. ``full_vocab`` is an explicit legacy mode: it takes the no-sort
    path when k is active and within the cap, otherwise falls back to sort +
    ``apply_top_k_top_p_with_sorted``.
    ``k_max`` avoids the host sync; ``no_sort=False`` forces the sort path
    (``additional_config.triton_topk_topp_disable_no_sort``).
    """
    _validate_top_p_mode(top_p_mode)
    if top_p_mode == "full_vocab" and no_sort:
        masked_logits = _try_apply_top_k_top_p_no_sort(logits, k, p, k_max)
        if masked_logits is not None:
            return masked_logits
    logits_sort, logits_idx = _sort_ascending(logits)
    return apply_top_k_top_p_with_sorted(logits_sort, logits_idx, k, p, top_p_mode=top_p_mode)
