# SPDX-License-Identifier: Apache-2.0
"""NPU GDN checkpoint/replay primitives, without scheduler or cache policy.

Q/K are already normalized, beta is activated, and g is log decay. State is
FP32 [slots, value_heads, value_dim, key_dim]; the FP32 replay record is
[slots, value_heads, window, value_dim + key_dim + 1]. Only the slot stride
may contain padding. Slots <= 0 are reserved/padding and never written.

The caller owns metadata validity and ordering: rows must select distinct
positive slots, 0 <= query_length <= window, and offsets must partition the
packed inputs. Checkpoint, record and output regions must not overlap.
Commit counts describe processed input tokens (including the
mandatory token), not accepted drafts or the newly sampled bonus token.
Commit must happen exactly once before the next verify of the same checkpoint.
"""

import math

import torch
from vllm.triton_utils import tl, triton

_MAX_WINDOW = 16
_MAX_DIM = 256
_VALUE_TILE = 64
_WIDE_KEY_VALUE_TILE = 32


@triton.jit
def _verify_kernel(
    S,
    Q,
    K,
    V,
    G,
    Beta,
    IDs,
    CU,
    Out,
    R,
    SS: tl.constexpr,
    RS: tl.constexpr,
    QT: tl.constexpr,
    KT: tl.constexpr,
    VT: tl.constexpr,
    GT: tl.constexpr,
    BT: tl.constexpr,
    W: tl.constexpr,
    NK: tl.constexpr,
    HV: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(IDs + row).to(tl.int64)
    start = tl.load(CU + row).to(tl.int64)
    end = tl.load(CU + row + 1).to(tl.int64)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < VD) & (kk[None, :] < KD)
    if slot > 0 and end > start:
        state_offset = slot * SS + head * VD * KD + vv[:, None] * KD + kk[None, :]
        state = tl.load(S + state_offset, mask=mask, other=0).to(tl.float32)
        qk_head = head // (HV // NK)
        for step in range(W):
            token = start + step
            if token < end:
                key = tl.load(K + token * KT + qk_head * KD + kk, mask=kk < KD, other=0).to(tl.float32)
                query = tl.load(Q + token * QT + qk_head * KD + kk, mask=kk < KD, other=0).to(tl.float32)
                query = query * SCALE
                value = tl.load(V + token * VT + head * VD + vv, mask=vv < VD, other=0).to(tl.float32)
                decay = tl.exp(tl.load(G + token * GT + head).to(tl.float32))
                beta = tl.load(Beta + token * BT + head).to(tl.float32)
                state = state * decay
                correction = (value - tl.sum(state * key[None, :], 1)) * beta
                state = state + correction[:, None] * key[None, :]
                output = tl.sum(state * query[None, :], 1)
                tl.store(Out + (token * HV + head) * VD + vv, output, mask=vv < VD)
                record = slot * RS + (head * W + step) * (VD + KD + 1)
                tl.store(R + record + vv, correction, mask=vv < VD)
                if tile == 0:
                    tl.store(R + record + VD + kk, key, mask=kk < KD)
                    tl.store(R + record + VD + KD, decay)
    elif end > start:
        for step in range(W):
            token = start + step
            if token < end:
                tl.store(Out + (token * HV + head) * VD + vv, 0, mask=vv < VD)


@triton.jit
def _commit_kernel(
    S,
    R,
    IDs,
    CU,
    Prefix,
    SS: tl.constexpr,
    RS: tl.constexpr,
    W: tl.constexpr,
    HV: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(IDs + row).to(tl.int64)
    available = tl.load(CU + row + 1) - tl.load(CU + row)
    length = tl.minimum(tl.maximum(tl.load(Prefix + row), 0), available)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < VD) & (kk[None, :] < KD)
    if slot > 0 and length > 0:
        offset = slot * SS + head * VD * KD + vv[:, None] * KD + kk[None, :]
        state = tl.load(S + offset, mask=mask, other=0).to(tl.float32)
        for step in range(W):
            if step < length:
                record = slot * RS + (head * W + step) * (VD + KD + 1)
                key = tl.load(R + record + VD + kk, mask=kk < KD, other=0)
                correction = tl.load(R + record + vv, mask=vv < VD, other=0)
                decay = tl.load(R + record + VD + KD)
                state = state * decay
                state = state + correction[:, None] * key[None, :]
        tl.store(S + offset, state, mask=mask)


@triton.jit
def _conv_history_kernel(
    Conv,
    Saved,
    IDs,
    CU,
    Prefix,
    CS: tl.constexpr,
    HS: tl.constexpr,
    HISTORY: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
    SAVE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(IDs + row).to(tl.int64)
    channels = tile * BLOCK + tl.arange(0, BLOCK)
    available = tl.load(CU + row + 1) - tl.load(CU + row)
    mask = channels < CHANNELS
    if slot > 0 and available > 0:
        if SAVE:
            for h in range(HISTORY):
                old = tl.load(Conv + slot * CS + h * CHANNELS + channels, mask=mask, other=0)
                tl.store(Saved + slot * HS + h * CHANNELS + channels, old, mask=mask)
        else:
            length = tl.minimum(tl.maximum(tl.load(Prefix + row), 0), available)
            # Load all history rows before overlapping writes to Conv[:HISTORY].
            hh = tl.arange(0, triton.next_power_of_2(HISTORY))
            if length > 0:
                offset = slot * CS + (length - 1 + hh[:, None]) * CHANNELS + channels[None, :]
                history = tl.load(Conv + offset, mask=(hh[:, None] < HISTORY) & mask[None, :], other=0)
            else:
                offset = slot * HS + hh[:, None] * CHANNELS + channels[None, :]
                history = tl.load(Saved + offset, mask=(hh[:, None] < HISTORY) & mask[None, :], other=0)
            destination = slot * CS + hh[:, None] * CHANNELS + channels[None, :]
            tl.store(Conv + destination, history, mask=(hh[:, None] < HISTORY) & mask[None, :])


def _check_layout(tensor: torch.Tensor, name: str) -> None:
    expected = 1
    for size, stride in zip(reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])):
        if stride != expected:
            raise ValueError(f"{name} must have contiguous inner dimensions")
        expected *= size
    if tensor.stride(0) < expected:
        raise ValueError(f"{name} slots/rows must not overlap")


def _check_metadata(state: torch.Tensor, indices: torch.Tensor, offsets: torch.Tensor) -> None:
    if indices.ndim != 1 or offsets.shape != (indices.numel() + 1,):
        raise ValueError("indices must be [batch] and offsets [batch + 1]")
    for tensor in (indices, offsets):
        if tensor.dtype not in (torch.int32, torch.int64) or not tensor.is_contiguous():
            raise ValueError("metadata must be contiguous int32/int64 tensors")
        if tensor.device != state.device:
            raise ValueError("metadata and cache must be on the same device")


def _check_cache(state: torch.Tensor, records: torch.Tensor) -> tuple[int, int, int, int]:
    if state.ndim != 4 or records.ndim != 4:
        raise ValueError("state and records must be four-dimensional")
    slots, heads, value_dim, key_dim = state.shape
    window = records.shape[2]
    if min(slots, heads, value_dim, key_dim) < 1:
        raise ValueError("cache dimensions must be positive")
    if not 1 <= window <= _MAX_WINDOW or max(value_dim, key_dim) > _MAX_DIM:
        raise ValueError("supported window is 1..16 and key/value dimensions are 1..256")
    if records.shape != (slots, heads, window, value_dim + key_dim + 1):
        raise ValueError("records must hold correction, key and decay for each slot/head/step")
    if state.dtype != torch.float32 or records.dtype != torch.float32:
        raise ValueError("checkpoint and records must be FP32")
    if state.device != records.device:
        raise ValueError("checkpoint and records must be on the same device")
    _check_layout(state, "state")
    _check_layout(records, "records")
    return heads, value_dim, key_dim, window


@torch.no_grad()
def gdn_recover_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    records: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    *,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify packed tokens; write outputs/records but never the checkpoint.

    Q/K: [tokens, qk_heads, key_dim], V: [tokens, value_heads, value_dim],
    g/beta: [tokens, value_heads]. value_heads must be divisible by qk_heads.
    All floating inputs are BF16/FP32 and already normalized/activated.
    Callers can supply a persistent contiguous output for graph capture.
    """
    heads, vd, kd, window = _check_cache(state, records)
    _check_metadata(state, indices, offsets)
    if q.ndim != 3 or k.shape != q.shape or q.shape[2] != kd or q.shape[1] < 1:
        raise ValueError("Q/K must have matching [tokens, qk_heads, key_dim] shapes")
    tokens, qk_heads, _ = q.shape
    if heads % qk_heads or v.shape != (tokens, heads, vd):
        raise ValueError("invalid grouped-head mapping or V shape")
    if g.shape != (tokens, heads) or beta.shape != g.shape:
        raise ValueError("g/beta must be [tokens, value_heads]")
    for tensor in (q, k, v, g, beta):
        if tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.device != state.device:
            raise ValueError("floating inputs must be BF16/FP32 on the cache device")
        _check_layout(tensor, "input")
    scale = kd**-0.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if out.shape != v.shape or out.dtype != v.dtype or out.device != v.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with V's shape, dtype and device")
    if indices.numel() == 0 or tokens == 0:
        return out
    value_tile = _WIDE_KEY_VALUE_TILE if kd > 128 else _VALUE_TILE
    _verify_kernel[(indices.numel(), heads, triton.cdiv(vd, value_tile))](
        state,
        q,
        k,
        v,
        g,
        beta,
        indices,
        offsets,
        out,
        records,
        SS=state.stride(0),
        RS=records.stride(0),
        QT=q.stride(0),
        KT=k.stride(0),
        VT=v.stride(0),
        GT=g.stride(0),
        BT=beta.stride(0),
        W=window,
        NK=qk_heads,
        HV=heads,
        KD=kd,
        VD=vd,
        BK=triton.next_power_of_2(kd),
        BV=value_tile,
        SCALE=scale,
        enable_fp_fusion=False,
    )
    return out


@torch.no_grad()
def gdn_recover_commit(
    state: torch.Tensor,
    records: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    processed_prefix_lengths: torch.Tensor,
) -> None:
    """Replay the selected input prefix once, in place; records stay read-only.

    Counts are clipped to [0, query_length]. Zero restores the unchanged SSM
    checkpoint; it does not mean zero accepted drafts. Metadata stays on device.
    """
    heads, vd, kd, window = _check_cache(state, records)
    _check_metadata(state, indices, offsets)
    _check_metadata(state, processed_prefix_lengths, offsets)
    if processed_prefix_lengths.shape != indices.shape:
        raise ValueError("prefix lengths must match the request rows")
    if indices.numel() == 0:
        return
    value_tile = _WIDE_KEY_VALUE_TILE if kd > 128 else _VALUE_TILE
    _commit_kernel[(indices.numel(), heads, triton.cdiv(vd, value_tile))](
        state,
        records,
        indices,
        offsets,
        processed_prefix_lengths,
        SS=state.stride(0),
        RS=records.stride(0),
        W=window,
        HV=heads,
        KD=kd,
        VD=vd,
        BK=triton.next_power_of_2(kd),
        BV=value_tile,
        enable_fp_fusion=False,
    )


def _check_conv(conv: torch.Tensor, saved: torch.Tensor) -> tuple[int, int]:
    if conv.ndim != 3 or saved.ndim != 3:
        raise ValueError("conv/saved must be [slots, rows, channels]")
    if conv.shape[0] != saved.shape[0] or conv.shape[2] != saved.shape[2]:
        raise ValueError("conv/saved slots and channels must match")
    history = saved.shape[1]
    window = conv.shape[1] - history + 1
    if not 1 <= history <= 16 or not 1 <= window <= _MAX_WINDOW or conv.shape[2] < 1:
        raise ValueError("invalid conv history/window geometry")
    if conv.dtype not in (torch.bfloat16, torch.float32) or saved.dtype != conv.dtype:
        raise ValueError("conv/saved must have the same BF16/FP32 dtype")
    if conv.device != saved.device:
        raise ValueError("conv/saved must be on the same device")
    _check_layout(conv, "conv")
    _check_layout(saved, "saved")
    return history, conv.shape[2]


@torch.no_grad()
def gdn_save_conv_history(
    conv: torch.Tensor,
    saved: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Save Conv[:history] before speculative convolution changes its state."""
    history, channels = _check_conv(conv, saved)
    _check_metadata(conv, indices, offsets)
    if indices.numel():
        _conv_history_kernel[(indices.numel(), triton.cdiv(channels, 128))](
            conv,
            saved,
            indices,
            offsets,
            offsets,
            CS=conv.stride(0),
            HS=saved.stride(0),
            HISTORY=history,
            CHANNELS=channels,
            BLOCK=128,
            SAVE=True,
        )


@torch.no_grad()
def gdn_commit_conv_history(
    conv: torch.Tensor,
    saved: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    processed_prefix_lengths: torch.Tensor,
) -> None:
    """Compact expanded conv history to the same prefix as the SSM commit.

    After verify's convolution, a positive prefix selects Conv[prefix-1:]
    for `history` rows; zero restores saved pre-verify history. Tail rows and
    saved storage are untouched. The caller must commit SSM and conv together.
    """
    history, channels = _check_conv(conv, saved)
    _check_metadata(conv, indices, offsets)
    _check_metadata(conv, processed_prefix_lengths, offsets)
    if processed_prefix_lengths.shape != indices.shape:
        raise ValueError("prefix lengths must match the request rows")
    if indices.numel():
        _conv_history_kernel[(indices.numel(), triton.cdiv(channels, 128))](
            conv,
            saved,
            indices,
            offsets,
            processed_prefix_lengths,
            CS=conv.stride(0),
            HS=saved.stride(0),
            HISTORY=history,
            CHANNELS=channels,
            BLOCK=128,
            SAVE=False,
        )
