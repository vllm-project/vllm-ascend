# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant recurrent-state window-flush operators (pure torch).

Reference: LeapQuant (arXiv 2609.38166), RFC vllm-ascend#21097 (M1).

The recurrent state of gated-delta-net style linear attention evolves as
``S_t = Diag(decay) · S_{t-1} + k̃_t u_tᵀ``.  Quantizing ``S`` every decode
step accumulates rounding error recursively; LeapQuant instead keeps the
materialized state in fp32 and re-quantizes it only at *window boundaries*
(every ``window`` tokens), which is algebraically equivalent to the paper's
window unroll because the in-window fp32 recurrence from a dequantized
boundary state ``Ŝ₀`` reproduces the window form exactly.

The boundary flush has three parts (paper eq. 5-7):

1. **Compensator tokens** — one round of subspace iteration (2-pass modified
   Gram-Schmidt) extracts the top-``r`` singular directions of ``S``.  The
   low-rank factor is kept in fp16 and never goes through int8.
2. **Smoothed residual int8** — the residual ``E = S − Q̃ᵀŨ`` is quantized
   with a per-value-row / per-key-column scale pair, so an outlier only
   coarsens its own row/column instead of the whole matrix.
3. **Round-trip** — the stored state is replaced by the reconstruction
   ``Ŝ = Q̃ᵀŨ + codes · (s_k/127) · s_v``.

The tensor layout follows the checkpoint contract of the reference GPU
kernel (huggingface.co/kernels/humanfia-lab/leapquant): state blocks are
``[V, K]`` per head, compensator factors ``q ∈ [R, V]`` fp16 and
``u ∈ [R, K]`` fp16, ``codes`` int8 ``[V, K]``, scales fp32.  This module is
device-agnostic (CPU unit tests, NPU runtime).

The numerical core is a faithful port of the validated torch reference in
``yanghaocsg/awesome_quant_algo`` ``turboquant_exp/colab_02_leapquant``.
"""

from __future__ import annotations

import torch

__all__ = [
    "QMAX",
    "EPS",
    "start_basis",
    "mgs_orthonormal_columns",
    "fit_compensators",
    "quantize_residual",
    "reconstruct_state",
    "leap_window_flush",
    "decode_flush_mask",
    "prefill_flush_mask",
]

# Symmetric int8 quantization uses the [-127, 127] range (no -128), matching
# the reference kernel contract "int8 codes must remain in [-127, 127]".
QMAX = 127.0
EPS = 1e-12

# Deterministic seed for the fixed start basis q0. The basis depends only on
# (V, R); the reference kernel keeps it fixed across windows and layers.
_START_BASIS_SEED = 26093816600 & 0x7FFFFFFF


def start_basis(v_dim: int, rank: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fixed orthonormal start basis ``q0 ∈ [V, R]`` for subspace iteration.

    The reference implementation uses a random but *fixed* basis rather than
    the previous window's factors, so the flush stays independent of the
    execution history (and of capture order).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(_START_BASIS_SEED + 17 * v_dim + rank)
    q0 = torch.randn(v_dim, rank, generator=g).to(dtype)
    q0, _ = torch.linalg.qr(q0)
    return q0.to(device=device, dtype=dtype).contiguous()


def mgs_orthonormal_columns(m: torch.Tensor, *, passes: int = 2) -> torch.Tensor:
    """Batched modified Gram-Schmidt over the last axis (paper eq. 5-6).

    ``m`` has shape ``[..., D, R]``; the ``R`` columns become orthonormal.
    Two projection passes keep the columns orthogonal even when the input
    columns are nearly parallel (numerically stabilized MGS).
    Port of the reference ``_mgs``.
    """
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(passes):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)


def fit_compensators(
    state: torch.Tensor,
    q0: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One round of subspace iteration -> top-``rank`` compensator tokens.

    Args:
        state: fp32 ``[N, V, K]`` state blocks (one per head/slot-row).
        q0: fixed orthonormal start basis ``[V, R]``.
        rank: number of compensator tokens (paper r = 4 at int8).

    Returns:
        ``(u, q)`` with ``u ∈ [N, R, K]`` fp16 and ``q ∈ [N, R, V]`` fp16 such
        that ``qᵀ·u ≈ S`` in the top-rank subspace.
    """
    x = state.transpose(-1, -2)  # [N, K, V]
    qk = mgs_orthonormal_columns(x @ q0)  # [N, K, R]
    qv = mgs_orthonormal_columns(x.transpose(-1, -2) @ qk)  # [N, V, R]
    uk = x @ qv  # [N, K, R]
    u = uk.transpose(-1, -2).to(torch.float16)  # [N, R, K]
    q = qv.transpose(-1, -2).to(torch.float16)  # [N, R, V]
    del rank  # rank is implicit in q0/qv shape; kept for API clarity
    return u, q


def quantize_residual(
    state: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smoothed two-scale int8 quantization of the residual (paper eq. 7).

    Args:
        state: fp32 ``[N, V, K]``.
        u: fp16 ``[N, R, K]`` compensator factors (key side).
        q: fp16 ``[N, R, V]`` compensator factors (value side).

    Returns:
        ``(codes int8 [N, V, K], s_k fp32 [N, K], s_v fp32 [N, V])`` with
        ``s_k = sqrt(mean_V |E|)`` (per key column) and
        ``s_v = absmax_K(|E| / s_k)`` (per value row).
    """
    residual = state - torch.einsum("nrv,nrk->nvk", q.float(), u.float())
    s_k = residual.abs().mean(-2).sqrt().clamp_min(EPS)  # [N, K]
    s_v = (residual.abs() / s_k[..., None, :]).amax(-1).clamp_min(EPS)  # [N, V]
    codes = torch.round(residual / (s_k[..., None, :] * s_v[..., :, None]) * QMAX)
    codes = codes.clamp(-QMAX, QMAX).to(torch.int8)
    return codes, s_k, s_v


def reconstruct_state(
    codes: torch.Tensor,
    s_k: torch.Tensor,
    s_v: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the fp32 state from a LeapQuant checkpoint (paper eq. 8-9)."""
    state = torch.einsum("nrv,nrk->nvk", q.float(), u.float())
    state = state + codes.float() * (s_k[..., None, :] / QMAX) * s_v[..., :, None]
    return state


def leap_window_flush(rows: torch.Tensor, *, compensator_tokens: int = 4) -> torch.Tensor:
    """Round-trip state rows through a LeapQuant checkpoint at a window boundary.

    Args:
        rows: ``[B, H, V, K]`` (or ``[N, V, K]``) fp32/fp16 state rows whose
            token count just reached a multiple of ``window``.
        compensator_tokens: number of compensator tokens (paper r).

    Returns:
        Tensor with the same shape/dtype/device: ``reconstruct(quantize(rows))``.
        This is the only point where quantization error enters the stream.
    """
    orig_shape = rows.shape
    orig_dtype = rows.dtype
    blocks = rows.reshape(-1, orig_shape[-2], orig_shape[-1]).float()  # [N, V, K]
    v_dim = blocks.shape[-2]

    q0 = start_basis(v_dim, compensator_tokens, device=blocks.device, dtype=blocks.dtype)
    u, q = fit_compensators(blocks, q0, compensator_tokens)
    codes, s_k, s_v = quantize_residual(blocks, u, q)
    flushed = reconstruct_state(codes, s_k, s_v, u, q)
    return flushed.to(orig_dtype).reshape(orig_shape)


def decode_flush_mask(context_lens: torch.Tensor, window: int) -> torch.Tensor:
    """Rows whose state hits a window multiple after this decode step.

    Each decode sequence contributes one token, so the post-update token
    count is ``context + 1``.  ``context_lens`` is a CPU int tensor aligned
    with the decode state-index rows.
    """
    ends = context_lens.to(torch.long) + 1
    return (ends > 0) & (ends % window == 0)


def prefill_flush_mask(context_lens: torch.Tensor, query_lens: torch.Tensor, window: int) -> torch.Tensor:
    """Rows whose post-prefill token count is an exact window multiple.

    ``context_lens``/``query_lens`` are CPU int tensors aligned with the
    prefill state-index rows.
    """
    ends = context_lens.to(torch.long) + query_lens.to(torch.long)
    return (ends > 0) & (ends % window == 0)
