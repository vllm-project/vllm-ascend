# Adapted from vllm/model_executor/layers/mamba/ops/mamba_ssm.py
# Provides a PyTorch-based selective_scan_fn for NPU platforms
# where torch.ops._C.selective_scan_fwd is unavailable.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F


def _add_delta_bias(delta: torch.Tensor, delta_bias: torch.Tensor) -> torch.Tensor:
    """Add delta_bias to delta in a shape-agnostic way.

    Some upstream call sites pass delta_bias as ``(dim,)`` while others
    may already have it broadcast to the same shape as ``delta``.
    """
    if delta.shape == delta_bias.shape:
        return delta + delta_bias
    # Assume delta_bias is (dim,) — broadcast over delta's trailing dim.
    shape = [1] * delta.ndim
    shape[-1] = -1
    return delta + delta_bias.reshape(shape)


def _prepare_delta(
    delta: torch.Tensor,
    delta_bias: torch.Tensor | None,
    delta_softplus: bool,
) -> torch.Tensor:
    if delta_bias is not None:
        delta = _add_delta_bias(delta, delta_bias)
    if delta_softplus:
        delta = F.softplus(delta)
    return delta


def _scan_sequence(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
) -> torch.Tensor:
    """Run selective scan for one sequence: inputs are ``(seqlen, dim/dstate)``."""
    seqlen, dim = u.shape
    dstate = A.shape[-1]

    if seqlen == 0:
        return u.new_empty((0, dim))

    # Prefetch decay / input terms for the whole sequence to cut per-step work.
    # dA: (L, dim, dstate), dBu: (L, dim, dstate)
    dA = torch.exp(A.unsqueeze(0) * delta.unsqueeze(-1))
    dBu = delta.unsqueeze(-1) * B.unsqueeze(1) * u.unsqueeze(-1)

    # Decode (L==1) is the hot path for serving; keep it allocation-light.
    if seqlen == 1:
        h = dBu[0]
        y = (C[0].unsqueeze(0) * h).sum(-1)
        if D is not None:
            y = y + D * u[0]
        if z is not None:
            y = y * torch.sigmoid(z[0])
        return y.unsqueeze(0)

    h = torch.zeros(dim, dstate, device=u.device, dtype=torch.float32)
    ys = torch.empty(seqlen, dim, device=u.device, dtype=torch.float32)
    for t in range(seqlen):
        h = dA[t] * h + dBu[t]
        y_t = (C[t].unsqueeze(0) * h).sum(-1)
        if D is not None:
            y_t = y_t + D * u[t]
        if z is not None:
            y_t = y_t * torch.sigmoid(z[t])
        ys[t] = y_t
    return ys


def _selective_scan_impl(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified selective scan supporting both 3-D batch and 2-D varlen."""
    delta = _prepare_delta(delta, delta_bias, delta_softplus)

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float() if D is not None else None
    z_f = z.float() if z is not None else None

    if query_start_loc is not None:
        # Packed varlen: (total_tokens, dim). Pull offsets to host once to avoid
        # per-sequence .item() syncs on the NPU hot path.
        starts_ends = query_start_loc.detach().to("cpu").tolist()
        out = torch.empty_like(u_f)
        for s in range(len(starts_ends) - 1):
            start = int(starts_ends[s])
            end = int(starts_ends[s + 1])
            if end <= start:
                continue
            out[start:end] = _scan_sequence(
                u_f[start:end],
                delta_f[start:end],
                A_f,
                B_f[start:end],
                C_f[start:end],
                D_f,
                None if z_f is None else z_f[start:end],
            )
        return out.to(u.dtype)

    # Batch mode: (batch, dim, seqlen)
    batch, dim, seqlen = u.shape

    if seqlen == 0:
        return u.new_empty((batch, dim, 0))

    # (batch, L, dim/dstate) layout is friendlier for the shared sequence kernel.
    u_bt = u_f.transpose(1, 2).contiguous()
    delta_bt = delta_f.transpose(1, 2).contiguous()
    B_bt = B_f.transpose(1, 2).contiguous()
    C_bt = C_f.transpose(1, 2).contiguous()
    z_bt = None if z_f is None else z_f.transpose(1, 2).contiguous()

    if seqlen == 1:
        # h0 = 0 ⇒ first step is just dBu; skip dA materialization.
        dt = delta_bt[:, 0]  # (batch, dim)
        ut = u_bt[:, 0]
        bt = B_bt[:, 0]
        ct = C_bt[:, 0]
        h = dt.unsqueeze(-1) * bt.unsqueeze(1) * ut.unsqueeze(-1)
        y = (ct.unsqueeze(1) * h).sum(-1)
        if D_f is not None:
            y = y + D_f.unsqueeze(0) * ut
        if z_bt is not None:
            y = y * torch.sigmoid(z_bt[:, 0])
        return y.unsqueeze(-1).to(u.dtype)

    out = torch.empty(batch, seqlen, dim, device=u.device, dtype=torch.float32)
    for b in range(batch):
        out[b] = _scan_sequence(
            u_bt[b],
            delta_bt[b],
            A_f,
            B_bt[b],
            C_bt[b],
            D_f,
            None if z_bt is None else z_bt[b],
        )
    return out.transpose(1, 2).contiguous().to(u.dtype)


def selective_scan_fn_npu(
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    """PyTorch-based Mamba selective scan for NPU.

    Replaces the CUDA-only ``torch.ops._C.selective_scan_fwd`` on Ascend NPU
    where that custom op is not available.

    Uses ``*args, **kwargs`` for maximum compatibility with the upstream
    ``selective_scan_fn`` signature.
    """

    def _get(name: str, idx: int) -> object:
        if name in kwargs:
            return kwargs[name]
        if idx < len(args):
            return args[idx]
        return None

    _u = _get("u", 0)
    _delta = _get("delta", 1)
    _A = _get("A", 2)
    _B = _get("B", 3)
    _C = _get("C", 4)
    _D = _get("D", 5)
    _z = _get("z", 6)
    _delta_bias = _get("delta_bias", 7)
    _delta_softplus = _get("delta_softplus", 8)
    _query_start_loc = _get("query_start_loc", 9)

    assert isinstance(_u, torch.Tensor), f"u must be a Tensor, got {type(_u)}"
    assert isinstance(_delta, torch.Tensor), f"delta must be a Tensor, got {type(_delta)}"
    assert isinstance(_A, torch.Tensor), f"A must be a Tensor, got {type(_A)}"
    assert isinstance(_B, torch.Tensor), f"B must be a Tensor, got {type(_B)}"
    assert isinstance(_C, torch.Tensor), f"C must be a Tensor, got {type(_C)}"

    u: torch.Tensor = _u
    delta: torch.Tensor = _delta
    A: torch.Tensor = _A
    B: torch.Tensor = _B
    C: torch.Tensor = _C
    D: torch.Tensor | None = _D if isinstance(_D, torch.Tensor) else None
    z: torch.Tensor | None = _z if isinstance(_z, torch.Tensor) else None
    delta_bias: torch.Tensor | None = _delta_bias if isinstance(_delta_bias, torch.Tensor) else None
    delta_softplus: bool = bool(_delta_softplus)
    query_start_loc: torch.Tensor | None = _query_start_loc if isinstance(_query_start_loc, torch.Tensor) else None

    return _selective_scan_impl(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        query_start_loc,
    )
