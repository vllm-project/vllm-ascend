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

    if query_start_loc is not None:
        # Packed varlen: (total_tokens, dim)
        dim = u.shape[-1]
        dstate = A.shape[-1]
        num_seqs = query_start_loc.shape[0] - 1

        if delta_bias is not None:
            delta = _add_delta_bias(delta, delta_bias)
        if delta_softplus:
            delta = F.softplus(delta)

        u_f = u.float()
        delta_f = delta.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()
        D_f = D.float() if D is not None else None
        z_f = z.float() if z is not None else None

        out = torch.zeros_like(u_f)

        for s in range(num_seqs):
            start = int(query_start_loc[s].item())
            end = int(query_start_loc[s + 1].item())
            if end <= start:
                continue

            h = torch.zeros(dim, dstate, device=u.device, dtype=torch.float32)

            for t in range(start, end):
                dt = delta_f[t]  # (dim,)
                ut = u_f[t]  # (dim,)
                bt = B_f[t]  # (dstate,)
                ct = C_f[t]  # (dstate,)

                dA = torch.exp(A_f * dt.unsqueeze(-1))  # (dim, dstate)
                h = dA * h + dt.unsqueeze(-1) * bt.unsqueeze(0) * ut.unsqueeze(-1)
                y_t = (ct.unsqueeze(0) * h).sum(-1)

                if D_f is not None:
                    y_t = y_t + D_f * ut
                if z_f is not None:
                    y_t = y_t * F.sigmoid(z_f[t])

                out[t] = y_t

        return out.to(u.dtype)

    else:
        # Batch mode: (batch, dim, seqlen)
        batch, dim, seqlen = u.shape
        dstate = A.shape[-1]

        if delta_bias is not None:
            delta = _add_delta_bias(delta, delta_bias)
        if delta_softplus:
            delta = F.softplus(delta)

        u_f = u.float()
        delta_f = delta.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()
        D_f = D.float() if D is not None else None
        z_f = z.float() if z is not None else None

        h = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
        out = torch.zeros(batch, dim, seqlen, device=u.device, dtype=torch.float32)

        for t in range(seqlen):
            dt = delta_f[:, :, t]  # (batch, dim)
            ut = u_f[:, :, t]  # (batch, dim)
            bt = B_f[:, :, t]  # (batch, dstate)
            ct = C_f[:, :, t]  # (batch, dstate)

            dA = torch.exp(A_f.view(1, -1, dstate) * dt.unsqueeze(-1))
            h = dA * h + dt.unsqueeze(-1) * bt.unsqueeze(1) * ut.unsqueeze(-1)
            y_t = (ct.unsqueeze(1) * h).sum(-1)

            if D_f is not None:
                y_t = y_t + D_f.unsqueeze(0) * ut
            if z_f is not None:
                y_t = y_t * F.sigmoid(z_f[:, :, t])

            out[:, :, t] = y_t

        return out.to(u.dtype)


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
    # Canonical parameter names in positional order (matches upstream).
    _names = (
        "u",
        "delta",
        "A",
        "B",
        "C",
        "D",
        "z",
        "delta_bias",
        "delta_softplus",
        "query_start_loc",
        "cache_seqlens",
        "head_dim",
    )

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
    D: torch.Tensor | None = _D if isinstance(_D, torch.Tensor) else None  # type: ignore[assignment]
    z: torch.Tensor | None = _z if isinstance(_z, torch.Tensor) else None  # type: ignore[assignment]
    delta_bias: torch.Tensor | None = _delta_bias if isinstance(_delta_bias, torch.Tensor) else None  # type: ignore[assignment]
    delta_softplus: bool = bool(_delta_softplus)
    query_start_loc: torch.Tensor | None = _query_start_loc if isinstance(_query_start_loc, torch.Tensor) else None  # type: ignore[assignment]

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
