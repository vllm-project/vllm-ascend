# SPDX-License-Identifier: Apache-2.0
"""GDN-only RMSNorm/gate for strided token/head Z views."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gdn_gated_norm_kernel(
    X,
    Z,
    W,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    ZT: tl.constexpr,
    ZH: tl.constexpr,
    EPS: tl.constexpr,
    BEFORE: tl.constexpr,
    SWISH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    mask = rows[:, None] < M
    xo = (rows // H) * XT + (rows % H) * XH
    zo = (rows // H) * ZT + (rows % H) * ZH
    x = tl.load(X + xo[:, None] + cols[None, :], mask, other=0).to(tl.float32)
    z = tl.load(Z + zo[:, None] + cols[None, :], mask, other=0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    gate = tl.sigmoid(z)
    if SWISH:
        gate *= z
    if not BEFORE:
        x *= gate
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)
    y = x * rstd[:, None] * w[None, :]
    if BEFORE:
        y *= gate
    tl.store(Y + rows[:, None] * D + cols[None, :], y, mask)


def gdn_gated_norm(x, z, weight, eps, norm_before_gate, activation):
    """Normalize GDN head rows without flattening/copying a QKVZ slice.

    The GDN dispatcher validates the restricted shape/layout contract.
    This inference-only kernel has no global registration or shared-norm hook.
    """
    tokens, heads, dim = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    block_m = 64
    _gdn_gated_norm_kernel[(triton.cdiv(tokens * heads, block_m),)](
        x,
        z,
        weight,
        out,
        tokens * heads,
        heads,
        dim,
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        eps,
        norm_before_gate,
        activation in ("swish", "silu"),
        block_m,
    )
    return out
