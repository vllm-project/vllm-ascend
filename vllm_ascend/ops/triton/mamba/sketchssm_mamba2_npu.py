# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM for Mamba-2: sketch storage, B/C rotation and the sketch build.

The state is kept key-major, ``(slot, head, dstate, dim)``, in coordinates
rotated by a calibrated frame R per group. Decode keeps B and C unrotated
(the B ring holds the BF16 B as ReplaySSM does, ``bc_pre`` is B·C) and
rotates in FP32 only what meets the rotated state: the query R·C and, at a
flush, the window keys R·B_t. Per request, a head of rank ``m``
(0 = dense) keeps ``u`` (its leading ``m`` state rows, all rows if dense),
``w`` (``min(m, 4)`` coefficient-map rows) and, for ``m > 4``, the FP32
factors ``ag``. Sketch buffers are indexed by the persistent request index.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

SKETCH_PIVOTS = 4
# Ridge of the sketch coefficient map solve (the build kernel inlines it).
SKETCH_RIDGE = 0.003


def sketch_rows(ranks: torch.Tensor, state_size: int) -> tuple[int, int, int]:
    """``(u rows, w rows, ag columns)`` of one layer's ranks."""
    ranks = ranks.long()
    return (
        int(torch.where(ranks == 0, state_size, ranks).sum()),
        int(ranks.clamp(max=SKETCH_PIVOTS).sum()),
        int(ranks.sum()),
    )


def sketch_shapes(
    ranks: torch.Tensor, head_dim: int, state_size: int
) -> tuple[tuple[int, ...], ...]:
    """Per-request ``(u, w, ag)`` shapes of one layer."""
    u_rows, w_rows, ag_cols = (max(n, 1) for n in sketch_rows(ranks, state_size))
    return (
        (u_rows, head_dim),
        (w_rows, state_size),
        (SKETCH_PIVOTS + 1, ag_cols),
    )


SKETCH_DTYPES = (torch.bfloat16, torch.bfloat16, torch.float32)


class SketchTables(torch.nn.Module):
    """Per-head offsets of one layer's packed sketch."""

    def __init__(self, ranks: torch.Tensor, state_size: int):
        super().__init__()
        ranks = ranks.to(torch.int32).cpu()
        u_rows = torch.where(ranks == 0, state_size, ranks)
        pivots = ranks.clamp(max=SKETCH_PIVOTS)
        tables = {
            "ranks": ranks,
            "ag_offsets": ranks.cumsum(0) - ranks,
            "u_offsets": u_rows.cumsum(0) - u_rows,
            "w_offsets": pivots.cumsum(0) - pivots,
            # Dense heads keep all their state rows in U.
            "dense_rows": torch.where(ranks == 0, state_size, 0),
        }
        device = torch.get_default_device()
        for name, value in tables.items():
            self.register_buffer(
                name, value.to(torch.int32).to(device), persistent=False
            )


@triton.jit(do_not_specialize=["T"])
def _rot_pair_kernel(
    x_ptr, y_ptr, r_ptr, T, SX, SY, N: tl.constexpr, BT: tl.constexpr,
    G: tl.constexpr,
):  # fmt: skip
    # Programs pg < G rotate B's groups, the rest C's.
    pt = tl.program_id(0)
    pg = tl.program_id(1)
    offs_t = pt * BT + tl.arange(0, BT)
    on = tl.arange(0, N)
    mask = offs_t[:, None] < T
    x = tl.load(x_ptr + offs_t[:, None] * SX + pg * N + on[None, :], mask, 0.0)
    r = tl.load(r_ptr + (pg % G) * N * N + on[:, None] * N + on[None, :])
    y = tl.dot(x.to(tl.float32), r, input_precision="ieee")
    tl.store(
        y_ptr + offs_t[:, None] * SY + pg * N + on[None, :],
        y.to(y_ptr.dtype.element_ty),
        mask,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _rotate_groups_kernel(
    x_ptr,
    y_ptr,
    frames_t_ptr,
    num_tokens,
    stride_x_token,
    stride_x_elem,
    stride_y_token,
    stride_y_elem,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr = None,
):
    # y[t, g, cols] <- x[t, g] @ R_g^T[:, cols] (BLOCK_C output columns).
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, N)
    if BLOCK_C is None:
        offs_c = offs_n
    else:
        offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_t[:, None] < num_tokens
    cols = pid_g * N + offs_n[None, :]
    x = tl.load(
        x_ptr + offs_t[:, None] * stride_x_token + cols * stride_x_elem, mask, 0.0
    )
    r = tl.load(frames_t_ptr + pid_g * N * N + offs_n[:, None] * N + offs_c[None, :])
    y = tl.dot(x.to(tl.float32), r, input_precision="ieee")
    cols = pid_g * N + offs_c[None, :]
    tl.store(
        y_ptr + offs_t[:, None] * stride_y_token + cols * stride_y_elem,
        y.to(y_ptr.dtype.element_ty),
        mask,
    )


def _rotate(B, C, frames_t, B_out, C_out):
    num_tokens = B.shape[0]
    groups, n, _ = frames_t.shape
    if num_tokens == 0:
        return
    if (
        B.stride() == C.stride()
        and B.stride(1) == 1
        and B.data_ptr() + B.shape[1] * B.element_size() == C.data_ptr()
        and B_out.stride() == C_out.stride()
        and B_out.data_ptr() + B_out.shape[1] * B_out.element_size() == C_out.data_ptr()
    ):
        # Adjacent B/C column slices: one launch for both.
        _rot_pair_kernel[(triton.cdiv(num_tokens, 32), 2 * groups)](
            B, B_out, frames_t, num_tokens, B.stride(0), B_out.stride(0), n, 32,
            groups, num_warps=4,
        )  # fmt: skip
        return
    for x, y in ((B, B_out), (C, C_out)):
        _rotate_groups_kernel[(triton.cdiv(num_tokens, 64), groups)](
            x, y, frames_t, num_tokens, x.stride(0), x.stride(1), y.stride(0),
            y.stride(1), n, 64, num_warps=4,
        )  # fmt: skip


def sketch_rotate(
    B: torch.Tensor, C: torch.Tensor, frames_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 copies of grouped B and C rotated by the transposed ``frames_t``."""
    BC = torch.empty(B.shape[0], 2 * B.shape[1], dtype=torch.float32, device=B.device)
    B_out, C_out = BC.split(B.shape[1], dim=1)
    _rotate(B, C, frames_t, B_out, C_out)
    return B_out, C_out


def sketch_rotate_(B: torch.Tensor, C: torch.Tensor, frames_t: torch.Tensor):
    """Rotate grouped B and C in place by the transposed ``frames_t``."""
    _rotate(B, C, frames_t, B, C)


@triton.jit
def _rotate_states_kernel(
    x_ptr,
    y_ptr,
    frames_t_ptr,
    heads,
    heads_per_group,
    stride_x_item,
    stride_x_head,
    stride_x_dim,
    stride_x_dstate,
    stride_y_item,
    stride_y_head,
    stride_y_dim,
    DIM: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INVERSE: tl.constexpr,
):
    # y[i, h, p, cols] = x[i, h, p] @ R_g^T[:, cols] (R_g[:, cols] if INVERSE).
    # NPU: "ieee" dots; the upstream tf32x3 path is unproven on triton-ascend.
    item = tl.program_id(0) // heads
    head = tl.program_id(0) % heads
    group = head // heads_per_group
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, N)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_p[:, None] < DIM
    x = tl.load(
        x_ptr
        + item * stride_x_item
        + head * stride_x_head
        + offs_p[:, None] * stride_x_dim
        + offs_n[None, :] * stride_x_dstate,
        mask=mask,
        other=0.0,
    )
    frame = frames_t_ptr + group * N * N
    if INVERSE:
        r = tl.load(frame + offs_c[None, :] * N + offs_n[:, None])
    else:
        r = tl.load(frame + offs_n[:, None] * N + offs_c[None, :])
    y = tl.dot(x, r, input_precision="ieee")
    tl.store(
        y_ptr
        + item * stride_y_item
        + head * stride_y_head
        + offs_p[:, None] * stride_y_dim
        + offs_c[None, :],
        y,
        mask=mask,
    )


def sketch_rotate_states(
    states: torch.Tensor, frames_t: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """States ``(items, heads, dim, N)`` FP32 in the rotated frame, ``S R^T``
    (``S R`` back to the model frame if ``inverse``).

    Prefill runs the unrotated chunk scan; its states enter the cache rotated.
    """
    items, heads, dim, n = states.shape
    assert states.dtype == torch.float32 and frames_t.is_contiguous()
    out = torch.empty(states.shape, dtype=torch.float32, device=states.device)
    if items == 0:
        return out
    block_p, block_c = 32, min(n, 64)
    grid = (items * heads, triton.cdiv(dim, block_p), n // block_c)
    _rotate_states_kernel[grid](
        states, out, frames_t, heads, heads // frames_t.shape[0],
        *states.stride(), *out.stride()[:3], DIM=dim, N=n, BLOCK_P=block_p,
        BLOCK_C=block_c, INVERSE=inverse, num_warps=4,
    )  # fmt: skip
    return out


def sketch_query(C: torch.Tensor, frames_t: torch.Tensor) -> torch.Tensor:
    """FP32 rotated query R·C ``(batch, groups, dstate)`` of the decode C."""
    batch, groups, n = C.shape
    assert C.stride(1) == n * C.stride(2)
    out = torch.empty(batch, groups, n, dtype=torch.float32, device=C.device)
    if batch:
        block_c = min(n, 64)
        _rotate_groups_kernel[(triton.cdiv(batch, 32), groups, n // block_c)](
            C, out, frames_t, batch, C.stride(0), C.stride(2), out.stride(0), 1,
            n, 32, block_c, num_warps=4,
        )  # fmt: skip
    return out


@triton.jit
def _window_keys_rows_kernel(
    B_ptr,
    B_cache_ptr,
    frames_t_ptr,
    out_ptr,
    write_pos_ptr,
    is_flush_ptr,
    slots_ptr,
    null_block_id,
    stride_B_batch,
    stride_B_group,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_out_batch,
    stride_out_group,
    stride_out_pos,
    W: tl.constexpr,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # out[row, g, t, cols] <- B_t @ R_g^T[:, cols] for t <= wp (ring rows,
    # then the current B) of one flush row; one program per (row, group,
    # column block). Row-parallel instead of walking a row list in a while
    # loop: the list walk with an in-loop dot hangs on triton-ascend.
    row = tl.program_id(0)
    if tl.load(is_flush_ptr + row) == 0:
        return
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot == null_block_id:
        return
    pid_g = tl.program_id(1)
    on = tl.arange(0, N)
    oc = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    r = tl.load(frames_t_ptr + pid_g * N * N + on[:, None] * N + oc[None, :])
    wp = tl.load(write_pos_ptr + row).to(tl.int32)
    ring = B_cache_ptr + slot * stride_B_cache_slot
    ring += pid_g * stride_B_cache_group
    dst = out_ptr + row.to(tl.int64) * stride_out_batch
    dst += pid_g * stride_out_group
    cur = tl.load(B_ptr + row.to(tl.int64) * stride_B_batch
                  + pid_g * stride_B_group + on)  # fmt: skip
    for t0 in tl.static_range(0, W, BLOCK_T):
        if t0 <= wp:
            ot = t0 + tl.arange(0, BLOCK_T)
            keys = tl.load(
                ring + ot[:, None] * stride_B_cache_pos + on[None, :],
                mask=ot[:, None] < wp,
                other=0.0,
            )
            keys = tl.where(ot[:, None] == wp, cur[None, :], keys)
            y = tl.dot(keys.to(tl.float32), r, input_precision="ieee")
            tl.store(
                dst + ot[:, None] * stride_out_pos + oc[None, :],
                y,
                mask=ot[:, None] <= wp,
            )


def sketch_window_keys(
    B: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    slots: torch.Tensor,
    frames_t: torch.Tensor,
    out: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
    *,
    is_flush: torch.Tensor,
) -> None:
    """FP32 rotated window keys ``out[row, group, t] = R B_t`` (t <= write_pos,
    from the unrotated ring and the current B) of every flush row."""
    batch, groups, n = B.shape
    if batch == 0:
        return
    assert B.stride(2) == 1 and B_cache.stride(3) == 1 and out.stride(3) == 1
    block_n = min(n, 64)
    _window_keys_rows_kernel[(batch, groups, n // block_n)](
        B, B_cache, frames_t, out, write_pos, is_flush, slots, null_block_id,
        B.stride(0), B.stride(1), *B_cache.stride()[:3],
        *out.stride()[:3], W=B_cache.shape[2], N=n, BLOCK_T=16,
        BLOCK_N=block_n, num_warps=4,
    )  # fmt: skip


def sketch_window_scratch(B: torch.Tensor, B_cache: torch.Tensor) -> torch.Tensor:
    """Per-step scratch of ``sketch_window_keys``: ``(batch, groups, W, N)``."""
    batch, groups, n = B.shape
    return torch.empty(
        batch, groups, B_cache.shape[2], n, dtype=torch.float32, device=B.device
    )


@triton.jit
def _inv_norm(w, u):
    # 1 / |w|, or 0 when w keeps almost nothing of u.
    e = tl.sum(w * w)
    keep = e > 1.0e-12 * tl.sum(u * u)
    return tl.where(keep, 1.0 / tl.sqrt(tl.where(e > 0.0, e, 1.0)), 0.0)


@triton.jit
def _pivot_scalars(u0, u1, u2, u3, m):
    # Gram-Schmidt coefficients of the pivot columns.
    z = tl.sum(u0 * 0.0)
    e0 = tl.sum(u0 * u0)
    n0 = 1.0 / tl.sqrt(tl.where(e0 > 0.0, e0, 1.0))
    p10, p11, n1 = z, z, z
    p20, p21, p22, p23, n2 = z, z, z, z, z
    p30, p31, p32, p33, p34, p35, n3 = z, z, z, z, z, z, z
    v0 = u0 * n0
    v1 = u0 * 0.0
    v2 = u0 * 0.0
    if m > 1:
        p10 = tl.sum(v0 * u1)
        w1 = u1 - v0 * p10
        p11 = tl.sum(v0 * w1)
        w1 = w1 - v0 * p11
        n1 = _inv_norm(w1, u1)
        v1 = w1 * n1
    if m > 2:
        p20 = tl.sum(v0 * u2)
        w2 = u2 - v0 * p20
        p21 = tl.sum(v1 * w2)
        w2 = w2 - v1 * p21
        p22 = tl.sum(v0 * w2)
        w2 = w2 - v0 * p22
        p23 = tl.sum(v1 * w2)
        w2 = w2 - v1 * p23
        n2 = _inv_norm(w2, u2)
        v2 = w2 * n2
    if m > 3:
        p30 = tl.sum(v0 * u3)
        w3 = u3 - v0 * p30
        p31 = tl.sum(v1 * w3)
        w3 = w3 - v1 * p31
        p32 = tl.sum(v2 * w3)
        w3 = w3 - v2 * p32
        p33 = tl.sum(v0 * w3)
        w3 = w3 - v0 * p33
        p34 = tl.sum(v1 * w3)
        w3 = w3 - v1 * p34
        p35 = tl.sum(v2 * w3)
        w3 = w3 - v2 * p35
        n3 = _inv_norm(w3, u3)
    return n0, p10, p11, n1, p20, p21, p22, p23, n2, p30, p31, p32, p33, p34, p35, n3


@triton.jit
def _pivot_directions(u0, u1, u2, u3, n0, p10, p11, n1, p20, p21, p22, p23, n2, p30,
                      p31, p32, p33, p34, p35, n3):  # fmt: skip
    v0 = u0 * n0
    v1 = (u1 - v0 * p10 - v0 * p11) * n1
    v2 = (u2 - v0 * p20 - v1 * p21 - v0 * p22 - v1 * p23) * n2
    v3 = (u3 - v0 * p30 - v1 * p31 - v2 * p32 - v0 * p33 - v1 * p34 - v2 * p35) * n3
    return v1, v2, v3


@triton.jit
def _sketch_head(
    base,
    meta,
    h,
    u,
    tail,
    ag,
    widths,
    offsets,
    u_offsets,
    w_offsets,
    SV: tl.constexpr,
    SK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    V_PAD: tl.constexpr,
    SM: tl.constexpr,
    UROWS: tl.constexpr,
    WROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Sketch of one head from its state at ``base``, streamed in value chunks.
    P: tl.constexpr = 4
    j = tl.arange(0, K)
    m = tl.load(widths + h).to(tl.int32)
    urow = u + (meta.to(tl.int64) * UROWS + tl.load(u_offsets + h) + j[None, :]) * V
    if m == 0:
        for c in tl.static_range(0, V, BLOCK_V):
            vc = c + tl.arange(0, BLOCK_V)
            mask = vc[:, None] < V
            s = tl.load(base + vc[:, None] * SV + j[None, :] * SK, mask, 0.0)
            tl.store(urow + vc[:, None], s.to(u.dtype.element_ty), mask)
    else:
        off = tl.load(offsets + h).to(tl.int32)
        pv = tl.arange(0, V_PAD)
        u0 = tl.load(base + pv * SV, pv < V, 0.0).to(tl.float32)
        u1 = tl.load(base + pv * SV + SK, pv < V, 0.0).to(tl.float32)
        u2 = tl.load(base + pv * SV + 2 * SK, pv < V, 0.0).to(tl.float32)
        u3 = tl.load(base + pv * SV + 3 * SK, pv < V, 0.0).to(tl.float32)
        n0, p10, p11, n1, p20, p21, p22, p23, n2, p30, p31, p32, p33, p34, p35, n3 = (
            _pivot_scalars(u0, u1, u2, u3, m)
        )
        energy = tl.full((K,), 0.0, tl.float32)
        cross = tl.full((K,), 0.0, tl.float32)
        z1 = tl.full((K,), 0.0, tl.float32)
        z2 = tl.full((K,), 0.0, tl.float32)
        z3 = tl.full((K,), 0.0, tl.float32)
        for c in tl.static_range(0, V, BLOCK_V):
            vc = c + tl.arange(0, BLOCK_V)
            vmask = vc < V
            s = tl.load(
                base + vc[:, None] * SV + j[None, :] * SK, vmask[:, None], 0.0
            ).to(tl.float32)
            tl.store(
                urow + vc[:, None],
                s.to(u.dtype.element_ty),
                vmask[:, None] & (j[None, :] < m),
            )
            c0 = tl.load(base + vc * SV, vmask, 0.0).to(tl.float32)
            energy += tl.sum(s * s, axis=0)
            cross += tl.sum(c0[:, None] * s, axis=0)
            if m > 1:
                c1 = tl.load(base + vc * SV + SK, vmask, 0.0).to(tl.float32)
                c2 = tl.load(base + vc * SV + 2 * SK, vmask, 0.0).to(tl.float32)
                c3 = tl.load(base + vc * SV + 3 * SK, vmask, 0.0).to(tl.float32)
                v1, v2, v3 = _pivot_directions(c0, c1, c2, c3, n0, p10, p11, n1, p20,
                                               p21, p22, p23, n2, p30, p31, p32, p33,
                                               p34, p35, n3)  # fmt: skip
                z1 += tl.sum(v1[:, None] * s, axis=0)
                if m > 2:
                    z2 += tl.sum(v2[:, None] * s, axis=0)
                if m > 3:
                    z3 += tl.sum(v3[:, None] * s, axis=0)
        mean = tl.sum(energy) / K
        safe_mean = tl.where(mean > 0.0, mean, 1.0)
        pivot_energy = tl.sum(tl.where(j == 0, energy, 0.0))
        e = energy / safe_mean
        z0 = (
            cross
            / tl.sqrt(tl.where(pivot_energy > 0.0, pivot_energy, 1.0))
            / tl.sqrt(safe_mean)
        )
        z0 = tl.where(pivot_energy > 0.0, z0, 0.0)
        z1 = z1 / tl.sqrt(safe_mean)
        z2 = z2 / tl.sqrt(safe_mean)
        z3 = z3 / tl.sqrt(safe_mean)
        residual = tl.maximum(e - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3, 0.0)
        residual = tl.where(j < tl.minimum(m, P), 0.0, residual)
        denominator = residual + 0.003
        a = residual / denominator
        b0 = tl.where(j < m, z0 / denominator, 0.0)
        g0 = b0 / (1.0 + tl.sum(z0 * b0))
        g1 = tl.full((K,), 0.0, tl.float32)
        g2 = tl.full((K,), 0.0, tl.float32)
        g3 = tl.full((K,), 0.0, tl.float32)
        if m > 1:
            b1 = tl.where(j < m, z1 / denominator, 0.0)
            l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
            l1_0 = (tl.sum(z1 * b0)) / l0_0
            l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
            y0 = b0 / l0_0
            y1 = (b1 - l1_0 * y0) / l1_1
            g1 = (y1) / l1_1
            g0 = (y0 - l1_0 * g1) / l0_0
            if m > 2:
                b2 = tl.where(j < m, z2 / denominator, 0.0)
                l2_0 = (tl.sum(z2 * b0)) / l0_0
                l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
                l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
                y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
                g2 = (y2) / l2_2
                g1 = (y1 - l2_1 * g2) / l1_1
                g0 = (y0 - l1_0 * g1 - l2_0 * g2) / l0_0
                if m > 3:
                    b3 = tl.where(j < m, z3 / denominator, 0.0)
                    l3_0 = (tl.sum(z3 * b0)) / l0_0
                    l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
                    l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
                    l3_3 = tl.sqrt(
                        1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
                    )
                    y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
                    g3 = (y3) / l3_3
                    g2 = (y2 - l3_2 * g3) / l2_2
                    g1 = (y1 - l2_1 * g2 - l3_1 * g3) / l1_1
                    g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3) / l0_0
        factor = tl.where(j < m, 0.003 / denominator, 1.0)
        tbase = tail + (meta.to(tl.int64) * WROWS + tl.load(w_offsets + h)) * K + j
        index = meta.to(tl.int64) * ((P + 1) * SM) + off + j
        if m <= 4:
            for n in range(m):
                f0 = tl.sum(tl.where(j == n, g0, 0.0))
                f1 = tl.sum(tl.where(j == n, g1, 0.0))
                f2 = tl.sum(tl.where(j == n, g2, 0.0))
                f3 = tl.sum(tl.where(j == n, g3, 0.0))
                merged = (f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3) * factor
                tl.store(tbase + n * K, merged.to(tail.dtype.element_ty))
        else:
            tl.store(tbase, (z0 * factor).to(tail.dtype.element_ty))
            tl.store(ag + index, a, j < m)
            tl.store(ag + index + SM, g0, j < m)
            if m > 1:
                tl.store(tbase + 1 * K, (z1 * factor).to(tail.dtype.element_ty))
                tl.store(ag + index + 2 * SM, g1, j < m)
            if m > 2:
                tl.store(tbase + 2 * K, (z2 * factor).to(tail.dtype.element_ty))
                tl.store(ag + index + 3 * SM, g2, j < m)
            if m > 3:
                tl.store(tbase + 3 * K, (z3 * factor).to(tail.dtype.element_ty))
                tl.store(ag + index + 4 * SM, g3, j < m)


@triton.jit(do_not_specialize=["slots", "metas", "flags", "rows", "batch"])
def _cold_build_kernel(
    state,
    u,
    tail,
    ag,
    widths,
    offsets,
    u_offsets,
    w_offsets,
    slots,
    metas,
    flags,
    rows,
    batch,
    null_block_id,
    SB: tl.constexpr,
    SH: tl.constexpr,
    SV: tl.constexpr,
    SK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    V_PAD: tl.constexpr,
    SM: tl.constexpr,
    UROWS: tl.constexpr,
    WROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ROW_LIST: tl.constexpr,
):
    # With ROW_LIST, programs walk ``rows`` and stop at the first -1 padding.
    h = tl.program_id(1)
    it = tl.program_id(0)
    if ROW_LIST:
        row = tl.load(rows + it, mask=it < batch, other=-1)
    else:
        row = tl.where(tl.load(flags + it) != 0, it, -1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int32)
        if slot != null_block_id:
            meta = tl.load(metas + row)
            _sketch_head(state + slot.to(tl.int64) * SB + h.to(tl.int64) * SH,
                         meta, h, u, tail, ag, widths, offsets, u_offsets,
                         w_offsets, SV, SK, K, V, V_PAD, SM, UROWS, WROWS,
                         BLOCK_V)  # fmt: skip
        if ROW_LIST:
            it += tl.num_programs(0)
            row = tl.load(rows + it, mask=it < batch, other=-1)
        else:
            row = row * 0 - 1


@triton.jit
def _bc_tile(
    B_ptr,
    C_ptr,
    B_cache_ptr,
    bc_pre_ptr,
    write_pos,
    stride_B_cache_pos,
    MAX_CACHE_LEN: tl.constexpr,
    DSTATE: tl.constexpr,
    DSTATE_PAD: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, DSTATE_PAD)
    nmask = offs_n < DSTATE
    B_cur = tl.load(B_ptr + offs_n, mask=nmask, other=0.0)
    C = tl.load(C_ptr + offs_n, mask=nmask, other=0.0)
    B_cache = tl.load(
        B_cache_ptr + offs_k[:, None] * stride_B_cache_pos + offs_n[None, :],
        mask=(offs_k[:, None] < write_pos) & nmask[None, :],
        other=0.0,
    )
    B_all = tl.where(offs_k[:, None] == write_pos, B_cur[None, :], B_cache)
    bc = tl.sum(B_all.to(tl.float32) * C[None, :].to(tl.float32), axis=1)
    tl.store(
        bc_pre_ptr + offs_k,
        bc,
        mask=(offs_k <= write_pos) & (offs_k < MAX_CACHE_LEN),
    )


@triton.jit
def _bc_pre_kernel(
    B_ptr,
    C_ptr,
    B_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    bc_pre_ptr,
    slots_ptr,
    null_block_id,
    batch,
    stride_B_batch,
    stride_B_group,
    stride_C_batch,
    stride_C_group,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    MAX_CACHE_LEN: tl.constexpr,
    DSTATE: tl.constexpr,
    DSTATE_PAD: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
):
    # Key tile size is chosen by the ring position.
    pid = tl.program_id(0)
    pid_b = pid % batch
    pid_g = pid // batch
    if tl.load(is_flush_ptr + pid_b) != 0:
        return
    slot = tl.load(slots_ptr + pid_b).to(tl.int64)
    if slot == null_block_id:
        return
    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)
    B_ptr += pid_b * stride_B_batch + pid_g * stride_B_group
    C_ptr += pid_b * stride_C_batch + pid_g * stride_C_group
    B_cache_ptr += slot * stride_B_cache_batch + pid_g * stride_B_cache_group
    bc_pre_ptr += pid_b * stride_bc_pre_batch + pid_g * stride_bc_pre_group
    if write_pos < 4:
        _bc_tile(B_ptr, C_ptr, B_cache_ptr, bc_pre_ptr, write_pos,
                 stride_B_cache_pos, MAX_CACHE_LEN, DSTATE, DSTATE_PAD, 4)  # fmt: skip
    elif write_pos < 8:
        _bc_tile(B_ptr, C_ptr, B_cache_ptr, bc_pre_ptr, write_pos,
                 stride_B_cache_pos, MAX_CACHE_LEN, DSTATE, DSTATE_PAD, 8)  # fmt: skip
    elif write_pos < 16:
        _bc_tile(B_ptr, C_ptr, B_cache_ptr, bc_pre_ptr, write_pos,
                 stride_B_cache_pos, MAX_CACHE_LEN, DSTATE, DSTATE_PAD, 16)  # fmt: skip
    else:
        _bc_tile(B_ptr, C_ptr, B_cache_ptr, bc_pre_ptr, write_pos,
                 stride_B_cache_pos, MAX_CACHE_LEN, DSTATE, DSTATE_PAD,
                 MAX_BLOCK)  # fmt: skip


def sketch_bc_pre(
    B: torch.Tensor,
    C: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    bc_pre: torch.Tensor,
    slots: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """``bc_pre[row, group, t] = B_t . C`` over each non-flush row's window."""
    batch, n_groups, dstate = B.shape
    if batch == 0:
        return
    _bc_pre_kernel[(batch * n_groups,)](
        B, C, B_cache, write_pos, is_flush, bc_pre, slots, null_block_id, batch,
        B.stride(0), B.stride(1), C.stride(0), C.stride(1), B_cache.stride(0),
        B_cache.stride(1), B_cache.stride(2), bc_pre.stride(0), bc_pre.stride(1),
        MAX_CACHE_LEN=B_cache.shape[2], DSTATE=dstate,
        DSTATE_PAD=triton.next_power_of_2(dstate),
        MAX_BLOCK=triton.next_power_of_2(B_cache.shape[2]), num_warps=2,
    )  # fmt: skip


@dataclass
class SketchArgs:
    """One layer's SketchSSM sketch, tables and frames for the decode."""

    u: torch.Tensor  # (num_reqs, u_rows, head_dim) bf16
    w: torch.Tensor  # (num_reqs, w_rows, state_size) bf16
    ag: torch.Tensor  # (num_reqs, 5, ag_cols) fp32
    tables: SketchTables
    frames_t: torch.Tensor  # (groups, state_size, state_size) fp32, R^T


# Rows per program when a launch walks a row list.
_ROWS_PER_PROGRAM = 4


def row_list_programs(batch: int, rows_per_program: int = _ROWS_PER_PROGRAM) -> int:
    return max(1, triton.cdiv(batch, rows_per_program))


def sketch_build(
    state: torch.Tensor,
    flags: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    rows: torch.Tensor | None = None,
    rows_per_program: int = _ROWS_PER_PROGRAM,
) -> None:
    """Build the sketch of every flagged row from its checkpoint state.

    ``rows`` is an optional ``flush_row_list`` of the flagged rows.
    """
    batch = flags.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    _, heads, head_dim, state_size = state.shape
    programs = batch if rows is None else row_list_programs(batch, rows_per_program)
    _cold_build_kernel[(programs, heads)](
        state, sketch.u, sketch.w, sketch.ag, t.ranks, t.ag_offsets, t.u_offsets,
        t.w_offsets, slots, meta, flags, flags if rows is None else rows, batch,
        null_block_id,
        SB=state.stride(0), SH=state.stride(1), SV=state.stride(2),
        SK=state.stride(3), K=state_size, V=head_dim,
        V_PAD=triton.next_power_of_2(head_dim),
        SM=sketch.ag.shape[2], UROWS=sketch.u.shape[1], WROWS=sketch.w.shape[1],
        BLOCK_V=32, ROW_LIST=rows is not None, num_warps=2,
    )  # fmt: skip


@functools.cache
def _flush_stream(device: torch.device):
    if device.type == "npu":
        return torch.npu.Stream(device)
    return torch.cuda.Stream(device)


def run_with_flush(
    flush: Callable[[], None],
    nonflush: Callable[[], None],
    device: torch.device | None = None,
) -> None:
    """Run ``flush`` on a side stream concurrently with ``nonflush``.

    They touch disjoint rows, state slots and sketches.
    """
    if device is None:
        device = torch.get_default_device()
    if device.type == "npu" or not torch.cuda.is_available():
        main = torch.npu.current_stream()
        side = _flush_stream(main.device)
        side.wait_stream(main)
        with torch.npu.stream(side):
            flush()
        nonflush()
        main.wait_stream(side)
        return
    main = torch.cuda.current_stream()
    side = _flush_stream(main.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        flush()
    nonflush()
    main.wait_stream(side)


def flush_row_list(is_flush: torch.Tensor) -> torch.Tensor:
    """``(batch,)`` int32 indices of the flush rows, then -1 padding."""
    rows = torch.full_like(is_flush, -1, dtype=torch.int32)
    idx = torch.nonzero(is_flush).flatten()
    rows[: idx.numel()] = idx.to(torch.int32)
    return rows
