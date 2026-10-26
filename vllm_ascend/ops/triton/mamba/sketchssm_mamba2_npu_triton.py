# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Mamba-2 decode kernels (Triton).

Non-flush rows read their sketch; flush rows replay the window into the full
state as ReplaySSM does, and their sketches are rebuilt afterwards. B and C
arrive unrotated: the ring and ``bc_pre`` use them as they are, the sketch and
state reads take the FP32 query R·C, and a flush folds the FP32 rotated
window keys R·B_t into the rotated state.
"""

import os

import torch
from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton.mamba.sketchssm_mamba2_npu import (
    SKETCH_PIVOTS,
    SketchArgs,
    run_with_flush,
    sketch_bc_pre,
    sketch_build,
    sketch_query,
    sketch_window_keys,
    sketch_window_scratch,
)

_PIVOTS = tl.constexpr(SKETCH_PIVOTS)


@triton.jit
def _window(
    dt_cache_ptr,
    x_cache_ptr,
    dt_cur,
    x_cur,
    A,
    wp,
    offs_m,
    dim,
    stride_dt_cache_pos,
    stride_x_cache_pos,
    stride_x_cache_dim,
    BLOCK_K: tl.constexpr,
):
    # Window weights s_t = dt_t exp(A (sum dt - cumsum dt)_t).
    offs_k = tl.arange(0, BLOCK_K)
    dt_all = tl.load(dt_cache_ptr + offs_k * stride_dt_cache_pos, mask=offs_k < wp, other=0.0).to(tl.float32)
    dt_all = tl.where(offs_k == wp, dt_cur, dt_all)
    total = A * tl.sum(dt_all, axis=0)
    scale = dt_all * tl.exp(total - A * tl.cumsum(dt_all, axis=0))
    scale = tl.where(offs_k <= wp, scale, 0.0)
    x_all = tl.load(
        x_cache_ptr + offs_m[:, None] * stride_x_cache_dim + offs_k[None, :] * stride_x_cache_pos,
        mask=(offs_m[:, None] < dim) & (offs_k[None, :] < wp),
        other=0.0,
    )
    x_all = tl.where(offs_k[None, :] == wp, x_cur[:, None], x_all).to(tl.float32)
    return tl.exp(total), scale, x_all


@triton.jit
def _sketch_decode_kernel(
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    bc_pre_ptr,
    write_pos_ptr,
    is_flush_ptr,
    slots_ptr,
    meta_ptr,
    u_ptr,
    w_ptr,
    ag_ptr,
    rank_ptr,
    u_off_ptr,
    w_off_ptr,
    ag_off_ptr,
    null_block_id,
    dim,
    dstate,
    heads_per_group,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_x_cache_slot,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_slot,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    U_ROWS: tl.constexpr,
    W_ROWS: tl.constexpr,
    AG_COLS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # C_ptr is the FP32 query R·C; B is unrotated (ring append only).
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    slot = tl.load(slots_ptr + pid_b).to(tl.int64)
    if slot == null_block_id:
        return
    group = pid_h // heads_per_group
    wp = tl.load(write_pos_ptr + pid_b).to(tl.int32)
    # Flush rows run in _sketch_flush_kernel.
    if tl.load(is_flush_ptr + pid_b) != 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mmask = offs_m < dim
    nmask = offs_n < dstate

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    B_ptr += pid_b * stride_B_batch + group * stride_B_group
    C_ptr += pid_b * stride_C_batch + group * stride_C_group
    x_cache_ptr += slot * stride_x_cache_slot + pid_h * stride_x_cache_head
    dt_cache_ptr += slot * stride_dt_cache_slot + pid_h * stride_dt_cache_head
    B_cache_ptr += slot * stride_B_cache_slot + group * stride_B_cache_group

    dt_cur = tl.load(dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head)
    dt_cur = dt_cur.to(tl.float32)
    dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if DT_SOFTPLUS:
        dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    x_cur = tl.load(x_ptr + offs_m * stride_x_dim, mask=mmask, other=0.0)
    C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=nmask, other=0.0)
    C = C.to(tl.float32)
    decay, scale, x_all = _window(
        dt_cache_ptr, x_cache_ptr, dt_cur, x_cur, A, wp, offs_m, dim,
        stride_dt_cache_pos, stride_x_cache_pos, stride_x_cache_dim, BLOCK_K,
    )  # fmt: skip

    # y = decay * (U c) + sum_t s_t (B_t . C) x_t (dense heads: c = C).
    offs_k = tl.arange(0, BLOCK_K)
    bc = tl.load(
        bc_pre_ptr + pid_b * stride_bc_pre_batch + group * stride_bc_pre_group + offs_k,
        mask=offs_k <= wp,
        other=0.0,
    )
    y = tl.sum(x_all * (scale * bc)[None, :], axis=1)
    meta = tl.load(meta_ptr + pid_b).to(tl.int64)
    m = tl.load(rank_ptr + pid_h)
    u_base = u_ptr + (meta * U_ROWS + tl.load(u_off_ptr + pid_h)) * dim
    ag_base = ag_ptr + meta * (_PIVOTS + 1) * AG_COLS
    ag_base += tl.load(ag_off_ptr + pid_h)
    offs_p = tl.arange(0, _PIVOTS)
    read = tl.zeros([BLOCK_M], dtype=tl.float32)
    # The rank is static per head, so a uniform branch per program selects a
    # straight-line read instead of a loop whose trip count is a runtime value.
    # A runtime bound keeps the row offsets symbolic, which scalarizes the u
    # tile loads; pinning the block widths keeps them vectorizable.
    if m == 0:
        for j0 in range(0, dstate, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            jmask = offs_j < dstate
            q_j = tl.load(
                C_ptr + offs_j * stride_C_dstate,
                mask=jmask,
                other=0.0,
            ).to(tl.float32)
            u_tile = tl.load(
                u_base + offs_j[:, None] * dim + offs_m[None, :],
                mask=jmask[:, None] & mmask[None, :],
                other=0.0,
            )
            read += tl.sum(u_tile.to(tl.float32) * q_j[:, None], axis=0)
    else:
        w_base = w_ptr + (meta * W_ROWS + tl.load(w_off_ptr + pid_h)) * dstate
        maps = tl.load(
            w_base + offs_p[:, None] * dstate + offs_n[None, :],
            mask=(offs_p[:, None] < tl.minimum(m, _PIVOTS)) & nmask[None, :],
            other=0.0,
        )
        dots = tl.sum(maps.to(tl.float32) * C[None, :], axis=1)
        if m <= _PIVOTS:
            offs_j = tl.arange(0, _PIVOTS)
            jmask = offs_j < m
            coef = tl.where(jmask, dots, 0.0)
            u_tile = tl.load(
                u_base + offs_j[:, None] * dim + offs_m[None, :],
                mask=jmask[:, None] & mmask[None, :],
                other=0.0,
            )
            read += tl.sum(u_tile.to(tl.float32) * coef[:, None], axis=0)
        elif m <= 2 * _PIVOTS:
            offs_j = tl.arange(0, 2 * _PIVOTS)
            jmask = offs_j < m
            q_j = tl.load(
                C_ptr + offs_j * stride_C_dstate,
                mask=jmask,
                other=0.0,
            ).to(tl.float32)
            a_j = tl.load(ag_base + offs_j, mask=jmask, other=0.0)
            g_j = tl.load(
                ag_base + (1 + offs_p[None, :]) * AG_COLS + offs_j[:, None],
                mask=jmask[:, None],
                other=0.0,
            )
            coef = a_j * q_j + tl.sum(g_j * dots[None, :], axis=1)
            u_tile = tl.load(
                u_base + offs_j[:, None] * dim + offs_m[None, :],
                mask=jmask[:, None] & mmask[None, :],
                other=0.0,
            )
            read += tl.sum(u_tile.to(tl.float32) * coef[:, None], axis=0)
        else:
            for j0 in range(0, m, BLOCK_J):
                offs_j = j0 + tl.arange(0, BLOCK_J)
                jmask = offs_j < m
                q_j = tl.load(
                    C_ptr + offs_j * stride_C_dstate,
                    mask=jmask,
                    other=0.0,
                ).to(tl.float32)
                a_j = tl.load(ag_base + offs_j, mask=jmask, other=0.0)
                g_j = tl.load(
                    ag_base + (1 + offs_p[None, :]) * AG_COLS + offs_j[:, None],
                    mask=jmask[:, None],
                    other=0.0,
                )
                coef = a_j * q_j + tl.sum(g_j * dots[None, :], axis=1)
                u_tile = tl.load(
                    u_base + offs_j[:, None] * dim + offs_m[None, :],
                    mask=jmask[:, None] & mmask[None, :],
                    other=0.0,
                )
                read += tl.sum(u_tile.to(tl.float32) * coef[:, None], axis=0)
    y += decay * read

    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        y += x_cur.to(tl.float32) * D
    tl.store(
        out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head
        + offs_m * stride_out_dim,
        y,
        mask=mmask,
    )  # fmt: skip
    # Append the current token to the ring.
    tl.store(x_cache_ptr + wp * stride_x_cache_pos + offs_m * stride_x_cache_dim,
             x_cur, mask=mmask)  # fmt: skip
    if pid_m == 0:
        tl.store(dt_cache_ptr + wp * stride_dt_cache_pos, dt_cur)
        if pid_h % heads_per_group == 0:
            B_cur = tl.load(B_ptr + offs_n * stride_B_dstate, mask=nmask)
            tl.store(
                B_cache_ptr + wp * stride_B_cache_pos + offs_n * stride_B_cache_dstate,
                B_cur,
                mask=nmask,
            )


@triton.jit
def _sketch_flush_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    BW_ptr,
    write_pos_ptr,
    flush_rows_ptr,
    slots_ptr,
    null_block_id,
    batch,
    dim,
    dstate,
    heads_per_group,
    stride_BW_batch,
    stride_BW_group,
    stride_BW_pos,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_x_cache_slot,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_slot,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    DT_SOFTPLUS: tl.constexpr,
    HAS_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # ReplaySSM flush of the rows in flush_rows (-1 padded) in the rotated
    # frame: S = decay * S_0 + sum_t s_t x_t (R B_t)^T, R B_t from BW.
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(2)
    group = pid_h // heads_per_group
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mmask = offs_m < dim
    nmask = offs_n < dstate
    # The list holds the flush rows first, so a program stops at its first -1.
    # Row-parallel: one program per (dim block, row, head), gated by
    # is_flush; the flush_rows list walk (a while loop around the dots)
    # times the AI Core out on triton-ascend when many rows share a
    # window origin (batch >= 32).
    row = tl.program_id(1)
    if tl.load(flush_rows_ptr + row) == 0:  # is_flush flags, not a row list
        return
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot != null_block_id:
        wp = tl.load(write_pos_ptr + row).to(tl.int32)
        x_ring = x_cache_ptr + slot * stride_x_cache_slot
        x_ring += pid_h * stride_x_cache_head
        dt_ring = dt_cache_ptr + slot * stride_dt_cache_slot
        dt_ring += pid_h * stride_dt_cache_head
        B_ring = B_cache_ptr + slot * stride_B_cache_slot
        B_ring += group * stride_B_cache_group
        B_row = B_ptr + row * stride_B_batch + group * stride_B_group
        dt_cur = tl.load(dt_ptr + row * stride_dt_batch + pid_h * stride_dt_head)
        dt_cur = dt_cur.to(tl.float32)
        dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
        if DT_SOFTPLUS:
            dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
        A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
        x_cur = tl.load(
            x_ptr + row * stride_x_batch + pid_h * stride_x_head
            + offs_m * stride_x_dim,
            mask=mmask,
            other=0.0,
        )  # fmt: skip
        C = tl.load(
            C_ptr + row * stride_C_batch + group * stride_C_group
            + offs_n * stride_C_dstate,
            mask=nmask,
            other=0.0,
        ).to(tl.float32)  # fmt: skip
        decay, scale, x_all = _window(
            dt_ring, x_ring, dt_cur, x_cur, A, wp, offs_m, dim,
            stride_dt_cache_pos, stride_x_cache_pos, stride_x_cache_dim,
            BLOCK_K,
        )  # fmt: skip
        B_all = tl.load(
            BW_ptr
            + row.to(tl.int64) * stride_BW_batch
            + group * stride_BW_group
            + offs_k[:, None] * stride_BW_pos
            + offs_n[None, :],
            mask=(offs_k[:, None] <= wp) & nmask[None, :],
            other=0.0,
        )
        s_ptrs = (
            state_ptr
            + slot * stride_state_slot
            + pid_h * stride_state_head
            + offs_m[:, None] * stride_state_dim
            + offs_n[None, :] * stride_state_dstate
        )
        smask = mmask[:, None] & nmask[None, :]
        s = tl.load(s_ptrs, mask=smask, other=0.0) * decay
        s += tl.dot(x_all * scale[None, :], B_all, input_precision=DOT_PRECISION)
        tl.store(s_ptrs, s, mask=smask)
        y = tl.sum(s * C[None, :], axis=1)
        if HAS_D:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
            y += x_cur.to(tl.float32) * D
        tl.store(
            out_ptr + row * stride_out_batch + pid_h * stride_out_head
            + offs_m * stride_out_dim,
            y,
            mask=mmask,
        )  # fmt: skip
        # Append the current token to the ring.
        tl.store(
            x_ring + wp * stride_x_cache_pos + offs_m * stride_x_cache_dim,
            x_cur,
            mask=mmask,
        )
        if pid_m == 0:
            tl.store(dt_ring + wp * stride_dt_cache_pos, dt_cur)
            if pid_h % heads_per_group == 0:
                B_cur = tl.load(B_row + offs_n * stride_B_dstate, mask=nmask)
                tl.store(
                    B_ring + wp * stride_B_cache_pos
                    + offs_n * stride_B_cache_dstate,
                    B_cur,
                    mask=nmask,
                )  # fmt: skip


def sketch_triton_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    bc_pre: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    flush_rows: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    out: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """One SketchSSM decode step of a Mamba-2 layer.

    Shapes follow ``selective_state_update_replayssm_output_only``; B/C are
    unrotated (rotated here by ``sketch.frames_t``) and ``flush_rows`` is -1
    padded.
    """
    batch = x.shape[0]
    if batch == 0:
        return
    if slots.dim() == 2:
        slots = slots[:, 0]
    _, heads, dim, dstate = state.shape
    L = x_cache.shape[2]
    t = sketch.tables
    sketch_bc_pre(B, C, B_cache, write_pos, is_flush, bc_pre, slots, null_block_id)
    query = sketch_query(C, sketch.frames_t)
    window_keys = sketch_window_scratch(B, B_cache)

    block_k = max(16, triton.next_power_of_2(L))
    block_n = triton.next_power_of_2(dstate)
    common = (
        *x.stride(), dt.stride(0), dt.stride(1), dt_bias.stride(0), A.stride(0),
        *B.stride(), *query.stride(), D.stride(0) if D is not None else 0,
        *out.stride(), x_cache.stride(0), x_cache.stride(1), x_cache.stride(2),
        x_cache.stride(3), *dt_cache.stride(), *B_cache.stride(),
    )  # fmt: skip

    def nonflush() -> None:
        block_m = min(64, triton.next_power_of_2(dim))
        _sketch_decode_kernel[(triton.cdiv(dim, block_m), batch, heads)](
            x, dt, dt_bias, A, B, query, D, out, x_cache, dt_cache, B_cache,
            bc_pre, write_pos, is_flush, slots, meta, sketch.u, sketch.w,
            sketch.ag, t.ranks, t.u_offsets, t.w_offsets, t.ag_offsets,
            null_block_id, dim, dstate, heads // B.shape[1], *common,
            bc_pre.stride(0), bc_pre.stride(1),
            U_ROWS=sketch.u.shape[1], W_ROWS=sketch.w.shape[1],
            AG_COLS=sketch.ag.shape[2], DT_SOFTPLUS=True, HAS_D=D is not None,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_J=16,
            num_warps=1,
        )  # fmt: skip

    def flush() -> None:
        block_m = min(32, triton.next_power_of_2(dim))
        if not os.environ.get("SKETCHSSM_DEBUG_SKIP_WINDOW_KEYS"):
            sketch_window_keys(B, B_cache, write_pos, slots,
                               sketch.frames_t, window_keys, null_block_id,
                               is_flush=is_flush)  # fmt: skip
        if not os.environ.get("SKETCHSSM_DEBUG_SKIP_FLUSH_KERNEL"):
            _sketch_flush_kernel[(triton.cdiv(dim, block_m), batch, heads)](
            state, x, dt, dt_bias, A, B, query, D, out, x_cache, dt_cache,
            B_cache, window_keys, write_pos, is_flush, slots, null_block_id,
            batch, dim, dstate, heads // B.shape[1], *window_keys.stride()[:3],
            *state.stride(), *common,
            DT_SOFTPLUS=True, HAS_D=D is not None, BLOCK_M=block_m,
            BLOCK_N=block_n, BLOCK_K=block_k,
            DOT_PRECISION="ieee",  # NPU subtree: tf32x3 is unsupported on triton-ascend
            num_warps=2,
        )  # fmt: skip
        if not os.environ.get("SKETCHSSM_DEBUG_SKIP_FLUSH_BUILD"):
            sketch_build(
                state,
                is_flush,
                slots,
                meta,
                sketch,
                null_block_id,
            )

    if has_flush_rows and not os.environ.get("SKETCHSSM_DEBUG_DISABLE_FLUSH"):
        if os.environ.get("SKETCHSSM_DEBUG_FLUSH_SYNC"):
            flush()
            nonflush()
        else:
            run_with_flush(flush, nonflush, device=state.device)
    else:
        nonflush()
