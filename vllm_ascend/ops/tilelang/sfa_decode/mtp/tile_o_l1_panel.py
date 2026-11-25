# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from .native_output import MXFP8_NZ_C0, make_native_weight_loader


def panel_mx(
    batch,
    width,
    out_width,
    bm,
    bk,
    stages,
    span,
    panel,
    output_rows=None,
    tail_only=False,
    *,
    uniform_n=False,
    packed_output=False,
    native_output=False,
):
    assert panel % bk == 0 and width % panel == 0
    output_rows = batch if output_rows is None else output_rows
    assert batch - bm < output_rows <= batch
    tail = output_rows % bm
    assert not tail_only or tail
    groups = 1 if tail_only else output_rows // bm
    assert groups > 0
    store_rows = tail if tail_only else bm
    m_rows = ((store_rows + 15) // 16) * 16
    first_group = output_rows // bm if tail_only else 0
    segments = out_width // span
    granularity = MXFP8_NZ_C0 if native_output else 16
    base, extra = divmod(span // granularity, 28)
    nsmall = base * granularity
    nlarge = nsmall + granularity
    assert span % 16 == out_width % span == batch % bm == 0
    assert not uniform_n or (nsmall > 0 and extra > 0)
    fp8 = T.float8_e4m3fn
    trim_weight_panel = uniform_n and m_rows == 16
    weight_policy = "NOTALLOC_KEEP" if uniform_n and batch == 32 else "NORMAL_FV"

    weight_shape = (width // panel, out_width, panel) if packed_output else (out_width, width)
    if native_output:
        weight_shape = (out_width // MXFP8_NZ_C0, width, MXFP8_NZ_C0)
    scale_shape = (width // panel, out_width, panel // 64) if packed_output else (width // 64, out_width)

    load_native = make_native_weight_loader(panel)

    @T.macro
    def load_codes(WO, w, outer, col, first, end):
        if packed_output:
            T.copy(WO[outer, col + first : col + end, :], w[first:end, :], l2_cache_ctrl=weight_policy)
        else:
            T.copy(
                WO[col + first : col + end, outer * panel : (outer + 1) * panel],
                w[first:end, :],
                l2_cache_ctrl=weight_policy,
            )

    @T.macro
    def load_scales(OS, ws, outer, col, first, end):
        if packed_output:
            T.copy(OS[outer, col + first : col + end, :], ws[first:end, :])
        else:
            T.copy(
                OS[outer * (panel // 64) : (outer + 1) * (panel // 64), col + first : col + end],
                ws[first:end, :],
                transpose=True,
            )

    @T.macro
    def tile(gr, col, bn, m_rows, store_rows, C3, S3, WO, OS, Y, short=False):
        a = T.alloc_l1((m_rows, panel), fp8)
        w = T.alloc_l1((panel, bn) if native_output else (bn, panel), fp8)
        xs = T.alloc_l1((m_rows, panel // 64), T.uint16)
        ws = T.alloc_l1((bn, panel // 64), T.uint16)
        a0 = T.alloc_l0a((m_rows, bk), fp8)
        b0 = T.alloc_l0b((bn, bk), fp8)
        s0 = T.alloc_l0a_sf(a0)
        t0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((m_rows, bn), T.float32)
        for outer in T.Pipelined(width // panel, num_stages=stages):
            T.copy(C3[gr * bm : gr * bm + m_rows, outer * panel : (outer + 1) * panel], a)
            if native_output:
                load_native(WO, w, outer, col, bn)
            elif trim_weight_panel:
                load_codes(WO, w, outer, col, 0, nsmall)
                if not short:
                    load_codes(WO, w, outer, col, nsmall, bn)
            else:
                load_codes(WO, w, outer, col, 0, bn)
            T.copy(
                S3[outer * (panel // 64) : (outer + 1) * (panel // 64), gr * bm : gr * bm + m_rows], xs, transpose=True
            )
            if trim_weight_panel:
                load_scales(OS, ws, outer, col, 0, nsmall)
                if not short:
                    load_scales(OS, ws, outer, col, nsmall, bn)
            else:
                load_scales(OS, ws, outer, col, 0, bn)
            for inner in T.serial(panel // bk):
                T.copy(a[:, inner * bk : (inner + 1) * bk], a0)
                if native_output:
                    T.copy(w[inner * bk : (inner + 1) * bk, :], b0, transpose=True)
                else:
                    T.copy(w[:, inner * bk : (inner + 1) * bk], b0)
                T.copy(xs[:, inner * (bk // 64) : (inner + 1) * (bk // 64)], s0)
                T.copy(ws[:, inner * (bk // 64) : (inner + 1) * (bk // 64)], t0)
                T.gemm_blockscaled(a0, b0, acc, s0, t0, transpose_B=True, clear_accum=outer == 0 and inner == 0)
        # Fixpipe derives its source pitch from the stored rows rounded to 16.
        if short:
            T.copy(acc[:store_rows, :nsmall], Y[gr * bm : gr * bm + store_rows, col : col + nsmall])
        else:
            T.copy(acc[:store_rows, :], Y[gr * bm : gr * bm + store_rows, col : col + bn])

    @T.prim_func
    def main(
        C3: T.Tensor((batch, width), fp8),
        S3: T.Tensor((width // 64, batch), T.uint16),
        WO: T.Tensor(weight_shape, fp8),
        OS: T.Tensor(scale_shape, T.uint16),
        Y: T.Tensor((output_rows, out_width), T.bfloat16),
    ):
        with T.Kernel(groups * segments * 28) as task:
            gr = first_group + task // (28 * segments)
            segment = (task // 28) % segments
            lane = task % 28
            if uniform_n:
                # Short lanes come first so a full-width read stays within the segment.
                short_lanes = 28 - extra
                col = segment * span + lane * nsmall + T.max(lane - short_lanes, 0) * granularity
                tile(gr, col, nlarge, m_rows, store_rows, C3, S3, WO, OS, Y, lane < short_lanes)
            elif lane < extra:
                tile(gr, segment * span + lane * nlarge, nlarge, m_rows, store_rows, C3, S3, WO, OS, Y)
            else:
                tile(
                    gr,
                    segment * span + extra * nlarge + (lane - extra) * nsmall,
                    nsmall,
                    m_rows,
                    store_rows,
                    C3,
                    S3,
                    WO,
                    OS,
                    Y,
                )

    return main
