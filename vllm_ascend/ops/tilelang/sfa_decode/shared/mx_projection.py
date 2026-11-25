# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]

from ..mtp.native_output import MXFP8_NZ_C0, make_native_weight_loader

OUTPUT_PANEL_CHANNELS = 1024


def mx_projection(
    rows,
    width,
    out_width,
    bm,
    *,
    output_rows=None,
    native_weight=False,
    packed_weight=False,
    split_width=None,
    bn=128,
    stages=2,
    pipeline_k=False,
    row_region=None,
    balance_columns=False,
):
    output_rows = rows if output_rows is None else output_rows
    storage_panel = OUTPUT_PANEL_CHANNELS if packed_weight else width
    panel = 512
    bk = 256 if split_width or balance_columns else 128
    if split_width:
        panel = 512 if native_weight and rows == 32 else bk
    split = split_width or width
    parts = (width + split - 1) // split
    last = width - (parts - 1) * split
    assert rows % bm == width % panel == out_width % bn == 0 and 0 < output_rows <= rows
    columns = out_width // bn
    groups, tail = divmod(output_rows, bm)
    # A partial M tile has its own L0C extent and event schedule.
    first = groups if row_region == "tail" else 0
    tiles_m = 1 if row_region == "tail" else groups + int(tail > 0 and row_region is None)
    assert tiles_m > 0
    granularity = MXFP8_NZ_C0 if native_weight else 16
    span = min(4096, out_width)
    units, extra = divmod(span // granularity, 28)
    narrow = units * granularity
    wide = narrow + granularity
    if balance_columns:
        assert units and out_width % span == 0 and not pipeline_k
        columns = out_width // span * 28
    fp8 = T.float8_e4m3fn
    load_native = make_native_weight_loader(panel)
    if native_weight:
        weight_shape = (out_width // MXFP8_NZ_C0, width, MXFP8_NZ_C0)
    else:
        weight_shape = (width // storage_panel, out_width, storage_panel)
    scale_shape = (
        (width // storage_panel, out_width, storage_panel // 64) if packed_weight else (width // 64, out_width)
    )

    def k_loop(steps):
        return T.Pipelined(steps, num_stages=2) if pipeline_k else T.serial(steps)

    @T.macro
    def tile(gr, col, bn, store_rows, part, span, C3, S3, WO, OS, Y):
        m_rows = ((store_rows + 15) // 16) * 16
        a = T.alloc_l1((m_rows, panel), fp8)
        w = T.alloc_l1((panel, bn) if native_weight else (bn, panel), fp8)
        xs = T.alloc_l1((m_rows, panel // 64), T.uint16)
        ws = T.alloc_l1((bn, panel // 64), T.uint16)
        a0 = T.alloc_l0a((m_rows, bk), fp8)
        b0 = T.alloc_l0b((bn, bk), fp8)
        s0 = T.alloc_l0a_sf(a0)
        t0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((m_rows, bn), T.float32)
        for step in T.Pipelined(span // panel, num_stages=stages):
            outer = part * (split // panel) + step
            T.copy(C3[gr * bm : gr * bm + m_rows, outer * panel : (outer + 1) * panel], a)
            if native_weight:
                load_native(WO, w, outer, col, bn)
            else:
                k = outer * panel % storage_panel
                T.copy(WO[outer * panel // storage_panel, col : col + bn, k : k + panel], w)
            T.copy(
                S3[outer * (panel // 64) : (outer + 1) * (panel // 64), gr * bm : gr * bm + m_rows], xs, transpose=True
            )
            if packed_weight:
                k = outer * panel % storage_panel // 64
                T.copy(OS[outer * panel // storage_panel, col : col + bn, k : k + panel // 64], ws)
            else:
                T.copy(OS[outer * (panel // 64) : (outer + 1) * (panel // 64), col : col + bn], ws, transpose=True)
            for inner in k_loop(panel // bk):
                T.copy(a[:, inner * bk : (inner + 1) * bk], a0)
                if native_weight:
                    T.copy(w[inner * bk : (inner + 1) * bk, :], b0, transpose=True)
                else:
                    T.copy(w[:, inner * bk : (inner + 1) * bk], b0)
                T.copy(xs[:, inner * (bk // 64) : (inner + 1) * (bk // 64)], s0)
                T.copy(ws[:, inner * (bk // 64) : (inner + 1) * (bk // 64)], t0)
                T.gemm_blockscaled(a0, b0, acc, s0, t0, transpose_B=True, clear_accum=step == 0 and inner == 0)
        T.copy(
            acc[:store_rows, :], Y[gr * bm : gr * bm + store_rows, part * out_width + col : part * out_width + col + bn]
        )

    @T.prim_func
    def main(
        C3: T.Tensor((rows, width), fp8),
        S3: T.Tensor((width // 64, rows), T.uint16),
        WO: T.Tensor(weight_shape, fp8),
        OS: T.Tensor(scale_shape, T.uint16),
        Y: T.Tensor((output_rows, parts * out_width), T.float32 if split_width else T.bfloat16),
    ):
        with T.Kernel(tiles_m * columns * parts) as task:
            gr = first + (task // parts % tiles_m if pipeline_k else task // (columns * parts))
            lane = task // parts % columns
            col = task // (parts * tiles_m) * bn if pipeline_k else lane * bn
            part = task % parts
            if balance_columns:
                shard = lane % 28
                begin = lane // 28 * span + shard * narrow + T.min(shard, extra) * granularity
                if shard < extra:
                    tile(gr, begin, wide, tail if row_region == "tail" else bm, part, split, C3, S3, WO, OS, Y)
                else:
                    tile(gr, begin, narrow, tail if row_region == "tail" else bm, part, split, C3, S3, WO, OS, Y)
            elif row_region is None and tail and gr == groups:
                tile(gr, col, bn, tail, part, split, C3, S3, WO, OS, Y)
            elif last != split and part == parts - 1:
                tile(gr, col, bn, bm, part, last, C3, S3, WO, OS, Y)
            else:
                tile(gr, col, bn, bm, part, split, C3, S3, WO, OS, Y)

    return main


def output_projections(rows, tokens, tile, *, bulk, native=False, packed=False):
    full, tail = divmod(tokens, tile)
    for count, region in ((full, "full"), (tail, "tail")):
        if count:
            pipeline = bulk and region == "full"
            yield mx_projection(
                rows,
                16384,
                4096,
                tile,
                output_rows=tokens,
                native_weight=native,
                packed_weight=packed,
                bn=256,
                pipeline_k=pipeline,
                row_region=region,
                balance_columns=not pipeline,
            )


def query_projection(rows, tile, *, bulk, native=False):
    return mx_projection(
        rows, 1536, 16384, tile, native_weight=native, bn=256, pipeline_k=bulk, balance_columns=not bulk
    )
