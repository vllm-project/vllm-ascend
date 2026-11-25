# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from ..shared.key_pipeline import make_key_pipeline
from ..shared.value_pipeline import make_value_pipeline
from . import DENSE_PREFIX_TOKENS, MIN_BULK_QUERY_TOKENS

DIM = 512

SCALE_GROUP = 128

PACKED_WORDS = DIM // 4 + DIM // SCALE_GROUP

SCORE_SCALE = 256**-0.5


def large_main_program(
    context,
    query_count,
    head_tile,
    requests,
    pages,
    width=128,
    *,
    index_width=2176,
    tokens=None,
    prepared_kv=False,
    score_rows_per_query=0,
):
    assert width == 128 and head_tile == 64
    assert head_tile in (16, 32, 64)
    groups = 64 // head_tile
    batch = requests * query_count if tokens is None else tokens
    page_size = 128
    query_rows, tile, dim_tile = head_tile, 128, width
    max_visible = DENSE_PREFIX_TOKENS
    if query_rows < 16 or query_rows % 16:
        raise ValueError("The flattened query-by-head dimension must be a multiple of 16.")
    if page_size % tile or max_visible % tile or tile % 32:
        raise ValueError("Each KV tile must stay within a page and have a width divisible by 32.")

    half_rows = query_rows // 2
    stride_words = (PACKED_WORDS + 7) // 8 * 8
    dtype = T.bfloat16
    f32 = T.float32
    neg = -1.0e30
    table_width = (context + page_size - 1) // page_size

    @T.macro
    def fetch(bx, owner, sid, kv, live, raw_len, Cache, Table, Indices, k_l1, k_ub, packed):
        pos = kv * tile
        # At 512 complete pools the indexer can reorder even a fully selected prefix.
        if raw_len < max_visible:
            row = sid * (tile // 2)
            remaining = T.min(T.max(live - pos - row, 0), tile // 2)
            if remaining > 0:
                page = T.cast(T.max(Table[owner, kv], 0), T.int64)
                T.copy(
                    Cache[page, row : row + remaining, :],
                    packed[:remaining, :PACKED_WORDS],
                )
        elif pos + (sid + 1) * (tile // 2) <= live:
            for group in T.serial(tile // 8):
                i = pos + sid * (tile // 2) + group * 4
                logical = T.if_then_else(i < live, T.max(Indices[bx, T.min(i, 2050)], 0), 0)
                page = T.cast(T.max(Table[owner, logical // page_size], 0), T.int64)
                row = logical % page_size
                T.copy(
                    Cache[page, row : row + 4, :],
                    packed[group * 4 : group * 4 + 4, :PACKED_WORDS],
                )
        else:
            for group in T.serial(tile // 8):
                i = pos + sid * (tile // 2) + group * 4
                if i < live:
                    logical = T.max(Indices[bx, i], 0)
                    page = T.cast(T.max(Table[owner, logical // page_size], 0), T.int64)
                    row = logical % page_size
                    T.copy(
                        Cache[page, row : row + 4, :],
                        packed[group * 4 : group * 4 + 4, :PACKED_WORDS],
                    )
        if (pos + (sid + 1) * (tile // 2) > live) and (pos != 2048):
            with T.SimtVF(threads=256):
                for r, c in T.Parallel(tile // 2, PACKED_WORDS):
                    if pos + sid * (tile // 2) + r >= live:
                        packed[r, c] = 0
        fp8_bytes = T.view(packed, (tile // 2, stride_words * 4), dtype=T.float8_e4m3fn)
        fp32_scales = T.view(packed, (tile // 2, stride_words), dtype=T.float32)
        if pos == 2048:
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                all_bf16 = S.pset(16, "PAT_ALL")
                input_mask = S.pset(8, "PAT_ALL")
                nz_stride = T.int32(((tile // 2 + 1) << 16) | 1)
                zeros = T.reinterpret(S.vdup(T.uint16(0), "uint16", all_bf16), "bfloat16x128")
                for j in range(4):
                    offset = j * 128
                    dest = S.make_ubuf_ptr(
                        T.access_ptr(k_ub[0, offset], "w", tile // 2 + 1, 128),
                        "bfloat16",
                    )

                    for r in range(4):
                        if sid == 0 and r < live - pos:
                            scale = S.vld(fp32_scales[r, DIM // 4 + j], "BRC_B32")
                            left = S.vld(fp8_bytes[r, offset], "UNPK4_B8")
                            right = S.vld(fp8_bytes[r, offset + 64], "UNPK4_B8")
                            left32 = S.vmul(S.vcvt(left, "float32", input_mask, part=0), scale, full)
                            right32 = S.vmul(
                                S.vcvt(right, "float32", input_mask, part=0),
                                scale,
                                full,
                            )
                            left16 = S.vcvt(left32, "bfloat16", full, part=0)
                            right16 = S.vcvt(right32, "bfloat16", full, part=0)
                            dense, unused = S.vdintlv(left16, right16)
                            dest = S.vsstb(dense, dest, nz_stride, all_bf16, update=True)
                        else:
                            dest = S.vsstb(zeros, dest, nz_stride, all_bf16, update=True)
                    for r in range(4, tile // 2):
                        dest = S.vsstb(zeros, dest, nz_stride, all_bf16, update=True)
        elif batch >= MIN_BULK_QUERY_TOKENS:
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                bfmask = S.pset(16, "PAT_ALL")
                bytes_mask = S.pset(8, "PAT_ALL")
                nz_stride = T.int32(((tile // 2 + 1) << 16) | 1)
                for channel in range(DIM // SCALE_GROUP):
                    offset = channel * SCALE_GROUP
                    dst = S.make_ubuf_ptr(
                        T.access_ptr(k_ub[0, offset], "w", tile // 2 + 1, SCALE_GROUP),
                        "bfloat16",
                    )
                    for pair in range(tile // 4):
                        scales = [
                            S.vld(
                                fp32_scales[pair * 2 + lane, DIM // 4 + channel],
                                "BRC_B32",
                            )
                            for lane in (0, 1)
                        ]
                        codes = [S.vld(fp8_bytes[pair * 2 + lane, offset], "UNPK_B8") for lane in (0, 1)]
                        for lane in (0, 1):
                            even = S.vmul(
                                S.vcvt(codes[lane], "float32", bytes_mask, part=0),
                                scales[lane],
                                full,
                            )
                            odd = S.vmul(
                                S.vcvt(codes[lane], "float32", bytes_mask, part=2),
                                scales[lane],
                                full,
                            )
                            lo = T.reinterpret(S.vcvt(even, "bfloat16", full, part=0), "uint16x128")
                            hi = T.reinterpret(S.vcvt(odd, "bfloat16", full, part=1), "uint16x128")
                            packed_row = T.reinterpret(S.vor(lo, hi, bfmask), "bfloat16x128")
                            dst = S.vsstb(packed_row, dst, nz_stride, bfmask, update=True)
        else:
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                all_bf16 = S.pset(16, "PAT_ALL")
                input_mask = S.pset(8, "PAT_ALL")
                nz_stride = T.int32(((tile // 2 + 1) << 16) | 1)
                for j in range(4):
                    offset = j * 128
                    dest = S.make_ubuf_ptr(
                        T.access_ptr(k_ub[0, offset], "w", tile // 2 + 1, 128),
                        "bfloat16",
                    )
                    for r in range(tile // 2):
                        scale = S.vld(fp32_scales[r, DIM // 4 + j], "BRC_B32")
                        left = S.vld(fp8_bytes[r, offset], "UNPK4_B8")
                        right = S.vld(fp8_bytes[r, offset + 64], "UNPK4_B8")
                        left32 = S.vmul(S.vcvt(left, "float32", input_mask, part=0), scale, full)
                        right32 = S.vmul(S.vcvt(right, "float32", input_mask, part=0), scale, full)
                        left16 = S.vcvt(left32, "bfloat16", full, part=0)
                        right16 = S.vcvt(right32, "bfloat16", full, part=0)
                        dense, unused = S.vdintlv(left16, right16)
                        dest = S.vsstb(dense, dest, nz_stride, all_bf16, update=True)

        T.dual_copy(k_ub[: tile // 2, :], k_l1)

    dot, _ = make_key_pipeline(query_rows, tile, dim_tile, interleave_loads=batch >= MIN_BULK_QUERY_TOKENS)

    @T.macro
    def normalize_block(kv, scores, limits, row_max, row_sum, alpha, prob, short):
        pos = kv * tile
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_ALL")
            lanes = S.vci(T.int32(0), "int32")
            zero = S.vdup(T.float32(0), "float32", full)
            negs = S.vdup(T.float32(neg), "float32", full)
            d0 = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            for r in range(half_rows):
                m0 = S.vcmps(lanes, limits[0, r] - pos, full, op="lt")
                m1 = S.vcmps(lanes, limits[0, r] - pos - 64, full, op="lt")
                a = S.vsel(
                    S.vmuls(S.vld(scores[0, r, 0], "NORM"), T.float32(SCORE_SCALE), full),
                    negs,
                    m0,
                )
                b = S.vsel(
                    S.vmuls(S.vld(scores[0, r, 64], "NORM"), T.float32(SCORE_SCALE), full),
                    negs,
                    m1,
                )
                old = S.vld(row_max[r], "BRC_B32")
                pairmax = S.vdupv(S.vcmax(S.vmax(a, b, full), full), full)
                mx = S.vmax(old, pairmax, full)
                adj = S.vexpdif(old, mx, full)
                a = S.vsel(S.vexpdif(a, mx, full), zero, m0)
                b = S.vsel(S.vexpdif(b, mx, full), zero, m1)
                if short:
                    single_mass = S.vadd(
                        S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full),
                        S.vcadd(a, full),
                        full,
                    )
                    S.vsts(row_sum[r], single_mass, scalar, "ONEPT_B32")
                else:
                    even_exp, odd_exp = S.vdintlv(a, b)
                    block_sum = S.vcadd(S.vadd(even_exp, odd_exp, full), full)
                    paired_mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), block_sum, full)
                    S.vsts(row_sum[r], paired_mass, scalar, "ONEPT_B32")
                S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                lo = S.vcvt(a, "bfloat16", full, round="ROUND_A", part=0)
                hi = S.vcvt(b, "bfloat16", full, round="ROUND_A", part=0)
                dense, unused = S.vdintlv(lo, hi)
                d0 = S.vsstb(dense, d0, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)

    @T.macro
    def normalize_full(kv, scores, limits, row_max, row_sum, alpha, prob, short):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_ALL")
            d0 = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            for r in range(half_rows):
                a = S.vmuls(S.vld(scores[0, r, 0], "NORM"), T.float32(SCORE_SCALE), full)
                b = S.vmuls(S.vld(scores[0, r, 64], "NORM"), T.float32(SCORE_SCALE), full)
                old = S.vld(row_max[r], "BRC_B32")
                pairmax = S.vdupv(S.vcmax(S.vmax(a, b, full), full), full)
                mx = S.vmax(old, pairmax, full)
                adj = S.vexpdif(old, mx, full)
                a = S.vexpdif(a, mx, full)
                b = S.vexpdif(b, mx, full)
                if short:
                    single_mass = S.vadd(
                        S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full),
                        S.vcadd(a, full),
                        full,
                    )
                    S.vsts(row_sum[r], single_mass, scalar, "ONEPT_B32")
                else:
                    even_exp, odd_exp = S.vdintlv(a, b)
                    block_sum = S.vcadd(S.vadd(even_exp, odd_exp, full), full)
                    paired_mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), block_sum, full)
                    S.vsts(row_sum[r], paired_mass, scalar, "ONEPT_B32")
                S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                lo = S.vcvt(a, "bfloat16", full, round="ROUND_A", part=0)
                hi = S.vcvt(b, "bfloat16", full, round="ROUND_A", part=0)
                dense, unused = S.vdintlv(lo, hi)
                d0 = S.vsstb(dense, d0, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)

    apply = make_value_pipeline(half_rows, dim_tile, interleave_loads=batch >= MIN_BULK_QUERY_TOKENS)

    @T.macro
    def consume_tile(
        kv,
        live,
        k_l1,
        scores_l0,
        score0,
        limits,
        row_max,
        row_sum,
        alpha,
        prob,
        p_l1,
        q_l0,
        pv_b,
        out_l0,
        partial,
        running,
        other_l0,
        other_ub,
        k_l0,
    ):
        T.dual_copy(scores_l0, score0[0, :, :])
        if live - kv * tile >= tile:
            normalize_full(kv, score0, limits, row_max, row_sum, alpha, prob, False)
        elif live - kv * tile <= 64:
            normalize_block(kv, score0, limits, row_max, row_sum, alpha, prob, True)
        else:
            normalize_block(kv, score0, limits, row_max, row_sum, alpha, prob, False)
        apply(
            prob,
            k_l1,
            p_l1,
            q_l0,
            pv_b,
            out_l0,
            partial,
            running,
            alpha,
            other_l0,
            other_ub,
            spare_b=k_l0,
        )

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 64, DIM), dtype),
        Cache: T.Tensor((pages, page_size, PACKED_WORDS), T.int32),
        Table: T.Tensor((requests, table_width), T.int32),
        Indices: T.Tensor((batch, index_width), T.int32),
        Positions: T.Tensor((batch,), T.int64),
        Starts: T.Tensor((requests + 1,), T.int32),
        Ends: T.Tensor((requests,), T.int64),
        Y: T.Tensor((batch, 64, DIM), dtype),
        Length: T.Tensor((requests,), T.int32),
        PreparedKV: T.Tensor(
            ((batch * score_rows_per_query if score_rows_per_query else requests * max_visible), DIM), dtype
        ),
    ):
        with T.MixedKernel(batch * groups, sids=2) as (cta, sid):
            bx = cta // groups
            head = (cta % groups) * query_rows
            q_l1 = T.alloc_l1((query_rows, DIM), dtype)
            k0_l1 = T.alloc_l1((tile, DIM), dtype)
            p_l1 = T.alloc_l1((query_rows, tile), dtype)
            q_l0 = T.alloc_l0a((query_rows, dim_tile), dtype)
            q_pin0 = T.alloc_l0a((query_rows, dim_tile), dtype)
            q_pin1 = T.alloc_l0a((query_rows, dim_tile), dtype)
            q_pin2 = T.alloc_l0a((query_rows, dim_tile), dtype)
            k_l0 = T.alloc_l0b((tile, dim_tile), dtype)
            pv_b = T.alloc_l0b((tile, dim_tile), dtype)
            scores_l0 = T.alloc_l0c((query_rows, tile), f32)
            out_l0 = T.alloc_l0c((query_rows, dim_tile), f32)
            packed = T.alloc_shared((tile // 2, stride_words), T.int32)
            k_ub = T.alloc_shared((tile // 2 + 1, DIM), dtype)
            T.annotate_layout({k_ub: make_ascend_compact_nz_layout(k_ub)})
            limits = T.alloc_shared((1, half_rows), T.int32)
            partial = T.alloc_shared((half_rows, dim_tile), f32)
            other_ub = T.alloc_shared((half_rows, dim_tile), f32)
            other_l0 = T.alloc_l0c((query_rows, dim_tile), f32)
            score0 = T.view(partial, (dim_tile // tile, half_rows, tile), dtype=f32)
            prob = T.alloc_shared((half_rows + 1, tile), dtype)
            T.annotate_layout({prob: make_ascend_compact_nz_layout(prob)})
            row_max = T.alloc_shared((half_rows,), f32)
            row_sum = T.alloc_shared((half_rows,), f32)
            alpha = T.alloc_shared((half_rows,), f32)

            running = T.alloc_shared((half_rows, DIM), f32)
            out = T.view(packed, (tile * stride_words // DIM, DIM), dtype=dtype)

            guess = bx // query_count
            owner = T.alloc_var(T.int32, init=guess)
            # Cube and vector must take the same number of QK/PV iterations.
            # Use the selector's input bounds on both engines.
            active = bx < Ends[requests - 1]
            if active and bx >= Starts[guess + 1]:
                high = T.alloc_var(T.int32, init=requests - 1)
                for step in T.serial(10):
                    middle = (owner + high + 1) // 2
                    if Starts[middle] <= bx:
                        owner = middle
                    else:
                        high = middle - 1
            raw_len = T.max(T.cast(Positions[bx] + 1, T.int32), 0)
            live = T.if_then_else(active, T.min(raw_len // 4, 512) * 4 + raw_len % 4, 0)
            selected_tiles = T.ceildiv(live, tile)
            T.copy(Q[bx, head : head + query_rows, :], q_l1)
            T.copy(q_l1[:, 0:128], q_pin0)
            T.copy(q_l1[:, 128:256], q_pin1)
            T.copy(q_l1[:, 256:384], q_pin2)
            with T.SimtVF(threads=32):
                for r in T.Parallel(half_rows):
                    limits[0, r] = live
            with T.SimtVF(threads=256):
                for r in T.Parallel(half_rows):
                    row_max[r] = neg
                    row_sum[r] = 0
                for r, c in T.Parallel(half_rows, DIM):
                    running[r, c] = 0

            score_start = Starts[owner] * score_rows_per_query if score_rows_per_query else owner * max_visible
            fits = (
                (
                    active
                    and Length[owner] < max_visible
                    and T.ceildiv(Length[owner], tile) * tile
                    <= (T.cast(Ends[owner], T.int32) - Starts[owner]) * score_rows_per_query
                )
                if score_rows_per_query
                else False
            )
            if prepared_kv or fits:
                k1_l1 = T.alloc_l1((tile, DIM), dtype)
                if selected_tiles > 0:
                    T.copy(PreparedKV[score_start : score_start + tile, :], k0_l1)
                for pair in T.serial(selected_tiles // 2):
                    left = 2 * pair
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    T.copy(PreparedKV[score_start + (left + 1) * tile : score_start + (left + 2) * tile, :], k1_l1)
                    consume_tile(
                        left,
                        live,
                        k0_l1,
                        scores_l0,
                        score0,
                        limits,
                        row_max,
                        row_sum,
                        alpha,
                        prob,
                        p_l1,
                        q_l0,
                        pv_b,
                        out_l0,
                        partial,
                        running,
                        other_l0,
                        other_ub,
                        k_l0,
                    )
                    dot(q_l1, k1_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    if left + 2 < selected_tiles:
                        T.copy(PreparedKV[score_start + (left + 2) * tile : score_start + (left + 3) * tile, :], k0_l1)
                    consume_tile(
                        left + 1,
                        live,
                        k1_l1,
                        scores_l0,
                        score0,
                        limits,
                        row_max,
                        row_sum,
                        alpha,
                        prob,
                        p_l1,
                        q_l0,
                        pv_b,
                        out_l0,
                        partial,
                        running,
                        other_l0,
                        other_ub,
                        k_l0,
                    )
                if selected_tiles % 2 == 1:
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    consume_tile(
                        selected_tiles - 1,
                        live,
                        k0_l1,
                        scores_l0,
                        score0,
                        limits,
                        row_max,
                        row_sum,
                        alpha,
                        prob,
                        p_l1,
                        q_l0,
                        pv_b,
                        out_l0,
                        partial,
                        running,
                        other_l0,
                        other_ub,
                        k_l0,
                    )
            else:
                for kv in T.serial(selected_tiles):
                    fetch(
                        bx,
                        owner,
                        sid,
                        kv,
                        live,
                        raw_len,
                        Cache,
                        Table,
                        Indices,
                        k0_l1,
                        k_ub,
                        packed,
                    )
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    consume_tile(
                        kv,
                        live,
                        k0_l1,
                        scores_l0,
                        score0,
                        limits,
                        row_max,
                        row_sum,
                        alpha,
                        prob,
                        p_l1,
                        q_l0,
                        pv_b,
                        out_l0,
                        partial,
                        running,
                        other_l0,
                        other_ub,
                        k_l0,
                    )
            with T.SimdVF():
                mask = S.pset(32)
                for row in range(half_rows):
                    mass = S.vmaxs(S.vld(row_sum[row], "BRC_B32"), T.float32(1e-30), mask)
                    for col in range(0, DIM, 64):
                        value = S.vdiv(
                            S.vld(running[row, col], "NORM"),
                            mass,
                            mask,
                            precision="ftz_true",
                        )
                        result = S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0)
                        S.vsts(out[row, col], result, mask, "PK_B32")
            T.dual_copy(out[:half_rows, :], Y[bx, head : head + query_rows, :])

    return main
