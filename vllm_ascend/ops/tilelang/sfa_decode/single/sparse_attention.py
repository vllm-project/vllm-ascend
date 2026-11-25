# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from ..shared.key_pipeline import make_key_pipeline
from ..shared.value_pipeline import make_value_pipeline
from .resident_sparse import resident_value_tiles

DIM = 512

SCALE_GROUP = 128

PACKED_WORDS = DIM // 4 + DIM // SCALE_GROUP

SCORE_SCALE = 256**-0.5

MIN_PAIRED_KV_BATCH = 64

MIN_STREAMING_BATCH = 128


def main_from_pool_ids(
    batch,
    pages,
    page_size,
    context,
    tail,
    *,
    table_columns,
    length_offset=0,
    prefill_search_bits=None,
    indexed_prefill=False,
    resident_cores=None,
    prefill_heads=64,
    physical_pool_ids=False,
):
    assert page_size == 128
    causal_prefill = prefill_search_bits is not None
    prefill = causal_prefill or indexed_prefill
    search_bits = prefill_search_bits if causal_prefill else 0
    requests = T.dynamic("requests") if prefill else batch
    heads = prefill_heads if prefill else 64
    head_tile = heads
    kv_policy = "NORMAL_FV"
    if not prefill:
        # Split independent heads while all request/head tiles still fit in one 28-Cube wave.
        if resident_cores is None:
            head_tile = 16 if batch <= 7 else 32 if batch <= 14 else 64
        if batch >= MIN_STREAMING_BATCH:
            kv_policy = "NOTALLOC_KEEP"
    width, groups = 128, heads // head_tile
    query_rows, tile, dim_tile = head_tile, 128, width
    max_visible = 2048
    if query_rows < 16 or query_rows % 16:
        raise ValueError("The flattened query-by-head dimension must be a multiple of 16.")
    if page_size % tile or max_visible % tile or tile % 32:
        raise ValueError("Each KV tile must stay within a page and have a width divisible by 32.")

    half_rows = query_rows // 2
    stride_words = (PACKED_WORDS + 7) // 8 * 8
    dtype = T.bfloat16
    f32 = T.float32
    neg = -1.0e30

    @T.macro
    def load_plane(values, scales, row, group):
        gain = S.vld(scales[row, DIM // 4 + group], "BRC_B32")
        codes = S.vld(values[row, group * SCALE_GROUP], "UNPK_B8")
        return gain, codes

    @T.macro
    def scale_plane(gain, codes, byte_mask, full):
        # UNPK_B8 places the even and odd FP8 channels in byte positions 0 and 2 of each word.
        even = S.vmul(S.vcvt(codes, "float32", byte_mask, part=0), gain, full)
        odd = S.vmul(S.vcvt(codes, "float32", byte_mask, part=2), gain, full)
        return even, odd

    @T.macro
    def pack_bf16(even, odd, full, bf16_mask):
        low = T.reinterpret(S.vcvt(even, "bfloat16", full, part=0), "uint16x128")
        high = T.reinterpret(S.vcvt(odd, "bfloat16", full, part=1), "uint16x128")
        return T.reinterpret(S.vor(low, high, bf16_mask), "bfloat16x128")

    paired_kv = indexed_prefill and query_rows in (16, 32) if prefill else batch >= MIN_PAIRED_KV_BATCH
    unpack_rows = 2 if paired_kv else 1
    plane_indices = tuple(range(unpack_rows * (DIM // SCALE_GROUP)))

    @T.macro
    def unpack(packed, k_ub, start, live, read_rows, masked):
        fp8_bytes = T.view(packed, (tile // 2, stride_words * 4), dtype=T.float8_e4m3fn)
        scales = T.view(packed, (tile // 2, stride_words), dtype=T.float32)
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            all_bf16 = S.pset(16, "PAT_ALL")
            input_mask = S.pset(8, "PAT_ALL")
            nz_stride = T.int32(((tile // 2 + 1) << 16) | 1)
            if masked:
                zeros = T.reinterpret(S.vdup(T.uint16(0), "uint16", all_bf16), "bfloat16x128")
                for j in range(DIM // SCALE_GROUP):
                    offset = j * SCALE_GROUP
                    dest = S.make_ubuf_ptr(
                        T.access_ptr(k_ub[0, offset], "w", tile // 2 + 1, 128),
                        "bfloat16",
                    )
                    for r in range(read_rows):
                        if start + r < live:
                            gain, codes = load_plane(fp8_bytes, scales, r, j)
                            even, odd = scale_plane(gain, codes, input_mask, full)
                            dense = pack_bf16(even, odd, full, all_bf16)
                            dest = S.vsstb(dense, dest, nz_stride, all_bf16, update=True)
                        else:
                            dest = S.vsstb(zeros, dest, nz_stride, all_bf16, update=True)
                    for r in range(read_rows, tile // 2):
                        dest = S.vsstb(zeros, dest, nz_stride, all_bf16, update=True)
            else:
                # Issue the four independent scale groups before consuming their converted registers.
                plane0_ptr = S.make_ubuf_ptr(
                    T.access_ptr(k_ub[0, 0], "w", tile // 2 + 1, SCALE_GROUP),
                    "bfloat16",
                )
                plane1_ptr = S.make_ubuf_ptr(
                    T.access_ptr(k_ub[0, SCALE_GROUP], "w", tile // 2 + 1, SCALE_GROUP),
                    "bfloat16",
                )
                plane2_ptr = S.make_ubuf_ptr(
                    T.access_ptr(k_ub[0, 2 * SCALE_GROUP], "w", tile // 2 + 1, SCALE_GROUP),
                    "bfloat16",
                )
                plane3_ptr = S.make_ubuf_ptr(
                    T.access_ptr(k_ub[0, 3 * SCALE_GROUP], "w", tile // 2 + 1, SCALE_GROUP),
                    "bfloat16",
                )
                # Load independent token and scale planes before issuing their conversions.
                for row in range(tile // (2 * unpack_rows)):
                    raw = [load_plane(fp8_bytes, scales, row * unpack_rows + k // 4, k % 4) for k in plane_indices]
                    scaled = [scale_plane(gain, codes, input_mask, full) for gain, codes in raw]
                    plane0_ptr = S.vsstb(
                        pack_bf16(scaled[0][0], scaled[0][1], full, all_bf16),
                        plane0_ptr,
                        nz_stride,
                        all_bf16,
                        update=True,
                    )
                    plane1_ptr = S.vsstb(
                        pack_bf16(scaled[1][0], scaled[1][1], full, all_bf16),
                        plane1_ptr,
                        nz_stride,
                        all_bf16,
                        update=True,
                    )
                    plane2_ptr = S.vsstb(
                        pack_bf16(scaled[2][0], scaled[2][1], full, all_bf16),
                        plane2_ptr,
                        nz_stride,
                        all_bf16,
                        update=True,
                    )
                    plane3_ptr = S.vsstb(
                        pack_bf16(scaled[3][0], scaled[3][1], full, all_bf16),
                        plane3_ptr,
                        nz_stride,
                        all_bf16,
                        update=True,
                    )
                    if unpack_rows == 2:
                        plane0_ptr = S.vsstb(
                            pack_bf16(scaled[4][0], scaled[4][1], full, all_bf16),
                            plane0_ptr,
                            nz_stride,
                            all_bf16,
                            update=True,
                        )
                        plane1_ptr = S.vsstb(
                            pack_bf16(scaled[5][0], scaled[5][1], full, all_bf16),
                            plane1_ptr,
                            nz_stride,
                            all_bf16,
                            update=True,
                        )
                        plane2_ptr = S.vsstb(
                            pack_bf16(scaled[6][0], scaled[6][1], full, all_bf16),
                            plane2_ptr,
                            nz_stride,
                            all_bf16,
                            update=True,
                        )
                        plane3_ptr = S.vsstb(
                            pack_bf16(scaled[7][0], scaled[7][1], full, all_bf16),
                            plane3_ptr,
                            nz_stride,
                            all_bf16,
                            update=True,
                        )

    @T.macro
    def fetch(bx, owner, sid, kv, live, raw_len, Cache, Table, Indices, k_l1, k_ub, packed):
        pos = kv * tile
        start = pos + sid * (tile // 2)
        if not causal_prefill and pos == max_visible:
            if sid == 0:
                logical = T.bitwise_and(raw_len, -4) if indexed_prefill else raw_len // 4 * 4
                page_col = T.shift_right(logical, 7) if indexed_prefill else logical // page_size
                page = T.cast(Table[owner, page_col], T.int64)
                row = T.bitwise_and(logical, 127) if indexed_prefill else logical % page_size
                T.copy(Cache[page, row : row + 4, :], packed[0:4, :PACKED_WORDS])
        elif causal_prefill or raw_len < max_visible:
            # All visible pools were selected in logical order. Each tile lies in one page.
            row = sid * (tile // 2)
            remaining = T.min(T.max(live - start, 0), tile // 2)
            if remaining > 0:
                page = T.cast(Table[owner, kv], T.int64)
                T.copy(
                    Cache[page, row : row + remaining, :], packed[:remaining, :PACKED_WORDS], l2_cache_ctrl=kv_policy
                )
        elif indexed_prefill and heads < 64:
            # Each selected pool supplies four tokens. Reuse the page lookup for adjacent pools.
            for pair in T.serial(tile // 16):
                index = start + pair * 8
                first = T.cast(Indices[bx, index], T.int64)
                second = T.cast(Indices[bx, index + 4], T.int64)
                first_col = T.shift_right(first, 7)
                second_col = T.shift_right(second, 7)
                first_row = T.bitwise_and(first, 127)
                second_row = T.bitwise_and(second, 127)
                first_page = T.cast(Table[owner, first_col], T.int64)
                T.copy(
                    Cache[first_page, first_row : first_row + 4, :],
                    packed[pair * 8 : pair * 8 + 4, :PACKED_WORDS],
                    l2_cache_ctrl=kv_policy,
                )
                second_page = T.if_then_else(
                    first_col == second_col, first_page, T.cast(Table[owner, second_col], T.int64)
                )
                T.copy(
                    Cache[second_page, second_row : second_row + 4, :],
                    packed[pair * 8 + 4 : pair * 8 + 8, :PACKED_WORDS],
                    l2_cache_ctrl=kv_policy,
                )
        else:
            for group in T.serial(tile // 8):
                if indexed_prefill:
                    logical = T.cast(Indices[bx, start + group * 4], T.int64)
                    page_col = T.shift_right(logical, 7)
                    row = T.bitwise_and(logical, 127)
                    page = T.cast(Table[owner, page_col], T.int64)
                elif physical_pool_ids:
                    physical = Indices[bx, start // 4 + group]
                    page = T.cast(T.shift_right(physical, 5), T.int64)
                    row = T.bitwise_and(physical, 31) * 4
                else:
                    logical = Indices[bx, start // 4 + group] * 4
                    page_col = logical // page_size
                    row = logical % page_size
                    page = T.cast(Table[owner, page_col], T.int64)
                T.copy(
                    Cache[page, row : row + 4, :],
                    packed[group * 4 : group * 4 + 4, :PACKED_WORDS],
                    l2_cache_ctrl=kv_policy,
                )
        if not causal_prefill and pos == max_visible:
            unpack(packed, k_ub, start, live, 4, True)
        elif (causal_prefill or raw_len < max_visible) and pos + tile > live:
            unpack(packed, k_ub, start, live, tile // 2, True)
        else:
            unpack(packed, k_ub, start, live, tile // 2, False)
        T.dual_copy(k_ub[: tile // 2, :], k_l1)

    dot, residual_dot = make_key_pipeline(query_rows, tile, dim_tile)

    @T.macro
    def normalize_residual(scores, live, row_max, row_sum, alpha, prob):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_ALL")
            lanes = S.vci(T.int32(0), "int32")
            valid = S.vcmps(lanes, live - max_visible, full, op="lt")
            negs = S.vdup(T.float32(neg), "float32", full)
            zero = S.vdup(T.float32(0), "float32", full)
            dest = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            for r in range(half_rows):
                a = S.vsel(S.vmuls(S.vld(scores[r, 0], "NORM"), T.float32(SCORE_SCALE), full), negs, valid)
                old = S.vld(row_max[r], "BRC_B32")
                mx = S.vmax(old, S.vdupv(S.vcmax(a, full), full), full)
                adj = S.vexpdif(old, mx, full)
                a = S.vsel(S.vexpdif(a, mx, full), zero, valid)
                mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), S.vcadd(a, full), full)
                S.vsts(row_sum[r], mass, scalar, "ONEPT_B32")
                S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                lo = S.vcvt(a, "bfloat16", full, round="ROUND_A", part=0)
                hi = S.vcvt(zero, "bfloat16", full, round="ROUND_A", part=0)
                dense, unused = S.vdintlv(lo, hi)
                dest = S.vsstb(dense, dest, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)

    @T.macro
    def normalize_residual_heads(scores, live, row_max, row_sum, alpha, prob):
        probability_words = T.view(prob, ((half_rows + 1) * tile // 2,), dtype=T.uint32)
        with T.SimdVF():
            heads_mask = S.pset(32, "PAT_VL32")
            bf16_pairs = S.pset(16, "PAT_VL64")
            lane = T.reinterpret(S.vci(T.int32(0), "int32"), "uint32x64")
            # FIX stores 16 scores per head. PV consumes one 16-token NZ plane.
            score_offsets = S.vshls(lane, 4, heads_mask)
            probability_offsets = S.vshls(lane, 3, heads_mask)
            remaining = S.vdup(T.cast(live - max_visible, T.int32), "int32", heads_mask)
            has_second = S.vcmps(remaining, T.int32(1), heads_mask, op="gt")
            has_third = S.vcmps(remaining, T.int32(2), heads_mask, op="gt")
            masked_score = S.vdup(T.float32(neg), "float32", heads_mask)
            zero = S.vdup(T.float32(0), "float32", heads_mask)
            score0 = S.vmuls(S.vgather2(scores[0, 0], score_offsets, heads_mask), T.float32(SCORE_SCALE), heads_mask)
            score1 = S.vsel(
                S.vmuls(S.vgather2(scores[0, 1], score_offsets, heads_mask), T.float32(SCORE_SCALE), heads_mask),
                masked_score,
                has_second,
            )
            score2 = S.vsel(
                S.vmuls(S.vgather2(scores[0, 2], score_offsets, heads_mask), T.float32(SCORE_SCALE), heads_mask),
                masked_score,
                has_third,
            )
            previous_peak = S.vld(row_max[0], "NORM")
            new_peak = S.vmax(S.vmax(previous_peak, score0, heads_mask), S.vmax(score1, score2, heads_mask), heads_mask)
            rescale = S.vexpdif(previous_peak, new_peak, heads_mask)
            exp0 = S.vexpdif(score0, new_peak, heads_mask)
            exp1 = S.vsel(S.vexpdif(score1, new_peak, heads_mask), zero, has_second)
            exp2 = S.vsel(S.vexpdif(score2, new_peak, heads_mask), zero, has_third)
            delta = S.vadd(S.vadd(exp0, exp1, heads_mask), exp2, heads_mask)
            mass = S.vadd(S.vmul(S.vld(row_sum[0], "NORM"), rescale, heads_mask), delta, heads_mask)
            S.vsts(row_max[0], new_peak, heads_mask, "NORM_B32")
            S.vsts(alpha[0], rescale, heads_mask, "NORM_B32")
            S.vsts(row_sum[0], mass, heads_mask, "NORM_B32")
            lo = T.reinterpret(S.vcvt(exp0, "bfloat16", heads_mask, round="ROUND_A", part=0), "uint16x128")
            hi = T.reinterpret(S.vcvt(exp1, "bfloat16", heads_mask, round="ROUND_A", part=1), "uint16x128")
            first_pair = T.reinterpret(S.vor(lo, hi, bf16_pairs), "uint32x64")
            last_pair = T.reinterpret(S.vcvt(exp2, "bfloat16", heads_mask, round="ROUND_A", part=0), "uint32x64")
            zeros = S.vdup(T.uint32(0), "uint32", heads_mask)
            S.vscatter(first_pair, probability_words[0], probability_offsets, heads_mask)
            S.vscatter(last_pair, probability_words[1], probability_offsets, heads_mask)
            for col in range(2, 8):
                S.vscatter(zeros, probability_words[col], probability_offsets, heads_mask)

    @T.macro
    def normalize_block(kv, scores, limits, row_max, row_sum, alpha, prob, short):
        pos = kv * tile
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_VL64")
            lanes = S.vci(T.int32(0), "int32")
            zero = S.vdup(T.float32(0), "float32", full)
            negs = S.vdup(T.float32(neg), "float32", full)
            d0 = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            d1 = S.make_ubuf_ptr(T.access_ptr(prob[0, 64], "w", half_rows + 1, tile), "bfloat16")
            for r in range(half_rows):
                m0 = S.vcmps(lanes, limits[0, r] - pos, full, op="lt")
                m1 = S.vcmps(lanes, limits[0, r] - pos - 64, full, op="lt")
                a = S.vsel(S.vmuls(S.vld(scores[0, r, 0], "NORM"), T.float32(SCORE_SCALE), full), negs, m0)
                b = S.vsel(S.vmuls(S.vld(scores[0, r, 64], "NORM"), T.float32(SCORE_SCALE), full), negs, m1)
                old = S.vld(row_max[r], "BRC_B32")
                pairmax = S.vdupv(S.vcmax(S.vmax(a, b, full), full), full)
                mx = S.vmax(old, pairmax, full)
                adj = S.vexpdif(old, mx, full)
                a = S.vsel(S.vexpdif(a, mx, full), zero, m0)
                b = S.vsel(S.vexpdif(b, mx, full), zero, m1)
                if short:
                    single_mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), S.vcadd(a, full), full)
                    S.vsts(row_sum[r], single_mass, scalar, "ONEPT_B32")
                else:
                    even_exp, odd_exp = S.vdintlv(a, b)
                    block_sum = S.vcadd(S.vadd(even_exp, odd_exp, full), full)
                    paired_mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), block_sum, full)
                    S.vsts(row_sum[r], paired_mass, scalar, "ONEPT_B32")
                S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                even0, odd0 = S.vdintlv(a, zero)
                lo0 = T.reinterpret(S.vcvt(even0, "bfloat16", full, round="ROUND_A", part=0), "uint16x128")
                hi0 = T.reinterpret(S.vcvt(odd0, "bfloat16", full, round="ROUND_A", part=1), "uint16x128")
                out0 = T.reinterpret(S.vor(lo0, hi0, bf), "bfloat16x128")
                d0 = S.vsstb(out0, d0, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)
                even1, odd1 = S.vdintlv(b, zero)
                lo1 = T.reinterpret(S.vcvt(even1, "bfloat16", full, round="ROUND_A", part=0), "uint16x128")
                hi1 = T.reinterpret(S.vcvt(odd1, "bfloat16", full, round="ROUND_A", part=1), "uint16x128")
                out1 = T.reinterpret(S.vor(lo1, hi1, bf), "bfloat16x128")
                d1 = S.vsstb(out1, d1, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)

    group_head_stats = not prefill and resident_cores is None and query_rows == 64

    @T.macro
    def normalize_full(kv, scores, limits, row_max, row_sum, alpha, prob, short):
        if group_head_stats:
            block_sums = T.alloc_shared((64,), f32)
            peaks = T.alloc_shared((64,), f32)
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_ALL")
            d0 = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            if group_head_stats:
                active_heads = S.pset(32, "PAT_VL32")
                for h in range(half_rows):
                    first = S.vld(scores[0, h, 0], "NORM")
                    second = S.vld(scores[0, h, 64], "NORM")
                    S.vsts(peaks[h], S.vcmax(S.vmax(first, second, full), full), scalar, "ONEPT_B32")
                S.mem_bar("VST_VLD")
                prev = S.vld(row_max[0], "NORM")
                peak = S.vmuls(S.vld(peaks[0], "NORM"), T.float32(SCORE_SCALE), active_heads)
                new_peak = S.vmax(prev, peak, active_heads)
                factors = S.vexpdif(prev, new_peak, active_heads)
                S.vsts(row_max[0], new_peak, active_heads, "NORM_B32")
                S.vsts(alpha[0], factors, active_heads, "NORM_B32")
                S.mem_bar("VST_VLD")
            for r in range(half_rows):
                a = S.vmuls(S.vld(scores[0, r, 0], "NORM"), T.float32(SCORE_SCALE), full)
                b = S.vmuls(S.vld(scores[0, r, 64], "NORM"), T.float32(SCORE_SCALE), full)
                if group_head_stats:
                    mx = S.vld(row_max[r], "BRC_B32")
                else:
                    old = S.vld(row_max[r], "BRC_B32")
                    pairmax = S.vdupv(S.vcmax(S.vmax(a, b, full), full), full)
                    mx = S.vmax(old, pairmax, full)
                    adj = S.vexpdif(old, mx, full)
                a = S.vexpdif(a, mx, full)
                b = S.vexpdif(b, mx, full)
                if short:
                    block_sum = S.vcadd(a, full)
                else:
                    even_exp, odd_exp = S.vdintlv(a, b)
                    block_sum = S.vcadd(S.vadd(even_exp, odd_exp, full), full)
                if group_head_stats:
                    S.vsts(block_sums[r], block_sum, scalar, "ONEPT_B32")
                else:
                    mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), block_sum, full)
                    S.vsts(row_sum[r], mass, scalar, "ONEPT_B32")
                    S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                    S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                lo = S.vcvt(a, "bfloat16", full, round="ROUND_A", part=0)
                hi = S.vcvt(b, "bfloat16", full, round="ROUND_A", part=0)
                dense, unused = S.vdintlv(lo, hi)
                d0 = S.vsstb(dense, d0, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)
            if group_head_stats:
                S.mem_bar("VST_VLD")
                active_heads = S.pset(32, "PAT_VL32")
                old_sum = S.vld(row_sum[0], "NORM")
                factors_sum = S.vld(alpha[0], "NORM")
                delta_sum = S.vld(block_sums[0], "NORM")
                S.vsts(
                    row_sum[0],
                    S.vadd(S.vmul(old_sum, factors_sum, active_heads), delta_sum, active_heads),
                    active_heads,
                    "NORM_B32",
                )

    if resident_cores is not None:
        assert 1 <= batch <= 4 and not prefill and resident_cores >= 17
        return resident_value_tiles(
            batch,
            pages,
            table_columns,
            length_offset,
            resident_cores,
            PACKED_WORDS,
            stride_words,
            SCORE_SCALE,
            neg,
            fetch,
            dot,
            normalize_full,
            normalize_block,
        )

    apply = make_value_pipeline(half_rows, dim_tile, interleave_loads=True)
    # An odd final tile has no following KV fetch, so its consumed packed buffer can hold the output.
    finalize_odd = not prefill and batch > 512
    apply_last = (
        make_value_pipeline(half_rows, dim_tile, interleave_loads=True, finalize=True) if finalize_odd else apply
    )
    # With contiguous causal keys the two metadata inputs carry TND query ends
    # and absolute positions. Decode supplies selected pools and request lengths.
    index_shape = (requests, 1) if prefill else (batch, 512)
    metadata_dtype = T.int64 if prefill else T.int32

    @T.macro
    def request_for_row(row, ends):
        owner = T.alloc_var(T.int32, init=0)
        for bit in T.unroll(search_bits):
            probe = owner + (1 << (search_bits - bit - 1))
            owner = T.if_then_else(probe < requests and row >= ends[T.min(probe, requests) - 1, 0], probe, owner)
        return owner

    @T.macro
    def indexed_owner(row, ends):
        lo = T.alloc_var(T.int32, init=0)
        hi = T.alloc_var(T.int32, init=requests - 1)
        while lo < hi:
            mid = T.shift_right(lo + hi, 1)
            if row >= ends[mid]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @T.macro
    def attend(Q, Cache, Table, Indices, Length, Ends, Y):
        with T.MixedKernel(batch * groups, sids=2) as (cta, sid):
            bx = cta // groups
            head = (cta % groups) * query_rows
            q_l1 = T.alloc_l1((query_rows, DIM), dtype)
            k0_l1 = T.alloc_l1((tile, DIM), dtype)
            k1_l1 = T.alloc_l1((tile, DIM), dtype)
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
            # The grouped update loads a full FP32 register and masks its 32 active heads.
            stats_width = 64 if group_head_stats else half_rows
            row_max = T.alloc_shared((stats_width,), f32)
            row_sum = T.alloc_shared((stats_width,), f32)
            alpha = T.alloc_shared((stats_width,), f32)

            running = T.alloc_shared((half_rows, DIM), f32)
            out = T.view(packed, (tile * stride_words // DIM, DIM), dtype=dtype)

            if indexed_prefill:
                owner = indexed_owner(bx, Ends)
                raw_len = T.if_then_else(bx < Ends[requests - 1], T.max(T.cast(Length[bx] + 1, T.int32), 0), 0)
            elif causal_prefill:
                owner = request_for_row(bx, Indices)
                raw_len = T.if_then_else(bx < Indices[requests - 1, 0], T.max(T.cast(Length[bx] + 1, T.int32), 0), 0)
            else:
                owner = bx
                raw_len = T.max(Length[length_offset + bx], 0)
            remainder = T.bitwise_and(raw_len, 3) if indexed_prefill else raw_len % 4
            live = T.min(raw_len // 4, 512) * 4 + remainder
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

            if selected_tiles > 0:
                fetch(bx, owner, sid, 0, live, raw_len, Cache, Table, Indices, k0_l1, k_ub, packed)
            for pair in T.serial(selected_tiles // 2):
                left = pair * 2
                dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                fetch(bx, owner, sid, left + 1, live, raw_len, Cache, Table, Indices, k1_l1, k_ub, packed)
                T.dual_copy(scores_l0, score0[0, :, :])
                normalize_full(left, score0, limits, row_max, row_sum, alpha, prob, False)
                apply(prob, k0_l1, p_l1, q_l0, pv_b, out_l0, partial, running, alpha, other_l0, other_ub, spare_b=k_l0)
                dot(q_l1, k1_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                if left + 2 < selected_tiles:
                    fetch(bx, owner, sid, left + 2, live, raw_len, Cache, Table, Indices, k0_l1, k_ub, packed)
                T.dual_copy(scores_l0, score0[0, :, :])
                if live - (left + 1) * tile >= tile:
                    normalize_full(left + 1, score0, limits, row_max, row_sum, alpha, prob, False)
                elif live - (left + 1) * tile <= 64:
                    normalize_block(left + 1, score0, limits, row_max, row_sum, alpha, prob, True)
                else:
                    normalize_block(left + 1, score0, limits, row_max, row_sum, alpha, prob, False)
                apply(prob, k1_l1, p_l1, q_l0, pv_b, out_l0, partial, running, alpha, other_l0, other_ub, spare_b=k_l0)
            odd = T.bitwise_and(selected_tiles, 1) if indexed_prefill else selected_tiles % 2
            if odd == 1:
                residual_pool = (
                    not causal_prefill
                    and (indexed_prefill or query_rows == 64)
                    and selected_tiles == max_visible // tile + 1
                )
                if residual_pool:
                    # A VF reads 64 scores from the start of each 16-column row.
                    tail_scores = T.alloc_shared((half_rows + 3, 16), f32)
                    tail_l0 = T.alloc_l0c((query_rows, 16), f32)
                    residual_dot(0, q_pin0, k0_l1, k_l0, tail_l0, True)
                    residual_dot(128, q_pin1, k0_l1, pv_b, tail_l0, False)
                    residual_dot(256, q_pin2, k0_l1, k_l0, tail_l0, False)
                    T.copy(q_l1[:, 384:512], q_l0)
                    residual_dot(384, q_l0, k0_l1, pv_b, tail_l0, False)
                    T.dual_copy(tail_l0, tail_scores[:half_rows, :])
                    if finalize_odd:
                        normalize_residual_heads(tail_scores, live, row_max, row_sum, alpha, prob)
                    else:
                        normalize_residual(tail_scores, live, row_max, row_sum, alpha, prob)
                else:
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    T.dual_copy(scores_l0, score0[0, :, :])
                    if live - (selected_tiles - 1) * tile >= tile:
                        normalize_full(selected_tiles - 1, score0, limits, row_max, row_sum, alpha, prob, False)
                    elif live - (selected_tiles - 1) * tile <= 64:
                        normalize_block(selected_tiles - 1, score0, limits, row_max, row_sum, alpha, prob, True)
                    else:
                        normalize_block(selected_tiles - 1, score0, limits, row_max, row_sum, alpha, prob, False)
                apply_last(
                    prob,
                    k0_l1,
                    p_l1,
                    q_l0,
                    pv_b,
                    out_l0,
                    partial,
                    running,
                    alpha,
                    other_l0,
                    other_ub,
                    residual_pool,
                    spare_b=k_l0,
                    row_sum=row_sum,
                    out=out,
                )
            if not finalize_odd or odd == 0:
                with T.SimdVF():
                    mask = S.pset(32)
                    for row in range(half_rows):
                        mass = S.vmaxs(S.vld(row_sum[row], "BRC_B32"), T.float32(1e-30), mask)
                        for col in range(0, DIM, 64):
                            value = S.vdiv(S.vld(running[row, col], "NORM"), mass, mask, precision="ftz_true")
                            result = S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0)
                            S.vsts(out[row, col], result, mask, "PK_B32")
            T.dual_copy(out[:half_rows, :], Y[bx, head : head + query_rows, :])

    if indexed_prefill:
        index_stride = T.dynamic("index_stride")

        @T.prim_func
        def indexed(
            Q: T.Tensor((batch, heads, DIM), dtype),
            Cache: T.Tensor((pages, page_size, PACKED_WORDS), T.int32),
            Table: T.Tensor((requests, table_columns), T.int32),
            Indices: T.StridedTensor((batch, max_visible), (index_stride, 1), T.int32),
            Ends: T.Tensor((requests,), T.int64),
            Positions: T.Tensor((batch,), T.int64),
            Y: T.Tensor((batch, heads, DIM), dtype),
        ):
            attend(Q, Cache, Table, Indices, Positions, Ends, Y)

        return indexed

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, DIM), dtype),
        Cache: T.Tensor((pages, page_size, PACKED_WORDS), T.int32),
        Table: T.Tensor((requests, table_columns), T.int32),
        Indices: T.Tensor(index_shape, metadata_dtype),
        Length: T.Tensor((length_offset + batch,), metadata_dtype),
        Y: T.Tensor((batch, heads, DIM), dtype),
    ):
        attend(Q, Cache, Table, Indices, Length, Indices, Y)

    return main
