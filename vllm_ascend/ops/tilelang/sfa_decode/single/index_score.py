# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from . import (
    FRAGMENTED_INDEX_SEGMENT,
    INDEX_VOTE_STRIDE,
    LINKED_PAIR_LEADER,
    SHARED_INDEX_SEGMENT,
    use_four_request_index_scan,
    use_segmented_index_scan,
)

# Stream large scans through L2 without evicting the projection weights on a miss.
STREAMING_INDEX_BYTES = 1 << 30
LONG_CONTEXT_STREAMING_BYTES = 1 << 28
KEYS_PER_DESCRIPTOR = 4096


def score_program(
    batch,
    pages,
    context,
    tile=512,
    group=8,
    stages=2,
    *,
    table_columns,
    length_offset=0,
    fragmented_only=False,
    allow_specialized=True,
    shared_only=False,
    quartets=False,
    allow_paired_loads=True,
    partial_rows_only=False,
):
    assert tile == 512
    if fragmented_only:
        tile *= 2
        group //= 2
    pools = context // 4
    threshold = LONG_CONTEXT_STREAMING_BYTES if context == 131072 else STREAMING_INDEX_BYTES
    key_policy = "NOTALLOC_KEEP" if batch * pools * 128 >= threshold else "NORMAL_FV"
    runs_in_arena = max(pages - 15, 1)
    tiles_per_descriptor = KEYS_PER_DESCRIPTOR // tile
    max_cores = 28
    # Split short scans across otherwise idle Cubes while the requests still fit in one wave.
    while group > 1 and 2 * batch * (pools // (tile * group)) <= max_cores:
        group //= 2
    stages = min(stages, group)
    splits = pools // (tile * group)
    shared_rows = 4 if quartets else 2
    tasks = (batch // shared_rows if shared_only else batch) * splits
    if partial_rows_only:
        assert group * tile == KEYS_PER_DESCRIPTOR
        tasks = batch
    # On full waves, distribute the eight 128K descriptors over 27 paired cores.
    rotate_full_wave = (
        fragmented_only
        and 128 <= batch < 512
        and context == 131072
        and group == 4
        and tasks % max_cores == 0
        and tasks % (max_cores - 1) != 0
    )
    cores = min(max_cores - int(rotate_full_wave), tasks)
    assert pools % (tile * group) == 0

    pin_query = batch < 512
    # The paired load uses 12 pages per Vector and 8 on Cube. It needs a paired GEMM/FIX schedule.
    pack_key_pairs = allow_paired_loads and fragmented_only and batch >= 128 and tasks % cores != 0 and group % 2 == 0
    vector_rows = tile // 4
    if fragmented_only:
        vector_rows = 384 if pack_key_pairs else 320
    cube_rows = tile - 2 * vector_rows
    pages_per_tile = tile // 32
    cache_page_ids = batch >= max_cores
    segment_flags = use_segmented_index_scan(batch, context)
    share_keys = allow_specialized and segment_flags
    share_four = use_four_request_index_scan(batch, context)

    def is_dense(flag):
        return flag == 1 if segment_flags else flag != 0

    @T.macro
    def reduce_run(product, columns, pair_stride, dense, reduced, coef, scales, scores):
        if dense:
            # Reduce all heads through contiguous loads, then gather the four logical key vectors.
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                for chunk in range(T.ceildiv(columns, 64)):
                    origin = T.min(chunk * 64, columns - 64)
                    total = S.alloc_var("float32")
                    total = S.vdup(T.float32(0.0), "float32", full)
                    for h in range(32):
                        value = S.vmaxs(S.vld(product[h, origin], "NORM"), T.float32(0.0), full)
                        S.vmula(total, value, S.vld(coef[h], "BRC_B32"), full)
                    S.vsts(reduced[origin], total, full, "NORM_B32")
        with T.SimdVF():
            mask = S.pset(32, "PAT_ALL")
            lane = T.reinterpret(S.vci(T.int32(0), "int32"), "uint32x64")
            index = S.vsel(
                S.vadds(lane, T.uint32(2), mask),
                lane,
                S.vcmps(lane, T.uint32(32), mask, op="ge"),
            )
            for chunk in range(tile // 128):
                at = S.vadds(index, T.uint32((chunk // 2) * pair_stride + (chunk % 2) * 68), mask)
                if dense:
                    total = S.vgather2(reduced[0], at, mask)
                else:
                    total = S.alloc_var("float32")
                    total = S.vdup(T.float32(0.0), "float32", mask)
                    for h in range(32):
                        v = S.vmaxs(S.vgather2(product[h, 0], at, mask), T.float32(0.0), mask)
                        S.vmula(total, v, S.vld(coef[h], "BRC_B32"), mask)
                weighted = S.vmuls(
                    S.vmul(total, S.vld(scales[chunk * 64], "NORM"), mask),
                    T.float32(128.0**-0.5),
                    mask,
                )
                S.vsts(
                    scores[0, chunk * 64],
                    S.vcvt(weighted, "bfloat16", mask, round="ROUND_A", part=0),
                    mask,
                    "PK_B32",
                )

    @T.macro
    def load_scattered_scales(bx, sid, block, KS, Table, page_ids, scales):
        local_start = (block % group) * pages_per_tile
        for j in T.serial(tile // 64):
            offset = sid * (tile // 64) + j
            page = T.cast(
                page_ids[local_start + offset] if cache_page_ids else Table[bx, block * pages_per_tile + offset],
                T.int64,
            )
            T.copy(KS[page, 1024:1056], scales[j * 32 : (j + 1) * 32], l2_cache_ctrl=key_policy)

    @T.macro
    def swizzle_pairs(words, planes, first, count, base):
        mask = S.pset(32, "PAT_ALL")
        dest = S.make_ubuf_ptr(T.access_ptr(planes[base], "w", (count + 2) * 32), "float32")
        for pair in range(count // 2):
            value = S.vld(words[first * 32 + pair * 64], "NORM")
            dest = S.vsstb(value, dest, T.int32(((count // 2 + 1) << 16) | 1), mask, update=True)

    @T.macro
    def copy_pair_band(keys, planes, first, count, base):
        plane_words = (count // 2 + 1) * 32
        for parity in T.unroll(2):
            T.ascend_nd2nz_post_copy(
                T.access_ptr(keys[first + parity * (count // 2), 0], "w", count // 2, 128),
                T.access_ptr(planes[base + parity * plane_words], "r", plane_words),
                count // 2,
                32,
                tile,
                "float",
            )

    @T.macro
    def load_scattered_keys(
        bx, sid, block, K, KS, Table, page_ids, ku, scales, key_words, nz_words, prefetch_scales=False
    ):
        local_start = (block % group) * pages_per_tile
        for j in T.serial(cube_rows // 32):
            page = T.cast(Table[bx, block * (tile // 32) + j], T.int64)
            T.copy(K[page, :32, :], ku[j * 32 : (j + 1) * 32, :], l2_cache_ctrl=key_policy)
        # The first 1024 words of the FP32 cache alias hold the packed FP8 keys.
        for j in T.serial(vector_rows // 32):
            offset = cube_rows // 32 + sid * (vector_rows // 32) + j
            page = T.cast(
                page_ids[local_start + offset] if cache_page_ids else Table[bx, block * pages_per_tile + offset],
                T.int64,
            )
            T.copy(KS[page, :1024], key_words[j * 1024 : (j + 1) * 1024], l2_cache_ctrl=key_policy)
        if prefetch_scales:
            load_scattered_scales(bx, sid, block, KS, Table, page_ids, scales)
        if pack_key_pairs:
            # A full SIMD register holds two rows. Split banks at the FIX boundary so each
            # Vector can interleave its own scores after the ordered head reduction.
            boundary = tile // 2 - cube_rows
            rest = vector_rows - boundary
            second = (boundary + 2) * 32
            if sid == 0:
                with T.SimdVF():
                    swizzle_pairs(key_words, nz_words, 0, boundary, 0)
                    swizzle_pairs(key_words, nz_words, boundary, rest, second)
                copy_pair_band(ku, nz_words, cube_rows, boundary, 0)
                copy_pair_band(ku, nz_words, tile // 2, rest, second)
            else:
                with T.SimdVF():
                    swizzle_pairs(key_words, nz_words, 0, vector_rows, 0)
                copy_pair_band(ku, nz_words, cube_rows + vector_rows, vector_rows, 0)
        else:
            # A 128-byte FP8 row is four NZ blocks. Move its bits through 32-bit lanes.
            with T.SimdVF():
                mask = S.pset(32, "PAT_VL32")
                dest = S.make_ubuf_ptr(T.access_ptr(nz_words[0], "w", (vector_rows + 1) * 32), "float32")
                for row in range(vector_rows):
                    words = S.vld(key_words[row * 32], "NORM")
                    dest = S.vsstb(words, dest, T.int32(((vector_rows + 1) << 16) | 1), mask, update=True)
            T.ascend_nd2nz_post_copy(
                T.access_ptr(ku[cube_rows + sid * vector_rows, 0], "w", vector_rows, 128),
                T.access_ptr(nz_words[0], "r", (vector_rows + 1) * 32),
                vector_rows,
                32,
                tile,
                "float",
            )
        if not prefetch_scales:
            load_scattered_scales(bx, sid, block, KS, Table, page_ids, scales)

    @T.macro
    def store_weighted_scores(total, scales, scores, column, mask):
        weighted = S.vmuls(
            S.vmul(total, S.vld(scales[column], "NORM"), mask),
            T.float32(128.0**-0.5),
            mask,
        )
        S.vsts(
            scores[0, column],
            S.vcvt(weighted, "bfloat16", mask, round="ROUND_A", part=0),
            mask,
            "PK_B32",
        )

    @T.macro
    def reduce_packed_scores(bx, block, prod, coef, scales, scores, Score):
        columns = scores.shape[1] * 2
        with T.SimdVF():
            mask = S.pset(32, "PAT_ALL")
            for chunk in range(columns // 128):
                total = S.alloc_var("float32")
                total = S.vdup(T.float32(0.0), "float32", mask)
                for h in range(32):
                    v = S.vmaxs(S.vld(prod[h, chunk * 64], "NORM"), T.float32(0.0), mask)
                    c = S.vld(coef[h], "BRC_B32")
                    S.vmula(total, v, c, mask)
                store_weighted_scores(total, scales, scores, chunk * 64, mask)
        T.dual_copy(scores, Score[bx, block * tile : block * tile + columns])

    @T.macro
    def sum_four_stripes(prod, coef, p0, p1, p2, p3, mask):
        first_sum = S.alloc_var("float32")
        first_sum = S.vdup(T.float32(0.0), "float32", mask)
        second_sum = S.alloc_var("float32")
        second_sum = S.vdup(T.float32(0.0), "float32", mask)
        third_sum = S.alloc_var("float32")
        third_sum = S.vdup(T.float32(0.0), "float32", mask)
        fourth_sum = S.alloc_var("float32")
        fourth_sum = S.vdup(T.float32(0.0), "float32", mask)
        for head in range(32):
            coefficient = S.vld(coef[head], "BRC_B32")
            value = S.vmaxs(S.vld(prod[head, p0], "NORM"), T.float32(0.0), mask)
            S.vmula(first_sum, value, coefficient, mask)
            value = S.vmaxs(S.vld(prod[head, p1], "NORM"), T.float32(0.0), mask)
            S.vmula(second_sum, value, coefficient, mask)
            value = S.vmaxs(S.vld(prod[head, p2], "NORM"), T.float32(0.0), mask)
            S.vmula(third_sum, value, coefficient, mask)
            value = S.vmaxs(S.vld(prod[head, p3], "NORM"), T.float32(0.0), mask)
            S.vmula(fourth_sum, value, coefficient, mask)
        return first_sum, second_sum, third_sum, fourth_sum

    @T.macro
    def store_score_pair(even, odd, scales, scores, column, mask):
        low, high = S.vintlv(even, odd)
        store_weighted_scores(low, scales, scores, column, mask)
        store_weighted_scores(high, scales, scores, column + 64, mask)

    @T.macro
    def reduce_pair_band(prod, coef, scales, scores, first, count, mask):
        odd = first + count // 2
        for group in range(count // 256):
            even_at = first + group * 128
            odd_at = odd + group * 128
            a, b, c, d = sum_four_stripes(prod, coef, even_at, odd_at, even_at + 64, odd_at + 64, mask)
            store_score_pair(a, b, scales, scores, first + group * 256, mask)
            store_score_pair(c, d, scales, scores, first + group * 256 + 128, mask)
        if count % 256:
            offset = count // 256 * 128
            even_sum = S.alloc_var("float32")
            odd_sum = S.alloc_var("float32")
            even_sum = S.vdup(T.float32(0.0), "float32", mask)
            odd_sum = S.vdup(T.float32(0.0), "float32", mask)
            for head in range(32):
                weight = S.vld(coef[head], "BRC_B32")
                even_value = S.vmaxs(S.vld(prod[head, first + offset], "NORM"), T.float32(0.0), mask)
                odd_value = S.vmaxs(S.vld(prod[head, odd + offset], "NORM"), T.float32(0.0), mask)
                S.vmula(even_sum, even_value, weight, mask)
                S.vmula(odd_sum, odd_value, weight, mask)
            store_score_pair(even_sum, odd_sum, scales, scores, first + offset * 2, mask)

    @T.macro
    def reduce_linear_stripe(prod, coef, scales, scores, first, mask):
        a, b, c, d = sum_four_stripes(prod, coef, first, first + 64, first + 128, first + 192, mask)
        store_weighted_scores(a, scales, scores, first, mask)
        store_weighted_scores(b, scales, scores, first + 64, mask)
        store_weighted_scores(c, scales, scores, first + 128, mask)
        store_weighted_scores(d, scales, scores, first + 192, mask)

    @T.macro
    def reduce_fragmented_scores(sid, bx, block, prod, coef, scales, scores, Score):
        if pack_key_pairs:
            boundary = tile // 2 - cube_rows
            rest = vector_rows - boundary
            if sid == 0:
                with T.SimdVF():
                    mask = S.pset(32, "PAT_ALL")
                    reduce_linear_stripe(prod, coef, scales, scores, 0, mask)
                    reduce_pair_band(prod, coef, scales, scores, cube_rows, boundary, mask)
            else:
                with T.SimdVF():
                    mask = S.pset(32, "PAT_ALL")
                    reduce_pair_band(prod, coef, scales, scores, 0, rest, mask)
                    reduce_pair_band(prod, coef, scales, scores, rest, vector_rows, mask)
        else:
            with T.SimdVF():
                mask = S.pset(32, "PAT_ALL")
                for stripe in range(tile // (2 * 256)):
                    reduce_linear_stripe(prod, coef, scales, scores, stripe * 256, mask)
        T.dual_copy(scores, Score[bx, block * tile : (block + 1) * tile])

    @T.macro
    def score_packed_keys(bx, block, ql, ku, dot, prod, coef, scales, scores, Score):
        T.gemm(ql, ku, dot, transpose_B=True, clear_accum=True)
        T.dual_copy(dot, prod)
        reduce_packed_scores(bx, block, prod, coef, scales, scores, Score)

    @T.macro
    def score_mapped_tail(bx, sid, block, valid, K, KS, Table, ql, keys, dot, prod, coef, scales, scores, Score):
        columns = keys.shape[0]
        first = block * pages_per_tile
        last = (valid - 1) // 32
        for j in T.serial(columns // 32):
            page = T.cast(Table[bx, T.min(first + j, last)], T.int64)
            T.copy(K[page, :32, :], keys[j * 32 : (j + 1) * 32, :], l2_cache_ctrl=key_policy)
        for j in T.serial(columns // 64):
            page = T.cast(Table[bx, T.min(first + sid * (columns // 64) + j, last)], T.int64)
            T.copy(KS[page, 1024:1056], scales[j * 32 : (j + 1) * 32], l2_cache_ctrl=key_policy)
        score_packed_keys(bx, block, ql, keys, dot, prod, coef, scales, scores, Score)

    @T.macro
    def publish_shared_products(ql, peer_q, keys, dot, prod, peer_prod):
        T.gemm(ql, keys, dot, transpose_B=True, clear_accum=True)
        T.dual_copy(dot, prod)
        T.gemm(peer_q, keys, dot, transpose_B=True, clear_accum=True)
        T.dual_copy(dot, peer_prod)

    @T.macro
    def reduce_shared_pair(bx, block, prod, peer_prod, coef, peer_coef, scales, scores, Score):
        reduce_packed_scores(bx, block, prod, coef, scales, scores, Score)
        reduce_packed_scores(bx + 1, block, peer_prod, peer_coef, scales, scores, Score)

    @T.macro
    def score_full_tile(
        bx,
        sid,
        block,
        Runs,
        K,
        KS,
        Table,
        page_ids,
        Score,
        ql,
        ku,
        dot,
        prod,
        kwide,
        dwide,
        pwide,
        kfour,
        dfour,
        pfour,
        scales,
        scores,
        coef,
        key_words,
        nz_words,
        dense,
        reduced,
    ):
        run0 = T.cast(Runs[bx, 4 * block + 0], T.int64)
        run1 = T.cast(Runs[bx, 4 * block + 1], T.int64)
        run2 = T.cast(Runs[bx, 4 * block + 2], T.int64)
        run3 = T.cast(Runs[bx, 4 * block + 3], T.int64)
        addr = T.if_then_else(run0 >= 0 and run1 == run0 + 4, run0, T.int64(-1))
        second = T.if_then_else(run2 >= 0 and run3 == run2 + 4, run2, T.int64(-1))
        if addr >= 0 and second >= 0:
            bulk = T.StridedTensor((runs_in_arena, 544, 128), (4352, 128, 1), T.float8_e4m3fn, data=K.data)
            first = addr
            if second == addr + 8:
                T.copy(bulk[first, :, :], kwide, l2_cache_ctrl=key_policy)
            else:
                halves = T.StridedTensor((max(pages - 7, 1), 272, 128), (4352, 128, 1), T.float8_e4m3fn, data=K.data)
                T.copy(halves[addr, :, :], kwide[0:272, :], l2_cache_ctrl=key_policy)
                T.copy(halves[second, :, :], kwide[272:544, :], l2_cache_ctrl=key_policy)
            for j in T.serial(tile // 64):
                T.copy(
                    KS[T.if_then_else(sid == 0, addr, second) + j, 1024:1056],
                    scales[j * 32 : (j + 1) * 32],
                    l2_cache_ctrl=key_policy,
                )
            T.gemm(ql, kwide, dwide, transpose_B=True, clear_accum=True)
            T.dual_copy(dwide, pwide)
            reduce_run(pwide, 272, 136, dense, reduced, coef, scales, scores)
            T.dual_copy(scores, Score[bx, block * tile : (block + 1) * tile])
        elif dense or (run0 >= 0 and run1 >= 0 and run2 >= 0 and run3 >= 0):
            quart = T.StridedTensor((max(pages - 3, 1), 136, 128), (4352, 128, 1), T.float8_e4m3fn, data=K.data)
            T.copy(quart[run0, :, :], kfour[0:136, :], l2_cache_ctrl=key_policy)
            T.copy(quart[run1, :, :], kfour[144:280, :], l2_cache_ctrl=key_policy)
            T.copy(quart[run2, :, :], kfour[288:424, :], l2_cache_ctrl=key_policy)
            T.copy(quart[run3, :, :], kfour[432:568, :], l2_cache_ctrl=key_policy)
            for j in T.serial(tile // 64):
                first4 = T.if_then_else(sid == 0, T.if_then_else(j < 4, run0, run1), T.if_then_else(j < 4, run2, run3))
                T.copy(KS[first4 + j % 4, 1024:1056], scales[j * 32 : (j + 1) * 32], l2_cache_ctrl=key_policy)
            T.gemm(ql, kfour, dfour, transpose_B=True, clear_accum=True)
            T.dual_copy(dfour, pfour)
            reduce_run(pfour, 288, 144, dense, reduced, coef, scales, scores)
            T.dual_copy(scores, Score[bx, block * tile : (block + 1) * tile])
        else:
            load_scattered_keys(bx, sid, block, K, KS, Table, page_ids, ku, scales, key_words, nz_words)
            score_packed_keys(bx, block, ql, ku, dot, prod, coef, scales, scores, Score)

    @T.macro
    def scan_task(task):
        wave = task // cores
        lane = task % cores
        # Rotate each wave to spread short and long rows across the scan workers.
        rotation = wave * 12 if rotate_full_wave else wave
        if tasks % cores:
            rem = tasks % cores
            logical = T.if_then_else(
                task < tasks - rem,
                wave * cores + (lane + rotation) % cores,
                tasks - rem + (lane + rotation) % rem,
            )
        else:
            logical = wave * cores + (lane + rotation) % cores
        return logical

    @T.macro
    def prepare_query(bx, start, Q, QS, W, Table, ql, qs, w, coef, page_ids):
        if cache_page_ids:
            T.copy(Table[bx, start * pages_per_tile : (start + group) * pages_per_tile], page_ids)
        T.copy(Q[bx, :, :], ql)
        T.copy(QS[bx, :], qs)
        T.copy(W[bx, :], w)
        with T.SimtVF(threads=32):
            for h in T.Parallel(32):
                coef[h] = qs[h] * T.cast(w[h], T.float32)

    @T.macro
    def multiply_fragment(query, keys, qa, qb, qc, qd, even, odd, result):
        if pin_query:
            # Each K32 load can overlap the preceding MAD on the other L0B bank.
            T.copy(keys[:, :32], even)
            T.copy(keys[:, 32:64], odd)
            T.gemm(qa, even, result, transpose_B=True, clear_accum=True)
            T.copy(keys[:, 64:96], even)
            T.gemm(qb, odd, result, transpose_B=True, clear_accum=False)
            T.copy(keys[:, 96:128], odd)
            T.gemm(qc, even, result, transpose_B=True, clear_accum=False)
            T.gemm(qd, odd, result, transpose_B=True, clear_accum=False)
        else:
            T.gemm(query, keys, result, transpose_B=True, clear_accum=True)

    if fragmented_only:

        @T.prim_func
        def fragmented_scan(
            Q: T.Tensor((batch, 32, 128), T.float8_e4m3fn),
            QS: T.Tensor((batch, 32), T.float32),
            K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
            KS: T.Tensor((pages, 1088), T.float32),
            W: T.Tensor((batch, 32), T.bfloat16),
            Table: T.Tensor((batch, table_columns), T.int32),
            Length: T.Tensor((length_offset + batch,), T.int32),
            Runs: T.Tensor((batch, pools // 128 + pools // KEYS_PER_DESCRIPTOR), T.int32),
            Score: T.Tensor((batch, pools), T.bfloat16),
        ):
            with T.MixedKernel(cores, sids=2) as (core, sid):
                page_ids = T.alloc_shared((group * pages_per_tile,), T.int32)
                ql = T.alloc_l1((32, 128), T.float8_e4m3fn)
                ku = T.alloc_l1((tile, 128), T.float8_e4m3fn)
                next_keys = T.alloc_l1((tile, 128), T.float8_e4m3fn)
                next_scales = T.alloc_shared((tile // 2,), T.float32)
                dot = T.alloc_l0c((32, tile), T.float32)
                prod = T.alloc_shared((32, tile // 2), T.float32)
                w = T.alloc_shared((32,), T.bfloat16)
                qs = T.alloc_shared((32,), T.float32)
                coef = T.alloc_shared((32,), T.float32)
                scales = T.alloc_shared((tile // 2,), T.float32)
                scores = T.alloc_shared((1, tile // 2), T.bfloat16)
                key_words = T.alloc_shared(((vector_rows + 1) * 32,), T.float32)
                nz_words = T.alloc_shared(((vector_rows + (4 if pack_key_pairs else 1)) * 32,), T.float32)
                qa = T.alloc_l0a((32, 32), T.float8_e4m3fn)
                qb = T.alloc_l0a((32, 32), T.float8_e4m3fn)
                qc = T.alloc_l0a((32, 32), T.float8_e4m3fn)
                qd = T.alloc_l0a((32, 32), T.float8_e4m3fn)
                even = T.alloc_l0b((tile, 32), T.float8_e4m3fn)
                odd = T.alloc_l0b((tile, 32), T.float8_e4m3fn)
                for task in T.Persistent([batch * splits], cores, core):
                    logical = scan_task(task)
                    bx = logical // splits
                    start = (logical % splits) * group
                    valid = Length[length_offset + bx] // 4
                    if (start + group) * tile <= T.ceildiv(valid, 32) * 32 and Runs[
                        bx, pools // 128 + start // tiles_per_descriptor
                    ] == FRAGMENTED_INDEX_SEGMENT:
                        prepare_query(bx, start, Q, QS, W, Table, ql, qs, w, coef, page_ids)
                        if pin_query:
                            T.copy(ql[:, :32], qa)
                            T.copy(ql[:, 32:64], qb)
                            T.copy(ql[:, 64:96], qc)
                            T.copy(ql[:, 96:128], qd)
                        load_scattered_keys(bx, sid, start, K, KS, Table, page_ids, ku, scales, key_words, nz_words)
                        if group == 1:
                            score_packed_keys(bx, start, ql, ku, dot, prod, coef, scales, scores, Score)
                        else:
                            # FIX publishes the current scores before the next GM-to-L1 transfer.
                            # Both banks reuse the same layout conversion scratch and L0 accumulator.
                            for pair in T.serial(group // 2):
                                block = start + pair * 2
                                multiply_fragment(ql, ku, qa, qb, qc, qd, even, odd, dot)
                                T.dual_copy(dot, prod)
                                load_scattered_keys(
                                    bx,
                                    sid,
                                    block + 1,
                                    K,
                                    KS,
                                    Table,
                                    page_ids,
                                    next_keys,
                                    next_scales,
                                    key_words,
                                    nz_words,
                                    prefetch_scales=True,
                                )
                                reduce_fragmented_scores(sid, bx, block, prod, coef, scales, scores, Score)
                                multiply_fragment(ql, next_keys, qa, qb, qc, qd, even, odd, dot)
                                T.dual_copy(dot, prod)
                                if pair + 1 < (group + 1) // 2:
                                    load_scattered_keys(
                                        bx,
                                        sid,
                                        block + 2,
                                        K,
                                        KS,
                                        Table,
                                        page_ids,
                                        ku,
                                        scales,
                                        key_words,
                                        nz_words,
                                        prefetch_scales=True,
                                    )
                                reduce_fragmented_scores(sid, bx, block + 1, prod, coef, next_scales, scores, Score)
                            if group % 2:
                                score_packed_keys(bx, start + group - 1, ql, ku, dot, prod, coef, scales, scores, Score)

        return fragmented_scan

    @T.macro
    def load_peer_query(row, Q, QS, W, ql, qs, w, coef):
        T.copy(Q[row, :, :], ql)
        T.copy(QS[row, :], qs)
        T.copy(W[row, :], w)
        with T.SimtVF(threads=32):
            for head in T.Parallel(32):
                coef[head] = qs[head] * T.cast(w[head], T.float32)

    if shared_only:

        @T.prim_func
        def shared_scan(
            Q: T.Tensor((batch, 32, 128), T.float8_e4m3fn),
            QS: T.Tensor((batch, 32), T.float32),
            K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
            KS: T.Tensor((pages, 1088), T.float32),
            W: T.Tensor((batch, 32), T.bfloat16),
            Table: T.Tensor((batch, table_columns), T.int32),
            Length: T.Tensor((length_offset + batch,), T.int32),
            Runs: T.Tensor((batch, pools // 128 + pools // KEYS_PER_DESCRIPTOR), T.int32),
            Score: T.Tensor((batch, pools), T.bfloat16),
        ):
            with T.MixedKernel(cores, sids=2) as (core, sid):
                page_ids = T.alloc_shared((group * pages_per_tile,), T.int32)
                key_words = T.alloc_shared(((vector_rows + 1) * 32,), T.float32)
                nz_words = T.alloc_shared(((vector_rows + 1) * 32,), T.float32)
                ql = T.alloc_l1((32, 128), T.float8_e4m3fn)
                ku = T.alloc_l1((tile, 128), T.float8_e4m3fn)
                dot = T.alloc_l0c((32, tile), T.float32)
                prod = T.alloc_shared((32, tile // 2), T.float32)
                qs = T.alloc_shared((32,), T.float32)
                w = T.alloc_shared((32,), T.bfloat16)
                coef = T.alloc_shared((32,), T.float32)
                scales = T.alloc_shared((tile // 2,), T.float32)
                scores = T.alloc_shared((1, tile // 2), T.bfloat16)
                for task in T.Persistent([tasks], cores, core):
                    logical = scan_task(task)
                    bx = shared_rows * (logical // splits)
                    start = (logical % splits) * group
                    column = pools // 128 + start // tiles_per_descriptor
                    flag = Runs[bx, column]
                    if (
                        flag == LINKED_PAIR_LEADER
                        if quartets
                        else flag >= SHARED_INDEX_SEGMENT
                        if share_four
                        else flag == SHARED_INDEX_SEGMENT
                    ):
                        if not quartets:
                            prepare_query(bx, start, Q, QS, W, Table, ql, qs, w, coef, page_ids)
                        peer_q = T.alloc_l1((32, 128), T.float8_e4m3fn)
                        peer_prod = T.alloc_shared((32, tile // 2), T.float32)
                        peer_coef = T.alloc_shared((32,), T.float32)
                        if not quartets:
                            load_peer_query(bx + 1, Q, QS, W, peer_q, qs, w, peer_coef)
                        if quartets:
                            extra_q = T.alloc_l1((32, 128), T.float8_e4m3fn)
                            last_q = T.alloc_l1((32, 128), T.float8_e4m3fn)
                            extra_prod = T.alloc_shared((32, tile // 2), T.float32)
                            last_prod = T.alloc_shared((32, tile // 2), T.float32)
                            extra_coef = T.alloc_shared((32,), T.float32)
                            last_coef = T.alloc_shared((32,), T.float32)
                            group_scales = T.alloc_shared((4, 32), T.float32)
                            group_weights = T.alloc_shared((4, 32), T.bfloat16)
                            T.copy(Table[bx, start * pages_per_tile : (start + group) * pages_per_tile], page_ids)
                            T.copy(Q[bx, :, :], ql)
                            T.copy(Q[bx + 1, :, :], peer_q)
                            T.copy(Q[bx + 2, :, :], extra_q)
                            T.copy(Q[bx + 3, :, :], last_q)
                            T.copy(QS[bx : bx + 4, :], group_scales)
                            T.copy(W[bx : bx + 4, :], group_weights)
                            with T.SimtVF(threads=32):
                                for head in T.Parallel(32):
                                    coef[head] = group_scales[0, head] * T.cast(group_weights[0, head], T.float32)
                                    peer_coef[head] = group_scales[1, head] * T.cast(group_weights[1, head], T.float32)
                                    extra_coef[head] = group_scales[2, head] * T.cast(group_weights[2, head], T.float32)
                                    last_coef[head] = group_scales[3, head] * T.cast(group_weights[3, head], T.float32)
                        next_keys = T.alloc_l1((tile, 128), T.float8_e4m3fn)
                        next_scales = T.alloc_shared((tile // 2,), T.float32)
                        load_scattered_keys(bx, sid, start, K, KS, Table, page_ids, ku, scales, key_words, nz_words)
                        # Each pair has its own FIX output while the next K bank fills.
                        for pair in T.serial(group // 2):
                            block = start + 2 * pair
                            publish_shared_products(ql, peer_q, ku, dot, prod, peer_prod)
                            if quartets:
                                publish_shared_products(extra_q, last_q, ku, dot, extra_prod, last_prod)
                            load_scattered_keys(
                                bx,
                                sid,
                                block + 1,
                                K,
                                KS,
                                Table,
                                page_ids,
                                next_keys,
                                next_scales,
                                key_words,
                                nz_words,
                            )
                            reduce_shared_pair(bx, block, prod, peer_prod, coef, peer_coef, scales, scores, Score)
                            if quartets:
                                reduce_shared_pair(
                                    bx + 2, block, extra_prod, last_prod, extra_coef, last_coef, scales, scores, Score
                                )
                            publish_shared_products(ql, peer_q, next_keys, dot, prod, peer_prod)
                            if quartets:
                                publish_shared_products(extra_q, last_q, next_keys, dot, extra_prod, last_prod)
                            if pair + 1 < (group + 1) // 2:
                                load_scattered_keys(
                                    bx, sid, block + 2, K, KS, Table, page_ids, ku, scales, key_words, nz_words
                                )
                            reduce_shared_pair(
                                bx, block + 1, prod, peer_prod, coef, peer_coef, next_scales, scores, Score
                            )
                            if quartets:
                                reduce_shared_pair(
                                    bx + 2,
                                    block + 1,
                                    extra_prod,
                                    last_prod,
                                    extra_coef,
                                    last_coef,
                                    next_scales,
                                    scores,
                                    Score,
                                )
                        if group % 2:
                            publish_shared_products(ql, peer_q, ku, dot, prod, peer_prod)
                            if quartets:
                                publish_shared_products(extra_q, last_q, ku, dot, extra_prod, last_prod)
                            reduce_shared_pair(
                                bx, start + group - 1, prod, peer_prod, coef, peer_coef, scales, scores, Score
                            )
                            if quartets:
                                reduce_shared_pair(
                                    bx + 2,
                                    start + group - 1,
                                    extra_prod,
                                    last_prod,
                                    extra_coef,
                                    last_coef,
                                    scales,
                                    scores,
                                    Score,
                                )

        return shared_scan

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 32, 128), T.float8_e4m3fn),
        QS: T.Tensor((batch, 32), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.Tensor((pages, 1088), T.float32),
        W: T.Tensor((batch, 32), T.bfloat16),
        Table: T.Tensor((batch, table_columns), T.int32),
        Length: T.Tensor((length_offset + batch,), T.int32),
        Runs: T.Tensor((batch, pools // 128 + pools // KEYS_PER_DESCRIPTOR), T.int32),
        Score: T.Tensor((batch, pools), T.bfloat16),
    ):
        with T.MixedKernel(cores, sids=2) as (core, sid):
            page_ids = T.alloc_shared((group * pages_per_tile,), T.int32)
            reduced = T.alloc_shared((288,), T.float32)
            key_words = T.alloc_shared(((vector_rows + 1) * 32,), T.float32)
            nz_words = T.alloc_shared(((vector_rows + 1) * 32,), T.float32)
            ql = T.alloc_l1((32, 128), T.float8_e4m3fn)
            ku = T.alloc_l1((tile, 128), T.float8_e4m3fn)
            dot = T.alloc_l0c((32, tile), T.float32)
            prod = T.alloc_shared((32, tile // 2), T.float32)
            qs = T.alloc_shared((32,), T.float32)
            w = T.alloc_shared((32,), T.bfloat16)
            coef = T.alloc_shared((32,), T.float32)
            scales = T.alloc_shared((tile // 2,), T.float32)
            scores = T.alloc_shared((1, tile // 2), T.bfloat16)
            kwide = T.alloc_l1((544, 128), T.float8_e4m3fn)
            dwide = T.alloc_l0c((32, 544), T.float32)
            pwide = T.alloc_shared((32, 272), T.float32)
            # Four 34-row pages occupy 136 rows. Round each run to 144 for the L1 NZ row alignment.
            kfour = T.alloc_l1((576, 128), T.float8_e4m3fn)
            dfour = T.alloc_l0c((32, 576), T.float32)
            pfour = T.alloc_shared((32, 288), T.float32)
            for task in T.Persistent([tasks], cores, core):
                logical = scan_task(task)
                bx = logical if partial_rows_only else logical // splits
                valid = Length[length_offset + bx] // 4
                start = (
                    (T.ceildiv(valid, 32) * 32 // KEYS_PER_DESCRIPTOR) * tiles_per_descriptor
                    if partial_rows_only
                    else (logical % splits) * group
                )
                if (
                    start * tile < valid
                    and valid >= 512
                    and (
                        partial_rows_only
                        or not share_keys
                        or (start + group) * tile > T.ceildiv(valid, 32) * 32
                        or (
                            Runs[bx, pools // 128 + start // tiles_per_descriptor] != FRAGMENTED_INDEX_SEGMENT
                            and (
                                Runs[bx, pools // 128 + start // tiles_per_descriptor] < SHARED_INDEX_SEGMENT
                                if share_four
                                else Runs[bx, pools // 128 + start // tiles_per_descriptor] != SHARED_INDEX_SEGMENT
                            )
                        )
                    )
                ):
                    # A mapped page contains 32 keys. Selection masks its unused suffix.
                    mapped_keys = T.ceildiv(valid, 32) * 32
                    prepare_query(bx, start, Q, QS, W, Table, ql, qs, w, coef, page_ids)
                    if not partial_rows_only and (start + group) * tile <= mapped_keys:
                        if is_dense(Runs[bx, pools // 128 + start // tiles_per_descriptor]):
                            for subtile in T.Pipelined(group, num_stages=stages):
                                block = start + subtile
                                score_full_tile(
                                    bx,
                                    sid,
                                    block,
                                    Runs,
                                    K,
                                    KS,
                                    Table,
                                    page_ids,
                                    Score,
                                    ql,
                                    ku,
                                    dot,
                                    prod,
                                    kwide,
                                    dwide,
                                    pwide,
                                    kfour,
                                    dfour,
                                    pfour,
                                    scales,
                                    scores,
                                    coef,
                                    key_words,
                                    nz_words,
                                    True,
                                    reduced,
                                )
                        else:
                            for subtile in T.serial(group):
                                block = start + subtile
                                score_full_tile(
                                    bx,
                                    sid,
                                    block,
                                    Runs,
                                    K,
                                    KS,
                                    Table,
                                    page_ids,
                                    Score,
                                    ql,
                                    ku,
                                    dot,
                                    prod,
                                    kwide,
                                    dwide,
                                    pwide,
                                    kfour,
                                    dfour,
                                    pfour,
                                    scales,
                                    scores,
                                    coef,
                                    key_words,
                                    nz_words,
                                    False,
                                    reduced,
                                )
                    else:
                        for subtile in T.serial((mapped_keys - start * tile) // tile):
                            block = start + subtile
                            score_full_tile(
                                bx,
                                sid,
                                block,
                                Runs,
                                K,
                                KS,
                                Table,
                                page_ids,
                                Score,
                                ql,
                                ku,
                                dot,
                                prod,
                                kwide,
                                dwide,
                                pwide,
                                kfour,
                                dfour,
                                pfour,
                                scales,
                                scores,
                                coef,
                                key_words,
                                nz_words,
                                False,
                                reduced,
                            )
                        if mapped_keys % tile != 0:
                            block = start + (valid - start * tile) // tile
                            if batch >= 128 and mapped_keys - block * tile <= tile // 2:
                                # Selection masks scores past the last live pool. Only load and score half a tile.
                                width = tile // 2
                                tail_keys = T.alloc_l1((width, 128), T.float8_e4m3fn)
                                tail_dot = T.alloc_l0c((32, width), T.float32)
                                tail_prod = T.alloc_shared((32, width // 2), T.float32)
                                tail_scales = T.alloc_shared((width // 2,), T.float32)
                                tail_scores = T.alloc_shared((1, width // 2), T.bfloat16)
                                score_mapped_tail(
                                    bx,
                                    sid,
                                    block,
                                    valid,
                                    K,
                                    KS,
                                    Table,
                                    ql,
                                    tail_keys,
                                    tail_dot,
                                    tail_prod,
                                    coef,
                                    tail_scales,
                                    tail_scores,
                                    Score,
                                )
                            else:
                                score_mapped_tail(
                                    bx, sid, block, valid, K, KS, Table, ql, ku, dot, prod, coef, scales, scores, Score
                                )

    return main


def select_index_schedule(batch, context):
    vote_offset = batch * (context // 512 + context // 16384)
    lanes = 256
    select_four = use_four_request_index_scan(batch, context)
    columns = context // 512 + context // 16384

    @T.prim_func
    def main(Mode: T.Tensor((vote_offset + batch + 1 + 2 * int(select_four),), T.int32)):
        with T.Kernel(1) as _:
            votes = T.alloc_shared((batch,), T.int32)
            T.copy(Mode[vote_offset : vote_offset + batch], votes)
            with T.SimtVF(threads=lanes):
                ordinary_rows = T.alloc_fragment((1, lanes), T.int32)
                has_ordinary = T.alloc_fragment((1,), T.int32)
                choices = T.alloc_fragment((1, lanes), T.int32)
                total = T.alloc_fragment((1,), T.int32)
                for lane in T.Parallel(lanes):
                    value = T.alloc_var(T.int32, init=0)
                    ordinary = T.alloc_var(T.int32, init=0)
                    for group in T.serial(T.ceildiv(batch, lanes)):
                        row = group * lanes + lane
                        if row < batch:
                            if select_four:
                                value += votes[row] >> 1
                                ordinary += votes[row] & 1
                            else:
                                value += votes[row]
                    choices[0, lane] = value
                    if select_four:
                        ordinary_rows[0, lane] = ordinary
                T.reduce_sum(choices, total, dim=1)
                if select_four:
                    T.reduce_sum(ordinary_rows, has_ordinary, dim=1)
                for lane in T.Parallel(1):
                    Mode[vote_offset + batch] = total[lane]
                    if select_four:
                        Mode[vote_offset + batch + 2] = has_ordinary[lane]

            if select_four:
                shared = Mode[vote_offset + batch] & (INDEX_VOTE_STRIDE - 1)
                if shared != 0:
                    with T.SimtVF(threads=lanes):
                        counts = T.alloc_fragment((1, lanes), T.int32)
                        quartets = T.alloc_fragment((1,), T.int32)
                        for lane in T.Parallel(lanes):
                            matches = T.alloc_var(T.int32, init=0)
                            row = lane * 4
                            if row + 3 < batch:
                                for segment in T.unroll(context // 16384):
                                    index = row * columns + context // 512 + segment
                                    matches += T.cast(
                                        Mode[index] == LINKED_PAIR_LEADER
                                        and Mode[index + 2 * columns] == SHARED_INDEX_SEGMENT,
                                        T.int32,
                                    )
                            counts[0, lane] = matches
                        T.reduce_sum(counts, quartets, dim=1)
                        for lane in T.Parallel(1):
                            Mode[vote_offset + batch + 1] = T.cast(quartets[lane] * 4 == shared, T.int32)

    return main


def reset_index_count(batch, context):
    offset = batch * (context // 512 + context // 16384)

    @T.prim_func
    def main(Mode: T.Tensor((offset + 1,), T.int32)):
        with T.Kernel(1) as _, T.SimtVF(threads=32):
            for lane in T.Parallel(1):
                Mode[offset + lane] = 0

    return main
