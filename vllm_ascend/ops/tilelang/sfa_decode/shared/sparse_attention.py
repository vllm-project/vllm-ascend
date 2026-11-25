# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]
from tilelang.ascend.language import simd as S  # type: ignore[import-not-found]
from tilelang.layout import make_ascend_compact_nz_layout  # type: ignore[import-not-found]

from ..mtp import DENSE_PREFIX_TOKENS
from .attention_mma import attention_mma

DIM = 512

SCALE_GROUP = 128

PACKED_WORDS = DIM // 4 + DIM // SCALE_GROUP

SCORE_SCALE = 256**-0.5


@T.macro
def dequantize_group(scales, codes, row, channel, full, byte_mask, bf_mask):
    gain = S.vld(scales[row, DIM // 4 + channel], "BRC_B32")
    data = S.vld(codes[row, channel * SCALE_GROUP], "UNPK_B8")
    even = S.vmul(S.vcvt(data, "float32", byte_mask, part=0), gain, full)
    odd = S.vmul(S.vcvt(data, "float32", byte_mask, part=2), gain, full)
    lo = T.reinterpret(S.vcvt(even, "bfloat16", full, part=0), "uint16x128")
    hi = T.reinterpret(S.vcvt(odd, "bfloat16", full, part=1), "uint16x128")
    return T.reinterpret(S.vor(lo, hi, bf_mask), "bfloat16x128")  # codespell:ignore vor


def paged_attention(
    batch,
    requests,
    pages,
    table_width,
    *,
    heads=64,
    query_count,
    index_width,
    length_offset=0,
    score_rows_per_query=0,
    causal_bits=None,
    indexed_prefill=False,
):
    if query_count <= 1 or indexed_prefill or causal_bits is not None:
        raise ValueError("Expected MTP target verification queries")
    causal = causal_bits is not None
    prefill = causal or indexed_prefill
    mtp = query_count > 1
    groups = 1 if prefill or mtp else next(g for g in (4, 2, 1) if g == 1 or g * batch <= 28)
    query_rows = heads // groups
    page_size = tile = dim_tile = 128
    max_visible = DENSE_PREFIX_TOKENS
    half_rows = query_rows // 2
    stride_words = (PACKED_WORDS + 7) // 8 * 8
    dtype, f32 = T.bfloat16, T.float32
    neg = -1.0e30

    @T.macro
    def fetch(kv, k_l1, request, cache):
        bx, owner, sid, live, raw_len = request
        Cache, Table, Indices, k_ub, packed = cache
        pos = kv * tile
        row = sid * (tile // 2)
        first = pos + row
        if causal or raw_len < max_visible:
            remaining = T.min(T.max(live - first, 0), tile // 2)
            if remaining > 0:
                page = T.cast(Table[owner, kv], T.int64)
                T.copy(Cache[page, row : row + remaining, :], packed[:remaining, :PACKED_WORDS])
        else:
            for group in T.serial(tile // 8):
                i = first + 4 * group
                if i < live:
                    logical = T.alloc_var(T.int32, init=raw_len // 4 * 4)
                    if pos < max_visible:
                        logical = (
                            Indices[bx, i] if indexed_prefill or mtp or index_width != 512 else Indices[bx, i // 4] * 4
                        )
                    page = T.cast(Table[owner, logical // page_size], T.int64)
                    offset = logical % page_size
                    T.copy(Cache[page, offset : offset + 4, :], packed[group * 4 : group * 4 + 4, :PACKED_WORDS])
        if first + tile // 2 > live:
            with T.SimtVF(threads=256):
                for r, c in T.Parallel(tile // 2, PACKED_WORDS):
                    if first + r >= live:
                        packed[r, c] = 0
        fp8_bytes = T.view(packed, (tile // 2, stride_words * 4), dtype=T.float8_e4m3fn)
        fp32_scales = T.view(packed, (tile // 2, stride_words), dtype=T.float32)
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
                    for lane in (0, 1):
                        row_index = pair * 2 + lane
                        packed_row = dequantize_group(
                            fp32_scales, fp8_bytes, row_index, channel, full, bytes_mask, bfmask
                        )
                        dst = S.vsstb(packed_row, dst, nz_stride, bfmask, update=True)

        T.dual_copy(k_ub[: tile // 2, :], k_l1)

    dot, apply = attention_mma(half_rows)

    @T.macro
    def normalize_block(kv, scores, limits, row_max, row_sum, alpha, prob, short, masked):
        pos = kv * tile
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            scalar = S.pset(32, "PAT_VL1")
            bf = S.pset(16, "PAT_ALL")
            if masked:
                lanes = S.vci(T.int32(0), "int32")
                zero = S.vdup(T.float32(0), "float32", full)
                negs = S.vdup(T.float32(neg), "float32", full)
            d0 = S.make_ubuf_ptr(T.access_ptr(prob[0, 0], "w", half_rows + 1, tile), "bfloat16")
            for r in range(half_rows):
                left = S.vmuls(S.vld(scores[0, r, 0], "NORM"), T.float32(SCORE_SCALE), full)
                right = S.vmuls(S.vld(scores[0, r, 64], "NORM"), T.float32(SCORE_SCALE), full)
                if masked:
                    m0 = S.vcmps(lanes, limits[0, r] - pos, full, op="lt")
                    m1 = S.vcmps(lanes, limits[0, r] - pos - 64, full, op="lt")
                    a = S.vsel(left, negs, m0)
                    b = S.vsel(right, negs, m1)
                else:
                    a = left
                    b = right
                old = S.vld(row_max[r], "BRC_B32")
                pairmax = S.vdupv(S.vcmax(S.vmax(a, b, full), full), full)
                mx = S.vmax(old, pairmax, full)
                adj = S.vexpdif(old, mx, full)
                if masked:
                    a = S.vsel(S.vexpdif(a, mx, full), zero, m0)
                    b = S.vsel(S.vexpdif(b, mx, full), zero, m1)
                else:
                    a = S.vexpdif(a, mx, full)
                    b = S.vexpdif(b, mx, full)
                if short:
                    block_sum = S.vcadd(a, full)
                else:
                    even_exp, odd_exp = S.vdintlv(a, b)
                    block_sum = S.vcadd(S.vadd(even_exp, odd_exp, full), full)
                mass = S.vadd(S.vmul(S.vld(row_sum[r], "BRC_B32"), adj, full), block_sum, full)
                S.vsts(row_sum[r], mass, scalar, "ONEPT_B32")
                S.vsts(row_max[r], mx, scalar, "ONEPT_B32")
                S.vsts(alpha[r], adj, scalar, "ONEPT_B32")
                lo = S.vcvt(a, "bfloat16", full, round="ROUND_A", part=0)
                hi = S.vcvt(b, "bfloat16", full, round="ROUND_A", part=0)
                dense, unused = S.vdintlv(lo, hi)
                d0 = S.vsstb(dense, d0, T.int32(((half_rows + 1) << 16) | 1), bf, update=True)

    @T.macro
    def consume_tile(kv, live, k_l1, scores_l0, softmax, value_tile):
        score0, limits, row_max, row_sum, alpha, prob = softmax
        T.dual_copy(scores_l0, score0[0, :, :])
        if live - kv * tile >= tile:
            normalize_block(kv, score0, limits, row_max, row_sum, alpha, prob, False, False)
        elif live - kv * tile <= 64:
            normalize_block(kv, score0, limits, row_max, row_sum, alpha, prob, True, True)
        else:
            normalize_block(kv, score0, limits, row_max, row_sum, alpha, prob, False, True)
        apply(prob, k_l1, alpha, value_tile)

    @T.macro
    def row_end(ends, row):
        return ends[row, 0] if causal else ends[row]

    @T.macro
    def owner_for_row(row, ends, guess):
        lo = T.alloc_var(T.int32, init=guess)
        if row >= row_end(ends, guess):
            hi = T.alloc_var(T.int32, init=requests - 1)
            while lo < hi:
                mid = T.shift_right(lo + hi, 1)
                if row >= row_end(ends, mid):
                    lo = mid + 1
                else:
                    hi = mid
        return lo

    @T.macro
    def attend(Q, Cache, Table, Indices, Length, Positions, Starts, Ends, PreparedKV, Y):
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

            if mtp or prefill:
                guess = bx // query_count if mtp else 0
                owner = owner_for_row(bx, Ends, guess)
                active = bx < row_end(Ends, requests - 1)
                raw_len = T.if_then_else(active, T.max(T.cast(Positions[bx] + 1, T.int32), 0), 0)
            else:
                owner = bx
                raw_len = T.max(Length[length_offset + bx], 0)
            live = T.min(raw_len // 4, 512) * 4 + raw_len % 4
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

            softmax = (score0, limits, row_max, row_sum, alpha, prob)
            value_tile = (p_l1, q_l0, pv_b, out_l0, partial, running, other_l0, other_ub, k_l0)
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
            if score_rows_per_query and fits:
                score_start = Starts[owner] * score_rows_per_query
                k1_l1 = T.alloc_l1((tile, DIM), dtype)
                if selected_tiles > 0:
                    T.copy(PreparedKV[score_start : score_start + tile, :], k0_l1)
                for pair in T.serial(selected_tiles // 2):
                    left = 2 * pair
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    T.copy(PreparedKV[score_start + (left + 1) * tile : score_start + (left + 2) * tile, :], k1_l1)
                    consume_tile(left, live, k0_l1, scores_l0, softmax, value_tile)
                    dot(q_l1, k1_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    if left + 2 < selected_tiles:
                        T.copy(PreparedKV[score_start + (left + 2) * tile : score_start + (left + 3) * tile, :], k0_l1)
                    consume_tile(left + 1, live, k1_l1, scores_l0, softmax, value_tile)
                if selected_tiles % 2 == 1:
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    consume_tile(selected_tiles - 1, live, k0_l1, scores_l0, softmax, value_tile)
            else:
                for kv in T.serial(selected_tiles):
                    fetch(kv, k0_l1, (bx, owner, sid, live, raw_len), (Cache, Table, Indices, k_ub, packed))
                    dot(q_l1, k0_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, pv_b, scores_l0)
                    consume_tile(kv, live, k0_l1, scores_l0, softmax, value_tile)
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
        PreparedKV: T.Tensor((max(1, batch * score_rows_per_query), DIM), dtype),
    ):
        attend(Q, Cache, Table, Indices, Length, Positions, Starts, Ends, PreparedKV, Y)

    return main
