# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

DIM = 512
FULL_TILES = 16
TILES_WITH_TAIL = FULL_TILES + 1


def resident_value_tiles(
    batch,
    pages,
    table_columns,
    length_offset,
    cores,
    packed_words,
    aligned_words,
    scale,
    negative,
    fetch,
    dot,
    normalize_full,
    normalize_block,
):
    tile = 128
    dtype, f32 = T.bfloat16, T.float32

    workers_per_request = cores // batch
    slots = tuple(range((TILES_WITH_TAIL + workers_per_request - 1) // workers_per_request))

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 64, DIM), dtype),
        Cache: T.Tensor((pages, tile, packed_words), T.int32),
        Table: T.Tensor((batch, table_columns), T.int32),
        Indices: T.Tensor((batch, 512), T.int32),
        Length: T.Tensor((length_offset + batch,), T.int32),
        Maxima: T.Tensor((batch, TILES_WITH_TAIL, 64), f32),
        Partial: T.Tensor((batch, 64, FULL_TILES * DIM), f32),
        TailValue: T.Tensor((batch, 64, DIM), f32),
        Stats: T.Tensor((batch, 2, TILES_WITH_TAIL, 64), f32),
    ):
        with T.MixedKernel(cores, sids=2) as (core, sid):
            request = core // workers_per_request
            first = core % workers_per_request
            length = T.max(Length[length_offset + T.min(request, batch - 1)], 0)
            live = T.min(length // 4, 512) * 4 + length % 4
            query_l1 = T.alloc_l1((64, DIM), dtype)
            query0 = T.alloc_l0a((64, 128), dtype)
            query1 = T.alloc_l0a((64, 128), dtype)
            query2 = T.alloc_l0a((64, 128), dtype)
            query_work = T.alloc_l0a((64, 128), dtype)
            key0 = T.alloc_l0b((tile, 128), dtype)
            key1 = T.alloc_l0b((tile, 128), dtype)
            score_l0 = T.alloc_l0c((64, tile), f32)
            packed = T.alloc_shared((tile // 2, aligned_words), T.int32)
            key_ub = T.alloc_shared((tile // 2 + 1, DIM), dtype)
            T.annotate_layout({key_ub: make_ascend_compact_nz_layout(key_ub)})
            tile_max = T.alloc_shared((32,), f32)
            prob_l1 = T.alloc_l1((64, tile), dtype)
            value_l0 = T.alloc_l0c((64, 128), f32)
            limits = T.alloc_shared((1, 32), T.int32)
            row_max = T.alloc_shared((64,), f32)
            mass = T.alloc_shared((32,), f32)
            alpha = T.alloc_shared((32,), f32)
            prob = T.alloc_shared((33, tile), dtype)
            T.annotate_layout({prob: make_ascend_compact_nz_layout(prob)})
            value_ub = T.alloc_shared((32, 128), f32)
            previous = T.alloc_shared((TILES_WITH_TAIL, 64), f32)
            keys = [T.alloc_l1((tile, DIM), dtype) for _ in slots]
            tiles = [T.alloc_shared((32, tile), f32) for _ in slots]
            if request < batch and first * tile < live:
                T.copy(Q[request, :, :], query_l1)
                T.copy(query_l1[:, 0:128], query0)
                T.copy(query_l1[:, 128:256], query1)
                T.copy(query_l1[:, 256:384], query2)
                for slot in slots:
                    tile_id = first + slot * workers_per_request
                    key_l1 = keys[slot]
                    score_rows = tiles[slot]
                    if tile_id * tile < live:
                        fetch(
                            request, request, sid, tile_id, live, length, Cache, Table, Indices, key_l1, key_ub, packed
                        )
                        dot(query_l1, key_l1, query0, query1, query2, query_work, key0, key1, score_l0)
                        T.dual_copy(score_l0, score_rows)
                        with T.SimdVF():
                            full = S.pset(32, "PAT_ALL")
                            scalar = S.pset(32, "PAT_VL1")
                            lanes = S.vci(T.int32(0), "int32")
                            valid_lo = S.vcmps(lanes, live - tile_id * tile, full, op="lt")
                            valid_hi = S.vcmps(lanes, live - tile_id * tile - 64, full, op="lt")
                            masked = S.vdup(T.float32(negative), "float32", full)
                            for r in range(32):
                                left = S.vsel(
                                    S.vmuls(S.vld(score_rows[r, 0], "NORM"), T.float32(scale), full),
                                    masked,
                                    valid_lo,
                                )
                                right = S.vsel(
                                    S.vmuls(S.vld(score_rows[r, 64], "NORM"), T.float32(scale), full),
                                    masked,
                                    valid_hi,
                                )
                                S.vsts(
                                    tile_max[r],
                                    S.vcmax(S.vmax(left, right, full), full),
                                    scalar,
                                    "ONEPT_B32",
                                )
                        T.copy(tile_max, Maxima[request, tile_id, sid * 32 : sid * 32 + 32])
            # Each Cube retains its KV tiles in L1 across the all-Vector prefix-max rendezvous.
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 11)
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE2", 11)
            if request < batch and first * tile < live:
                T.copy(Maxima[request, :, :], previous)
                for slot in slots:
                    tile_id = first + slot * workers_per_request
                    key_l1 = keys[slot]
                    score_rows = tiles[slot]
                    if tile_id * tile < live:
                        scores = T.view(score_rows, (1, 32, tile), dtype=f32)
                        with T.SimdVF():
                            half = S.pset(32, "PAT_VL32")
                            peak = S.alloc_var("float32")
                            peak = S.vdup(T.float32(negative), "float32", half)
                            for i in range(tile_id):
                                peak = S.vmax(peak, S.vld(previous[i, sid * 32], "NORM"), half)
                            S.vsts(row_max[0], peak, half, "NORM_B32")
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(32):
                                mass[i] = 0
                                limits[0, i] = live
                        if live - tile_id * tile >= tile:
                            normalize_full(tile_id, scores, limits, row_max, mass, alpha, prob, False)
                        elif live - tile_id * tile <= 64:
                            normalize_block(tile_id, scores, limits, row_max, mass, alpha, prob, True)
                        else:
                            normalize_block(tile_id, scores, limits, row_max, mass, alpha, prob, False)
                        T.dual_copy(prob[:32, :], prob_l1)
                        T.copy(prob_l1, query_work)
                        for segment in T.unroll(DIM // tile):
                            col = segment * 128
                            T.copy(key_l1[:, col : col + 128], key1)
                            T.gemm(query_work, key1, value_l0, clear_accum=True)
                            dst = tile_id * DIM + col
                            if batch == 1:
                                T.dual_copy(value_l0, value_ub)
                                if tile_id < FULL_TILES:
                                    T.copy(value_ub, Partial[request, sid * 32 : sid * 32 + 32, dst : dst + 128])
                                else:
                                    T.copy(value_ub, TailValue[request, sid * 32 : sid * 32 + 32, col : col + 128])
                            elif tile_id < FULL_TILES:
                                # FIX writes all heads while Vector prepares the next tile.
                                T.copy(value_l0, Partial[request, :, dst : dst + 128])
                            else:
                                T.copy(value_l0, TailValue[request, :, col : col + 128])
                        T.copy(alpha, Stats[request, 0, tile_id, sid * 32 : sid * 32 + 32])
                        T.copy(mass, Stats[request, 1, tile_id, sid * 32 : sid * 32 + 32])

    return main


def merge_value_tiles(batch, length_offset):
    tile = 128
    dtype, f32 = T.bfloat16, T.float32

    @T.prim_func
    def main(
        Partial: T.Tensor((batch, 64, FULL_TILES, DIM), f32),
        TailValue: T.Tensor((batch, 64, DIM), f32),
        Stats: T.Tensor((batch, 2, TILES_WITH_TAIL, 64), f32),
        Length: T.Tensor((length_offset + batch,), T.int32),
        Y: T.Tensor((batch, 64, DIM), dtype),
    ):
        with T.Kernel(batch * 64) as block:
            request, h = block // 64, block % 64
            length = T.max(Length[length_offset + request], 0)
            live = T.min(length // 4, 512) * 4 + length % 4
            tiles = T.ceildiv(live, tile)
            parts = T.alloc_shared((TILES_WITH_TAIL, DIM), f32)
            stats = T.alloc_shared((2, TILES_WITH_TAIL, 64), f32)
            out = T.alloc_shared((DIM,), dtype)
            if tiles >= FULL_TILES:
                T.copy(Partial[request, h, :, :], parts[:FULL_TILES, :])
            elif tiles > 0:
                T.copy(Partial[request, h, :tiles, :], parts[:tiles, :])
            if tiles == TILES_WITH_TAIL:
                T.copy(TailValue[request, h, :], parts[FULL_TILES, :])
            T.copy(Stats[request, :, :, :], stats)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                total = S.alloc_var("float32")
                total = S.vdup(T.float32(0), "float32", full)
                for k in range(tiles):
                    adj = S.vld(stats[0, k, h], "BRC_B32")
                    mass = S.vld(stats[1, k, h], "BRC_B32")
                    total = S.vadd(S.vmul(total, adj, full), mass, full)
                denominator = S.vmaxs(total, T.float32(1e-30), full)
                for j in range(0, DIM, 64):
                    value = S.alloc_var("float32")
                    value = S.vdup(T.float32(0), "float32", full)
                    for k in range(tiles):
                        adj = S.vld(stats[0, k, h], "BRC_B32")
                        value = S.vadd(S.vmul(value, adj, full), S.vld(parts[k, j], "NORM"), full)
                    normalized = S.vdiv(value, denominator, full, precision="ftz_true")
                    bf = S.vcvt(normalized, "bfloat16", full, round="ROUND_A", part=0)
                    S.vsts(out[j], bf, full, "PK_B32")
            T.copy(out, Y[request, h, :])

    return main
