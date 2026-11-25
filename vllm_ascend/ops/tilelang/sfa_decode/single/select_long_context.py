# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .select_pool_radix_fused import make_pool_ranker

CHUNK_POOLS = 32768
TOP_K = 512


def make_long_rankers(pools):
    chunks = pools // CHUNK_POOLS
    assert pools % CHUNK_POOLS == 0 and 1 < chunks <= 32
    candidates = chunks * TOP_K
    local_rank = make_pool_ranker(CHUNK_POOLS, TOP_K, True, preloaded=True)
    merge_rank = make_pool_ranker(candidates, TOP_K, True, preloaded=True)

    @T.macro
    def rank_chunk(X, bx, start, count, chunk, raw, ords, hist, bounds, local_ids, shortlist, words, ids):
        if count >= TOP_K:
            span = T.ceildiv(count, 256) * 256
            T.copy(X[bx, start : start + span], raw[:span])
            local_rank(X, bx, count, raw, ords, hist, bounds, local_ids)
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        j = half * 256 + lane
                        index = local_ids[j]
                        key = ords[T.cast(index, T.int32)]
                        words[chunk, j] = T.bitwise_xor(
                            key, T.if_then_else(key >= T.uint16(32768), T.uint16(32768), T.uint16(65535))
                        )
                        ids[chunk, j] = index
        else:
            T.copy(
                X[bx, start : start + TOP_K],
                shortlist[0, chunk * TOP_K : (chunk + 1) * TOP_K],
            )
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        j = half * 256 + lane
                        ids[chunk, j] = T.cast(j, T.uint16)

    @T.macro
    def merge_chunks(X, bx, visible, merged, merged_ords, ids, hist, bounds, local_ids, out):
        # The global cutoff is at least each full chunk's cutoff. Its greater scores
        # and selected ties therefore retain their original order within each chunk.
        live = visible // CHUNK_POOLS * TOP_K + T.min(visible % CHUNK_POOLS, TOP_K)
        merge_rank(X, bx, live, merged, merged_ords, hist, bounds, local_ids)
        with T.SimtVF(threads=256):
            for half in T.serial(2):
                for lane in T.Parallel(256):
                    j = half * 256 + lane
                    index = T.cast(local_ids[j], T.int32)
                    chunk = index // TOP_K
                    out[j] = chunk * CHUNK_POOLS + T.cast(ids[chunk, index % TOP_K], T.int32)

    return rank_chunk, merge_chunks


def make_long_selector(pools):
    chunks = pools // CHUNK_POOLS
    candidates = chunks * TOP_K
    rank_chunk, merge_chunks = make_long_rankers(pools)

    @T.macro
    def select(X, bx, visible, out):
        raw = T.alloc_shared((CHUNK_POOLS,), T.bfloat16)
        ords = T.view(raw, (CHUNK_POOLS,), dtype=T.uint16)
        local_ids = T.alloc_shared((TOP_K,), T.uint16)
        hist = T.alloc_shared((256,), T.uint16)
        bounds = T.alloc_shared((8,), T.int32)
        shortlist = T.alloc_shared((1, candidates), T.bfloat16)
        words = T.view(shortlist, (chunks, TOP_K), dtype=T.uint16)
        ids = T.alloc_shared((chunks, TOP_K), T.uint16)
        merged = T.view(shortlist, (candidates,), dtype=T.bfloat16)
        merged_ords = T.view(shortlist, (candidates,), dtype=T.uint16)
        if visible >= TOP_K:
            for chunk in T.serial(T.ceildiv(visible, CHUNK_POOLS)):
                count = T.min(visible - chunk * CHUNK_POOLS, CHUNK_POOLS)
                rank_chunk(
                    X, bx, chunk * CHUNK_POOLS, count, chunk, raw, ords, hist, bounds, local_ids, shortlist, words, ids
                )
            merge_chunks(X, bx, visible, merged, merged_ords, ids, hist, bounds, local_ids, out)
        else:
            with T.SimdVF():
                mask = S.pset(32)
                lanes = S.vci(T.int32(0), "int32")
                for j in range(TOP_K // 64):
                    S.vsts(out[j * 64], S.vadds(lanes, j * 64, mask), mask, "NORM_B32")

    return select


def select_long_context(batch, context, length_offset=0):
    pools = context // 4
    select = make_long_selector(pools)

    @T.prim_func
    def main(
        X: T.Tensor((batch, pools), T.bfloat16),
        L: T.Tensor((length_offset + batch,), T.int32),
        Y: T.Tensor((batch, TOP_K), T.int32),
    ):
        with T.Kernel(batch) as bx:
            out = T.alloc_shared((TOP_K,), T.int32)
            visible = T.min(T.max(L[length_offset + bx] // 4, 0), pools)
            select(X, bx, visible, out)
            T.copy(out, Y[bx, :])

    return main
