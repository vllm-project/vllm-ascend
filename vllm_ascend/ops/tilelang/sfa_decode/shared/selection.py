# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]
from tilelang.ascend.language import simd as S  # type: ignore[import-not-found]

from .compiler import phase, vector
from .pool_rank import make_pool_ranker

CHUNK_POOLS = 32768
TOP_K = 512


@T.macro
def publish_indices(top, length, Y, bx, width):
    visible = length // 4
    token_ids = T.alloc_shared((2176,), T.int32)
    history = T.min(visible, 512) * 4
    with T.SimtVF(threads=128):
        for fragment in T.serial(2176 // 128):
            for lane in T.Parallel(128):
                col = fragment * 128 + lane
                full_id = T.cast(top[T.min(col // 4, 511)], T.int32) * 4 + col % 4
                suffix = col - history
                token_ids[col] = T.if_then_else(
                    col < history, full_id, T.if_then_else(suffix < length % 4, visible * 4 + suffix, -1)
                )
    T.copy(token_ids[:width], Y[bx, :])


def pack_dense_indices(batch, requests, width):
    padded_width = (width + 63) // 64 * 64

    @T.prim_func
    def dense(
        Pos: T.Tensor((batch,), T.int64),
        Ends: T.Tensor((requests,), T.int64),
        Y: T.Tensor((batch, width), T.int32),
    ):
        with T.Kernel(batch) as bx:
            values = T.alloc_shared((padded_width,), T.int32)
            length = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32), 0)
            with T.SimdVF():
                mask = S.pset(32)
                lanes = S.vci(T.int32(0), "int32")
                invalid = S.vdup(T.int32(-1), "int32", mask)
                for tile in range(padded_width // 64):
                    cols = S.vadds(lanes, tile * 64, mask)
                    valid = S.vcmps(cols, length, mask, op="lt")
                    S.vsts(values[tile * 64], S.vsel(cols, invalid, valid), mask, "NORM_B32")
            T.copy(values[:width], Y[bx, :])

    return dense


def _selection_stages(batch, requests, context, width, *, mtp=True, length_offset=0):
    pools = context // 4
    # Histogram counts and output indices are uint16 inside one vector tile.
    chunked = pools >= 1 << 16
    chunks = pools // CHUNK_POOLS if chunked else 1
    assert not chunked or pools % CHUNK_POOLS == 0
    candidates = chunks * TOP_K
    capacity = CHUNK_POOLS if chunked else pools
    rank = make_pool_ranker(capacity, TOP_K, True, preloaded=True)
    metadata_type = T.int64 if mtp else T.int32

    @T.macro
    def length(row, Pos, Ends):
        if mtp:
            value = T.if_then_else(row < Ends[requests - 1], T.cast(Pos[row] + 1, T.int32), 0)
        else:
            value = Pos[length_offset + row]
        return T.min(T.max(value, 0), context)

    @T.macro
    def publish(top, n, Y, row):
        if width == TOP_K:
            ids = T.alloc_shared((TOP_K,), T.int32)
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        i = half * 256 + lane
                        ids[i] = T.cast(top[i], T.int32)
            T.copy(ids, Y[row, :])
        else:
            publish_indices(top, n, Y, row, width)

    @T.macro
    def select(X, row, begin, count, raw, words, top):
        hist = T.alloc_shared((256,), T.uint16)
        bounds = T.alloc_shared((8,), T.int32)
        if count >= TOP_K:
            span = T.ceildiv(count, 256) * 256
            T.copy(X[row, begin : begin + span], raw[:span])
        rank(X, row, count, raw, words, hist, bounds, top)

    if not chunked:

        @T.prim_func
        def direct(
            X: T.Tensor((batch, pools), T.bfloat16),
            Pos: T.Tensor((length_offset + batch,), metadata_type),
            Ends: T.Tensor((requests,), T.int64),
            Y: T.Tensor((batch, width), T.int32),
        ):
            with T.Kernel(batch) as row:
                raw = T.alloc_shared((capacity,), T.bfloat16)
                words = T.view(raw, (capacity,), dtype=T.uint16)
                top = T.alloc_shared((TOP_K,), T.uint16)
                n = length(row, Pos, Ends)
                select(X, row, 0, n // 4, raw, words, top)
                publish(top, n, Y, row)

        return None, direct

    @T.prim_func
    def partial(
        X: T.Tensor((batch, pools), T.bfloat16),
        Pos: T.Tensor((length_offset + batch,), metadata_type),
        Ends: T.Tensor((requests,), T.int64),
        Scores: T.Tensor((batch, candidates), T.bfloat16),
        LocalIds: T.Tensor((batch, candidates), T.uint16),
    ):
        with T.Kernel(batch * chunks) as task:
            row = task // chunks
            chunk = task % chunks
            begin = chunk * capacity
            visible = length(row, Pos, Ends) // 4
            count = T.min(visible - begin, capacity)
            if count > 0 and visible >= TOP_K:
                raw = T.alloc_shared((capacity,), T.bfloat16)
                words = T.view(raw, (capacity,), dtype=T.uint16)
                ids = T.alloc_shared((TOP_K,), T.uint16)
                scores = T.alloc_shared((TOP_K,), T.bfloat16)
                bits = T.view(scores, (TOP_K,), dtype=T.uint16)
                select(X, row, begin, count, raw, words, ids)
                if count >= TOP_K:
                    with T.SimtVF(threads=256):
                        for half in T.serial(2):
                            for lane in T.Parallel(256):
                                i = half * 256 + lane
                                key = words[T.cast(ids[i], T.int32)]
                                bits[i] = T.bitwise_xor(
                                    key, T.if_then_else(key >= T.uint16(32768), T.uint16(32768), T.uint16(65535))
                                )
                else:
                    T.copy(X[row, begin : begin + TOP_K], scores)
                start = chunk * TOP_K
                T.copy(scores, Scores[row, start : start + TOP_K])
                T.copy(ids, LocalIds[row, start : start + TOP_K])

    merge_rank = make_pool_ranker(candidates, TOP_K, True, preloaded=True)

    @T.prim_func
    def merge(
        Scores: T.Tensor((batch, candidates), T.bfloat16),
        LocalIds: T.Tensor((batch, candidates), T.uint16),
        Pos: T.Tensor((length_offset + batch,), metadata_type),
        Ends: T.Tensor((requests,), T.int64),
        Y: T.Tensor((batch, width), T.int32),
    ):
        with T.Kernel(batch) as row:
            n = length(row, Pos, Ends)
            visible = n // 4
            out = T.alloc_shared((TOP_K,), T.int32)
            ids = T.alloc_shared((TOP_K,), T.uint16)
            scores = T.alloc_shared((candidates,), T.bfloat16)
            words = T.view(scores, (candidates,), dtype=T.uint16)
            by_chunk = T.alloc_shared((candidates,), T.uint16)
            hist = T.alloc_shared((256,), T.uint16)
            bounds = T.alloc_shared((8,), T.int32)
            live = visible // capacity * TOP_K + T.min(visible % capacity, TOP_K)
            if visible >= TOP_K:
                span = T.ceildiv(visible, capacity) * TOP_K
                T.copy(Scores[row, :span], scores[:span])
                T.copy(LocalIds[row, :span], by_chunk[:span])
            merge_rank(Scores, row, live, scores, words, hist, bounds, ids)
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        i = half * 256 + lane
                        if visible >= TOP_K:
                            j = T.cast(ids[i], T.int32)
                            out[i] = j // TOP_K * capacity + T.cast(by_chunk[j], T.int32)
                        else:
                            out[i] = i
            publish(out, n, Y, row)

    return partial, merge


def selection_phases(batch, requests, context, width, *, inputs, output, scratch, guard=None, concurrent=(), **mode):
    partial, select = _selection_stages(batch, requests, context, width, **mode)
    if partial is None:
        return [phase(*concurrent, vector(select, X="Score", Y=output, guard=guard, **inputs))]
    return [
        phase(*concurrent, vector(partial, X="Score", guard=guard, **inputs, **scratch), barrier=guard),
        phase(vector(select, Y=output, guard=guard, **inputs, **scratch)),
    ]
