# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def score_program(
    batch, tokens, pages, columns, joint=6, tile=128, group=4, stages=2, *, index_cols=None, length_offset=0
):
    index_cols = columns if index_cols is None else index_cols
    pools = ((columns * 32 + group * tile - 1) // (group * tile)) * group * tile
    splits = pools // (tile * group)
    heads = 32 * joint
    padded = tokens + joint - 1
    cores = min(28, batch * splits)
    # A stride coprime to 28 distributes short requests without a loop-carried core offset.
    shards = min(splits, cores)
    core_stride = 11
    scale_offset = 1024 if joint == 1 else 0

    chunks = tile // 128

    @T.macro
    def score_tile(block, edge, inputs):
        bx, sid, qrow, end, valid, cache, operands, reduction, Score = inputs
        Table, K, KS = cache
        ql, ku, dot, prod = operands
        coef, scales, scores = reduction
        for j in T.serial(tile // 32):
            col = block * (tile // 32) + j
            page = T.cast(Table[bx, T.min(col, (valid - 1) // 32) if edge else col], T.int64)
            T.copy(K[page, :32, :], ku[j * 32 : (j + 1) * 32, :])
        for j in T.serial(tile // 64):
            col = block * (tile // 32) + sid * (tile // 64) + j
            page = T.cast(Table[bx, T.min(col, (valid - 1) // 32) if edge else col], T.int64)
            T.copy(KS[page, scale_offset : scale_offset + 32], scales[j * 32 : (j + 1) * 32])
        T.gemm(ql, ku, dot, transpose_B=True, clear_accum=True)
        T.dual_copy(dot, prod)
        with T.SimdVF():
            mask = S.pset(32, "PAT_ALL")
            sums = S.alloc_local((joint * chunks,), "float32")
            for i in T.unroll(joint * chunks):
                sums[i] = S.vdup(T.float32(0), "float32", mask)
            for head in range(32):
                for query in T.unroll(joint):
                    row = query * 32 + head
                    weight = S.vld(coef[row], "BRC_B32")
                    for chunk in T.unroll(chunks):
                        value = S.vmaxs(S.vld(prod[row, chunk * 64], "NORM"), T.float32(0), mask)
                        S.vmula(sums[query * chunks + chunk], value, weight, mask)
            for chunk in T.unroll(chunks):
                scale = S.vld(scales[chunk * 64], "NORM")
                for query in T.unroll(joint):
                    value = S.vmuls(S.vmul(sums[query * chunks + chunk], scale, mask), T.float32(128.0**-0.5), mask)
                    S.vsts(
                        scores[query, chunk * 64],
                        S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0),
                        mask,
                        "PK_B32",
                    )
        for oo in T.unroll(joint):
            if qrow + oo < end:
                T.dual_copy(
                    scores[oo : oo + 1, :],
                    Score[qrow + oo, block * tile : (block + 1) * tile],
                )

    @T.prim_func
    def main(
        Q: T.Tensor((padded * 32, 128), T.float8_e4m3fn),
        QS: T.Tensor((padded * 32,), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.StridedTensor((pages, scale_offset + 32), (1088, 1), T.float32),
        W: T.Tensor((padded * 32,), T.bfloat16),
        Table: T.Tensor((batch, index_cols), T.int32),
        Length: T.Tensor((length_offset + batch,), T.int32),
        Ends: T.Tensor((batch,), T.int64),
        Score: T.Tensor((tokens, pools), T.bfloat16),
    ):
        with T.MixedKernel(cores, sids=2) as (core, sid):
            ql = T.alloc_l1((heads, 128), T.float8_e4m3fn)
            ku = T.alloc_l1((tile, 128), T.float8_e4m3fn)
            dot = T.alloc_l0c((heads, tile), T.float32)
            prod = T.alloc_shared((heads, tile // 2), T.float32)
            qs = T.alloc_shared((heads,), T.float32)
            w = T.alloc_shared((heads,), T.bfloat16)
            coef = T.alloc_shared((heads,), T.float32)
            scales = T.alloc_shared((tile // 2,), T.float32)
            scores = T.alloc_shared((joint, tile // 2), T.bfloat16)
            cache = (Table, K, KS)
            operands = (ql, ku, dot, prod)
            reduction = (coef, scales, scores)
            for task in T.Persistent([batch * shards], cores, core):
                bx = task // shards
                lane = task % shards
                valid = T.min(Length[length_offset + bx] // 4, pools)
                begin = bx if joint == 1 else T.if_then_else(bx == 0, 0, T.cast(Ends[T.max(0, bx - 1)], T.int32))
                end = bx + 1 if joint == 1 else T.cast(Ends[bx], T.int32)
                active_splits = T.ceildiv(valid, tile * group)
                if valid >= 512 and end > begin:
                    first = (lane + shards - (bx * core_stride % shards)) % shards
                    if first < active_splits:
                        qrow = begin
                        T.copy(Q[qrow * 32 : (qrow + joint) * 32, :], ql)
                        T.copy(QS[qrow * 32 : (qrow + joint) * 32], qs)
                        T.copy(W[qrow * 32 : (qrow + joint) * 32], w)
                        with T.SimtVF(threads=32):
                            for h in T.Parallel(heads):
                                coef[h] = qs[h] * T.cast(w[h], T.float32)
                        inputs = (bx, sid, qrow, end, valid, cache, operands, reduction, Score)
                        for span_i in T.serial(T.ceildiv(active_splits - first, shards)):
                            start = (first + span_i * shards) * group
                            if (start + group) * tile <= valid:
                                for subtile in T.Pipelined(group, num_stages=stages):
                                    score_tile(start + subtile, False, inputs)
                            else:
                                for subtile in T.serial(valid // tile - start):
                                    score_tile(start + subtile, False, inputs)
                                if valid % tile != 0:
                                    score_tile(valid // tile, True, inputs)

    return main
