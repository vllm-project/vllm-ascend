# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def score_program(batch, tokens, pages, columns, joint=6, tile=128, group=4, stages=2, *, index_cols=None):
    index_cols = columns if index_cols is None else index_cols
    pools = ((columns * 32 + group * tile - 1) // (group * tile)) * group * tile
    splits = pools // (tile * group)
    heads = 32 * joint
    padded = tokens + joint - 1
    cores = min(28, batch * splits)
    # A stride coprime to 28 distributes short requests without a loop-carried core offset.
    core_stride = 11

    @T.macro
    def score_tile(
        block, bx, sid, qrow, end, valid, edge, Table, K, KS, Score, ql, ku, dot, prod, coef, scales, scores
    ):
        for j in T.serial(tile // 32):
            col = block * (tile // 32) + j
            page = T.cast(Table[bx, T.min(col, (valid - 1) // 32) if edge else col], T.int64)
            T.copy(K[page, :32, :], ku[j * 32 : (j + 1) * 32, :])
        for j in T.serial(tile // 64):
            col = block * (tile // 32) + sid * (tile // 64) + j
            page = T.cast(Table[bx, T.min(col, (valid - 1) // 32) if edge else col], T.int64)
            T.copy(KS[page, :], scales[j * 32 : (j + 1) * 32])
        T.gemm(ql, ku, dot, transpose_B=True, clear_accum=True)
        T.dual_copy(dot, prod)
        with T.SimdVF():
            mask = S.pset(32, "PAT_ALL")
            for cohort in range(joint // 3):
                for chunk in range(tile // 128):
                    sc = S.vld(scales[chunk * 64], "NORM")
                    t0 = S.alloc_var("float32")
                    t0 = S.vdup(T.float32(0.0), "float32", mask)
                    t1 = S.alloc_var("float32")
                    t1 = S.vdup(T.float32(0.0), "float32", mask)
                    t2 = S.alloc_var("float32")
                    t2 = S.vdup(T.float32(0.0), "float32", mask)
                    for h in range(32):
                        v0 = S.vmaxs(
                            S.vld(prod[(cohort * 3 + 0) * 32 + h, chunk * 64], "NORM"),
                            T.float32(0.0),
                            mask,
                        )
                        c0 = S.vld(coef[(cohort * 3 + 0) * 32 + h], "BRC_B32")
                        S.vmula(t0, v0, c0, mask)
                        v1 = S.vmaxs(
                            S.vld(prod[(cohort * 3 + 1) * 32 + h, chunk * 64], "NORM"),
                            T.float32(0.0),
                            mask,
                        )
                        c1 = S.vld(coef[(cohort * 3 + 1) * 32 + h], "BRC_B32")
                        S.vmula(t1, v1, c1, mask)
                        v2 = S.vmaxs(
                            S.vld(prod[(cohort * 3 + 2) * 32 + h, chunk * 64], "NORM"),
                            T.float32(0.0),
                            mask,
                        )
                        c2 = S.vld(coef[(cohort * 3 + 2) * 32 + h], "BRC_B32")
                        S.vmula(t2, v2, c2, mask)
                    u0 = S.vmuls(S.vmul(t0, sc, mask), T.float32(128.0**-0.5), mask)
                    S.vsts(
                        scores[cohort * 3 + 0, chunk * 64],
                        S.vcvt(u0, "bfloat16", mask, round="ROUND_A", part=0),
                        mask,
                        "PK_B32",
                    )
                    u1 = S.vmuls(S.vmul(t1, sc, mask), T.float32(128.0**-0.5), mask)
                    S.vsts(
                        scores[cohort * 3 + 1, chunk * 64],
                        S.vcvt(u1, "bfloat16", mask, round="ROUND_A", part=0),
                        mask,
                        "PK_B32",
                    )
                    u2 = S.vmuls(S.vmul(t2, sc, mask), T.float32(128.0**-0.5), mask)
                    S.vsts(
                        scores[cohort * 3 + 2, chunk * 64],
                        S.vcvt(u2, "bfloat16", mask, round="ROUND_A", part=0),
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
        KS: T.StridedTensor((pages, 32), (1088, 1), T.float32),
        W: T.Tensor((padded * 32,), T.bfloat16),
        Table: T.Tensor((batch, index_cols), T.int32),
        Length: T.Tensor((batch,), T.int32),
        Ends: T.Tensor((batch,), T.int64),
        Runs: T.Tensor((batch, columns // (tile // 32)), T.int32),
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
            for bx in T.serial(batch):
                valid = Length[bx] // 4
                begin = T.if_then_else(bx == 0, 0, T.cast(Ends[T.max(0, bx - 1)], T.int32))
                end = T.cast(Ends[bx], T.int32)
                active_splits = T.ceildiv(valid, tile * group)
                # Only full pools contribute scores. Shorter prefixes use dense token indices.
                if valid >= 512 and end > begin:
                    first = (core + cores - (bx * core_stride % cores)) % cores
                    if first < active_splits:
                        qrow = begin
                        T.copy(Q[qrow * 32 : (qrow + joint) * 32, :], ql)
                        T.copy(QS[qrow * 32 : (qrow + joint) * 32], qs)
                        T.copy(W[qrow * 32 : (qrow + joint) * 32], w)
                        with T.SimtVF(threads=32):
                            for h in T.Parallel(heads):
                                coef[h] = qs[h] * T.cast(w[h], T.float32)
                        for span_i in T.serial(T.ceildiv(active_splits - first, cores)):
                            start = (first + span_i * cores) * group
                            if (start + group) * tile <= valid:
                                for subtile in T.Pipelined(group, num_stages=stages):
                                    score_tile(
                                        start + subtile,
                                        bx,
                                        sid,
                                        qrow,
                                        end,
                                        valid,
                                        False,
                                        Table,
                                        K,
                                        KS,
                                        Score,
                                        ql,
                                        ku,
                                        dot,
                                        prod,
                                        coef,
                                        scales,
                                        scores,
                                    )
                            else:
                                for subtile in T.serial(valid // tile - start):
                                    score_tile(
                                        start + subtile,
                                        bx,
                                        sid,
                                        qrow,
                                        end,
                                        valid,
                                        False,
                                        Table,
                                        K,
                                        KS,
                                        Score,
                                        ql,
                                        ku,
                                        dot,
                                        prod,
                                        coef,
                                        scales,
                                        scores,
                                    )
                                if valid % tile != 0:
                                    score_tile(
                                        valid // tile,
                                        bx,
                                        sid,
                                        qrow,
                                        end,
                                        valid,
                                        True,
                                        Table,
                                        K,
                                        KS,
                                        Score,
                                        ql,
                                        ku,
                                        dot,
                                        prod,
                                        coef,
                                        scales,
                                        scores,
                                    )

    return main
