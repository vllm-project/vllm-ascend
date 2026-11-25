# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend.language import simd as S


def attention_mma(half_rows):
    query_rows = 2 * half_rows
    tile = dim_tile = 128

    @T.macro
    def load_bank(k_l1, offset, other_k, transpose):
        # QK loads keys into the non-transposed L0B layout, including the bank shared with PV.
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_load_cbuf_to_cb"),
                T.access_ptr(other_k[0, 0], "w", tile, dim_tile),
                T.access_ptr(k_l1[0, 0], "r", tile, k_l1.shape[1]),
                0,
                offset // 16,
                tile // 16,
                dim_tile // 16,
                tile // 16,
                tile // 16,
                transpose,
            )
        )

    @T.macro
    def multiply(query, other_k, scores_l0, clear):
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_mad"),
                T.access_ptr(scores_l0[0, 0], "w", query_rows, tile),
                T.access_ptr(query[0, 0], "r", query_rows, dim_tile),
                T.access_ptr(other_k[0, 0], "r", tile, dim_tile),
                query_rows,
                dim_tile,
                tile,
                0,
                1,
                0,
                clear,
            )
        )

    @T.macro
    def dot(q_l1, k_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, other_k, scores_l0):
        T.copy(k_l1[:, 0:128], k_l0)
        load_bank(k_l1, 128, other_k, 0)
        T.gemm(q_pin0, k_l0, scores_l0, transpose_B=True, clear_accum=True)
        T.copy(k_l1[:, 256:384], k_l0)
        multiply(q_pin1, other_k, scores_l0, False)
        T.copy(q_l1[:, 384:512], q_l0)
        load_bank(k_l1, 384, other_k, 0)
        T.gemm(q_pin2, k_l0, scores_l0, transpose_B=True, clear_accum=False)
        multiply(q_l0, other_k, scores_l0, False)

    @T.macro
    def accumulate_value(offset, partial, running, alpha):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for r in range(half_rows):
                adjust = S.vld(alpha[r], "BRC_B32")
                for c in range(0, dim_tile, 64):
                    col = offset + c
                    old = S.vld(running[r, col], "NORM")
                    cur = S.vld(partial[r, c], "NORM")
                    value = S.vadd(S.vmul(old, adjust, full), cur, full)
                    S.vsts(running[r, col], value, full, "NORM_B32")

    @T.macro
    def apply(
        prob,
        k_l1,
        alpha,
        value_tile,
    ):
        p_l1, pv_a, pv_b, out_l0, partial, running, other_l0, other_ub, spare_b = value_tile
        T.dual_copy(prob[:half_rows, :], p_l1)
        T.copy(p_l1, pv_a)
        load_bank(k_l1, 128, spare_b, 1)
        T.copy(k_l1[:, :128], pv_b)
        T.gemm(pv_a, pv_b, out_l0, clear_accum=True)
        T.dual_copy(out_l0, partial)
        multiply(pv_a, spare_b, other_l0, True)
        accumulate_value(0, partial, running, alpha)
        T.dual_copy(other_l0, other_ub)
        load_bank(k_l1, 384, spare_b, 1)
        T.copy(k_l1[:, 256:384], pv_b)
        T.gemm(pv_a, pv_b, out_l0, clear_accum=True)
        accumulate_value(128, other_ub, running, alpha)
        T.dual_copy(out_l0, partial)
        multiply(pv_a, spare_b, other_l0, True)
        accumulate_value(256, partial, running, alpha)
        T.dual_copy(other_l0, other_ub)
        accumulate_value(384, other_ub, running, alpha)

    return dot, apply
