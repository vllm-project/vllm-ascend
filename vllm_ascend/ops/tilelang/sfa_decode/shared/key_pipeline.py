# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang import tvm


def make_key_pipeline(query_rows, tile, dim_tile, *, interleave_loads=True):
    @T.macro
    def load_next_key(k_l1, offset, other_k, columns):
        # QK loads keys into the non-transposed L0B layout, including the bank shared with PV.
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_load_cbuf_to_cb"),
                T.access_ptr(other_k[0, 0], "w", columns, dim_tile),
                T.access_ptr(k_l1[0, 0], "r", tile, k_l1.shape[1]),
                0,
                offset // 16,
                columns // 16,
                dim_tile // 16,
                tile // 16,
                columns // 16,
                0,
            )
        )

    @T.macro
    def accumulate_key(query, other_k, scores_l0, columns, clear):
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_mad"),
                T.access_ptr(scores_l0[0, 0], "w", query_rows, columns),
                T.access_ptr(query[0, 0], "r", query_rows, dim_tile),
                T.access_ptr(other_k[0, 0], "r", columns, dim_tile),
                query_rows,
                dim_tile,
                columns,
                0,
                1,
                0,
                clear,
            )
        )

    @T.macro
    def dot(q_l1, k_l1, q_pin0, q_pin1, q_pin2, q_l0, k_l0, other_k, scores_l0):
        if interleave_loads:
            T.copy(k_l1[:, 0:128], k_l0)
            load_next_key(k_l1, 128, other_k, tile)
            T.gemm(q_pin0, k_l0, scores_l0, transpose_B=True, clear_accum=True)
            T.copy(k_l1[:, 256:384], k_l0)
            accumulate_key(q_pin1, other_k, scores_l0, tile, False)
            T.copy(q_l1[:, 384:512], q_l0)
            load_next_key(k_l1, 384, other_k, tile)
            T.gemm(q_pin2, k_l0, scores_l0, transpose_B=True, clear_accum=False)
            accumulate_key(q_l0, other_k, scores_l0, tile, False)
        else:
            T.copy(k_l1[:, 0:128], k_l0)
            T.gemm(q_pin0, k_l0, scores_l0, transpose_B=True, clear_accum=True)
            T.copy(k_l1[:, 128:256], k_l0)
            T.gemm(q_pin1, k_l0, scores_l0, transpose_B=True, clear_accum=False)
            T.copy(k_l1[:, 256:384], k_l0)
            T.gemm(q_pin2, k_l0, scores_l0, transpose_B=True, clear_accum=False)
            T.copy(q_l1[:, 384:512], q_l0)
            T.copy(k_l1[:, 384:512], k_l0)
            T.gemm(q_l0, k_l0, scores_l0, transpose_B=True, clear_accum=False)

    @T.macro
    def residual_dot(offset, query, keys, bank, scores, clear):
        load_next_key(keys, offset, bank, 16)
        accumulate_key(query, bank, scores, 16, clear)

    return dot, residual_dot
