# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend.language import simd as S


def make_value_pipeline(half_rows, dim_tile, *, interleave_loads=False, finalize=False):
    rows = 2 * half_rows

    @T.macro
    def load_value(offset, k_l1, bank, depth):
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_load_cbuf_to_cb"),
                T.access_ptr(bank[0, 0], "w", depth, dim_tile),
                T.access_ptr(k_l1[0, 0], "r", 128, 512),
                0,
                offset // 16,
                depth // 16,
                dim_tile // 16,
                8,
                dim_tile // 16,
                1,
            )
        )

    @T.macro
    def multiply_value(pv_a, bank, acc, depth):
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_mad"),
                T.access_ptr(acc[0, 0], "w", rows, dim_tile),
                T.access_ptr(pv_a[0, 0], "r", rows, depth),
                T.access_ptr(bank[0, 0], "r", depth, dim_tile),
                rows,
                depth,
                dim_tile,
                0,
                1,
                0,
                True,
            )
        )

    @T.macro
    def project_value(offset, k_l1, pv_a, pv_b, acc, residual_pool):
        if residual_pool:
            load_value(offset, k_l1, pv_b, 16)
            multiply_value(pv_a, pv_b, acc, 16)
        else:
            T.copy(k_l1[:, offset : offset + dim_tile], pv_b)
            T.gemm(pv_a, pv_b, acc, clear_accum=True)

    @T.macro
    def accumulate_value(offset, partial, running, alpha, row_sum, out):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for r in range(half_rows):
                adjust = S.vld(alpha[r], "BRC_B32")
                if finalize:
                    mass = S.vmaxs(S.vld(row_sum[r], "BRC_B32"), T.float32(1e-30), full)
                for c in range(0, dim_tile, 64):
                    col = offset + c
                    old = S.vld(running[r, col], "NORM")
                    cur = S.vld(partial[r, c], "NORM")
                    value = S.vadd(S.vmul(old, adjust, full), cur, full)
                    if finalize:
                        scaled = S.vdiv(value, mass, full, precision="ftz_true")
                        result = S.vcvt(scaled, "bfloat16", full, round="ROUND_A", part=0)
                        S.vsts(out[r, col], result, full, "PK_B32")
                    else:
                        S.vsts(running[r, col], value, full, "NORM_B32")

    @T.macro
    def apply(
        prob,
        k_l1,
        p_l1,
        pv_a,
        pv_b,
        out_l0,
        partial,
        running,
        alpha,
        other_l0,
        other_ub,
        residual_pool=False,
        spare_b=None,
        row_sum=None,
        out=None,
    ):
        T.dual_copy(prob[:half_rows, :], p_l1)
        if residual_pool:
            # The final one to three tokens fit in the first 16-wide K block of L0A.
            T.evaluate(
                T.call_intrin(
                    "void",
                    tvm.ir.Op.get("tl.ascend_load_cbuf_to_ca"),
                    T.access_ptr(pv_a[0, 0], "w", rows, 16),
                    T.access_ptr(p_l1[0, 0], "r", rows, 128),
                    0,
                    0,
                    rows // 16,
                    1,
                    rows // 16,
                    rows // 16,
                    0,
                )
            )
        else:
            T.copy(p_l1, pv_a)
        if interleave_loads and not residual_pool:
            load_value(128, k_l1, spare_b, 128)
        project_value(0, k_l1, pv_a, pv_b, out_l0, residual_pool)
        T.dual_copy(out_l0, partial)
        if interleave_loads and not residual_pool:
            multiply_value(pv_a, spare_b, other_l0, 128)
        else:
            project_value(128, k_l1, pv_a, pv_b, other_l0, residual_pool)
        accumulate_value(0, partial, running, alpha, row_sum, out)
        T.dual_copy(other_l0, other_ub)
        if interleave_loads and not residual_pool:
            load_value(384, k_l1, spare_b, 128)
        project_value(256, k_l1, pv_a, pv_b, out_l0, residual_pool)
        accumulate_value(128, other_ub, running, alpha, row_sum, out)
        T.dual_copy(out_l0, partial)
        if interleave_loads and not residual_pool:
            multiply_value(pv_a, spare_b, other_l0, 128)
        else:
            project_value(384, k_l1, pv_a, pv_b, other_l0, residual_pool)
        accumulate_value(256, partial, running, alpha, row_sum, out)
        T.dual_copy(other_l0, other_ub)
        accumulate_value(384, other_ub, running, alpha, row_sum, out)

    return apply
