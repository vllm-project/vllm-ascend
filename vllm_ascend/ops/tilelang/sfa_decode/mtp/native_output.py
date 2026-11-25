# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx

MXFP8_NZ_C0 = 32


def make_native_weight_loader(panel):
    @T.macro
    def load(weight, tile, outer, col, columns):
        # The compiler schedules one MTE2 write to the whole L1 tile. Each
        # physical [K, C0] plane is a matrix in the strided hardware copy.
        T.evaluate(
            T.call_intrin(
                "void",
                tvm.ir.Op.get("tl.ascend_copy_gm_to_cbuf"),
                T.access_ptr(tile[0, 0], "w", panel, columns),
                T.access_ptr(weight[col // MXFP8_NZ_C0, outer * panel, 0], "r", panel, MXFP8_NZ_C0),
                0,
                MXFP8_NZ_C0,
                0,
                panel,
                columns,
                weight.shape[1] * MXFP8_NZ_C0,
                0,
                0,
                panel,
                "",
            )
        )

    return load


def lower_native_weight_copy(stage):
    def lower(statement):
        copy = statement.value
        if not isinstance(copy, tirx.Call) or getattr(copy.op, "name", None) != "tl.ascend_copy_gm_to_cbuf":
            return None
        dst, src, _, pitch, cache, rows, columns, matrix_stride, _, _, height, _ = copy.args
        if not isinstance(matrix_stride, tirx.IntImm) or int(matrix_stride) == 0:
            return None
        # A contiguous C0 matrix occupies 'rows' 32-byte blocks in NZ L1.
        matrices = int(columns) // MXFP8_NZ_C0
        assert int(columns) % MXFP8_NZ_C0 == 0 and int(pitch) == MXFP8_NZ_C0
        return tirx.SeqStmt(
            [
                tirx.Evaluate(tirx.call_extern("void", "asc_set_gm2l1_nz_para", matrices, 1, height, rows)),
                tirx.Evaluate(
                    tirx.call_extern(
                        "void",
                        "asc_copy_gm2l1_nd2nz",
                        dst,
                        src,
                        pitch,
                        tirx.call_extern("uint8", "static_cast<asc_load_l2_cache_mode>", cache),
                        rows,
                        MXFP8_NZ_C0,
                        matrix_stride,
                        False,
                    )
                ),
                tirx.Evaluate(tirx.call_extern("void", "asc_set_gm2l1_nz_para", 1, 1, height, 0)),
            ]
        )

    return stage.with_body(tirx.stmt_functor.ir_transform(stage.body, None, lower, ["tirx.Evaluate"]))
