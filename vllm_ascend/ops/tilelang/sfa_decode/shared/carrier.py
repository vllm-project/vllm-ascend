# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tvm import tirx


@T.macro
def mixed_signature(left, right, output):
    # Both engines must be present in the lowered host stub.
    with T.MixedKernel(28, sids=2):
        x = T.alloc_l1((32, 128), T.float8_e4m3fn)
        w = T.alloc_l1((32, 128), T.float8_e4m3fn)
        acc = T.alloc_l0c((32, 32), T.float32)
        value = T.alloc_shared((32, 16), T.float32)
        out = T.alloc_shared((32,), T.bfloat16)
        T.copy(left, x)
        T.copy(right, w)
        T.gemm(x, w, acc, transpose_B=True, clear_accum=True)
        T.dual_copy(acc, value)
        marker = T.call_extern("int32", "sfa_keep_parameters")
        with T.SimtVF(threads=32):
            for lane in T.Parallel(32):
                out[lane] = T.cast(value[lane, 0] + T.cast(marker, T.float32), T.bfloat16)
        T.copy(out, output[0, :32])


def keep_parameters(signature):
    """Keep every declared GM argument until compile_fused replaces the stub."""
    buffers = list(signature.buffer_map.values())
    marker = sum(tirx.Cast("int32", tirx.BufferLoad(buf, [0] * len(buf.shape))) for buf in buffers)

    def visit(call):
        if call.op.name == "tirx.call_extern" and call.args[0].value == "sfa_keep_parameters":
            return marker
        return None

    body = tirx.stmt_functor.ir_transform(signature.body, None, visit, ["tirx.Call"])
    return signature.with_body(body)
