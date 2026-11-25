# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from . import mtp_projection_rows


def abi_program(
    requests,
    tokens,
    pages,
    columns,
    joint,
    tile,
    group,
    stages,
    main_pages,
    main_cols,
    ring_pages,
    ring_table_stride,
    index_width=2176,
    index_cols=None,
    native_output=False,
    native_head_weights=False,
):
    index_cols = columns if index_cols is None else index_cols
    uv_shape = (64, 512, 256) if native_head_weights else (64, 256, 512)
    uk_shape = (64, 256, 512) if native_head_weights else (64, 512, 256)
    output_weight_shape = (16384, 4096) if native_output else (4096, 16384)
    nr = mtp_projection_rows(tokens)
    output_rows = nr
    pools = columns * 32
    padded = (tokens + joint - 1) * 32

    @T.prim_func
    def main(
        Q: T.Tensor((padded, 128), T.float8_e4m3fn),
        QS: T.Tensor((padded,), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.StridedTensor((pages, 32), (1088, 1), T.float32),
        W: T.Tensor((nr * 32,), T.bfloat16),
        Table: T.Tensor((requests, index_cols), T.int32),
        Length: T.Tensor((requests,), T.int32),
        Ends: T.Tensor((requests,), T.int64),
        Runs: T.Tensor((requests, columns // 8), T.int32),
        Score: T.Tensor((tokens, pools), T.bfloat16),
        Raw: T.Tensor((nr, 4096), T.bfloat16),
        Pos: T.Tensor((tokens,), T.int64),
        Packed: T.Tensor((tokens, index_width), T.int32),
        MQ: T.Tensor((nr, 64, 512), T.bfloat16),
        MK: T.Tensor((main_pages, 128, 132), T.int32),
        MT: T.Tensor((requests, main_cols), T.int32),
        MS: T.Tensor((requests + 1,), T.int32),
        MY: T.Tensor((output_rows, 64, 512), T.bfloat16),
        WV: T.Tensor(uv_shape, T.bfloat16),
        V: T.Tensor((output_rows, 16384), T.bfloat16),
        OC: T.Tensor((output_rows, 16384), T.float8_e4m3fn),
        XS: T.Tensor((256, output_rows), T.int16),
        WO: T.Tensor(output_weight_shape, T.float8_e4m3fn),
        OS: T.Tensor((256, 4096), T.int16),
        Final: T.Tensor((tokens, 4096), T.bfloat16),
        X: T.Tensor((tokens, 4096), T.bfloat16),
        AC: T.Tensor((nr, 4096), T.float8_e4m3fn),
        AX: T.Tensor((64, nr), T.int16),
        AW: T.Tensor((2048, 4096), T.float8_e4m3fn),
        AWX: T.Tensor((64, 2048), T.int16),
        QKV: T.Tensor((nr, 2048), T.bfloat16),
        GQ: T.Tensor((1536,), T.bfloat16),
        GK: T.Tensor((512,), T.bfloat16),
        Slot: T.Tensor((tokens, 1), T.int32),
        Norm: T.Tensor((nr, 1536), T.bfloat16),
        BC: T.Tensor((nr, 1536), T.float8_e4m3fn),
        BX: T.Tensor((24, nr), T.int16),
        BW: T.Tensor((16384, 1536), T.float8_e4m3fn),
        BWX: T.Tensor((24, 16384), T.int16),
        QB: T.Tensor((nr, 16384), T.bfloat16),
        UK: T.Tensor(uk_shape, T.bfloat16),
        IQuery: T.Tensor((1, 4096, 1536), T.bfloat16),
        IWeight: T.Tensor((32, 4096), T.bfloat16),
        Panel: T.Tensor((nr + 32, 4096), T.float32),
        Parts: T.Tensor((nr // 32, 32, 32, 32), T.float32),
        GateWeight: T.Tensor((1, 256, 4096), T.bfloat16),
        KG: T.Tensor((nr, 256), T.bfloat16),
        Gamma: T.Tensor((128,), T.float32),
        Beta: T.Tensor((128,), T.float32),
        Projected: T.Tensor((tokens, 256), T.float32),
        Ape: T.Tensor((4, 128), T.float32),
        RingPage: T.StridedTensor((requests,), (ring_table_stride,), T.int32),
        Ring: T.StridedTensor((ring_pages, 9, 256), (289 * 256, 256, 1), T.float32),
        IndexSlots: T.Tensor((tokens,), T.int64),
        Writes: T.Tensor((requests, 6), T.int64),
    ):
        with T.MixedKernel(28, sids=2) as (core, sid):
            ql = T.alloc_l1((32, 128), T.float8_e4m3fn)
            kl = T.alloc_l1((32, 128), T.float8_e4m3fn)
            acc = T.alloc_l0c((32, 32), T.float32)
            out = T.alloc_shared((32, 16), T.float32)
            rows = T.alloc_shared((32,), T.int32)
            sval = T.alloc_shared((32,), T.bfloat16)
            first = T.alloc_shared((32,), T.float32)
            second = T.alloc_shared((32,), T.float32)
            head = T.alloc_shared((32,), T.bfloat16)
            inp = T.alloc_shared((32,), T.bfloat16)
            T.copy(Q[0:32, :], ql)
            T.copy(K[0, 0:32, :], kl)
            T.gemm(ql, kl, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, out)
            T.copy(QS[:32], first)
            T.copy(KS[0, :], second)
            T.copy(W[:32], head)
            T.copy(Raw[0, :32], inp)
            # Keep the live cache arguments in the lowered ABI until the fused stage bodies replace this body.
            bias = Table[0, 0] + Length[0] + Runs[0, 0] + T.cast(Ends[0] + Pos[0], T.int32)
            bias += MT[0, 0] + MS[0] + MK[0, 0, 0]
            mqtmp = T.alloc_shared((32,), T.bfloat16)
            T.copy(MQ[0, 0, :32], mqtmp)
            bias += (
                T.cast(WV[0, 0, 0], T.int32)
                + T.cast(V[0, 0], T.int32)
                + T.cast(OC[0, 0], T.int32)
                + T.cast(WO[0, 0], T.int32)
                + T.cast(XS[0, 0], T.int32)
                + T.cast(OS[0, 0], T.int32)
            )
            bias += T.cast(X[0, 0], T.int32)
            bias += T.cast(AC[0, 0], T.int32)
            bias += T.cast(AX[0, 0], T.int32)
            bias += T.cast(AW[0, 0], T.int32)
            bias += T.cast(AWX[0, 0], T.int32)
            bias += T.cast(QKV[0, 0], T.int32)
            bias += T.cast(GQ[0], T.int32)
            bias += T.cast(GK[0], T.int32)
            bias += T.cast(Slot[0, 0], T.int32)
            bias += T.cast(Norm[0, 0], T.int32)
            bias += T.cast(BC[0, 0], T.int32)
            bias += T.cast(BX[0, 0], T.int32)
            bias += T.cast(BW[0, 0], T.int32)
            bias += T.cast(BWX[0, 0], T.int32)
            bias += T.cast(QB[0, 0], T.int32)
            bias += T.cast(UK[0, 0, 0], T.int32)
            bias += T.cast(IQuery[0, 0, 0], T.int32)
            bias += T.cast(IWeight[0, 0], T.int32)
            bias += T.cast(Panel[0, 0], T.int32)
            bias += T.cast(Parts[0, 0, 0, 0], T.int32)
            bias += T.cast(GateWeight[0, 0, 0], T.int32)
            bias += T.cast(KG[0, 0], T.int32)
            bias += T.cast(Gamma[0], T.int32)
            bias += T.cast(Beta[0], T.int32)
            bias += T.cast(Projected[0, 0], T.int32)
            bias += T.cast(Ape[0, 0], T.int32)
            bias += T.cast(RingPage[0], T.int32)
            bias += T.cast(Ring[0, 0, 0], T.int32) + T.cast(IndexSlots[0] + Writes[0, 0], T.int32)
            with T.SimtVF(threads=32):
                for lane in T.Parallel(32):
                    sval[lane] = T.cast(
                        out[lane, 0]
                        + first[lane]
                        + second[lane]
                        + T.cast(head[lane], T.float32)
                        + T.cast(inp[lane], T.float32)
                        + T.cast(mqtmp[lane], T.float32),
                        T.bfloat16,
                    )
                    rows[lane] = T.cast(sval[lane], T.int32) + bias
            T.copy(sval, Final[0, :32])
            T.copy(sval, MY[0, 0, :32])
            T.copy(rows, Packed[0, 0:32])
            T.copy(sval, Score[0, 0:32])

    return main
