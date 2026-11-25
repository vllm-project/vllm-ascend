# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from . import OUTPUT_PANEL_CHANNELS, head_panel_layout, projection_rows, query_workspace_rows


def full_decode_program(
    batch,
    pages,
    page_size,
    context,
    include_tail,
    *,
    capacity=4,
    index_width,
    prepacked_heads,
    index_pages,
    table_columns,
    tail_layout,
    mask_padding,
    resident_workspace=False,
    packed_output=False,
):
    """Lower the host calling convention. The compiler replaces this device body with the fused phases."""
    dtype = T.bfloat16
    main_columns, index_columns = table_columns
    rows = projection_rows(batch, prepacked_heads)
    query_rows = query_workspace_rows(batch, rows, resident_workspace)
    tail_count, tail_stride, tail_id_stride = tail_layout
    scale_rows = 66 if mask_padding else 64
    pools = context // 4
    parts, _, aligned = head_panel_layout(batch)
    weight_shape = (parts * 32, aligned) if prepacked_heads else (32, 4096)
    weight_dtype = T.float32 if prepacked_heads else dtype
    panels = 16384 // OUTPUT_PANEL_CHANNELS
    output_shape = (panels, 4096, OUTPUT_PANEL_CHANNELS) if packed_output else (4096, 16384)
    scale_shape = (panels, 4096, OUTPUT_PANEL_CHANNELS // 64) if packed_output else (256, 4096)

    @T.prim_func
    def main(
        X: T.Tensor((batch, 4096), dtype),
        A: T.Tensor((2048, 4096), T.float8_e4m3fn),
        AS: T.Tensor((64, 2048), T.int16),
        B: T.Tensor((16384, 1536), T.float8_e4m3fn),
        BS: T.Tensor((24, 16384), T.int16),
        WK: T.Tensor((64, 512, 256), dtype),
        WV: T.Tensor((64, 256, 512), dtype),
        WO: T.Tensor(output_shape, T.float8_e4m3fn),
        OS: T.Tensor(scale_shape, T.int16),
        GQ: T.Tensor((1536,), dtype),
        GK: T.Tensor((512,), dtype),
        KGate: T.Tensor((256, 4096), dtype),
        IG: T.Tensor((128,), T.float32),
        IB: T.Tensor((128,), T.float32),
        Ape: T.Tensor((4, 128), T.float32),
        IQuery: T.Tensor((1, 4096, 1536), dtype),
        IWeight: T.Tensor(weight_shape, weight_dtype),
        Cache: T.Tensor((pages, page_size, 132), T.int32),
        Table: T.Tensor((batch, main_columns), T.int32),
        Slot: T.Tensor((batch, 1), T.int32),
        Length: T.Tensor((batch,), T.int32),
        Tail: T.StridedTensor((tail_count, capacity, 256), (tail_stride, 256, 1), T.float32),
        TailPage: T.StridedTensor((batch,), (tail_id_stride,), T.int32),
        IT: T.Tensor((batch, index_columns), T.int32),
        IK: T.Tensor((index_pages, 34, 128), T.float8_e4m3fn),
        IKS: T.Tensor((index_pages, 1088), T.float32),
        C1: T.Tensor((rows, 4096), T.float8_e4m3fn),
        S1: T.Tensor((scale_rows, rows), T.int16),
        C2: T.Tensor((rows, 1536), T.float8_e4m3fn),
        S2: T.Tensor((24, rows), T.int16),
        C3: T.Tensor((rows, 16384), T.float8_e4m3fn),
        S3: T.Tensor((256, rows), T.int16),
        QKV: T.Tensor((rows, 2048), dtype),
        Norm: T.Tensor((rows, 1536), dtype),
        QB: T.Tensor((rows, 16384), dtype),
        QA: T.Tensor((query_rows, 32768), dtype),
        Latent: T.Tensor((rows, 32768), dtype),
        Value: T.Tensor((rows, 16384), dtype),
        IQRaw: T.Tensor((rows, 4096), dtype),
        IQ: T.Tensor((rows, 32, 128), T.float8_e4m3fn),
        IQS: T.Tensor((rows, 32), T.float32),
        IW: T.Tensor((rows, 32), dtype),
        Score: T.Tensor((batch, pools), T.bfloat16),
        Selected: T.Tensor((batch, index_width), T.int32),
        Y: T.Tensor((batch, 4096), dtype),
    ):
        CacheRows = T.Tensor((pages * page_size, 132), T.int32, Cache.data)
        Q3 = T.Tensor((rows, 64, 512), dtype, QA.data)
        Latent3 = T.Tensor((rows, 64, 512), dtype, Latent.data)
        with T.MixedKernel(rows // 32, sids=2) as (core, sid):
            left = T.alloc_l1((32, 128), T.float8_e4m3fn)
            right = T.alloc_l1((32, 128), T.float8_e4m3fn)
            acc = T.alloc_l0c((32, 32), T.float32)
            value = T.alloc_shared((32, 16), T.float32)
            result = T.alloc_shared((32,), T.bfloat16)
            T.copy(C1[0:32, 0:128], left)
            T.copy(A[0:32, 0:128], right)
            T.gemm(left, right, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, value)
            address = T.cast(X[0, 0], T.int32)
            address += T.cast(AS[0, 0], T.int32)
            address += T.cast(B[0, 0], T.int32)
            address += T.cast(BS[0, 0], T.int32)
            address += T.cast(WK[0, 0, 0], T.int32)
            address += T.cast(WV[0, 0, 0], T.int32)
            if packed_output:
                address += T.cast(WO[0, 0, 0], T.int32) + T.cast(OS[0, 0, 0], T.int32)
            else:
                address += T.cast(WO[0, 0], T.int32) + T.cast(OS[0, 0], T.int32)
            address += T.cast(GQ[0], T.int32)
            address += T.cast(GK[0], T.int32)
            address += T.cast(KGate[0, 0], T.int32)
            address += T.cast(IG[0], T.int32)
            address += T.cast(IB[0], T.int32)
            address += T.cast(Ape[0, 0], T.int32)
            address += T.cast(IQuery[0, 0, 0], T.int32)
            address += T.cast(IWeight[0, 0], T.int32)
            address += T.cast(CacheRows[0, 0], T.int32)
            address += T.cast(Table[0, 0], T.int32)
            address += T.cast(Slot[0, 0], T.int32)
            address += T.cast(Length[0], T.int32)
            address += T.cast(Tail[0, 0, 0], T.int32)
            address += T.cast(TailPage[0], T.int32)
            address += T.cast(IT[0, 0], T.int32)
            address += T.cast(IK[0, 0, 0], T.int32)
            address += T.cast(IKS[0, 0], T.int32)
            address += T.cast(S1[0, 0], T.int32)
            address += T.cast(C2[0, 0], T.int32)
            address += T.cast(S2[0, 0], T.int32)
            address += T.cast(C3[0, 0], T.int32)
            address += T.cast(S3[0, 0], T.int32)
            address += T.cast(QKV[0, 0], T.int32)
            address += T.cast(Norm[0, 0], T.int32)
            address += T.cast(QB[0, 0], T.int32)
            address += T.cast(Q3[0, 0, 0], T.int32)
            address += T.cast(Latent3[0, 0, 0], T.int32)
            address += T.cast(Value[0, 0], T.int32)
            # Reference each scratch argument so lowering retains the buffers used by the fused stages.
            address += T.cast(IQRaw[0, 0], T.int32)
            address += T.cast(IQ[0, 0, 0], T.int32)
            address += T.cast(IQS[0, 0], T.int32)
            address += T.cast(IW[0, 0], T.int32)
            address += T.cast(Score[0, 0], T.int32)
            address += T.cast(Selected[0, 0], T.int32)
            with T.SimtVF(threads=32):
                for lane in T.Parallel(32):
                    result[lane] = T.cast(value[lane, 0] + T.cast(address, T.float32), T.bfloat16)
            T.copy(result, Y[0, :32])

    return main
