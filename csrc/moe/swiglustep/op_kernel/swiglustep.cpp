/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglustep.cpp
 * \brief SwigluStep fused kernel (A2: single x[M,2N] input):
 *        out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
 *        where gate = x[..., :N], up = x[..., N:] (row-interleaved [M,2N]).
 *
 *        Kernel reads x contiguous, splits gate/up per-row in UB (each row's first N
 *        and last N are contiguous segments), so the host side no longer needs
 *        x.chunk(2,-1).contiguous() — those two GM->GM copies are eliminated.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;              // double buffer, hide GM<->UB latency

template <typename DATA_T>                      // DATA_T = bfloat16 / half
class KernelSwiglustep {
public:
    __aicore__ inline KernelSwiglustep() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out,
                                const SwiglustepTilingData* tilingData) {
        // 1) read tiling (row semantics: totalLength=M, tileLength=tileM, N=col width)
        N            = tilingData->N;
        tileM        = tilingData->tileLength;
        formerNum    = tilingData->formerNum;
        formerLength = tilingData->formerLength;     // rows
        tailNum      = tilingData->tailNum;
        tailLength   = tilingData->tailLength;       // rows
        limit        = tilingData->limit;

        // 2) segment handled by this core (former: formerLength rows / tail: tailLength rows)
        blockIdx = GetBlockIdx();
        int64_t usedCoreNum = formerNum + tailNum;
        coreRows = (blockIdx == usedCoreNum - 1) ? tailLength : formerLength;
        coreRowOffset = (blockIdx < formerNum) ? blockIdx * formerLength
                                               : formerNum * formerLength;

        // 3) GM tensors: x [M,2N] row-major, out [M,N] row-major (this core's row range)
        xGm.SetGlobalBuffer((__gm__ DATA_T*)x   + coreRowOffset * 2 * N, coreRows * 2 * N);
        outGm.SetGlobalBuffer((__gm__ DATA_T*)out + coreRowOffset * N,   coreRows * N);

        // 4) UB buffers (per out-element: inQueueX 8B + outQueue 4B + 5 fp32 20B = 32B)
        int64_t tileEle = tileM * N;             // out elements per tile
        pipe.InitBuffer(inQueueX,    BUFFER_NUM, tileM * 2 * N * sizeof(DATA_T));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileEle * sizeof(DATA_T));
        pipe.InitBuffer(gateFp32Buf, tileEle * sizeof(float));
        pipe.InitBuffer(upFp32Buf,   tileEle * sizeof(float));
        pipe.InitBuffer(outFp32Buf,  tileEle * sizeof(float));
        pipe.InitBuffer(tmpABuf,     tileEle * sizeof(float));
        pipe.InitBuffer(tmpBBuf,     tileEle * sizeof(float));
    }

    __aicore__ inline void Process() {
        // empty/edge-case guard: coreRows<=0 would make tileNum=0 and underflow tailTileRows
        if (coreRows <= 0) {
            return;
        }
        int64_t tileNum     = (coreRows + tileM - 1) / tileM;
        int64_t tailTileRows = coreRows - (tileNum - 1) * tileM;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileM);
            Compute(i, tileM);
            CopyOut(i, tileM);
        }
        // tail tile (rows < tileM)
        CopyIn(tileNum - 1, tailTileRows);
        Compute(tileNum - 1, tailTileRows);
        CopyOut(tileNum - 1, tailTileRows);
    }

private:
    // ---- CopyIn: read x[rows, 2N] contiguous (one DataCopy over the full tile) ----
    __aicore__ inline void CopyIn(int64_t progress, int64_t rows) {
        LocalTensor<DATA_T> xLocal = inQueueX.AllocTensor<DATA_T>();
        DataCopy(xLocal, xGm[progress * tileM * 2 * N], rows * 2 * N);
        inQueueX.EnQue(xLocal);
    }

    // ---- Compute: split gate/up per row (contiguous N-segs), then silu+clamp+mul ----
    __aicore__ inline void Compute(int64_t progress, int64_t rows) {
        LocalTensor<DATA_T> xLocal  = inQueueX.DeQue<DATA_T>();
        LocalTensor<DATA_T> outLocal = outQueueOut.AllocTensor<DATA_T>();

        LocalTensor<float> gateFp32 = gateFp32Buf.Get<float>();
        LocalTensor<float> upFp32   = upFp32Buf.Get<float>();
        LocalTensor<float> outFp32  = outFp32Buf.Get<float>();
        LocalTensor<float> tmpA     = tmpABuf.Get<float>();   // sigmoid then silu
        LocalTensor<float> tmpB     = tmpBBuf.Get<float>();   // upClamped

        // per-row upcast: gate = xUB row's first N, up = row's last N (both contiguous).
        // xUB rows are 2N apart in memory, so this loop is required (no stride API).
        for (int64_t r = 0; r < rows; ++r) {
            Cast(gateFp32[r * N], xLocal[r * 2 * N],         RoundMode::CAST_NONE, N);
            Cast(upFp32[r * N],   xLocal[r * 2 * N + N],     RoundMode::CAST_NONE, N);
        }

        // gateFp32/upFp32 are now contiguous [rows, N]; compute the whole tile at once
        const int64_t calEle = rows * N;
        Sigmoid(tmpA, gateFp32, calEle);              // sigmoid(gate)
        Mul(tmpA, gateFp32, tmpA, calEle);            // silu = gate * sigmoid(gate)
        Mins(tmpA, tmpA, limit, calEle);              // clamp silu upper bound (lower ~ -0.278)
        Mins(tmpB, upFp32, limit, calEle);            // clamp up both bounds
        Maxs(tmpB, tmpB, -limit, calEle);
        Mul(outFp32, tmpA, tmpB, calEle);             // out = silu_c * up_c

        Cast(outLocal, outFp32, RoundMode::CAST_RINT, calEle);   // downcast fp32 -> bf16/fp16

        inQueueX.FreeTensor(xLocal);
        outQueueOut.EnQue<DATA_T>(outLocal);
    }

    // ---- CopyOut: write out[rows, N] contiguous ----
    __aicore__ inline void CopyOut(int64_t progress, int64_t rows) {
        LocalTensor<DATA_T> outLocal = outQueueOut.DeQue<DATA_T>();
        DataCopy(outGm[progress * tileM * N], outLocal, rows * N);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;      // x [tileM, 2N]
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;   // out [tileM, N]
    TBuf<TPosition::VECCALC> gateFp32Buf, upFp32Buf, outFp32Buf, tmpABuf, tmpBBuf;
    GlobalTensor<DATA_T> xGm, outGm;

    int64_t blockIdx, coreRowOffset, coreRows, tileM, N;
    int64_t formerNum, formerLength, tailNum, tailLength;
    float limit;
};

// ---- kernel entry: x[M,2N] -> out[M,N] ----
extern "C" __global__ __aicore__ void swiglustep(GM_ADDR x, GM_ADDR out,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelSwiglustep<DTYPE_X> op;
    op.Init(x, out, &tilingData);
    op.Process();
}
