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
 * \brief SwigluStep fused kernel: out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;              // double buffer, hide GM<->UB latency

template <typename DATA_T>                      // DATA_T = bfloat16 / half
class KernelSwiglustep {
public:
    __aicore__ inline KernelSwiglustep() {}

    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR out,
                                const SwiglustepTilingData* tilingData) {
        // 1) read tiling
        formerNum     = tilingData->formerNum;
        formerLength  = tilingData->formerLength;
        tailNum       = tilingData->tailNum;
        tailLength    = tilingData->tailLength;
        tileLength    = tilingData->tileLength;
        limit         = tilingData->limit;

        // 2) segment handled by this core (former core: formerLength / tail core: tailLength)
        blockIdx   = GetBlockIdx();
        int64_t usedCoreNum = formerNum + tailNum;
        coreLength = (blockIdx == usedCoreNum - 1) ? tailLength : formerLength;
        gmOffset   = (blockIdx < formerNum) ? blockIdx * formerLength
                                            : formerNum * formerLength;

        // 3) GM tensors (this core's segment)
        gateGm.SetGlobalBuffer((__gm__ DATA_T*)gate + gmOffset, coreLength);
        upGm.SetGlobalBuffer((__gm__ DATA_T*)up   + gmOffset, coreLength);
        outGm.SetGlobalBuffer((__gm__ DATA_T*)out + gmOffset, coreLength);

        // 4) UB queues + independent fp32 TBufs
        pipe.InitBuffer(inQueueGate, BUFFER_NUM, tileLength * sizeof(DATA_T));
        pipe.InitBuffer(inQueueUp,   BUFFER_NUM, tileLength * sizeof(DATA_T));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileLength * sizeof(DATA_T));
        pipe.InitBuffer(gateFp32Buf, tileLength * sizeof(float));
        pipe.InitBuffer(upFp32Buf,   tileLength * sizeof(float));
        pipe.InitBuffer(outFp32Buf,  tileLength * sizeof(float));
        pipe.InitBuffer(tmpABuf,     tileLength * sizeof(float));
        pipe.InitBuffer(tmpBBuf,     tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        // empty/edge-case guard: coreLength<=0 would make tileNum=0, tailTileLen
        // positive, and CopyIn run with a negative progress (out-of-bounds)
        if (coreLength <= 0) {
            return;
        }
        int64_t tileNum    = (coreLength + tileLength - 1) / tileLength;
        int64_t tailTileLen = coreLength - (tileNum - 1) * tileLength;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileLength);
            Compute(i, tileLength);
            CopyOut(i, tileLength);
        }
        // tail tile
        CopyIn(tileNum - 1, tailTileLen);
        Compute(tileNum - 1, tailTileLen);
        CopyOut(tileNum - 1, tailTileLen);
    }

private:
    // ---- CopyIn: GM -> UB (gate/up) ----
    __aicore__ inline void CopyIn(int64_t progress, int64_t len) {
        LocalTensor<DATA_T> gateLocal = inQueueGate.AllocTensor<DATA_T>();
        LocalTensor<DATA_T> upLocal   = inQueueUp.AllocTensor<DATA_T>();
        DataCopy(gateLocal, gateGm[progress * tileLength], len);
        DataCopy(upLocal,   upGm[progress * tileLength],   len);
        inQueueGate.EnQue(gateLocal);
        inQueueUp.EnQue(upLocal);
    }

    // ---- Compute: silu(gate).clamp(max=limit) * up.clamp(-limit,limit) ----
    __aicore__ inline void Compute(int64_t progress, int64_t len) {
        LocalTensor<DATA_T> gateLocal = inQueueGate.DeQue<DATA_T>();
        LocalTensor<DATA_T> upLocal   = inQueueUp.DeQue<DATA_T>();
        LocalTensor<DATA_T> outLocal  = outQueueOut.AllocTensor<DATA_T>();

        // independent fp32 buffers (one per TBuf)
        LocalTensor<float> gateFp32 = gateFp32Buf.Get<float>();
        LocalTensor<float> upFp32   = upFp32Buf.Get<float>();
        LocalTensor<float> outFp32  = outFp32Buf.Get<float>();
        LocalTensor<float> tmpA     = tmpABuf.Get<float>();   // sigmoid then silu
        LocalTensor<float> tmpB     = tmpBBuf.Get<float>();   // upClamped

        // upcast bf16/fp16 -> fp32
        Cast(gateFp32, gateLocal, RoundMode::CAST_NONE, len);
        Cast(upFp32,   upLocal,   RoundMode::CAST_NONE, len);

        // 1) sigmoid(gate)
        Sigmoid(tmpA, gateFp32, len);
        // 2) silu = gate * sigmoid
        Mul(tmpA, gateFp32, tmpA, len);
        // 3) clamp silu: only upper bound needs capping, silu lower bound ~ -0.278
        Mins(tmpA, tmpA, limit, len);
        // 4) clamp up (both bounds)
        Mins(tmpB, upFp32, limit, len);
        Maxs(tmpB, tmpB, -limit, len);
        // 5) out = silu_c * up_c
        Mul(outFp32, tmpA, tmpB, len);

        // downcast fp32 -> bf16/fp16
        Cast(outLocal, outFp32, RoundMode::CAST_RINT, len);

        inQueueGate.FreeTensor(gateLocal);
        inQueueUp.FreeTensor(upLocal);
        outQueueOut.EnQue<DATA_T>(outLocal);
    }

    // ---- CopyOut: UB -> GM (out) ----
    __aicore__ inline void CopyOut(int64_t progress, int64_t len) {
        LocalTensor<DATA_T> outLocal = outQueueOut.DeQue<DATA_T>();
        DataCopy(outGm[progress * tileLength], outLocal, len);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueGate;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueUp;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    // independent fp32 TBufs
    TBuf<TPosition::VECCALC> gateFp32Buf, upFp32Buf, outFp32Buf, tmpABuf, tmpBBuf;
    GlobalTensor<DATA_T> gateGm, upGm, outGm;

    int64_t blockIdx, gmOffset, coreLength, tileLength;
    int64_t formerNum, formerLength, tailNum, tailLength;
    float limit;
};

// ---- kernel entry (GET_TILING_DATA reads tiling) ----
extern "C" __global__ __aicore__ void swiglustep(GM_ADDR gate, GM_ADDR up, GM_ADDR out,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelSwiglustep<DTYPE_GATE> op;
    op.Init(gate, up, out, &tilingData);
    op.Process();
}
