/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310.h
 * \brief
 */
#ifndef FUSED_GDN_GATING_V310_H
#define FUSED_GDN_GATING_V310_H

#include "kernel_operator.h"

namespace optiling {

constexpr uint32_t QUEUE_DEPTH_DOUBLE = 2;
constexpr uint32_t QUEUE_DEPTH_SINGLE = 1;

constexpr uint32_t DEFAULT_UINT_ZERO = 0;
constexpr uint32_t DEFAULT_UINT_ONE = 1;
constexpr float DEFAULT_FLOAT_ZERO = 0.0f;
constexpr float DEFAULT_FLOAT_ONE = 1.0f;

class KernelFusedGdnGating {
public:
    __aicore__ inline KernelFusedGdnGating() {}

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR A_log, GM_ADDR dt_bias,
        GM_ADDR g, GM_ADDR beta_output,
        uint32_t usedCoreNum, uint32_t alignedLength, uint32_t tailLength,
        uint32_t numHeads, uint32_t tileRows,
        float beta, float inv_beta, float threshold);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t tileOffset, uint32_t curElements);
    __aicore__ inline void Compute(uint32_t tileOffset, uint32_t curElements);
    __aicore__ inline void CopyOut(uint32_t tileOffset, uint32_t curElements);

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH_DOUBLE> inQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH_DOUBLE> inQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH_DOUBLE> outQueueG;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH_DOUBLE> outQueueBeta;

    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH_SINGLE> inQueueALog;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH_SINGLE> inQueueDtBias;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufTmp2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufSoftplus;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufMaskFloat;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufBroadcastBias;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufBroadcastExpALog;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufBeta;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufALogFloat;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufDtBiasFloat;

    AscendC::GlobalTensor<half> aGm, bGm, betaOutGm, aLogGm, dtBiasGm;
    AscendC::GlobalTensor<float> gGm;

    uint32_t numHeads = DEFAULT_UINT_ZERO;
    uint32_t linesPerCore = DEFAULT_UINT_ZERO;
    uint32_t tileRows = DEFAULT_UINT_ONE;
    float beta = DEFAULT_FLOAT_ONE;
    float inv_beta = DEFAULT_FLOAT_ONE;
    float threshold = DEFAULT_FLOAT_ZERO;
    uint32_t coreGlobalOffset = DEFAULT_UINT_ZERO;
    uint32_t floatLen = DEFAULT_UINT_ZERO;
};
} // namespace optiling

extern "C" __global__ __aicore__ void fused_gdn_gating_v310(
    GM_ADDR a, GM_ADDR b, GM_ADDR A_log, GM_ADDR dt_bias,
    GM_ADDR g, GM_ADDR beta_output, GM_ADDR workspace, GM_ADDR tiling);

#endif // FUSED_GDN_GATING_V310_H
