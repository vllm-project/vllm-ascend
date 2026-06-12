/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310.cpp
 * \brief
 */
#include "fused_gdn_gating_v310.h"
namespace {
    constexpr uint32_t ALIGN_HALF_NUM = 16;
    constexpr uint32_t ALIGN_HALF_OFFSET = 15;
    constexpr uint32_t QUEUE_DEPTH_DOUBLE = 2;
    constexpr uint32_t ALIGN_FLOAT_NUM = 8;
}

__aicore__ inline void SafeCopyInHalf(AscendC::LocalTensor<half>& dst, const AscendC::GlobalTensor<half>& src,
    uint32_t gmOffset, uint32_t num)
    {
    uint32_t blockLen = num / ALIGN_HALF_NUM ;
    uint32_t tail = num % ALIGN_HALF_NUM ;
    if (blockLen > 0) {
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(blockLen), 0, 0};
        AscendC::DataCopy(dst, src[gmOffset], copyParams);
    }
    if (tail > 0) {
        uint32_t align = blockLen * ALIGN_HALF_NUM ;
        for (uint32_t i = 0; i < tail; ++i) dst.SetValue(align + i, src.GetValue(gmOffset + align + i));
        uint32_t padEnd = ((num + ALIGN_HALF_OFFSET) / ALIGN_HALF_NUM) * ALIGN_HALF_NUM;
        for (uint32_t i = tail; i < padEnd - align; ++i) dst.SetValue(align + i, (half)0.0f);
    }
}

__aicore__ inline void SafeCopyOutHalf(AscendC::GlobalTensor<half>& dst, const AscendC::LocalTensor<half>& src,
    uint32_t gmOffset, uint32_t num)
    {
    uint32_t blockLen = num / ALIGN_HALF_NUM ;
    uint32_t tail = num % ALIGN_HALF_NUM ;
    if (blockLen > 0) {
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(blockLen), 0, 0};
        AscendC::DataCopy(dst[gmOffset], src, copyParams);
    }
    if (tail > 0) {
        uint32_t align = blockLen * ALIGN_HALF_NUM ;
        for (uint32_t i = 0; i < tail; ++i) dst.SetValue(gmOffset + align + i, src.GetValue(align + i));
    }
}

__aicore__ inline void SafeCopyOutFloat(AscendC::GlobalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    uint32_t gmOffset, uint32_t num)
    {
    uint32_t blockLen = num / ALIGN_FLOAT_NUM;
    uint32_t tail = num % ALIGN_FLOAT_NUM;
    if (blockLen > 0) {
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(blockLen), 0, 0};
        AscendC::DataCopy(dst[gmOffset], src, copyParams);
    }
    if (tail > 0) {
        uint32_t align = blockLen * ALIGN_FLOAT_NUM;
        for (uint32_t i = 0; i < tail; ++i) dst.SetValue(gmOffset + align + i, src.GetValue(align + i));
    }
}

namespace optiling {

__aicore__ inline void KernelFusedGdnGating::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR A_log, GM_ADDR dt_bias, GM_ADDR g, GM_ADDR beta_output,
    uint32_t usedCoreNum, uint32_t alignedLength, uint32_t tailLength,
    uint32_t numHeads, uint32_t tileRows, float beta, float inv_beta, float threshold)
{
    this->beta = (beta < 0.001f || beta > 100.0f) ? 1.0f : beta;
    this->inv_beta = (inv_beta < 0.001f || inv_beta > 1000.0f) ? 1.0f : inv_beta;
    this->threshold = (threshold < 1.0f || threshold > 100.0f) ? 20.0f : threshold;

    this->numHeads = numHeads;
    this->tileRows = tileRows;

    const uint32_t coreId = AscendC::GetBlockIdx();
    this->linesPerCore = (coreId == usedCoreNum - 1) ? tailLength : alignedLength;
    this->coreGlobalOffset = coreId * alignedLength;

    this->aGm.SetGlobalBuffer((__gm__ half*)a);
    this->bGm.SetGlobalBuffer((__gm__ half*)b);
    this->aLogGm.SetGlobalBuffer((__gm__ half*)A_log);
    this->dtBiasGm.SetGlobalBuffer((__gm__ half*)dt_bias);
    this->gGm.SetGlobalBuffer((__gm__ float*)g);
    this->betaOutGm.SetGlobalBuffer((__gm__ half*)beta_output);

    this->floatLen = ((this->numHeads + ALIGN_HALF_OFFSET) / ALIGN_HALF_NUM) * ALIGN_HALF_NUM ;
    const uint32_t floatBytes = this->floatLen * sizeof(float);
    const uint32_t halfBytes = this->floatLen * sizeof(half);

    uint32_t alignTileRows = ((this->tileRows + ALIGN_HALF_OFFSET) / ALIGN_HALF_NUM) * ALIGN_HALF_NUM ;
    const uint32_t computeFloatBytes = alignTileRows * sizeof(float);
    const uint32_t computeHalfBytes = alignTileRows * sizeof(half);

    this->pipe.InitBuffer(this->inQueueA, QUEUE_DEPTH_DOUBLE, computeHalfBytes);
    this->pipe.InitBuffer(this->inQueueB, QUEUE_DEPTH_DOUBLE, computeHalfBytes);
    this->pipe.InitBuffer(this->outQueueG, QUEUE_DEPTH_DOUBLE, computeFloatBytes);
    this->pipe.InitBuffer(this->outQueueBeta, QUEUE_DEPTH_DOUBLE, computeHalfBytes);

    this->pipe.InitBuffer(this->inQueueALog, 1, halfBytes);
    this->pipe.InitBuffer(this->inQueueDtBias, 1, halfBytes);

    this->pipe.InitBuffer(this->calcBufX, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufB, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufTmp, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufTmp2, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufSoftplus, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufMaskFloat, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufBroadcastBias, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufBroadcastExpALog, computeFloatBytes);
    this->pipe.InitBuffer(this->calcBufBeta, computeFloatBytes);
    
    this->pipe.InitBuffer(this->calcBufALogFloat, floatBytes);
    this->pipe.InitBuffer(this->calcBufDtBiasFloat, floatBytes);
}

__aicore__ inline void KernelFusedGdnGating::CopyIn(uint32_t tileOffset, uint32_t curElements)
{
    AscendC::LocalTensor<half> bufA = this->inQueueA.AllocTensor<half>();
    AscendC::LocalTensor<half> bufB = this->inQueueB.AllocTensor<half>();
    uint32_t gmOffset = this->coreGlobalOffset + tileOffset;

    SafeCopyInHalf(bufA, this->aGm, gmOffset, curElements);
    SafeCopyInHalf(bufB, this->bGm, gmOffset, curElements);

    this->inQueueA.EnQue(bufA);
    this->inQueueB.EnQue(bufB);
}

__aicore__ inline void KernelFusedGdnGating::Compute(uint32_t tileOffset, uint32_t curElements)
{
    AscendC::LocalTensor<half> bufA = this->inQueueA.DeQue<half>();
    AscendC::LocalTensor<half> bufB = this->inQueueB.DeQue<half>();

    AscendC::LocalTensor<float> bufG = this->outQueueG.AllocTensor<float>();
    AscendC::LocalTensor<half> bufBetaOut = this->outQueueBeta.AllocTensor<half>();

    AscendC::LocalTensor<float> rX = this->calcBufX.Get<float>();
    AscendC::LocalTensor<float> rB = this->calcBufB.Get<float>();
    AscendC::LocalTensor<float> rBetaX = this->calcBufBeta.Get<float>();
    AscendC::LocalTensor<float> rSoftplus = this->calcBufSoftplus.Get<float>();
    AscendC::LocalTensor<float> rMask = this->calcBufMaskFloat.Get<float>();
    AscendC::LocalTensor<float> rTmp1 = this->calcBufTmp.Get<float>();
    AscendC::LocalTensor<float> rTmp2 = this->calcBufTmp2.Get<float>();

    AscendC::LocalTensor<float> bcastDtBias = this->calcBufBroadcastBias.Get<float>();
    AscendC::LocalTensor<float> bcastNegExpALog = this->calcBufBroadcastExpALog.Get<float>();

    uint32_t vecElements = ((curElements + ALIGN_HALF_OFFSET) / ALIGN_HALF_NUM) * ALIGN_HALF_NUM ;

    AscendC::Cast(rX, bufA, AscendC::RoundMode::CAST_NONE, vecElements);
    AscendC::Cast(rB, bufB, AscendC::RoundMode::CAST_NONE, vecElements);

    AscendC::Muls(rB, rB, -1.0f, vecElements);
    AscendC::Maxs(rB, rB, -50.0f, vecElements);
    AscendC::Mins(rB, rB,  50.0f, vecElements);
    AscendC::Exp(rBetaX, rB, vecElements);
    AscendC::Adds(rBetaX, rBetaX, 1.0f, vecElements);
    
    AscendC::Duplicate(rTmp1, 1.0f, vecElements);
    AscendC::Div(rB, rTmp1, rBetaX, vecElements);
    AscendC::Cast(bufBetaOut, rB, AscendC::RoundMode::CAST_NONE, vecElements);

    AscendC::Add(rX, rX, bcastDtBias, vecElements);
    AscendC::Muls(rBetaX, rX, this->beta, vecElements);

    AscendC::Abs(rTmp1, rBetaX, vecElements);
    AscendC::Muls(rTmp2, rTmp1, -1.0f, vecElements);
    AscendC::Exp(rTmp2, rTmp2, vecElements);

    AscendC::Adds(rSoftplus, rTmp2, 1.0f, vecElements);
    AscendC::Ln(rSoftplus, rSoftplus, vecElements);

    AscendC::Mul(rB, rTmp2, rTmp2, vecElements);
    AscendC::Muls(rB, rB, -0.5f, vecElements);
    AscendC::Add(rB, rTmp2, rB, vecElements);

    AscendC::Adds(rMask, rTmp1, -7.0f, vecElements);
    AscendC::Maxs(rMask, rMask, 0.0f, vecElements);
    AscendC::Muls(rMask, rMask, 1000000.0f, vecElements);
    AscendC::Mins(rMask, rMask, 1.0f, vecElements);

    AscendC::LocalTensor<float> rInvMask = rTmp1;
    AscendC::Muls(rInvMask, rMask, -1.0f, vecElements);
    AscendC::Adds(rInvMask, rInvMask, 1.0f, vecElements);

    AscendC::Mul(rB, rB, rMask, vecElements);
    AscendC::Mul(rSoftplus, rSoftplus, rInvMask, vecElements);
    AscendC::Add(rTmp2, rB, rSoftplus, vecElements);

    AscendC::Maxs(rSoftplus, rBetaX, 0.0f, vecElements);
    AscendC::Add(rSoftplus, rSoftplus, rTmp2, vecElements);
    AscendC::Muls(rSoftplus, rSoftplus, this->inv_beta, vecElements);

    AscendC::Adds(rMask, rBetaX, -this->threshold, vecElements);
    AscendC::Maxs(rMask, rMask, 0.0f, vecElements);
    AscendC::Muls(rMask, rMask, 1000000.0f, vecElements);
    AscendC::Mins(rMask, rMask, 1.0f, vecElements);

    AscendC::Muls(rTmp1, rMask, -1.0f, vecElements);
    AscendC::Adds(rTmp1, rTmp1, 1.0f, vecElements);

    AscendC::Mul(rSoftplus, rSoftplus, rTmp1, vecElements);
    AscendC::Mul(rX, rX, rMask, vecElements);
    AscendC::Add(rSoftplus, rSoftplus, rX, vecElements);

    AscendC::Mul(bufG, rSoftplus, bcastNegExpALog, vecElements);

    this->outQueueG.EnQue(bufG);
    this->outQueueBeta.EnQue(bufBetaOut);

    this->inQueueA.FreeTensor(bufA);
    this->inQueueB.FreeTensor(bufB);
}

__aicore__ inline void KernelFusedGdnGating::CopyOut(uint32_t tileOffset, uint32_t curElements)
{
    AscendC::LocalTensor<float> bufG = this->outQueueG.DeQue<float>();
    AscendC::LocalTensor<half> bufBetaOut = this->outQueueBeta.DeQue<half>();
    uint32_t gmOffset = this->coreGlobalOffset + tileOffset;

    SafeCopyOutFloat(this->gGm, bufG, gmOffset, curElements);
    SafeCopyOutHalf(this->betaOutGm, bufBetaOut, gmOffset, curElements);

    this->outQueueG.FreeTensor(bufG);
    this->outQueueBeta.FreeTensor(bufBetaOut);
}

__aicore__ inline void KernelFusedGdnGating::Process()
{
    if (this->linesPerCore == 0 || this->tileRows == 0) return;

    AscendC::LocalTensor<half> bufALogHalf = this->inQueueALog.AllocTensor<half>();
    SafeCopyInHalf(bufALogHalf, this->aLogGm, 0, this->numHeads);
    this->inQueueALog.EnQue(bufALogHalf);

    AscendC::LocalTensor<half> aLogHalfReady = this->inQueueALog.DeQue<half>();
    AscendC::LocalTensor<float> aLogFloatReady = this->calcBufALogFloat.Get<float>();
    AscendC::Cast(aLogFloatReady, aLogHalfReady, AscendC::RoundMode::CAST_NONE, this->floatLen);
    this->inQueueALog.FreeTensor(aLogHalfReady);

    AscendC::LocalTensor<half> bufDtBiasHalf = this->inQueueDtBias.AllocTensor<half>();
    SafeCopyInHalf(bufDtBiasHalf, this->dtBiasGm, 0, this->numHeads);
    this->inQueueDtBias.EnQue(bufDtBiasHalf);

    AscendC::LocalTensor<half> dtBiasHalfReady = this->inQueueDtBias.DeQue<half>();
    AscendC::LocalTensor<float> dtBiasFloatReady = this->calcBufDtBiasFloat.Get<float>();
    AscendC::Cast(dtBiasFloatReady, dtBiasHalfReady, AscendC::RoundMode::CAST_NONE, this->floatLen);
    this->inQueueDtBias.FreeTensor(dtBiasHalfReady);

    AscendC::LocalTensor<float> tmpFloatALog = this->calcBufTmp.Get<float>();
    AscendC::Exp(tmpFloatALog, aLogFloatReady, this->floatLen);
    AscendC::Muls(tmpFloatALog, tmpFloatALog, -1.0f, this->floatLen);

    AscendC::LocalTensor<float> bcastDtBias = this->calcBufBroadcastBias.Get<float>();
    AscendC::LocalTensor<float> bcastExpALog = this->calcBufBroadcastExpALog.Get<float>();
    
    uint32_t alignTileRows = ((this->tileRows + ALIGN_HALF_OFFSET) / ALIGN_HALF_NUM) * ALIGN_HALF_NUM ;
    
    if (this->numHeads % ALIGN_FLOAT_NUM == 0) {
        AscendC::Adds(bcastDtBias[0], dtBiasFloatReady[0], 0.0f, this->numHeads);
        AscendC::Adds(bcastExpALog[0], tmpFloatALog[0], 0.0f, this->numHeads);
        
        uint32_t currentLen = this->numHeads;
        while (currentLen < alignTileRows) {
            uint32_t copyLen = currentLen;
            if (currentLen + copyLen > alignTileRows) {
                copyLen = alignTileRows - currentLen;
            }
            AscendC::Adds(bcastDtBias[currentLen], bcastDtBias[0], 0.0f, copyLen);
            AscendC::Adds(bcastExpALog[currentLen], bcastExpALog[0], 0.0f, copyLen);
            currentLen += copyLen;
        }
    } else {
        uint32_t headIdx = 0;
        for (uint32_t i = 0; i < alignTileRows; ++i) {
            bcastDtBias.SetValue(i, dtBiasFloatReady.GetValue(headIdx));
            bcastExpALog.SetValue(i, tmpFloatALog.GetValue(headIdx));
            headIdx++;
            if (headIdx == this->numHeads) headIdx = 0;
        }
    }

    uint32_t totalTiles = (this->linesPerCore + this->tileRows - 1) / this->tileRows;
    uint32_t tailElements = this->linesPerCore % this->tileRows;
    if (tailElements == 0) tailElements = this->tileRows;

    for (uint32_t tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
        uint32_t curElements = (tileIdx == totalTiles - 1) ? tailElements : this->tileRows;
        uint32_t tileOffset = tileIdx * this->tileRows;
        CopyIn(tileOffset, curElements);
        Compute(tileOffset, curElements);
        CopyOut(tileOffset, curElements);
    }
}

} // namespace optiling

extern "C" __global__ __aicore__ void fused_gdn_gating_v310(
    GM_ADDR a, GM_ADDR b, GM_ADDR A_log, GM_ADDR dt_bias, GM_ADDR g, GM_ADDR beta_output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    optiling::KernelFusedGdnGating op;
    
    op.Init(a, b, A_log, dt_bias, g, beta_output,
            tiling_data.usedCoreNum, tiling_data.alignedLength, tiling_data.tailLength,
            tiling_data.numHeads, tiling_data.tileRows,
            tiling_data.beta, tiling_data.inv_beta, tiling_data.threshold);
    op.Process();
}