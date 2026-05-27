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
 * \file rotary_position_embedding_reg_ab_mixed.h
 * \brief Mixed precision kernel for AB layout: x is half/bfloat16, cos/sin are float
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_AB_MIXED_H
#define ROTARY_POSITION_EMBEDDING_REG_AB_MIXED_H

#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename TX>
class RotaryPositionEmbeddingABMixed {
public:
    __aicore__ inline RotaryPositionEmbeddingABMixed(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace,
        const RopeRegbaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t xGmOffset, LocalTensor<float> cosBuffer, LocalTensor<float> sinBuffer,
        int64_t ubIdx, int64_t bsCount, int64_t nCount);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;

    GlobalTensor<TX> xGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<TX> yGm_;
    const RopeRegbaseTilingData *tilingData_;
    DataCopyPadExtParams<TX> padParams_ = {false, 0, 0, static_cast<TX>(0)};
    DataCopyPadExtParams<float> padParamsFloat_ = {false, 0, 0, 0};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSizeTX_ = 0;
    uint32_t dSplitSizeFloat_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t sliceAlignTX_ = 0;
    int64_t sliceAlignFloat_ = 0;
};

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingABMixed<TX>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
    GM_ADDR workspace, const RopeRegbaseTilingData *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    dSplitSizeTX_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TX);
    dSplitSizeFloat_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(float);
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    int64_t offset = blockDimBS * tilingData_->blockFactorBS * tilingData_->D;
    int64_t xOffset =
        offset * tilingData_->N + blockDimN * tilingData_->blockFactorN * tilingData_->D + tilingData_->sliceStart;
    this->cosGm_.SetGlobalBuffer((__gm__ float *)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ float *)sin + cosOffset);
    this->xGm_.SetGlobalBuffer((__gm__ TX *)x + xOffset);
    this->yGm_.SetGlobalBuffer((__gm__ TX *)y + xOffset);

    sliceAlignTX_ =
        ops::CeilDiv(tilingData_->sliceLength * sizeof(TX), GetUbBlockSize()) * GetUbBlockSize() / sizeof(TX);
    sliceAlignFloat_ =
        ops::CeilDiv(tilingData_->sliceLength * sizeof(float), GetUbBlockSize()) * GetUbBlockSize() / sizeof(float);
    int64_t bufferSizeTX = sliceAlignTX_ * sizeof(TX) * tilingData_->ubFactorBS;
    int64_t bufferSizeFloat = sliceAlignFloat_ * sizeof(float) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(xInQueue_, DB_FLAG, bufferSizeTX * tilingData_->ubFactorN);
    pipe_->InitBuffer(yOutQueue_, DB_FLAG, bufferSizeTX * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, bufferSizeFloat);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, bufferSizeFloat);
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingABMixed<TX>::Process()
{
    uint32_t bsLoopCnt = ops::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    uint32_t nLoopCnt = ops::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->N * tilingData_->D;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS
                                                          : bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        DataCopyExtParams cosParams = {
            static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSizeFloat_, 0, 0, 0};

        LocalTensor<float> cosBuffer = cosInQueue_.AllocTensor<float>();
        LocalTensor<float> sinBuffer = sinInQueue_.AllocTensor<float>();
        DataCopyPad(cosBuffer,
            cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength],
            cosParams,
            padParamsFloat_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<float>();
        DataCopyPad(sinBuffer,
            sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength],
            cosParams,
            padParamsFloat_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<float>();

        for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN
                                                          : nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingABMixed<TX>::ProcessLoop(int64_t xGmOffset, LocalTensor<float> cosBuffer,
    LocalTensor<float> sinBuffer, int64_t ubIdx, int64_t bsCount, int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef),
        dSplitSizeTX_,
        static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
        0,
        0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef),
        dSplitSizeTX_,
        0,
        static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
        0};
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        inParams = {static_cast<uint16_t>(totalCount),
            tilingData_->D * sizeof(TX),
            static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
            0,
            0};
    }

    LocalTensor<TX> inBuffer = xInQueue_.AllocTensor<TX>();
    LocalTensor<TX> outBuffer = yOutQueue_.AllocTensor<TX>();

    DataCopyPad(inBuffer, xGm_[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], inParams, padParams_);

    xInQueue_.EnQue(inBuffer);
    inBuffer = xInQueue_.DeQue<TX>();

    InterleaveModeVFMixed<TX>(inBuffer, cosBuffer, sinBuffer, outBuffer, tilingData_->sliceLength, bsCount, nCount);

    yOutQueue_.EnQue(outBuffer);
    outBuffer = yOutQueue_.DeQue<TX>();
    xInQueue_.FreeTensor(inBuffer);

    DataCopyPad(yGm_[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], outBuffer, outParams);

    yOutQueue_.FreeTensor(outBuffer);
}

}  // namespace InplacePartialRotaryMul

#endif  // ROTARY_POSITION_EMBEDDING_REG_AB_MIXED_H
