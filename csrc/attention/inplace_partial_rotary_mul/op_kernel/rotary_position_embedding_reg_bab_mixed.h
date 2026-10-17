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
 * \file rotary_position_embedding_reg_bab_mixed.h
 * \brief Mixed precision kernel: x is half/bfloat16, cos/sin are float
 */

#ifndef ROTARY_POSITION_EMBEDDING_REG_BAB_MIXED_H
#define ROTARY_POSITION_EMBEDDING_REG_BAB_MIXED_H

#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename TX>
class RotaryPositionEmbeddingBABMixed {
public:
    __aicore__ inline RotaryPositionEmbeddingBABMixed(TPipe *pipe, const RopeRegbaseTilingData *tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const RopeRegbaseTilingData *tilingData_;
    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = 1;
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignFloat_ = 0;
    int64_t bIdx_ = 0;
    int64_t sIdx_ = 0;
    int64_t bNum_ = 0;
    int64_t sNum_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<TX> xGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<TX> yOutGm_;

    TQue<QuePosition::VECIN, bufferNum> xInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, bufferNum> yOutQue_;

private:
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void Compute(const LocalTensor<float> &sinTensor, const LocalTensor<float> &cosTensor,
        const LocalTensor<TX> &inTensor, const LocalTensor<TX> &outTensor, const uint32_t currSNum,
        const uint32_t currDNum);
    __aicore__ inline void ProcessN(const LocalTensor<float> &sinTensor, const LocalTensor<float> &cosTensor,
        const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
};

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y)
{
    this->blockIdx_ = GetBlockIdx();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->dSplitSize_ = tilingData_->sliceLength / dSplitCoef_ * sizeof(TX);
    this->dAlign_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(TX)) * dSplitCoef_;
    this->dAlignFloat_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(float)) * dSplitCoef_;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;
    this->xGm_.SetGlobalBuffer((__gm__ TX *)x);
    this->cosGm_.SetGlobalBuffer((__gm__ float *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ float *)sin);
    this->yOutGm_.SetGlobalBuffer((__gm__ TX *)y);
    this->pipe_->InitBuffer(xInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(TX));
    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlignFloat_ * sizeof(float));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlignFloat_ * sizeof(float));
    this->pipe_->InitBuffer(yOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(TX));
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->S % tilingData_->blockFactorS;
    }
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::Process()
{
    PrePareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
        uint32_t sLoopCnt = ops::CeilDiv(sNum_, ubFactorS_);
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::ProcessNLoop(
    const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<float> sinTensor = sinInQue_.AllocTensor<float>();
    LocalTensor<float> cosTensor = cosInQue_.AllocTensor<float>();
    int64_t offset = sIdx * tilingData_->sliceLength;
    uint32_t dSplitSizeFloat = tilingData_->sliceLength / dSplitCoef_ * sizeof(float);
    DataCopyExtParams copyParams{static_cast<uint16_t>(currSNum * dSplitCoef_), dSplitSizeFloat, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<float>();
    cosTensor = cosInQue_.DeQue<float>();
    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum);
    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::ProcessN(const LocalTensor<float> &sinTensor,
    const LocalTensor<float> &cosTensor, const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<TX> xTensor;
    LocalTensor<TX> yTensor;
    int64_t baseOffset = (bIdx * tilingData_->S + sIdx) * tilingData_->N * tilingData_->D + tilingData_->sliceStart;
    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currDNum = (idxN == tilingData_->ubLoopNumN - 1) ? tilingData_->ubTailFactorN : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->D;
        xTensor = xInQue_.AllocTensor<TX>();
        DataCopyExtParams copyInParams{static_cast<uint16_t>(currSNum * currDNum * dSplitCoef_),
            dSplitSize_,
            static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
            0,
            0};
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(currSNum * currDNum * dSplitCoef_),
            dSplitSize_,
            0,
            static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
            0};
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
            copyInParams = {static_cast<uint16_t>(currSNum * currDNum),
                tilingData_->sliceLength * sizeof(TX),
                static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(TX)),
                0,
                0};
        }
        DataCopyPadExtParams<TX> padParams{false, 0, 0, 0};
        DataCopyPad(xTensor, xGm_[offset], copyInParams, padParams);
        xInQue_.EnQue(xTensor);
        xTensor = xInQue_.DeQue<TX>();
        yTensor = yOutQue_.AllocTensor<TX>();
        Compute(sinTensor, cosTensor, xTensor, yTensor, currSNum, currDNum);
        xInQue_.FreeTensor(xTensor);
        yOutQue_.EnQue(yTensor);
        yTensor = yOutQue_.DeQue<TX>();
        DataCopyPad(yOutGm_[offset], yTensor, copyOutParams);
        yOutQue_.FreeTensor(yTensor);
    }
}

template <typename TX>
__aicore__ inline void RotaryPositionEmbeddingBABMixed<TX>::Compute(const LocalTensor<float> &sinTensor,
    const LocalTensor<float> &cosTensor, const LocalTensor<TX> &inTensor, const LocalTensor<TX> &outTensor,
    const uint32_t currSNum, const uint32_t currDNum)
{
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeVFMixed<TX>(
            inTensor, cosTensor, sinTensor, outTensor, tilingData_->sliceLength, currSNum, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        // For HALF mode, need to implement HalfAlignVFMixed
        InterleaveModeVFMixed<TX>(
            inTensor, cosTensor, sinTensor, outTensor, tilingData_->sliceLength, currSNum, currDNum);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        // For QUARTER mode, need to implement QuarterAlignVFMixed
        InterleaveModeVFMixed<TX>(
            inTensor, cosTensor, sinTensor, outTensor, tilingData_->sliceLength, currSNum, currDNum);
    }
}

}  // namespace InplacePartialRotaryMul
#endif  // ROTARY_POSITION_EMBEDDING_REG_BAB_MIXED_H