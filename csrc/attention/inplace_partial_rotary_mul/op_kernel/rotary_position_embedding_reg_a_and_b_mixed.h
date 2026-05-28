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
 * \file rotary_position_embedding_reg_a_and_b_mixed.h
 * \brief Mixed precision kernel for AAndB layout: x is half/bfloat16, cos/sin are float
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_A_AND_B_MIXED_H
#define ROTARY_POSITION_EMBEDDING_REG_A_AND_B_MIXED_H

#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename TX, bool IsBoardCast>
class RotaryPositionEmbeddingAAndBMixed {
public:
    __aicore__ inline RotaryPositionEmbeddingAAndBMixed(){};

    __aicore__ inline ~RotaryPositionEmbeddingAAndBMixed(){};

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace,
        const RopeRegbaseTilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    __aicore__ inline void ProcessInLoop(
        LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInQ(GlobalTensor<TX> &source, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyOutQ(GlobalTensor<TX> &target, int64_t bStart, int64_t bLength);

    __aicore__ inline void Compute(LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t bLength);

private:
    static constexpr uint32_t COS_DB_BUFFER = IsBoardCast ? 1 : DOUBLE_BUFFER;

    TPipe *pipe_;

    GlobalTensor<TX> qGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<TX> qOutGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> qInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> qOutQueue_;

    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;

    const RopeRegbaseTilingData *tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignFloat_ = 0;
    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1;
    uint64_t ubCopyInStride = 0;
};

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR qOut, GM_ADDR workspace, const RopeRegbaseTilingData *tilingData, TPipe *pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(q, cos, sin, qOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::InitAllGlobalBuffer(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ TX *)q);
    this->cosGm_.SetGlobalBuffer((__gm__ float *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ float *)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ TX *)qOut);
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->D_ = this->tilingData_->D;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->copyInQSplitCoef_ = dSplitCoef_;
    this->dAlign_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(TX)) * dSplitCoef_;
    this->dAlignFloat_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(float)) * dSplitCoef_;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->copyInQSplitCoef_ = 1;
        if constexpr (!IsBoardCast) {
            this->ubCopyInStride =
                (this->dAlign_ * sizeof(TX) -
                    ops::CeilAlign<int64_t>(tilingData_->sliceLength * sizeof(TX), BLOCK_TYPE_SIZE)) /
                BLOCK_TYPE_SIZE;
        }
    }

    this->pipe_->InitBuffer(this->qInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(TX));
    this->pipe_->InitBuffer(this->qOutQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(TX));
    if constexpr (IsBoardCast) {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, dAlignFloat_ * sizeof(float));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, dAlignFloat_ * sizeof(float));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlignFloat_ * sizeof(float));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlignFloat_ * sizeof(float));
    }
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::InitLoopParams()
{
    this->bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    this->bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::Process()
{
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBoardCast) {
        this->CopyInCosAndSin(0, 1);
        LocalTensor<float> cosUb = this->cosInQueue_.template DeQue<float>();
        LocalTensor<float> sinUb = this->sinInQueue_.template DeQue<float>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInLoop(cosUb,
                sinUb,
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
        }
        this->cosInQueue_.FreeTensor(cosUb);
        this->sinInQueue_.FreeTensor(sinUb);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<float> cosUb = this->cosInQueue_.template DeQue<float>();
            LocalTensor<float> sinUb = this->sinInQueue_.template DeQue<float>();
            this->ProcessInLoop(cosUb,
                sinUb,
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::ProcessInLoop(
    LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t bUbStart, int64_t bUbLength)
{
    CopyInQ(qGm_, bUbStart, bUbLength);
    Compute(cos, sin, bUbLength);
    CopyOutQ(qOutGm_, bUbStart, bUbLength);
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::CopyInCosAndSin(
    int64_t bStart, int64_t bLength)
{
    LocalTensor<float> cosUb = this->cosInQueue_.template AllocTensor<float>();
    LocalTensor<float> sinUb = this->sinInQueue_.template AllocTensor<float>();
    DataCopyPadExtParams<float> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    if constexpr (IsBoardCast) {
        copyExtParams.blockCount = 1 * dSplitCoef_;
    } else {
        copyExtParams.blockCount = bLength * dSplitCoef_;
    }
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(float) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(cosUb, this->cosGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, this->sinGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::CopyInQ(
    GlobalTensor<TX> &source, int64_t bStart, int64_t bLength)
{
    LocalTensor<TX> target = this->qInQueue_.template AllocTensor<TX>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * copyInQSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(TX) / copyInQSplitCoef_;
    copyExtParams.srcStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(TX);
    copyExtParams.dstStride = ubCopyInStride;
    DataCopyPadExtParams<TX> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(target, source[bStart * D_ + tilingData_->sliceStart], copyExtParams, copyPadExtparams);
    this->qInQueue_.template EnQue(target);
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::CopyOutQ(
    GlobalTensor<TX> &target, int64_t bStart, int64_t bLength)
{
    LocalTensor<TX> source = this->qOutQueue_.template DeQue<TX>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(TX) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(TX);
    DataCopyPad(target[bStart * D_ + tilingData_->sliceStart], source, copyExtParams);
    this->qOutQueue_.FreeTensor(source);
}

template <typename TX, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndBMixed<TX, IsBoardCast>::Compute(
    LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t bLength)
{
    LocalTensor<TX> inUb = this->qInQueue_.template DeQue<TX>();
    LocalTensor<TX> outUb = this->qOutQueue_.template AllocTensor<TX>();
    if constexpr (IsBoardCast) {
        InterleaveModeVFMixed<TX>(inUb, cos, sin, outUb, tilingData_->sliceLength, 1, bLength);
    } else {
        BatchInterleaveModeVFMixed<TX, IsBoardCast>((__local_mem__ TX *)inUb.GetPhyAddr(),
            (__local_mem__ float *)cos.GetPhyAddr(),
            (__local_mem__ float *)sin.GetPhyAddr(),
            (__local_mem__ TX *)outUb.GetPhyAddr(),
            bLength,
            1,
            1,
            tilingData_->sliceLength,
            dAlign_,
            dAlignFloat_,
            ubFactorB_,
            1);
    }

    this->qInQueue_.FreeTensor(inUb);
    this->qOutQueue_.template EnQue(outUb);
}
}  // namespace InplacePartialRotaryMul

#endif
