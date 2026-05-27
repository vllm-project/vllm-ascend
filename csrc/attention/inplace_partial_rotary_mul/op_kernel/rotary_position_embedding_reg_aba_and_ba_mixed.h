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
 * \file rotary_position_embedding_reg_aba_and_ba_mixed.h
 * \brief Mixed precision kernel for ABA/BA layout: x is half/bfloat16, cos/sin are float
 */

#ifndef ROTARY_POSITION_EMBEDDING_REG_ABA_AND_BA_MIXED_H
#define ROTARY_POSITION_EMBEDDING_REG_ABA_AND_BA_MIXED_H

#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename TX, bool IsBBoardcast>
class RotaryPositionEmbeddingABAAndBAMixed {
public:
    __aicore__ inline RotaryPositionEmbeddingABAAndBAMixed(){};
    __aicore__ inline ~RotaryPositionEmbeddingABAAndBAMixed(){};

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace,
        const RopeRegbaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    __aicore__ inline void ProcessInSLoop(int64_t sUbStart, int64_t sUbLength);
    __aicore__ inline void ProcessInSBLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
        LocalTensor<TX> &cos, LocalTensor<TX> &sin);
    __aicore__ inline void ProcessInSBNLoop(int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength,
        int64_t nUbStart, int64_t nUbLength, int64_t nTotalSize, LocalTensor<float> &cosFloat,
        LocalTensor<float> &sinFloat, GlobalTensor<TX> &in, GlobalTensor<TX> &out);
    __aicore__ inline void CopyInCosAndSin(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInQ(GlobalTensor<TX> &source, int64_t sStart, int64_t sLength, int64_t bStart,
        int64_t bLength, int64_t nStart, int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void CopyOutQ(GlobalTensor<TX> &target, int64_t sStart, int64_t sLength, int64_t bStart,
        int64_t bLength, int64_t nStart, int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void Compute(
        LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t sLength, int64_t bLength, int64_t nLength);

private:
    TPipe *pipe_;

    GlobalTensor<TX> qGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<TX> qOutGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> qInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> qOutQueue_;

    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;
    int64_t sBlockStart_ = 0;
    int64_t sBlockLength_ = 0;

    const RopeRegbaseTilingData *tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignFloat_ = 0;
    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1;
    uint64_t ubCopyInStride = 0;
};

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::Init(GM_ADDR q, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR qOut, GM_ADDR workspace, const RopeRegbaseTilingData *tilingData, TPipe *pipe)
{
    this->tilingData_ = tilingData;
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    this->InitAllGlobalBuffer(q, cos, sin, qOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::InitAllGlobalBuffer(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ TX *)q);
    this->cosGm_.SetGlobalBuffer((__gm__ float *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ float *)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ TX *)qOut);
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->ubFactorS_ = this->tilingData_->ubFactorS;
    this->ubFactorN_ = this->tilingData_->ubFactorN;
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
        this->ubCopyInStride = (this->dAlign_ * sizeof(TX) -
                                   ops::CeilAlign<int64_t>(tilingData_->sliceLength * sizeof(TX), BLOCK_TYPE_SIZE)) /
                               BLOCK_TYPE_SIZE;
    }
    this->pipe_->InitBuffer(
        this->qInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(TX));
    this->pipe_->InitBuffer(
        this->qOutQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(TX));
    if constexpr (IsBBoardcast) {
        this->pipe_->InitBuffer(this->cosInQueue_, DOUBLE_BUFFER, ubFactorS_ * dAlignFloat_ * sizeof(float));
        this->pipe_->InitBuffer(this->sinInQueue_, DOUBLE_BUFFER, ubFactorS_ * dAlignFloat_ * sizeof(float));
    } else {
        this->pipe_->InitBuffer(
            this->cosInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * dAlignFloat_ * sizeof(float));
        this->pipe_->InitBuffer(
            this->sinInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * dAlignFloat_ * sizeof(float));
    }
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::InitLoopParams()
{
    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    int64_t sIdx = blockIdx_ / tilingData_->blockNumB;
    this->bBlockLength_ = tilingData_->blockFactorB;
    this->sBlockLength_ = tilingData_->blockFactorS;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        this->sBlockLength_ = tilingData_->S % tilingData_->blockFactorS;
    }
    this->bBlockStart_ = bIdx * tilingData_->blockFactorB;
    this->sBlockStart_ = sIdx * tilingData_->blockFactorS;
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::Process()
{
    int64_t ubLoopCount = ops::CeilDiv(sBlockLength_, ubFactorS_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        this->ProcessInSLoop(sBlockStart_ + ubLoopIdx * ubFactorS_,
            ubLoopIdx != ubLoopCount - 1 ? ubFactorS_ : sBlockLength_ - ubLoopIdx * ubFactorS_);
    }
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::ProcessInSLoop(
    int64_t sUbStart, int64_t sUbLength)
{
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBBoardcast) {
        this->CopyInCosAndSin(sUbStart, sUbLength, 0, 1);
        LocalTensor<float> cosUbFloat = this->cosInQueue_.template DeQue<float>();
        LocalTensor<float> sinUbFloat = this->sinInQueue_.template DeQue<float>();
        LocalTensor<TX> cosUb = cosUbFloat.template ReinterpretCast<TX>();
        LocalTensor<TX> sinUb = sinUbFloat.template ReinterpretCast<TX>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInSBLoop(sUbStart,
                sUbLength,
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
                cosUb,
                sinUb);
        }
        this->sinInQueue_.FreeTensor(sinUbFloat);
        this->cosInQueue_.FreeTensor(cosUbFloat);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(sUbStart,
                sUbLength,
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<float> cosUbFloat = this->cosInQueue_.template DeQue<float>();
            LocalTensor<float> sinUbFloat = this->sinInQueue_.template DeQue<float>();
            LocalTensor<TX> cosUb = cosUbFloat.template ReinterpretCast<TX>();
            LocalTensor<TX> sinUb = sinUbFloat.template ReinterpretCast<TX>();
            this->ProcessInSBLoop(sUbStart,
                sUbLength,
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
                cosUb,
                sinUb);
            this->cosInQueue_.FreeTensor(cosUbFloat);
            this->sinInQueue_.FreeTensor(sinUbFloat);
        }
    }
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::ProcessInSBLoop(int64_t sUbStart,
    int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, LocalTensor<TX> &cos, LocalTensor<TX> &sin)
{
    int64_t qUbLoopCount = ops::CeilDiv(tilingData_->N, ubFactorN_);
    LocalTensor<float> cosFloat = cos.template ReinterpretCast<float>();
    LocalTensor<float> sinFloat = sin.template ReinterpretCast<float>();
    for (int64_t ubLoopIdx = 0; ubLoopIdx < qUbLoopCount; ubLoopIdx++) {
        this->ProcessInSBNLoop(sUbStart,
            sUbLength,
            bUbStart,
            bUbLength,
            ubLoopIdx * ubFactorN_,
            ubLoopIdx != qUbLoopCount - 1 ? ubFactorN_ : tilingData_->N - ubLoopIdx * ubFactorN_,
            tilingData_->N,
            cosFloat,
            sinFloat,
            qGm_,
            qOutGm_);
    }
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::ProcessInSBNLoop(int64_t sUbStart,
    int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength, int64_t nTotalSize,
    LocalTensor<float> &cosFloat, LocalTensor<float> &sinFloat, GlobalTensor<TX> &in, GlobalTensor<TX> &out)
{
    CopyInQ(in, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
    Compute(cosFloat, sinFloat, sUbLength, bUbLength, nUbLength);
    CopyOutQ(out, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::CopyInCosAndSin(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength)
{
    LocalTensor<float> cosUb = this->cosInQueue_.template AllocTensor<float>();
    LocalTensor<float> sinUb = this->sinInQueue_.template AllocTensor<float>();
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = bLength;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->S * tilingData_->sliceLength * sizeof(float);
    loopParams.loop1DstStride = ubFactorS_ * dAlignFloat_ * sizeof(float);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<float> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(float) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(cosUb,
        this->cosGm_[bStart * tilingData_->S * tilingData_->sliceLength + sStart * tilingData_->sliceLength],
        copyExtParams,
        copyPadExtparams);
    DataCopyPad(sinUb,
        this->sinGm_[bStart * tilingData_->S * tilingData_->sliceLength + sStart * tilingData_->sliceLength],
        copyExtParams,
        copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::CopyInQ(GlobalTensor<TX> &source,
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart, int64_t nLength,
    int64_t nTotalSize)
{
    LocalTensor<TX> target = this->qInQueue_.template AllocTensor<TX>();
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = nTotalSize * tilingData_->S * D_ * sizeof(TX);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorS_ * dAlign_ * sizeof(TX);
    loopParams.loop1SrcStride = tilingData_->S * D_ * sizeof(TX);
    loopParams.loop1DstStride = ubFactorS_ * dAlign_ * sizeof(TX);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength * copyInQSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(TX) / copyInQSplitCoef_;
    copyExtParams.srcStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(TX);
    copyExtParams.dstStride = ubCopyInStride;
    DataCopyPadExtParams<TX> copyPadExtparams;
    copyPadExtparams.isPad = false;
    int64_t offset = bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_ +
                     tilingData_->sliceStart;
    DataCopyPad(target, source[offset], copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->qInQueue_.template EnQue(target);
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::CopyOutQ(GlobalTensor<TX> &target,
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart, int64_t nLength,
    int64_t nTotalSize)
{
    LocalTensor<TX> source = this->qOutQueue_.template DeQue<TX>();
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = ubFactorN_ * ubFactorS_ * dAlign_ * sizeof(TX);
    loopParams.loop2DstStride = nTotalSize * tilingData_->S * D_ * sizeof(TX);
    loopParams.loop1SrcStride = ubFactorS_ * dAlign_ * sizeof(TX);
    loopParams.loop1DstStride = tilingData_->S * D_ * sizeof(TX);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(TX) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(TX);
    int64_t offset = bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_ +
                     tilingData_->sliceStart;
    DataCopyPad(target[offset], source, copyExtParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    this->qOutQueue_.FreeTensor(source);
}

template <typename TX, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBAMixed<TX, IsBBoardcast>::Compute(
    LocalTensor<float> &cos, LocalTensor<float> &sin, int64_t sLength, int64_t bLength, int64_t nLength)
{
    LocalTensor<TX> inUb = this->qInQueue_.template DeQue<TX>();
    LocalTensor<TX> outUb = this->qOutQueue_.template AllocTensor<TX>();
    int64_t totalLength = sLength * bLength * nLength * tilingData_->sliceLength;

    BatchInterleaveModeVFMixed<TX, IsBBoardcast>((__local_mem__ TX *)inUb.GetPhyAddr(),
        (__local_mem__ float *)cos.GetPhyAddr(),
        (__local_mem__ float *)sin.GetPhyAddr(),
        (__local_mem__ TX *)outUb.GetPhyAddr(),
        sLength,
        bLength,
        nLength,
        tilingData_->sliceLength,
        dAlign_,
        dAlignFloat_,
        ubFactorS_,
        ubFactorN_);

    this->qInQueue_.FreeTensor(inUb);
    this->qOutQueue_.template EnQue(outUb);
}

}  // namespace InplacePartialRotaryMul

#endif  // ROTARY_POSITION_EMBEDDING_REG_ABA_AND_BA_MIXED_H