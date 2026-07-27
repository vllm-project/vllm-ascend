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
 * \file hc_post_float32.h
 * \brief
 */
#ifndef HC_POST_FLOAT32_H
#define HC_POST_FLOAT32_H

#include "kernel_operator.h"

namespace HcPostRegBase {
using namespace AscendC;

template <typename T>
class HcPostRegBaseFloat32 {
public:
    __aicore__ inline HcPostRegBaseFloat32() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y, GM_ADDR workspace,
        const HcPostTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void DataCopyInX(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DataCopyInPost(int64_t batchIndex);
    __aicore__ inline void DataCopyInResidual(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DataCopyInComb(int64_t batchIndex);
    __aicore__ inline void DataCopyOut(int64_t batchIndex, int64_t hcIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DoProcess(int64_t batchSize);
    __aicore__ inline void DoCompute(LocalTensor<float> sumTempBuf, LocalTensor<T> postUb, LocalTensor<T> combUb, int64_t batchIndex, int64_t dOffset, int64_t dDealing);
    __aicore__ inline void DoMulAndAdd(LocalTensor<float> xUb, LocalTensor<T> postUb, LocalTensor<float> residualUb, LocalTensor<T> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex, int64_t dDealing);

private:
    TPipe* pipe_;
    const HcPostTilingData* tiling_;
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

    int32_t blkIdx_ = -1;
    int64_t batch_ = 0;
    int64_t hcParam_ = 0;
    int64_t dParam_ = 0;
    int64_t batchOneCoreTail_ = 0;
    int64_t batchOneCore_ = 0;
    int64_t isFrontCore_ = 0;
    int64_t dParamAlign_ = 0;
    int64_t dParamOnceAlign_ = 0;
    int64_t dOnceDealing_ = 0;
    int64_t dLastDealing_ = 0;
    int64_t dSplitTime_ = 0;
    static constexpr int32_t ONE_BLOCK_SIZE = 32;
    int32_t perBlock32 = ONE_BLOCK_SIZE / sizeof(float);

    GlobalTensor<float> xGm_;
    GlobalTensor<float> residualGm_;
    GlobalTensor<T> postGm_;
    GlobalTensor<T> combGm_;
    GlobalTensor<float> yGm_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> residualQue_;
    TQue<QuePosition::VECIN, 1> postQue_;
    TQue<QuePosition::VECIN, 1> combQue_;
    TQue<QuePosition::VECOUT, 1> sumQue_;
    TBuf<QuePosition::VECCALC> sumTempBuf_;
};

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y,
    GM_ADDR workspace, const HcPostTilingData *tilingData, TPipe *pipe)
{
    blkIdx_ = GetBlockIdx();
    if (blkIdx_ >= tilingData->usedCoreNum) {
        return;
    }
    tiling_ = tilingData;
    pipe_ = pipe;
    hcParam_ = tilingData->hcParam;
    dParam_ = tilingData->dParam;
    batchOneCoreTail_ = tilingData->batchOneCoreTail;
    batchOneCore_ = tilingData->batchOneCore;
    isFrontCore_ = blkIdx_ < tilingData->frontCore;
    int64_t frontCore = tilingData->frontCore;
    dOnceDealing_ = tilingData->dOnceDealing;
    dLastDealing_ = tilingData->dLastDealing;
    dSplitTime_ = tilingData->dSplitTime;
    dParamAlign_ = (dParam_ + perBlock32 - 1) / perBlock32 * perBlock32;
    dParamOnceAlign_ = (dOnceDealing_ + perBlock32 - 1) / perBlock32 * perBlock32;

    int64_t xOffset = blkIdx_ * batchOneCore_ * dParam_;
    int64_t residualOffset = blkIdx_ * batchOneCore_ * hcParam_ * dParam_;
    int64_t postOffset = blkIdx_ * batchOneCore_ * hcParam_;
    int64_t combOffset = blkIdx_ * batchOneCore_ * hcParam_ * hcParam_;
    int64_t yOffset = blkIdx_ * batchOneCore_ * hcParam_ * dParam_;
    if (!isFrontCore_) {
        xOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * dParam_;
        residualOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * hcParam_ * dParam_;
        postOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * hcParam_;
        combOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * hcParam_ * hcParam_;
        yOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * hcParam_ * dParam_;
    }
    xGm_.SetGlobalBuffer((__gm__ float *)x + xOffset);
    residualGm_.SetGlobalBuffer((__gm__ float *)residual + residualOffset);
    postGm_.SetGlobalBuffer((__gm__ T *)post + postOffset);
    combGm_.SetGlobalBuffer((__gm__ T *)comb + combOffset);
    yGm_.SetGlobalBuffer((__gm__ float *)y + yOffset);

    pipe_->InitBuffer(xQue_, 2, dParamOnceAlign_ * sizeof(float));
    pipe_->InitBuffer(residualQue_, 2, hcParam_ * dParamOnceAlign_ * sizeof(float));
    pipe_->InitBuffer(postQue_, 2, hcParam_ * sizeof(T));
    pipe_->InitBuffer(combQue_, 2, hcParam_ * hcParam_ * sizeof(T));
    pipe_->InitBuffer(sumQue_, 2, dParamOnceAlign_ * sizeof(float));
    pipe_->InitBuffer(sumTempBuf_, dParamOnceAlign_ * sizeof(float));
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::Process()
{
    if (blkIdx_ >= tiling_->usedCoreNum) {
        return;
    }
    if (isFrontCore_) {
        DoProcess(tiling_->batchOneCore);
    } else {
        DoProcess(tiling_->batchOneCoreTail);
    }
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInX(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<float> xUb = xQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dOnceDealing * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[batchIndex * dParam_ + dOffset], copyParams, dataCopyPadParams);
    xQue_.EnQue<float>(xUb);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInPost(int64_t batchIndex)
{
    LocalTensor<T> postUb = postQue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(postUb, postGm_[batchIndex * hcParam_], copyParams, dataCopyPadParams);
    postQue_.EnQue<T>(postUb);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInResidual(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<float> residualUb = residualQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = hcParam_;
    copyParams.blockLen = dOnceDealing * sizeof(float);
    copyParams.srcStride = (dParamAlign_ - dOnceDealing) * sizeof(float);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(residualUb, residualGm_[batchIndex * hcParam_ * dParam_ + dOffset], copyParams, dataCopyPadParams);
    residualQue_.EnQue<float>(residualUb);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInComb(int64_t batchIndex)
{
    LocalTensor<T> combUb = combQue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * hcParam_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(combUb, combGm_[batchIndex * hcParam_ * hcParam_], copyParams, dataCopyPadParams);
    combQue_.EnQue<T>(combUb);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyOut(int64_t batchIndex, int64_t hcIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<float> outBuf = sumQue_.DeQue<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dOnceDealing * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGm_[batchIndex * hcParam_ * dParam_ + hcIndex * dParam_ + dOffset], outBuf, copyParams);
    sumQue_.FreeTensor(outBuf);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DoMulAndAdd(LocalTensor<float> xUb, LocalTensor<T> postUb, LocalTensor<float> residualUb, LocalTensor<T> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex, int64_t dOnceDealing)
{
    uint16_t aTimes = hcParam_;
    uint32_t xDealNumAlign = (dOnceDealing + perBlock32 - 1) / perBlock32 * perBlock32;
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (dOnceDealing + vfLen - 1) / vfLen;

    auto residualAddr = (__ubuf__ float*)residualUb.GetPhyAddr();
    auto combAddr = (__ubuf__ T*)combUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
    auto xAddr = (__ubuf__ float*)xUb.GetPhyAddr();
    auto postAddr = (__ubuf__ T*)postUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(dOnceDealing);
        AscendC::MicroAPI::RegTensor<T> combReg0;
        AscendC::MicroAPI::RegTensor<T> combReg1;
        AscendC::MicroAPI::RegTensor<T> combReg2;
        AscendC::MicroAPI::RegTensor<T> combReg3;
        AscendC::MicroAPI::RegTensor<float> residualRegFloat0;
        AscendC::MicroAPI::RegTensor<float> residualRegFloat1;
        AscendC::MicroAPI::RegTensor<float> residualRegFloat2;
        AscendC::MicroAPI::RegTensor<float> residualRegFloat3;
        AscendC::MicroAPI::RegTensor<float> combRegFloat0;
        AscendC::MicroAPI::RegTensor<float> combRegFloat1;
        AscendC::MicroAPI::RegTensor<float> combRegFloat2;
        AscendC::MicroAPI::RegTensor<float> combRegFloat3;
        AscendC::MicroAPI::RegTensor<float> sumRegFloat;
        AscendC::MicroAPI::RegTensor<float> sumTempReg0;
        AscendC::MicroAPI::RegTensor<float> sumTempReg1;
        AscendC::MicroAPI::RegTensor<float> sumTempReg2;
        AscendC::MicroAPI::RegTensor<float> sumTempReg3;

        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<T> postReg;
        AscendC::MicroAPI::RegTensor<float> xRegFloat;
        AscendC::MicroAPI::RegTensor<float> postRegFloat;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        if constexpr (sizeof(T) == 2) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg0, combAddr+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg1, combAddr+hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg2, combAddr+2*hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg3, combAddr+3*hcParam_+hcIndex);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(combRegFloat0, combReg0, pregMain);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(combRegFloat1, combReg1, pregMain);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(combRegFloat2, combReg2, pregMain);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(combRegFloat3, combReg3, pregMain);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(postReg, postAddr + hcIndex);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(postRegFloat, postReg, pregMain);
        } else {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat0, combAddr+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat1, combAddr+hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat2, combAddr+2*hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat3, combAddr+3*hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(postRegFloat, postAddr + hcIndex);
        }

        for (uint16_t j = 0; j < repeatTimes; j++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            AscendC::MicroAPI::DataCopy(xRegFloat, xAddr+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat0, residualAddr+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat1, residualAddr+xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat2, residualAddr+2*xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat3, residualAddr+3*xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::Mul(sumRegFloat, residualRegFloat0, combRegFloat0, pMask);
            AscendC::MicroAPI::MulAddDst(sumRegFloat, residualRegFloat3, combRegFloat3, pMask);
            AscendC::MicroAPI::MulAddDst(sumRegFloat, residualRegFloat1, combRegFloat1, pMask);
            AscendC::MicroAPI::MulAddDst(sumRegFloat, residualRegFloat2, combRegFloat2, pMask);
            AscendC::MicroAPI::MulAddDst(sumRegFloat, xRegFloat, postRegFloat, pMask);
            AscendC::MicroAPI::DataCopy(sumAddr+j*vfLen, sumRegFloat, pMask);
        }
    }
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DoCompute(LocalTensor<float> sumTempBuf, LocalTensor<T> postUb, LocalTensor<T> combUb, int64_t batchIndex, int64_t dOffset, int64_t dDealing)
{
    DataCopyInX(batchIndex, dDealing, dOffset);
    LocalTensor<float> xUb = xQue_.DeQue<float>();
    DataCopyInResidual(batchIndex, dDealing, dOffset);
    LocalTensor<float> residualUb = residualQue_.DeQue<float>();
    for (int64_t hc1Index = 0; hc1Index < hcParam_; hc1Index++) {
        DoMulAndAdd(xUb, postUb, residualUb, combUb, sumTempBuf, hc1Index, dDealing);
        LocalTensor<float> sumUb = sumQue_.AllocTensor<float>();
        AscendC::Copy(sumUb, sumTempBuf, dDealing);
        sumQue_.EnQue<float>(sumUb);
        DataCopyOut(batchIndex, hc1Index, dDealing, dOffset);
    }
    residualQue_.FreeTensor(residualUb);
    xQue_.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DoProcess(int64_t batchSize)
{
    LocalTensor<float> sumTempBuf = sumTempBuf_.Get<float>();
    for (int64_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        DataCopyInPost(batchIndex);
        LocalTensor<T> postUb = postQue_.DeQue<T>();
        DataCopyInComb(batchIndex);
        LocalTensor<T> combUb = combQue_.DeQue<T>();
        int64_t dOffset = 0;
        for (int64_t dIndex = 0; dIndex < dSplitTime_; dIndex++) {
            dOffset = dIndex*dOnceDealing_;
            DoCompute(sumTempBuf, postUb, combUb, batchIndex, dOffset, dOnceDealing_);
        }
        if (dLastDealing_ != 0) {
            dOffset = dSplitTime_*dOnceDealing_;
            DoCompute(sumTempBuf, postUb, combUb, batchIndex, dOffset, dLastDealing_);
        }
        combQue_.FreeTensor(combUb);
        postQue_.FreeTensor(postUb);
    }
}

}
#endif