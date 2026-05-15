/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    __aicore__ inline void DataCopyInX(int64_t batchIndex);
    __aicore__ inline void DataCopyInPost(int64_t batchIndex);
    __aicore__ inline void DataCopyInResidual(int64_t batchIndex);
    __aicore__ inline void DataCopyInComb(int64_t batchIndex);
    __aicore__ inline void DataCopyOut(int64_t batchIndex, int64_t hcIndex);
    __aicore__ inline void doProcess(int64_t batchSize);
    __aicore__ inline void doPostMulX(LocalTensor<float> xUb, LocalTensor<T> postUb, LocalTensor<float> sumTempBuf, int64_t hcIndex);
    __aicore__ inline void doMulAndAdd(LocalTensor<float> residualUb, LocalTensor<T> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex);

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

    pipe_->InitBuffer(xQue_, 2, dParam_ * sizeof(float));
    pipe_->InitBuffer(residualQue_, 2, hcParam_ * dParam_ * sizeof(float));
    pipe_->InitBuffer(postQue_, 2, hcParam_ * sizeof(T));
    pipe_->InitBuffer(combQue_, 2, hcParam_ * hcParam_ * sizeof(T));
    pipe_->InitBuffer(sumQue_, 2, dParam_ * sizeof(float));
    pipe_->InitBuffer(sumTempBuf_, dParam_ * sizeof(float));
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::Process()
{
    if (blkIdx_ >= tiling_->usedCoreNum) {
        return;
    }
    if (isFrontCore_) {
        doProcess(tiling_->batchOneCore);
    } else {
        doProcess(tiling_->batchOneCoreTail);
    }
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInX(int64_t batchIndex)
{
    LocalTensor<float> xUb = xQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dParam_ * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[batchIndex * dParam_], copyParams, dataCopyPadParams);
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
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyInResidual(int64_t batchIndex)
{
    LocalTensor<float> residualUb = residualQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * dParam_ * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(residualUb, residualGm_[batchIndex * hcParam_ * dParam_], copyParams, dataCopyPadParams);
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
__aicore__ inline void HcPostRegBaseFloat32<T>::DataCopyOut(int64_t batchIndex, int64_t hcIndex)
{
    LocalTensor<float> outBuf = sumQue_.DeQue<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dParam_ * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGm_[batchIndex * hcParam_ * dParam_ + hcIndex * dParam_], outBuf, copyParams);
    sumQue_.FreeTensor(outBuf);
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::doPostMulX(LocalTensor<float> xUb, LocalTensor<T> postUb, LocalTensor<float> sumTempBuf, int64_t hcIndex)
{
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (dParam_ + vfLen - 1) / vfLen;

    auto xAddr = (__ubuf__ float*)xUb.GetPhyAddr();
    auto postAddr = (__ubuf__ T*)postUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(dParam_);
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<T> postReg;
        AscendC::MicroAPI::RegTensor<float> xRegFloat;
        AscendC::MicroAPI::RegTensor<float> postRegFloat;
        AscendC::MicroAPI::RegTensor<float> sumRegFloat;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        if constexpr (sizeof(T) == 2) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(postReg, postAddr + hcIndex);
            AscendC::MicroAPI::Cast<float, T, castB16ToB32>(postRegFloat, postReg, pregMain);
        } else {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(postRegFloat, postAddr + hcIndex);
        }
        for (uint16_t i = 0; i < repeatTimes; i++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            AscendC::MicroAPI::DataCopy(xRegFloat, xAddr+i*vfLen);
            AscendC::MicroAPI::Mul(sumRegFloat, xRegFloat, postRegFloat, pMask);
            AscendC::MicroAPI::DataCopy(sumAddr+i*vfLen, sumRegFloat, pMask);
        }
    }
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::doMulAndAdd(LocalTensor<float> residualUb, LocalTensor<T> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex)
{
    uint16_t aTimes = hcParam_;
    uint32_t xDealNumAlign = (dParam_ + perBlock32 - 1) / perBlock32 * perBlock32;
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (dParam_ + vfLen - 1) / vfLen;

    auto residualAddr = (__ubuf__ float*)residualUb.GetPhyAddr();
    auto combAddr = (__ubuf__ T*)combUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(dParam_);
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
        } else {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat0, combAddr+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat1, combAddr+hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat2, combAddr+2*hcParam_+hcIndex);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat3, combAddr+3*hcParam_+hcIndex);
        }

        for (uint16_t j = 0; j < repeatTimes; j++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            AscendC::MicroAPI::DataCopy(sumRegFloat, sumAddr+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat0, residualAddr+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat1, residualAddr+xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat2, residualAddr+2*xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::DataCopy(residualRegFloat3, residualAddr+3*xDealNumAlign+j*vfLen);
            AscendC::MicroAPI::Mul(sumTempReg0, residualRegFloat0, combRegFloat0, pMask);
            AscendC::MicroAPI::Mul(sumTempReg1, residualRegFloat1, combRegFloat1, pMask);
            AscendC::MicroAPI::Mul(sumTempReg2, residualRegFloat2, combRegFloat2, pMask);
            AscendC::MicroAPI::Mul(sumTempReg3, residualRegFloat3, combRegFloat3, pMask);
            AscendC::MicroAPI::Add(sumTempReg0, sumTempReg0, sumTempReg1, pMask);
            AscendC::MicroAPI::Add(sumTempReg2, sumTempReg2, sumTempReg3, pMask);
            AscendC::MicroAPI::Add(sumRegFloat, sumRegFloat, sumTempReg0, pMask);
            AscendC::MicroAPI::Add(sumRegFloat, sumRegFloat, sumTempReg2, pMask);
            AscendC::MicroAPI::DataCopy(sumAddr+j*vfLen, sumRegFloat, pMask);
        }
    }
}

template <typename T>
__aicore__ inline void HcPostRegBaseFloat32<T>::doProcess(int64_t batchSize)
{
    LocalTensor<float> sumTempBuf = sumTempBuf_.Get<float>();
    for (int64_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        DataCopyInX(batchIndex);
        DataCopyInPost(batchIndex);
        LocalTensor<float> xUb = xQue_.DeQue<float>();
        LocalTensor<T> postUb = postQue_.DeQue<T>();
        DataCopyInResidual(batchIndex);
        DataCopyInComb(batchIndex);
        LocalTensor<float> residualUb = residualQue_.DeQue<float>();
        LocalTensor<T> combUb = combQue_.DeQue<T>();
        for (int64_t hc1Index = 0; hc1Index < hcParam_; hc1Index++) {
            doPostMulX(xUb, postUb, sumTempBuf, hc1Index);
            doMulAndAdd(residualUb, combUb, sumTempBuf, hc1Index);
            LocalTensor<float> sumUb = sumQue_.AllocTensor<float>();
            AscendC::Copy(sumUb, sumTempBuf, dParam_);
            sumQue_.EnQue<float>(sumUb);
            DataCopyOut(batchIndex, hc1Index);
        }
        combQue_.FreeTensor(combUb);
        residualQue_.FreeTensor(residualUb);
        postQue_.FreeTensor(postUb);
        xQue_.FreeTensor(xUb);
    }
}

}
#endif