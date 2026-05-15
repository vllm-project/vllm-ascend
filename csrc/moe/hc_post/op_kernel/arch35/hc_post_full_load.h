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
 * \file hc_post_full_load.h
 * \brief
 */
#ifndef HC_POST_FULL_LOAD_H
#define HC_POST_FULL_LOAD_H

#include "kernel_operator.h"

namespace HcPostRegBase {
using namespace AscendC;

template <typename T1, typename T2>
class HcPostRegBaseFullLoad {
public:
    __aicore__ inline HcPostRegBaseFullLoad() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y, GM_ADDR workspace,
        const HcPostTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void DataCopyInX(int64_t batchIndex);
    __aicore__ inline void DataCopyInPost(int64_t batchIndex);
    __aicore__ inline void DataCopyInResidual(int64_t batchIndex);
    __aicore__ inline void DataCopyInComb(int64_t batchIndex);
    __aicore__ inline void DataCopyOut(int64_t batchIndex, int64_t hcIndex);
    __aicore__ inline void doProcess(int64_t batchSize);
    __aicore__ inline void doPostMulX(LocalTensor<T1> xUb, LocalTensor<T2> postUb, LocalTensor<float> sumTempBuf, int64_t hcIndex);
    __aicore__ inline void doMulAndAdd(LocalTensor<T1> residualUb, LocalTensor<T2> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex);

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

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> residualGm_;
    GlobalTensor<T2> postGm_;
    GlobalTensor<T2> combGm_;
    GlobalTensor<T1> yGm_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> residualQue_;
    TQue<QuePosition::VECIN, 1> postQue_;
    TQue<QuePosition::VECIN, 1> combQue_;
    TQue<QuePosition::VECOUT, 1> sumQue_;
    TBuf<QuePosition::VECCALC> sumTempBuf_;

};

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y,
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
    xGm_.SetGlobalBuffer((__gm__ T1 *)x + xOffset);
    residualGm_.SetGlobalBuffer((__gm__ T1 *)residual + residualOffset);
    postGm_.SetGlobalBuffer((__gm__ T2 *)post + postOffset);
    combGm_.SetGlobalBuffer((__gm__ T2 *)comb + combOffset);
    yGm_.SetGlobalBuffer((__gm__ T1 *)y + yOffset);

    pipe_->InitBuffer(xQue_, 2, dParam_ * sizeof(T1));
    pipe_->InitBuffer(residualQue_, 2, hcParam_ * dParam_ * sizeof(T1));
    pipe_->InitBuffer(postQue_, 2, hcParam_ * sizeof(T2));
    pipe_->InitBuffer(combQue_, 2, hcParam_ * hcParam_ * sizeof(T2));
    pipe_->InitBuffer(sumQue_, 2, dParam_ * sizeof(T1));
    pipe_->InitBuffer(sumTempBuf_, dParam_ * sizeof(float));
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::Process()
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

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::DataCopyInX(int64_t batchIndex)
{
    LocalTensor<T1> xUb = xQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dParam_ * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[batchIndex * dParam_], copyParams, dataCopyPadParams);
    xQue_.EnQue<T1>(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::DataCopyInPost(int64_t batchIndex)
{
    LocalTensor<T2> postUb = postQue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(postUb, postGm_[batchIndex * hcParam_], copyParams, dataCopyPadParams);
    postQue_.EnQue<T2>(postUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::DataCopyInResidual(int64_t batchIndex)
{
    LocalTensor<T1> residualUb = residualQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * dParam_ * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(residualUb, residualGm_[batchIndex * hcParam_ * dParam_], copyParams, dataCopyPadParams);
    residualQue_.EnQue<T1>(residualUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::DataCopyInComb(int64_t batchIndex)
{
    LocalTensor<T2> combUb = combQue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = hcParam_ * hcParam_ * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(combUb, combGm_[batchIndex * hcParam_ * hcParam_], copyParams, dataCopyPadParams);
    combQue_.EnQue<T2>(combUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::DataCopyOut(int64_t batchIndex, int64_t hcIndex)
{
    LocalTensor<T1> outBuf = sumQue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dParam_ * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGm_[batchIndex * hcParam_ * dParam_ + hcIndex * dParam_], outBuf, copyParams);
    sumQue_.FreeTensor(outBuf);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::doPostMulX(LocalTensor<T1> xUb, LocalTensor<T2> postUb, LocalTensor<float> sumTempBuf, int64_t hcIndex)
{
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (dParam_ + vfLen - 1) / vfLen;

    auto xAddr = (__ubuf__ T1*)xUb.GetPhyAddr();
    auto postAddr = (__ubuf__ T2*)postUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(dParam_);
        AscendC::MicroAPI::RegTensor<T1> xReg;
        AscendC::MicroAPI::RegTensor<T2> postReg;
        AscendC::MicroAPI::RegTensor<float> xReg32;
        AscendC::MicroAPI::RegTensor<float> postReg32;
        AscendC::MicroAPI::RegTensor<float> sumReg32;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddr+i*vfLen);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(xReg32, xReg, pMask);
            } else {
                AscendC::MicroAPI::DataCopy(xReg32, xAddr+i*vfLen);
            }
            if constexpr (sizeof(T2) == 2) {
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(postReg, postAddr + hcIndex);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(postReg32, postReg, pMask);
            } else {
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(postReg32, postAddr + hcIndex);
            }
            AscendC::MicroAPI::Mul(sumReg32, xReg32, postReg32, pMask);
            AscendC::MicroAPI::DataCopy(sumAddr+i*vfLen, sumReg32, pMask);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::doMulAndAdd(LocalTensor<T1> residualUb, LocalTensor<T2> combUb, LocalTensor<float> sumTempBuf, int64_t hcIndex)
{
    uint16_t aTimes = hcParam_;
    uint32_t xDealNumAlign = (dParam_ + perBlock32 - 1) / perBlock32 * perBlock32;
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (dParam_ + vfLen - 1) / vfLen;

    auto residualAddr = (__ubuf__ T1*)residualUb.GetPhyAddr();
    auto combAddr = (__ubuf__ T2*)combUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(dParam_);
        AscendC::MicroAPI::RegTensor<T1> residualReg;
        AscendC::MicroAPI::RegTensor<T2> combReg;
        AscendC::MicroAPI::RegTensor<float> residualReg32;
        AscendC::MicroAPI::RegTensor<float> combReg32;
        AscendC::MicroAPI::RegTensor<float> sumReg32;
        AscendC::MicroAPI::RegTensor<float> sumTempReg32;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t j = 0; j < repeatTimes; j++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            AscendC::MicroAPI::DataCopy(sumReg32, sumAddr+j*vfLen);
            for (uint16_t i = 0; i < aTimes; i++) {
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg, residualAddr+i*xDealNumAlign+j*vfLen);
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualReg32, residualReg, pMask);
                } else {
                    AscendC::MicroAPI::DataCopy(residualReg32, residualAddr+i*xDealNumAlign+j*vfLen);
                }
                if constexpr (sizeof(T2) == 2) {
                    AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg, combAddr+i*hcParam_+hcIndex);
                    AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(combReg32, combReg, pregMain);
                } else {
                    AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combReg32, combAddr+i*hcParam_+hcIndex);
                }
                AscendC::MicroAPI::Mul(sumTempReg32, residualReg32, combReg32, pMask);
                AscendC::MicroAPI::Add(sumReg32, sumReg32, sumTempReg32, pMask);
            }
            AscendC::MicroAPI::DataCopy(sumAddr+j*vfLen, sumReg32, pMask);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseFullLoad<T1, T2>::doProcess(int64_t batchSize)
{
    LocalTensor<float> sumTempBuf = sumTempBuf_.Get<float>();
    for (int64_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        DataCopyInX(batchIndex);
        DataCopyInPost(batchIndex);
        LocalTensor<T1> xUb = xQue_.DeQue<T1>();
        LocalTensor<T2> postUb = postQue_.DeQue<T2>();
        DataCopyInResidual(batchIndex);
        DataCopyInComb(batchIndex);
        LocalTensor<T1> residualUb = residualQue_.DeQue<T1>();
        LocalTensor<T2> combUb = combQue_.DeQue<T2>();
        for (int64_t hc1Index = 0; hc1Index < hcParam_; hc1Index++) {
            doPostMulX(xUb, postUb, sumTempBuf, hc1Index);
            doMulAndAdd(residualUb, combUb, sumTempBuf, hc1Index);
            LocalTensor<T1> sumUb = sumQue_.AllocTensor<T1>();
            if constexpr (sizeof(T1) == 2) {
                AscendC::Cast(sumUb, sumTempBuf, AscendC::RoundMode::CAST_RINT, dParam_);
            } else {
                AscendC::Copy(sumUb, sumTempBuf, dParam_);
            }
            sumQue_.EnQue<T1>(sumUb);
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