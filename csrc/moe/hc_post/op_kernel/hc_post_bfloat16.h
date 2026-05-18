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
 * \file hc_post_bfloat16.h
 * \brief
 */
#ifndef HC_POST_BFLOAT16_H
#define HC_POST_BFLOAT16_H

#include "kernel_operator.h"

namespace HcPostRegBase {
using namespace AscendC;

template <typename T1, typename T2>
class HcPostRegBaseBfloat16 {
public:
    __aicore__ inline HcPostRegBaseBfloat16() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y, GM_ADDR workspace,
        const HcPostTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void DataCopyInX(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DataCopyInPost(int64_t batchIndex);
    __aicore__ inline void DataCopyInResidual(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DataCopyInComb(int64_t batchIndex);
    __aicore__ inline void DataCopyOut(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset);
    __aicore__ inline void DoProcess(int64_t batchSize);
    __aicore__ inline void DoCompute(LocalTensor<float> sumTempBuf, LocalTensor<T2> postUb, LocalTensor<T2> combUb, int64_t batchIndex, int64_t dOffset, int64_t dDealing);
    __aicore__ inline void DoMulAndAdd(LocalTensor<T1> xUb, LocalTensor<T2> postUb, LocalTensor<T1> residualUb, LocalTensor<T2> combUb, LocalTensor<float> sumTempBuf, int64_t dOnceDealing);

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
    int64_t dOnceDealing_ = 0;
    int64_t dLastDealing_ = 0;
    int64_t dSplitTime_ = 0;
    int64_t dParamOnceAlign_ = 0;
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
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y,
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
    xGm_.SetGlobalBuffer((__gm__ T1 *)x + xOffset);
    residualGm_.SetGlobalBuffer((__gm__ T1 *)residual + residualOffset);
    postGm_.SetGlobalBuffer((__gm__ T2 *)post + postOffset);
    combGm_.SetGlobalBuffer((__gm__ T2 *)comb + combOffset);
    yGm_.SetGlobalBuffer((__gm__ T1 *)y + yOffset);

    pipe_->InitBuffer(xQue_, 2, dParamOnceAlign_ * sizeof(T1));
    pipe_->InitBuffer(residualQue_, 2, hcParam_ * dParamOnceAlign_ * sizeof(T1));
    pipe_->InitBuffer(postQue_, 2, hcParam_ * sizeof(T2));
    pipe_->InitBuffer(combQue_, 2, hcParam_ * hcParam_ * sizeof(T2));
    pipe_->InitBuffer(sumQue_, 2, hcParam_* dParamOnceAlign_ * sizeof(T1));
    pipe_->InitBuffer(sumTempBuf_, hcParam_ * dParamOnceAlign_ * sizeof(float));
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::Process()
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

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DataCopyInX(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<T1> xUb = xQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dOnceDealing * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[batchIndex * dParam_ + dOffset], copyParams, dataCopyPadParams);
    xQue_.EnQue<T1>(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DataCopyInPost(int64_t batchIndex)
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
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DataCopyInResidual(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<T1> residualUb = residualQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = hcParam_;
    copyParams.blockLen = dOnceDealing * sizeof(T1);
    copyParams.srcStride = (dParamAlign_ - dOnceDealing) * sizeof(T1);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(residualUb, residualGm_[batchIndex * hcParam_ * dParam_ + dOffset], copyParams, dataCopyPadParams);
    residualQue_.EnQue<T1>(residualUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DataCopyInComb(int64_t batchIndex)
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
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DataCopyOut(int64_t batchIndex, int64_t dOnceDealing, int64_t dOffset)
{
    LocalTensor<T1> outBuf = sumQue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = hcParam_;
    copyParams.blockLen = dOnceDealing * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = (dParamAlign_ - dOnceDealing) * sizeof(T1);
    AscendC::DataCopyPad(yGm_[batchIndex * hcParam_ * dParam_ + dOffset], outBuf, copyParams);
    sumQue_.FreeTensor(outBuf);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DoMulAndAdd(LocalTensor<T1> xUb, LocalTensor<T2> postUb, LocalTensor<T1> residualUb, LocalTensor<T2> combUb, LocalTensor<float> sumTempBuf, int64_t dOnceDealing)
{
    uint16_t aTimes = hcParam_;
    uint32_t xDealNumAlign = (dOnceDealing + perBlock32 - 1) / perBlock32 * perBlock32;
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = dOnceDealing / vfLen;
    uint16_t hcTimes = hcParam_;
    uint32_t tailNum = dOnceDealing % vfLen;
    uint16_t tailLoopTimes = tailNum == 0 ? 0 : 1;

    auto residualAddr = (__ubuf__ T1*)residualUb.GetPhyAddr();
    auto combAddr = (__ubuf__ T2*)combUb.GetPhyAddr();
    auto sumAddr = (__ubuf__ float*)sumTempBuf.GetPhyAddr();
	auto xAddr = (__ubuf__ T1*)xUb.GetPhyAddr();
    auto postAddr = (__ubuf__ T2*)postUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t xDealNum = static_cast<uint32_t>(hcParam_ * dOnceDealing);
		AscendC::MicroAPI::RegTensor<T1> xReg;
        AscendC::MicroAPI::RegTensor<T2> postReg;
        AscendC::MicroAPI::RegTensor<float> xRegFloat;
        AscendC::MicroAPI::RegTensor<float> postRegFloat;
        AscendC::MicroAPI::RegTensor<T1> residualReg0;
        AscendC::MicroAPI::RegTensor<T1> residualReg1;
        AscendC::MicroAPI::RegTensor<T1> residualReg2;
        AscendC::MicroAPI::RegTensor<T1> residualReg3;
        AscendC::MicroAPI::RegTensor<T2> combReg0;
        AscendC::MicroAPI::RegTensor<T2> combReg1;
        AscendC::MicroAPI::RegTensor<T2> combReg2;
        AscendC::MicroAPI::RegTensor<T2> combReg3;
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
        for (uint16_t hcIndex = 0; hcIndex < hcTimes; hcIndex++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(xDealNum);
            if constexpr (sizeof(T2) == 2) {
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg0, combAddr+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg1, combAddr+hcParam_+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg2, combAddr+2*hcParam_+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(combReg3, combAddr+3*hcParam_+hcIndex);
				AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(postReg, postAddr + hcIndex);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(combRegFloat0, combReg0, pregMain);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(combRegFloat1, combReg1, pregMain);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(combRegFloat2, combReg2, pregMain);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(combRegFloat3, combReg3, pregMain);
                AscendC::MicroAPI::Cast<float, T2, castB16ToB32>(postRegFloat, postReg, pregMain);
            } else {
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat0, combAddr+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat1, combAddr+hcParam_+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat2, combAddr+2*hcParam_+hcIndex);
                AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(combRegFloat3, combAddr+3*hcParam_+hcIndex);
				AscendC::MicroAPI::DataCopy<T2, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(postRegFloat, postAddr + hcIndex);
            }

            for (uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddr+j*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg0, residualAddr+j*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg1, residualAddr+xDealNumAlign+j*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg2, residualAddr+2*xDealNumAlign+j*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg3, residualAddr+3*xDealNumAlign+j*vfLen);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat0, residualReg0, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat1, residualReg1, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat2, residualReg2, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat3, residualReg3, pMask);
				AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(xRegFloat, xReg, pMask);
                AscendC::MicroAPI::Mul(sumTempReg0, residualRegFloat0, combRegFloat0, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat3, combRegFloat3, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat1, combRegFloat1, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat2, combRegFloat2, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, xRegFloat, postRegFloat, pMask);
                AscendC::MicroAPI::DataCopy(sumAddr+hcIndex*xDealNumAlign+j*vfLen, sumTempReg0, pMask);
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddr+repeatTimes*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg0, residualAddr+repeatTimes*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg1, residualAddr+xDealNumAlign+repeatTimes*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg2, residualAddr+2*xDealNumAlign+repeatTimes*vfLen);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualReg3, residualAddr+3*xDealNumAlign+repeatTimes*vfLen);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat0, residualReg0, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat1, residualReg1, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat2, residualReg2, pMask);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(residualRegFloat3, residualReg3, pMask);
				AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(xRegFloat, xReg, pMask);
                AscendC::MicroAPI::Mul(sumTempReg0, residualRegFloat0, combRegFloat0, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat3, combRegFloat3, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat1, combRegFloat1, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, residualRegFloat2, combRegFloat2, pMask);
                AscendC::MicroAPI::MulAddDst(sumTempReg0, xRegFloat, postRegFloat, pMask);
                AscendC::MicroAPI::DataCopy(sumAddr+hcIndex*xDealNumAlign+repeatTimes*vfLen, sumTempReg0, pMask);
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DoCompute(LocalTensor<float> sumTempBuf, LocalTensor<T2> postUb, LocalTensor<T2> combUb, int64_t batchIndex, int64_t dOffset, int64_t dDealing)
{
    DataCopyInX(batchIndex, dDealing, dOffset);
    LocalTensor<T1> xUb = xQue_.DeQue<T1>();
    DataCopyInResidual(batchIndex, dDealing, dOffset);
    LocalTensor<T1> residualUb = residualQue_.DeQue<T1>();
    DoMulAndAdd(xUb, postUb, residualUb, combUb, sumTempBuf, dDealing);
    LocalTensor<T1> sumUb = sumQue_.AllocTensor<T1>();
    AscendC::Cast(sumUb, sumTempBuf, AscendC::RoundMode::CAST_RINT, hcParam_ * dOnceDealing_);
    sumQue_.EnQue<T1>(sumUb);
    DataCopyOut(batchIndex, dDealing, dOffset);
    residualQue_.FreeTensor(residualUb);
    xQue_.FreeTensor(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void HcPostRegBaseBfloat16<T1, T2>::DoProcess(int64_t batchSize)
{
    LocalTensor<float> sumTempBuf = sumTempBuf_.Get<float>();
    for (int64_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        DataCopyInPost(batchIndex);
        LocalTensor<T2> postUb = postQue_.DeQue<T2>();
        DataCopyInComb(batchIndex);
        LocalTensor<T2> combUb = combQue_.DeQue<T2>();
        int64_t dOffset = 0;
        for (int64_t dIndex = 0; dIndex < dSplitTime_; dIndex++) {
            dOffset = dIndex*dOnceDealing_;
            DoCompute(sumTempBuf, postUb, combUb, batchIndex, dOffset, dOnceDealing_);
        }
        if (dLastDealing_ != 0) {
            dOffset = dSplitTime_ * dOnceDealing_;
            DoCompute(sumTempBuf, postUb, combUb, batchIndex, dOffset, dLastDealing_);
        }
        combQue_.FreeTensor(combUb);
        postQue_.FreeTensor(postUb);
    }
}

}
#endif