/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file situ_mx_quant_axis_last.h
 * \brief Regbase implementation for Situ + MX quantization (activate_dim=-1, axis=-1)
 */

#ifndef SITU_MX_QUANT_AXIS_LAST_H
#define SITU_MX_QUANT_AXIS_LAST_H

#include "situ_mx_quant_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SituMxQuant {
using namespace AscendC;

template <typename T, typename U, bool hasLinearBeta>
class SituMxQuantAxisLast {
public:
    __aicore__ inline SituMxQuantAxisLast(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
                                const SituMxQuantTilingData* __restrict tilingData, AscendC::TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t dim0Size, int64_t dim1Size, int64_t dim1AlignSize);
    __aicore__ inline void CopyIn(int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize);
    __aicore__ inline void CopyOut(int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize,
                                   int64_t dim1OnceSizeAlgin);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    const SituMxQuantTilingData* tiling_;
    AscendC::TPipe* pipe_;
    int32_t blockIdx_ = 0;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQuex_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQuey_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueScale_;

    TBuf<QuePosition::VECCALC> situBuffer_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> halfScaleBuffer_;

    int64_t realCoreNum_ = 0;
    int64_t activateLeft_ = 0;
    float beta_ = 1.0f;
    float invBeta_ = 1.0f;
    float linearBeta_ = 0.0f;
    float invLinearBeta_ = 0.0f;

    int64_t dimM_ = 0;
    int64_t dim2N_ = 0;
    int64_t dimN_ = 0;
    int64_t factorDim0Size_ = 0;
    int64_t factorDim1Size_ = 0;

    int64_t mStart_ = 0;
    int64_t nStart_ = 0;
    int64_t loopTimesPerBatch_ = 0;
    int64_t tailPerBatch_ = 0;
    int64_t loopTimesN_ = 0;
    int64_t tailN_ = 0;
    uint16_t f8Emax_ = 0;
    int64_t outputScaleRowBytes_ = 0;
};

template <typename T, typename U, bool hasLinearBeta>
__aicore__ inline void SituMxQuantAxisLast<T, U, hasLinearBeta>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
    const SituMxQuantTilingData* __restrict tilingData, AscendC::TPipe* pipe)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tiling_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxscale);

    dimM_ = tiling_->inputDim1;
    dimN_ = tiling_->inputDim2;
    dim2N_ = dimN_ * CONST_2;

    int64_t dimNBlockNum = tiling_->dimNBlockNum;
    realCoreNum_ = tiling_->usedCoreNum;
    factorDim0Size_ = tiling_->maxBasicNumUbDim1;
    factorDim1Size_ = tiling_->maxBasicNumUbDim2;

    activateLeft_ = tiling_->activateLeft;
    beta_ = tiling_->beta;
    invBeta_ = 1.0f / beta_;
    linearBeta_ = tiling_->linearBeta;
    if constexpr (hasLinearBeta) {
        invLinearBeta_ = 1.0f / linearBeta_;
    }

    // Initialize pipe buffers
    int32_t factorSize = factorDim0Size_ * factorDim1Size_;
    pipe_->InitBuffer(inQuex_, CONST_2, factorSize * X_ONCE_NUM * sizeof(T));
    pipe_->InitBuffer(outQuey_, CONST_2, (factorSize * QUANT_ONCE_NUM) * sizeof(uint8_t));
    int32_t scaleUbSize = factorSize * SCALE_ONCE_NUM;
    scaleUbSize = ((scaleUbSize + CONST_64 - 1) / CONST_64) * CONST_64;
    pipe_->InitBuffer(outQueScale_, CONST_2, scaleUbSize);

    pipe_->InitBuffer(situBuffer_, factorSize * QUANT_ONCE_NUM * sizeof(T));
    int32_t maxExpUbSize = factorSize * SCALE_ONCE_NUM * sizeof(uint16_t);
    maxExpUbSize = ((maxExpUbSize + ONE_BLOCK_UB - 1) / ONE_BLOCK_UB) * ONE_BLOCK_UB;
    pipe_->InitBuffer(maxExpBuffer_, maxExpUbSize);
    pipe_->InitBuffer(halfScaleBuffer_, maxExpUbSize);

    // Core grid distribution (axis=-1 only)
    int64_t mCorePerB = tiling_->mCorePerB;
    int64_t nCoreNum = tiling_->nCoreNum;

    int64_t nIdx = blockIdx_ % nCoreNum;
    int64_t mIdx = blockIdx_ / nCoreNum;

    int64_t mHeadCores = dimM_ % mCorePerB;
    int64_t mBase = dimM_ / mCorePerB;
    mStart_ = (mIdx < mHeadCores) ? mIdx * (mBase + 1) : mHeadCores * (mBase + 1) + (mIdx - mHeadCores) * mBase;
    int64_t mRows = (mIdx < mHeadCores) ? mBase + 1 : mBase;

    loopTimesPerBatch_ = ops::CeilDiv(mRows, factorDim0Size_);
    tailPerBatch_ = mRows - (loopTimesPerBatch_ - 1) * factorDim0Size_;

    int64_t nHeadCores = dimNBlockNum % nCoreNum;
    int64_t blockPerNCore = dimNBlockNum / nCoreNum;
    int64_t nFrontCore = blockPerNCore + 1;
    nStart_ = (nIdx < nHeadCores) ? nIdx * nFrontCore : nHeadCores * nFrontCore + (nIdx - nHeadCores) * blockPerNCore;
    int64_t loopPerCoreN = (nIdx < nHeadCores) ? nFrontCore : blockPerNCore;
    loopTimesN_ = ops::CeilDiv(loopPerCoreN, factorDim1Size_);
    if (nIdx < nCoreNum - 1) {
        tailN_ = loopPerCoreN * 256 - (loopTimesN_ - 1) * factorDim1Size_ * 256;
    } else {
        tailN_ = dimN_ - nStart_ * 256 - (loopTimesN_ - 1) * factorDim1Size_ * 256;
    }

    outputScaleRowBytes_ = ((dimN_ + 64 - 1) / 64) * 2;
    if constexpr (ops::IsSame<U, fp8_e4m3fn_t>::value) {
        f8Emax_ = FP8_E4M3_MAX_EXP;
    }
    if constexpr (ops::IsSame<U, fp8_e5m2_t>::value) {
        f8Emax_ = FP8_E5M2_MAX_EXP;
    }
}

template <typename T, typename U, bool hasLinearBeta>
__aicore__ inline void SituMxQuantAxisLast<T, U, hasLinearBeta>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    int64_t dim1Size = factorDim1Size_ * QUANT_ONCE_NUM;
    int64_t dim1AlignSize = ((tailN_ + CONST_64 - 1) / CONST_64) * CONST_64;
    for (int64_t mGroup = 0; mGroup < loopTimesPerBatch_; mGroup++) {
        int64_t dim0Size = (mGroup == loopTimesPerBatch_ - 1) ? tailPerBatch_ : factorDim0Size_;
        int64_t rowOffset = mStart_ + mGroup * factorDim0Size_;
        for (int64_t nLoop = 0; nLoop < loopTimesN_; nLoop++) {
            int64_t colOffset = nStart_ + nLoop * factorDim1Size_;
            bool isTailDim1 = (nLoop == loopTimesN_ - 1);
            int64_t dim1SizeNow = isTailDim1 ? tailN_ : dim1Size;
            int64_t dim1AlignSizeNow = isTailDim1 ? dim1AlignSize : dim1Size;
            CopyIn(rowOffset, colOffset, dim0Size, dim1SizeNow);
            Compute(dim0Size, dim1SizeNow, dim1AlignSizeNow);
            CopyOut(rowOffset, colOffset, dim0Size, dim1SizeNow, dim1AlignSizeNow);
        }
    }
}

template <typename T, typename U, bool hasLinearBeta>
__aicore__ inline void SituMxQuantAxisLast<T, U, hasLinearBeta>::Compute(
    int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    LocalTensor<T> xlocal = inQuex_.DeQue<T>();
    auto x1UbAddr = (__ubuf__ T*)xlocal.GetPhyAddr();
    auto x2UbAddr = (__ubuf__ T*)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();

    // Determine gate and up based on activateLeft
    // activateLeft=true: gate=first half (x1), up=second half (x2)
    // activateLeft=false: gate=second half (x2), up=first half (x1)
    __ubuf__ T* gateUbAddr = x1UbAddr;
    __ubuf__ T* upUbAddr = x2UbAddr;
    if (activateLeft_ == 0) {
        gateUbAddr = x2UbAddr;
        upUbAddr = x1UbAddr;
    }

    LocalTensor<T> situUb = situBuffer_.Get<T>();
    auto situUbAddr = (__ubuf__ T*)situUb.GetPhyAddr();

    // Step 1: Situ activation
    ComputeVfSitu<T, hasLinearBeta>(gateUbAddr, upUbAddr, situUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize,
                                    beta_, invBeta_, linearBeta_, invLinearBeta_);
    inQuex_.FreeTensor(xlocal);

    // Step 2: MxQuant - extract max exponent per 32-element block
    LocalTensor<uint16_t> maxExpUb = maxExpBuffer_.Get<uint16_t>();
    auto maxExpUbAddr = (__ubuf__ uint16_t*)maxExpUb.GetPhyAddr();
    ComputeVfMaxExpVfLast<T>(situUbAddr, maxExpUbAddr, dim0OnceSize, dim1AlignSize);

    // Step 3: MxQuant - compute E8M0 scale and reciprocal scale
    LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
    auto mxScaleLocalAddr = (__ubuf__ uint16_t*)mxScaleLocal.GetPhyAddr();
    LocalTensor<uint16_t> halfScaleLocal = halfScaleBuffer_.Get<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());
    ComputeScaleLast<T>(f8Emax_, maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1AlignSize);
    outQueScale_.EnQue(mxScaleLocal);

    // Step 4: MxQuant - quantize to FP8
    LocalTensor<int8_t> outLocal = outQuey_.AllocTensor<int8_t>();
    auto outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
    ComputeDataF8Last<T, U>(situUbAddr, halfScaleLocalAddr, outLocalAddr, dim0OnceSize, dim1AlignSize);
    outQuey_.EnQue(outLocal);
}

template <typename T, typename U, bool hasLinearBeta>
__aicore__ inline void SituMxQuantAxisLast<T, U, hasLinearBeta>::CopyIn(
    int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize)
{
    LocalTensor<T> xlocal = inQuex_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> copyPadParams = {false, 0, 0, 0};
    // Load two halves of input: gate (first H) and up (second H)
    // Input x shape: [..., 2H], first half = gate, second half = up
    int64_t offset = rowOffset * dim2N_ + colBlockStart * QUANT_ONCE_NUM;
    copyInParam.blockCount = dim0OnceSize;
    copyInParam.blockLen = dim1OnceSize * sizeof(T);
    copyInParam.srcStride = (dim2N_ - dim1OnceSize) * sizeof(T);
    DataCopyPad(xlocal, xGm_[offset], copyInParam, copyPadParams);
    DataCopyPad(xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM], xGm_[offset + dimN_], copyInParam,
                copyPadParams);
    inQuex_.EnQue(xlocal);
}

template <typename T, typename U, bool hasLinearBeta>
__aicore__ inline void SituMxQuantAxisLast<T, U, hasLinearBeta>::CopyOut(
    int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1OnceSizeAlgin)
{
    LocalTensor<uint8_t> mxScaleLocal = outQueScale_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQuey_.DeQue<uint8_t>();

    // Copy FP8 output
    DataCopyExtParams copyOutParamData = {0, 0, 0, 0, 0};
    copyOutParamData.blockCount = dim0OnceSize;
    copyOutParamData.blockLen = dim1OnceSize;
    copyOutParamData.srcStride = (dim1OnceSizeAlgin - copyOutParamData.blockLen) / ONE_BLOCK_UB;
    copyOutParamData.dstStride = dimN_ - copyOutParamData.blockLen;
    int64_t offset = rowOffset * dimN_ + colBlockStart * 256;
    DataCopyPad(yGm_[offset], outLocal, copyOutParamData);

    // Copy E8M0 scale output
    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    uint32_t usedFactorDim1 = dim1OnceSizeAlgin / ONE_BLOCK_UB;
    copyOutParamScale.blockCount = dim0OnceSize;
    copyOutParamScale.blockLen = usedFactorDim1;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = outputScaleRowBytes_ - copyOutParamScale.blockLen;
    int64_t offsetScale = rowOffset * outputScaleRowBytes_ + colBlockStart * SCALE_ONCE_NUM;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[offsetScale], mxScaleLocal, copyOutParamScale);

    outQuey_.FreeTensor(outLocal);
    outQueScale_.FreeTensor(mxScaleLocal);
}
} // namespace SituMxQuant
#endif // SITU_MX_QUANT_AXIS_LAST_H
