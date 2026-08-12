/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_weight_nz_performance_base.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_BASE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_constant.h"

using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::UnaryRepeatParams;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
class Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel(){};

    __aicore__ inline void InitInputs(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                      GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y);
    __aicore__ inline void InitUbOffset();
    __aicore__ inline void InitTilingData();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2NzTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void CopyInBias(LocalTensor<xType> biasLocal, int64_t biasSrcOffset, int32_t biasDstOffset,
                                      int32_t biasLen);
    __aicore__ inline void CopyInAddMul(int64_t addOffset, int32_t realNLen, int64_t groupIdx);
    __aicore__ inline void CopyInTensorAub(int64_t ubOffset, int32_t blockCount, int32_t blockLen, int32_t srcStride,
                                           LocalTensor<xType> xLocal);
    __aicore__ inline void AntiQuantCompute(int32_t bubNLen, int32_t bubKLen, int32_t antiquanOffset,
                                            LocalTensor<half> weight, int64_t bubKOffset);
    __aicore__ inline void TransNd2Nz(LocalTensor<xType> srcUbLocal, LocalTensor<xType> dstUbLocal, int32_t outerDim,
                                      int32_t innerDim, int32_t innerLen);
    __aicore__ inline void CopyVecOut2L1(LocalTensor<xType> l1Local, LocalTensor<xType> ubLocal, int32_t blockCount,
                                         int32_t blockLen, int32_t dstStride);
    __aicore__ inline void PostProcess(int32_t mL0Len, int32_t nL0Len, int32_t mAL1Offset, int32_t nBL1Offset);
    __aicore__ inline void CopyVec2Out(int64_t yGmOffset, int64_t mReal, int64_t nReal, LocalTensor<yType> resCNd);
    __aicore__ inline void TransNz2Nd(LocalTensor<xType> cubLocal);
    __aicore__ inline void BiasProcess(LocalTensor<float> biasFp32Local, int64_t nBlockOffset);
    __aicore__ inline void AL1Process(LocalTensor<xType> aL1Local, int64_t mAL1Offset, int64_t kBlockOffset,
                                      int32_t kL1Len, int32_t mL0Len);
    __aicore__ inline void BL1Process(LocalTensor<xType> bL1Local, int64_t nBL1Offset, int64_t kBlockOffset,
                                      int32_t kL1Len, int32_t nL0Len);
    __aicore__ inline void BL1TailProcess(LocalTensor<xType> bL1Local, int64_t nBL1Offset, int64_t kBlockOffset,
                                          int32_t nL0Len, int32_t bubNLoopIdx, int32_t kLoopNum,
                                          LocalTensor<wType> wLocalPing, LocalTensor<wType> wLocalPong,
                                          LocalTensor<half> weight16Ping, LocalTensor<half> weight16Pong);
    __aicore__ inline void BL1SubProcess(LocalTensor<wType> wLocalPing, LocalTensor<half> weight16Ping,
                                         LocalTensor<xType> bL1Local, int64_t bubKOffset, int64_t bubNOffset,
                                         int32_t blockCount, int32_t blockLen, int32_t srcStride, int32_t bubNLen,
                                         int32_t bubKLen, int32_t loopIdx, int32_t loopMax, int32_t bubNLoopIdx,
                                         int32_t bubKLoopIdx);
    __aicore__ inline void CubeProcess(LocalTensor<xType> aL1Local, LocalTensor<xType> bL1Local,
                                       LocalTensor<float> biasFp32Local, int32_t mL0Len, int32_t nL0Len, int32_t kL1Len,
                                       int32_t kFactorIdx);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline int64_t CeilDiv(int64_t x, int64_t y);
    __aicore__ inline int32_t max(int32_t x, int32_t y);
    __aicore__ inline int32_t min(int32_t x, int32_t y);

    using inputXType = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, aTrans>;
    using inputWType = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, bTrans>;
    using outputYType = MatmulType<TPosition::VECIN, CubeFormat::NZ, yType>;
    using inputBiasType = MatmulType<TPosition::VECOUT, CubeFormat::ND, float>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj_;

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2NzTilingData *tiling_;

    bool biasFlag_ = false;
    bool isTailNBlock_;
    bool isTailMBlock_;
    bool isMAligned_;
    bool isKaAligned_;
    bool isKbAligned_;
    bool isNAligned_;
    bool innerAxisUnAlignedA_;
    bool innerAxisAlignedOpti_;
    bool innerAxisUnAlignedB_;

    xType scaleValue_;
    xType offsetValue_;

    int32_t curBlockIdx_;
    int32_t nDimIdx_;
    int32_t mDimIdx_;
    int32_t mCoreLoopNum_;
    int32_t nCoreLoopNum_;

    int64_t mL0Size_;
    int64_t kL0Size_;
    int64_t nL0Size_;
    int64_t al0DataSize_;
    int64_t bl0DataSize_;
    int64_t groupNum_;

    int64_t mAubSize_;
    int64_t kAubSize_;
    int64_t nBubSize_;
    int64_t kBubSize_;
    int64_t groupBubSize_;
    int64_t mCubSize_;
    int64_t nCubSize_;
    int64_t aubDataSize_;
    int64_t bubDataSize_;

    int64_t mAL1Size_;
    int64_t kAL1Size_;
    int64_t nBL1Size_;
    int64_t kBL1Size_;
    int64_t kL1Size_;
    int64_t aL1DataSize_;
    int64_t bL1DataSize_;

    int64_t mSingleCore_;
    int64_t kSingleCore_;
    int64_t nSingleCore_;

    int64_t mTailSize_;
    int64_t nTailSize_;

    int64_t mFactorTail_;
    int64_t kFactorTail_;
    int64_t nFactorTail_;

    int64_t aubMFactor_;
    int64_t aubKFactor_;
    int64_t bubNFactor_;
    int64_t bubKFactor_;

    int64_t mAL1Factor_;
    int64_t nBL1Factor_;
    int64_t kL1Factor_;

    int64_t mBlockOffset_;
    int64_t nBlockOffset_;

    // A矩阵TBuf分配
    int64_t elemsAubCopyInPing_;
    int64_t offsetAubCopyInPing_;

    int64_t elemsAubCopyInPong_;
    int64_t offsetAubCopyInPong_;

    int64_t elemsAubNd2NzPing_;
    int64_t offsetAubNd2NzPing_;

    int64_t elemsAubNd2NzPong_;
    int64_t offsetAubNd2NzPong_;

    // B矩阵TBuf分配
    int64_t elemsBubCopyInPing_;
    int64_t offsetBubCopyInPing_;

    int64_t elemsBubWeight16Ping_;
    int64_t offsetBubWeight16Ping_;

    int64_t elemsBubCopyInPong_;
    int64_t offsetBubCopyInPong_;

    int64_t elemsBubWeight16Pong_;
    int64_t offsetBubWeight16Pong_;

    // Antiquant参数TBuf分配
    int64_t elemsAddCopyIn_;
    int64_t offsetAddCopyIn_;

    int64_t elemsMulCopyIn_;
    int64_t offsetMulCopyIn_;

    int64_t elemsAddCompute_;
    int64_t offsetAddCompute_;
    int64_t elemsMulCompute_;
    int64_t offsetMulCompute_;

    // C矩阵TBuf分配
    int64_t resCNzElem_;
    int64_t resCNzOffset_;

    int64_t resCNdElem_;
    int64_t resCNdOffset_;

    int64_t elemsBiasCopyIn_;
    int64_t offsetBiasCopyIn_;

    int64_t elemsBiasFp32_;
    int64_t offsetBiasFp32_;

    GM_ADDR xGM_;
    GM_ADDR weightGM_;
    GM_ADDR biasGM_;
    GM_ADDR yGM_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> addGlobal_;
    GlobalTensor<xType> mulGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;

    TBuf<TPosition::A1> InBufAL1_;
    TBuf<TPosition::B1> InBufBL1_;
    TBuf<> apiTmpBuf_;

protected:
    static constexpr int32_t BLOCK_REDUCE_HALF = 16;
    static constexpr int32_t BLOCK_REDUCE_INT8 = 32;
    static constexpr int32_t REPEAT_BLOCK_NUM = 8;
    static constexpr int32_t BLOCK_CUBE = 16;
    static constexpr int32_t FOUR_BANK = 128 * 128;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::InitInputs(GM_ADDR x, GM_ADDR weight,
                                                                                         GM_ADDR antiquantScale,
                                                                                         GM_ADDR antiquantOffset,
                                                                                         GM_ADDR quantScale,
                                                                                         GM_ADDR quantOffset,
                                                                                         GM_ADDR bias, GM_ADDR y)
{
    xGM_ = x;
    weightGM_ = weight;
    biasGM_ = bias;
    yGM_ = y;

    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x), tiling_->mSize * tiling_->kSize);
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight), tiling_->kSize * tiling_->nSize);
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), tiling_->mSize * tiling_->nSize);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling_->nSize);

    groupNum_ = 1;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        groupNum_ = CeilDiv(tiling_->kSize, tiling_->groupSize);
    }
    mulGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), groupNum_ * tiling_->nSize);
    addGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), groupNum_ * tiling_->nSize);

    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        scaleValue_ = mulGlobal_.GetValue(0);
        if constexpr (hasAntiQuantOffset) {
            offsetValue_ = addGlobal_.GetValue(0);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::InitUbOffset()
{
    elemsAubCopyInPing_ = aubDataSize_;
    offsetAubCopyInPing_ = 0;

    // B矩阵TBuf分配
    elemsBubCopyInPing_ = bubDataSize_;
    offsetBubCopyInPing_ = elemsAubCopyInPing_ * 2;

    elemsBubWeight16Ping_ = bubDataSize_;
    offsetBubWeight16Ping_ = offsetBubCopyInPing_ + elemsBubCopyInPing_ * sizeof(wType);

    elemsBubCopyInPong_ = bubDataSize_;
    offsetBubCopyInPong_ = offsetBubWeight16Ping_ + elemsBubWeight16Ping_ * sizeof(half);

    elemsBubWeight16Pong_ = bubDataSize_;
    offsetBubWeight16Pong_ = offsetBubCopyInPong_ + elemsBubCopyInPong_ * sizeof(wType);

    // Antiquant TBuf分配
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        elemsAddCopyIn_ = nBubSize_;
        offsetAddCopyIn_ = offsetBubWeight16Pong_ + elemsBubWeight16Pong_ * sizeof(half);

        elemsMulCopyIn_ = nBubSize_;
        offsetMulCopyIn_ = offsetAddCopyIn_ + elemsAddCopyIn_ * sizeof(half);

        elemsAddCompute_ = nBubSize_ * groupBubSize_ * 16;
        offsetAddCompute_ = offsetMulCopyIn_ + elemsMulCopyIn_ * sizeof(half);

        elemsMulCompute_ = nBubSize_ * groupBubSize_ * 16;
        offsetMulCompute_ = offsetAddCompute_ + elemsAddCompute_ * sizeof(half);
    } else {
        elemsAddCopyIn_ = nBL1Size_;
        offsetAddCopyIn_ = offsetBubWeight16Pong_ + elemsBubWeight16Pong_ * sizeof(half);

        elemsMulCopyIn_ = nBL1Size_;
        offsetMulCopyIn_ = offsetAddCopyIn_ + elemsAddCopyIn_ * sizeof(half);

        elemsAddCompute_ = nBL1Size_ * 16;
        offsetAddCompute_ = offsetMulCopyIn_ + elemsMulCopyIn_ * sizeof(half);

        elemsMulCompute_ = nBL1Size_ * 16;
        offsetMulCompute_ = offsetAddCompute_ + elemsAddCompute_ * sizeof(half);
    }

    elemsAubCopyInPong_ = aubDataSize_;
    offsetAubCopyInPong_ = offsetMulCompute_ + elemsMulCompute_ * sizeof(half);

    elemsAubNd2NzPing_ = aubDataSize_;
    offsetAubNd2NzPing_ = offsetAubCopyInPong_ + elemsAubCopyInPong_ * sizeof(xType);

    elemsAubNd2NzPong_ = aubDataSize_;
    offsetAubNd2NzPong_ = offsetAubNd2NzPing_ + elemsAubNd2NzPing_ * sizeof(xType);

    elemsBiasCopyIn_ = nBL1Size_;
    offsetBiasCopyIn_ = offsetAubNd2NzPong_ + elemsAubNd2NzPong_ * 2;

    elemsBiasFp32_ = nBL1Size_;
    offsetBiasFp32_ = offsetBiasCopyIn_ + elemsBiasCopyIn_ * 2;

    resCNzElem_ = mL0Size_ * tiling_->matmulTiling.baseN;
    resCNzOffset_ = 0;

    resCNdElem_ = mL0Size_ * tiling_->matmulTiling.baseN;
    resCNdOffset_ = resCNzOffset_ + resCNzElem_ * sizeof(yType);

    int64_t endOffset = resCNdOffset_ + resCNdElem_ * sizeof(yType);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::InitTilingData()
{
    // core tiling_ data
    mSingleCore_ = CeilDiv(tiling_->mSize, tiling_->cubeBlockDimM * tiling_->mAL1Size);
    kSingleCore_ = CeilDiv(tiling_->kSize, max(tiling_->kAL1Size, tiling_->kBL1Size));
    nSingleCore_ = CeilDiv(tiling_->nSize, tiling_->cubeBlockDimN * tiling_->nBL1Size);
    mTailSize_ = tiling_->mSize - (tiling_->cubeBlockDimM - 1) * mSingleCore_ * tiling_->mAL1Size;
    nTailSize_ = tiling_->nSize - (tiling_->cubeBlockDimN - 1) * nSingleCore_ * tiling_->nBL1Size;
    nL0Size_ = tiling_->matmulTiling.baseN;
    nBL1Size_ = tiling_->nBL1Size;

    // L0 tiling_ data
    mL0Size_ = tiling_->matmulTiling.baseM;
    kL0Size_ = tiling_->matmulTiling.baseK;

    al0DataSize_ = mL0Size_ * kL0Size_;
    bl0DataSize_ = kL0Size_ * nL0Size_;

    // ub tiling_ data
    mAubSize_ = tiling_->mAubSize;
    kAubSize_ = tiling_->kAubSize;
    nBubSize_ = tiling_->nBubSize;
    kBubSize_ = tiling_->kBubSize;
    groupBubSize_ = 1;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        groupBubSize_ = CeilDiv(kBubSize_, tiling_->groupSize) + 1;
    }
    mCubSize_ = mL0Size_;
    nCubSize_ = tiling_->matmulTiling.baseN;
    aubDataSize_ = CeilDiv(mAubSize_ * kAubSize_, FOUR_BANK) * FOUR_BANK;
    bubDataSize_ = nBubSize_ * CeilDiv(kBubSize_, 32) * 32;

    // L1 tiling_ data
    mAL1Size_ = tiling_->mAL1Size;
    kAL1Size_ = tiling_->kAL1Size;

    kBL1Size_ = tiling_->kBL1Size;
    kL1Size_ = min(kAL1Size_, kBL1Size_);
    aL1DataSize_ = mAL1Size_ * kAL1Size_;
    bL1DataSize_ = nBL1Size_ * kBL1Size_;

    kL1Factor_ = kL1Size_ / kL0Size_;
    aubMFactor_ = mL0Size_ / mAubSize_;
    aubKFactor_ = kL1Size_ / kAubSize_;
    bubNFactor_ = nL0Size_ / nBubSize_;
    bubKFactor_ = CeilDiv(kL1Size_, kBubSize_);

    mAL1Factor_ = mAL1Size_ / mL0Size_;
    nBL1Factor_ = nBL1Size_ / nL0Size_;

    mCoreLoopNum_ = isTailMBlock_ ? CeilDiv(mTailSize_, mAL1Size_) : mSingleCore_;
    nCoreLoopNum_ = isTailNBlock_ ? CeilDiv(nTailSize_, nBL1Size_) : nSingleCore_;

    mBlockOffset_ = mDimIdx_ * mSingleCore_ * mAL1Size_;
    nBlockOffset_ = nDimIdx_ * nSingleCore_ * tiling_->nBL1Size;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
    hasAntiQuantOffset>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                              GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                              const Mc2WeightQuantBatchMatmulV2NzTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;

    biasFlag_ = static_cast<bool>(tiling_->matmulTiling.isBias);
    curBlockIdx_ = GetBlockIdx();
    nDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimM;
    mDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimM;
    isTailNBlock_ = nDimIdx_ == tiling_->cubeBlockDimN - 1;
    isTailMBlock_ = mDimIdx_ == tiling_->cubeBlockDimM - 1;
    isMAligned_ = tiling_->mSize % BLOCK_CUBE == 0;
    isKaAligned_ = tiling_->kSize % BLOCK_REDUCE_HALF == 0;
    isKbAligned_ = tiling_->kSize % BLOCK_REDUCE_INT8 == 0;
    isNAligned_ = tiling_->nSize % BLOCK_REDUCE_INT8 == 0;
    innerAxisUnAlignedA_ = aTrans ? !isMAligned_ : !isKaAligned_;
    innerAxisAlignedOpti_ = bTrans ? tiling_->kBubSize == tiling_->kSize : tiling_->nBubSize == tiling_->nSize;
    innerAxisUnAlignedB_ = (bTrans ? !isKbAligned_ : !isNAligned_) && !innerAxisAlignedOpti_;

    InitTilingData();

    InitInputs(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);

    uint32_t ubLength =
        tiling_->matmulTiling.isBias ? 256 * 1024 - tiling_->matmulTiling.baseN * sizeof(float) : 256 * 1024;
    pipe_->InitBuffer(apiTmpBuf_, ubLength);
    pipe_->InitBuffer(InBufAL1_, aL1DataSize_ * sizeof(xType));
    pipe_->InitBuffer(InBufBL1_, bL1DataSize_ * sizeof(xType));

    InitUbOffset();

    mmObj_.Init(&tiling_->matmulTiling, tPipe);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CopyInBias(LocalTensor<xType> biasLocal,
                                                                                         int64_t biasSrcOffset,
                                                                                         int32_t biasDstOffset,
                                                                                         int32_t biasLen)
{
    DataCopyParams intriParamsBias;
    intriParamsBias.blockCount = 1;
    intriParamsBias.blockLen = CeilDiv(biasLen, BLOCK_CUBE);
    intriParamsBias.srcStride = 0;
    intriParamsBias.dstStride = 0;
    DataCopy(biasLocal[biasDstOffset], biasGlobal_[biasSrcOffset], intriParamsBias);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CopyInAddMul(int64_t addOffset,
                                                                                           int32_t realNLen,
                                                                                           int64_t groupIdx)
{
    DataCopyParams intriParamsAdd;
    intriParamsAdd.blockCount = 1;
    intriParamsAdd.blockLen = CeilDiv(realNLen, BLOCK_CUBE);
    intriParamsAdd.srcStride = 0;
    intriParamsAdd.dstStride = 0;
    LocalTensor<half> mulLocal = apiTmpBuf_.template GetWithOffset<half>(elemsMulCopyIn_, offsetMulCopyIn_);
    DataCopy(mulLocal, mulGlobal_[addOffset], intriParamsAdd);

    event_t eventIdMte2ToV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV1);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV1);

    uint32_t mulComputeSize = elemsMulCompute_;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        mulComputeSize = nBubSize_ * 16;
    }
    LocalTensor<xType> mulTensor =
        apiTmpBuf_.template GetWithOffset<xType>(mulComputeSize, offsetMulCompute_ + nBubSize_ * groupIdx * 16);
    Brcb(mulTensor, mulLocal, CeilDiv(realNLen, 8), {1, 8});

    if constexpr (hasAntiQuantOffset) {
        LocalTensor<half> addLocal = apiTmpBuf_.template GetWithOffset<half>(elemsAddCopyIn_, offsetAddCopyIn_);
        DataCopy(addLocal, addGlobal_[addOffset], intriParamsAdd);
        uint32_t addComputeSize = elemsAddCompute_;
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
            addComputeSize = nBubSize_ * 16;
        }
        LocalTensor<xType> addTensor =
            apiTmpBuf_.template GetWithOffset<xType>(addComputeSize, offsetAddCompute_ + nBubSize_ * groupIdx * 16);
        event_t eventIdMte2ToV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV2);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV2);

        Brcb(addTensor, addLocal, CeilDiv(realNLen, 8), {1, 8});
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CopyInTensorAub(int64_t ubOffset,
                                                                                              int32_t blockCount,
                                                                                              int32_t blockLen,
                                                                                              int32_t srcStride,
                                                                                              LocalTensor<xType> xLocal)
{
    DataCopyParams intriParams;
    intriParams.dstStride = 0;
    intriParams.blockLen = blockLen;
    if (innerAxisUnAlignedA_) {
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        for (int32_t copyLoopIdx = 0; copyLoopIdx < blockCount; copyLoopIdx++) {
            int32_t aubDstOffset = copyLoopIdx * blockLen * BLOCK_CUBE;
            DataCopy(xLocal[aubDstOffset], xGlobal_[ubOffset], intriParams);
            ubOffset += aTrans ? tiling_->mSize : tiling_->kSize;
        }
    } else {
        intriParams.srcStride = srcStride;
        intriParams.blockCount = blockCount;
        DataCopy(xLocal, xGlobal_[ubOffset], intriParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::AntiQuantCompute(int32_t bubNLen,
                                                                                               int32_t bubKLen,
                                                                                               int32_t antiquantOffset,
                                                                                               LocalTensor<half> weight,
                                                                                               int64_t bubKOffset)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        if constexpr (hasAntiQuantOffset) {
            Adds(weight, weight, offsetValue_, weight.GetSize());
            PipeBarrier<PIPE_V>();
        }

        Muls(weight, weight, scaleValue_, weight.GetSize());
    } else if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        int64_t groupOffset = bubKOffset / tiling_->groupSize;
        int64_t groupLen = (bubKOffset + bubKLen - 1) / tiling_->groupSize - groupOffset + 1;
        if constexpr (hasAntiQuantOffset) {
            for (int64_t groupIdx = 0; groupIdx < groupLen; groupIdx++) {
                LocalTensor<half> addComputeTensor = apiTmpBuf_.template GetWithOffset<half>(
                    bubNLen * 16, offsetAddCompute_ + nBubSize_ * groupIdx * 16);

                int64_t weightOffset = groupIdx == 0 ? bubKOffset : tiling_->groupSize * (groupOffset + groupIdx);
                int64_t weightEnd =
                    groupIdx == groupLen - 1 ? bubKOffset + bubKLen : tiling_->groupSize * (groupOffset + groupIdx + 1);
                int64_t computeOffset = bubNLen * (weightOffset - bubKOffset);
                for (int i = 0; i < CeilDiv(weightEnd - weightOffset, 16); i++) {
                    Add(weight[computeOffset], weight[computeOffset], addComputeTensor, bubNLen * 16);
                    computeOffset += bubNLen * 16;
                }
            }
        }

        PipeBarrier<PIPE_V>();

        for (int64_t groupIdx = 0; groupIdx < groupLen; groupIdx++) {
            LocalTensor<half> mulComputeTensor =
                apiTmpBuf_.template GetWithOffset<half>(bubNLen * 16, offsetMulCompute_ + nBubSize_ * groupIdx * 16);

            int64_t weightOffset = groupIdx == 0 ? bubKOffset : tiling_->groupSize * (groupOffset + groupIdx);
            int64_t weightEnd =
                groupIdx == groupLen - 1 ? bubKOffset + bubKLen : tiling_->groupSize * (groupOffset + groupIdx + 1);
            int64_t computeOffset = bubNLen * (weightOffset - bubKOffset);
            for (int i = 0; i < CeilDiv(weightEnd - weightOffset, 16); i++) {
                Mul(weight[computeOffset], weight[computeOffset], mulComputeTensor, bubNLen * 16);
                computeOffset += bubNLen * 16;
            }
        }
    } else {
        if constexpr (hasAntiQuantOffset) {
            LocalTensor<half> addComputeTensor =
                apiTmpBuf_.template GetWithOffset<half>(elemsAddCompute_, offsetAddCompute_);
            for (int i = 0; i < CeilDiv(bubKLen, 16); i++) {
                Add(weight[i * bubNLen * 16], weight[i * bubNLen * 16], addComputeTensor[antiquantOffset],
                    bubNLen * 16);
            }
        }

        PipeBarrier<PIPE_V>();

        LocalTensor<half> mulComputeTensor =
            apiTmpBuf_.template GetWithOffset<half>(elemsMulCompute_, offsetMulCompute_);
        for (int i = 0; i < CeilDiv(bubKLen, 16); i++) {
            Mul(weight[i * bubNLen * 16], weight[i * bubNLen * 16], mulComputeTensor[antiquantOffset], bubNLen * 16);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
inline __aicore__ void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::TransNd2Nz(LocalTensor<xType> srcUbLocal,
                                                                                         LocalTensor<xType> dstUbLocal,
                                                                                         int32_t outerDim,
                                                                                         int32_t innerDim,
                                                                                         int32_t innerLen)
{
    int64_t mask = 128;
    int32_t repeatCnt = 2 * outerDim;
    uint8_t repeatLoop = repeatCnt / 255;
    uint8_t repeatTail = repeatCnt % 255;
    UnaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.srcBlkStride = innerLen;
    repeatParams.dstRepStride = REPEAT_BLOCK_NUM;
    repeatParams.srcRepStride = REPEAT_BLOCK_NUM * innerLen;
    for (int32_t repeatIdx = 0; repeatIdx < repeatLoop; ++repeatIdx) {
        for (int32_t loopIdx = 0; loopIdx < innerDim; loopIdx++) {
            int32_t srcOffset = loopIdx * BLOCK_CUBE + 255 * repeatIdx * (aTrans ? mAubSize_ : kAubSize_) * 8;
            int32_t dstOffset = loopIdx * outerDim * BLOCK_CUBE * BLOCK_CUBE + 255 * repeatIdx * BLOCK_CUBE * 8;
            Muls(dstUbLocal[dstOffset], srcUbLocal[srcOffset], static_cast<xType>(1.0), mask, 255, repeatParams);
        }
    }
    if (repeatTail != 0) {
        for (int32_t loopIdx = 0; loopIdx < innerDim; loopIdx++) {
            int32_t srcOffset = loopIdx * BLOCK_CUBE + 255 * repeatLoop * (aTrans ? mAubSize_ : kAubSize_) * 8;
            int32_t dstOffset = loopIdx * outerDim * BLOCK_CUBE * BLOCK_CUBE + 255 * repeatLoop * BLOCK_CUBE * 8;
            Muls(dstUbLocal[dstOffset], srcUbLocal[srcOffset], static_cast<xType>(1.0), mask, repeatTail, repeatParams);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::TransNz2Nd(LocalTensor<xType> cubLocal)
{
    uint64_t mask = min(128, nCubSize_);
    uint8_t repeat = min(255, mCubSize_);
    int maskLoop = nCubSize_ / mask;
    int maskTail = nCubSize_ % mask;
    int repeatLoop = mCubSize_ / repeat;
    int repeatTail = mCubSize_ % repeat;

    UnaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.srcBlkStride = mCubSize_;
    repeatParams.dstRepStride = nCubSize_ / BLOCK_CUBE;
    repeatParams.srcRepStride = 1;

    LocalTensor<xType> resCNd = apiTmpBuf_.template GetWithOffset<xType>(resCNdElem_, resCNdOffset_);

    for (int mIdx = 0; mIdx < repeatLoop; mIdx++) {
        for (int nIdx = 0; nIdx < maskLoop; nIdx++) {
            int dstOffset = mIdx * repeat * nCubSize_ + nIdx * mask;
            int srcOffset = mCubSize_ * mask * nIdx + mIdx * repeat * BLOCK_CUBE;
            Muls(resCNd[dstOffset], cubLocal[srcOffset], static_cast<yType>(1.0), mask, repeat, repeatParams);
        }

        if (unlikely(maskTail > 0)) {
            int dstOffset = mIdx * repeat * nCubSize_ + maskLoop * mask;
            int srcOffset = mCubSize_ * mask * maskLoop + mIdx * repeat * BLOCK_CUBE;
            Muls(resCNd[dstOffset], cubLocal[srcOffset], static_cast<yType>(1.0), maskTail, repeat, repeatParams);
        }
    }

    if (unlikely(repeatTail > 0)) {
        for (int nIdx = 0; nIdx < maskLoop; nIdx++) {
            int dstOffset = repeatLoop * repeat * nCubSize_ + nIdx * mask;
            int srcOffset = mCubSize_ * mask * nIdx + repeatLoop * repeat * BLOCK_CUBE;
            Muls(resCNd[dstOffset], cubLocal[srcOffset], static_cast<yType>(1.0), mask, repeatTail, repeatParams);
        }

        if (unlikely(maskTail > 0)) {
            int dstOffset = repeatLoop * repeat * nCubSize_ + maskLoop * mask;
            int srcOffset = mCubSize_ * mask * maskLoop + repeatLoop * repeat * BLOCK_CUBE;
            Muls(resCNd[dstOffset], cubLocal[srcOffset], static_cast<yType>(1.0), maskTail, repeatTail, repeatParams);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CopyVecOut2L1(LocalTensor<xType> l1Local,
                                                                                            LocalTensor<xType> ubLocal,
                                                                                            int32_t blockCount,
                                                                                            int32_t blockLen,
                                                                                            int32_t dstStride)
{
    DataCopyParams intriParams;
    intriParams.blockLen = blockLen;
    intriParams.blockCount = blockCount;
    intriParams.srcStride = 0;
    intriParams.dstStride = dstStride;
    DataCopy(l1Local, ubLocal, intriParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
    hasAntiQuantOffset>::BiasProcess(LocalTensor<float> biasFp32Local, int64_t nBlockOffset)
{
    LocalTensor<xType> biasLocal = apiTmpBuf_.template GetWithOffset<xType>(elemsBiasCopyIn_, offsetBiasCopyIn_);
    if (tiling_->nSize <= nBlockOffset) {
        return;
    }

    int32_t biasLen = min(tiling_->nSize - nBlockOffset, nL0Size_);
    CopyInBias(biasLocal, nBlockOffset, 0, biasLen);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    Cast(biasFp32Local, biasLocal, RoundMode::CAST_NONE, nBL1Size_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::AL1Process(LocalTensor<xType> aL1Local,
                                                                                         int64_t mAL1Offset,
                                                                                         int64_t kBlockOffset,
                                                                                         int32_t kL1Len, int32_t mL0Len)
{
    LocalTensor<xType> xLocalPing = apiTmpBuf_.template GetWithOffset<xType>(elemsAubCopyInPing_, offsetAubCopyInPing_);
    LocalTensor<xType> xLocalPong = apiTmpBuf_.template GetWithOffset<xType>(elemsAubCopyInPong_, offsetAubCopyInPong_);

    LocalTensor<xType> aubNzLocalPing =
        apiTmpBuf_.template GetWithOffset<xType>(elemsAubNd2NzPing_, offsetAubNd2NzPing_);
    LocalTensor<xType> aubNzLocalPong =
        apiTmpBuf_.template GetWithOffset<xType>(elemsAubNd2NzPong_, offsetAubNd2NzPong_);

    int32_t ml0Align = CeilDiv(mL0Len, BLOCK_CUBE) * BLOCK_CUBE;
    int32_t kL1LenAlign = CeilDiv(kL1Len, BLOCK_CUBE) * BLOCK_CUBE;
    int32_t mLoopNum = min(CeilDiv(tiling_->mSize - mAL1Offset, mAubSize_), aubMFactor_);
    int32_t kLoopNum = min(CeilDiv(tiling_->kSize - kBlockOffset, kAubSize_), aubKFactor_);
    int32_t loopMax = mLoopNum * kLoopNum;

    for (int32_t aubMLoopIdx = 0; aubMLoopIdx < mLoopNum; aubMLoopIdx++) {
        int64_t aubMOffset = mAL1Offset + aubMLoopIdx * mAubSize_;
        if (tiling_->mSize <= aubMOffset) {
            break;
        }
        int32_t aubMLen = min(tiling_->mSize - aubMOffset, mAubSize_);
        for (int32_t aubKLoopIdx = 0; aubKLoopIdx < kLoopNum; aubKLoopIdx++) {
            // aub process
            int64_t aubKOffset = kBlockOffset + aubKLoopIdx * kAubSize_;
            if (tiling_->kSize <= aubKOffset) {
                break;
            }

            int loopIdx = aubMLoopIdx * kLoopNum + aubKLoopIdx;
            int32_t aubKLen = min(tiling_->kSize - aubKOffset, kAubSize_);
            int64_t aubOffset = aubMOffset * tiling_->kSize + aubKOffset;
            int32_t blockCount = aTrans ? aubKLen : aubMLen;
            int32_t blockLen = CeilDiv((aTrans ? aubMLen : aubKLen), BLOCK_CUBE);
            int32_t srcStride = CeilDiv(tiling_->kSize, BLOCK_CUBE) - CeilDiv(aubKLen, BLOCK_CUBE);
            int32_t dstStride = mL0Size_ - CeilDiv(aubMLen, BLOCK_CUBE) * BLOCK_CUBE;
            if constexpr (aTrans) {
                aubOffset = aubKOffset * tiling_->mSize + aubMOffset;
                srcStride = CeilDiv(tiling_->mSize, BLOCK_CUBE) - CeilDiv(aubMLen, BLOCK_CUBE);
                dstStride = kL1LenAlign - CeilDiv(aubKLen, BLOCK_CUBE) * BLOCK_CUBE;
            }

            if (loopIdx % 2 == 0) {
                event_t eventIdVToMte2Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                if (loopIdx / 2 > 0) {
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);
                }

                CopyInTensorAub(aubOffset, blockCount, blockLen, srcStride, xLocalPing);

                event_t eventIdMte2ToVPing = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVPing);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVPing);

                int32_t outerDim = CeilDiv(blockCount, BLOCK_CUBE);
                int32_t innerDim = blockLen;

                TransNd2Nz(xLocalPing, aubNzLocalPing, outerDim, innerDim, blockLen);

                if (loopIdx < loopMax - 2) {
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);
                }

                event_t eventIdVToMte3Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping);

                int32_t kAL1Offset = aubKLoopIdx * kAubSize_;
                int32_t l1Offset = 0;
                if constexpr (aTrans) {
                    // m1, k1, k0, m0
                    l1Offset += kAL1Offset * BLOCK_CUBE + aubMLoopIdx * mAubSize_ * kL1LenAlign;
                } else {
                    // k1, m1, m0, k0
                    l1Offset += aubMLoopIdx * mAubSize_ * BLOCK_CUBE + aubKLoopIdx * kAubSize_ * mL0Size_;
                }

                CopyVecOut2L1(aL1Local[l1Offset], aubNzLocalPing, innerDim, outerDim * BLOCK_CUBE, dstStride);
            } else {
                event_t eventIdVToMte2Pong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                if (loopIdx / 2 > 0) {
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong);
                }
                CopyInTensorAub(aubOffset, blockCount, blockLen, srcStride, xLocalPong);

                event_t eventIdMte2ToVPong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVPong);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVPong);

                int32_t outerDim = CeilDiv(blockCount, BLOCK_CUBE);
                int32_t innerDim = blockLen;

                TransNd2Nz(xLocalPong, aubNzLocalPong, outerDim, innerDim, blockLen);

                if (loopIdx < loopMax - 2) {
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong);
                }

                event_t eventIdVToMte3Pong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Pong);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Pong);

                int32_t kAL1Offset = aubKLoopIdx * kAubSize_;
                int32_t l1Offset = 0;
                if constexpr (aTrans) {
                    // m1, k1, k0, m0
                    l1Offset += kAL1Offset * BLOCK_CUBE + aubMLoopIdx * mAubSize_ * kL1LenAlign;
                } else {
                    // k1, m1, m0, k0
                    l1Offset += aubMLoopIdx * mAubSize_ * BLOCK_CUBE + aubKLoopIdx * kAubSize_ * mL0Size_;
                }

                CopyVecOut2L1(aL1Local[l1Offset], aubNzLocalPong, innerDim, outerDim * BLOCK_CUBE, dstStride);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::BL1Process(LocalTensor<xType> bL1Local,
                                                                                         int64_t nBL1Offset,
                                                                                         int64_t kBlockOffset,
                                                                                         int32_t kL1Len, int32_t nL0Len)
{
    int32_t nL0Align = CeilDiv(nL0Len, BLOCK_CUBE) * BLOCK_CUBE;

    LocalTensor<wType> wLocalPing = apiTmpBuf_.template GetWithOffset<wType>(elemsBubCopyInPing_, offsetBubCopyInPing_);
    LocalTensor<wType> wLocalPong = apiTmpBuf_.template GetWithOffset<wType>(elemsBubCopyInPong_, offsetBubCopyInPong_);

    LocalTensor<half> weight16Ping =
        apiTmpBuf_.template GetWithOffset<half>(elemsBubWeight16Ping_, offsetBubWeight16Ping_);
    LocalTensor<half> weight16Pong =
        apiTmpBuf_.template GetWithOffset<half>(elemsBubWeight16Pong_, offsetBubWeight16Pong_);

    int32_t bubNLoopIdx = 0;
    int32_t nLoopNum = min(CeilDiv(tiling_->nSize - nBL1Offset, nBubSize_), bubNFactor_);
    int32_t kLoopNum = min(CeilDiv(tiling_->kSize - kBlockOffset, kBubSize_), bubKFactor_);
    int32_t loopMax = nLoopNum * kLoopNum;

    for (; bubNLoopIdx < nLoopNum; bubNLoopIdx++) {
        int64_t bubNOffset = nBL1Offset + bubNLoopIdx * nBubSize_;
        int32_t bubNLen = min(tiling_->nSize - bubNOffset, nBubSize_);

        for (int32_t bubKLoopIdx = 0; bubKLoopIdx < kLoopNum; bubKLoopIdx++) {
            // bub process
            int64_t bubKOffset = kBlockOffset + bubKLoopIdx * kBubSize_;
            int32_t loopIdx = bubNLoopIdx * kLoopNum + bubKLoopIdx;

            int32_t bubKLen = min(tiling_->kSize - bubKOffset, kBubSize_);
            int32_t blockCount = CeilDiv(bubKLen, 32);
            int64_t srcStrideSize = CeilDiv(tiling_->nSize, BLOCK_CUBE) * BLOCK_CUBE - bubNLen;
#if defined(__CCE_KT_TEST__)
            ASCENDC_ASSERT(srcStrideSize > 65535,
                           { KERNEL_LOG(KERNEL_ERROR, "srcStride must <= 65535, actual is %ld", srcStrideSize); });
#endif
            int32_t srcStride = static_cast<int32_t>(srcStrideSize);
            int32_t dstStride = nL0Align - CeilDiv(bubNLen, BLOCK_CUBE) * BLOCK_CUBE;
            int32_t blockLen = bubNLen;

            if (loopIdx % 2 == 0) {
                BL1SubProcess(wLocalPing, weight16Ping, bL1Local, bubKOffset, bubNOffset, blockCount, blockLen,
                              srcStride, bubNLen, bubKLen, loopIdx, loopMax, bubNLoopIdx, bubKLoopIdx);
            } else {
                BL1SubProcess(wLocalPong, weight16Pong, bL1Local, bubKOffset, bubNOffset, blockCount, blockLen,
                              srcStride, bubNLen, bubKLen, loopIdx, loopMax, bubNLoopIdx, bubKLoopIdx);
            }
        }
    }
    BL1TailProcess(bL1Local, nBL1Offset, kBlockOffset, nL0Len, bubNLoopIdx, kLoopNum, wLocalPing, wLocalPong,
                   weight16Ping, weight16Pong);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
inline __aicore__ void Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
    hasAntiQuantOffset>::BL1TailProcess(LocalTensor<xType> bL1Local, int64_t nBL1Offset, int64_t kBlockOffset,
                                        int32_t nL0Len, int32_t bubNLoopIdx, int32_t kLoopNum,
                                        LocalTensor<wType> wLocalPing, LocalTensor<wType> wLocalPong,
                                        LocalTensor<half> weight16Ping, LocalTensor<half> weight16Pong)
{
    int nBL1Tail = nL0Size_ % nBubSize_;
    if (likely(nBL1Tail == 0)) {
        return;
    }

    PipeBarrier<PIPE_ALL>();

    int32_t nL0Align = CeilDiv(nL0Len, BLOCK_CUBE) * BLOCK_CUBE;
    int64_t bubNOffset = nBL1Offset + bubNLoopIdx * nBubSize_;

    for (int32_t bubKLoopIdx = 0; bubKLoopIdx < kLoopNum; bubKLoopIdx++) {
        // bub process
        int64_t bubKOffset = kBlockOffset + bubKLoopIdx * kBubSize_;

        int32_t bubKLen = min(tiling_->kSize - bubKOffset, kBubSize_);
        int32_t blockCount = CeilDiv(bubKLen, 32);
        int32_t srcStride = CeilDiv(tiling_->nSize, BLOCK_CUBE) * BLOCK_CUBE - nBL1Tail;
        int32_t dstStride = nL0Align - CeilDiv(nBL1Tail, BLOCK_CUBE) * BLOCK_CUBE;
        int32_t blockLen = nBL1Tail;

        if (bubKLoopIdx % 2 == 0) {
            BL1SubProcess(wLocalPing, weight16Ping, bL1Local, bubKOffset, bubNOffset, blockCount, blockLen, srcStride,
                          nBL1Tail, bubKLen, bubKLoopIdx, kLoopNum, bubNLoopIdx, bubKLoopIdx);
        } else {
            BL1SubProcess(wLocalPong, weight16Pong, bL1Local, bubKOffset, bubNOffset, blockCount, blockLen, srcStride,
                          nBL1Tail, bubKLen, bubKLoopIdx, kLoopNum, bubNLoopIdx, bubKLoopIdx);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
inline __aicore__ void Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
    hasAntiQuantOffset>::BL1SubProcess(LocalTensor<wType> wLocalPing, LocalTensor<half> weight16Ping,
                                       LocalTensor<xType> bL1Local, int64_t bubKOffset, int64_t bubNOffset,
                                       int32_t blockCount, int32_t blockLen, int32_t srcStride, int32_t bubNLen,
                                       int32_t bubKLen, int32_t loopIdx, int32_t loopMax, int32_t bubNLoopIdx,
                                       int32_t bubKLoopIdx)
{
    int64_t bubOffset = bubKOffset * CeilDiv(tiling_->nSize, BLOCK_CUBE) * BLOCK_CUBE + bubNOffset * 32;

    DataCopyParams intriParams;
    intriParams.dstStride = 0;
    intriParams.blockCount = blockCount;
    intriParams.blockLen = blockLen;
    intriParams.srcStride = srcStride;

    DataCopy(wLocalPing, wGlobal_[bubOffset], intriParams);

    int64_t groupOffset = 0;
    int64_t groupLen = 1;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        groupOffset = bubKOffset / tiling_->groupSize;
        groupLen = (bubKOffset + bubKLen - 1) / tiling_->groupSize - groupOffset + 1;
        for (int64_t groupIdx = 0; groupIdx < groupLen; groupIdx++) {
            CopyInAddMul((groupOffset + groupIdx) * tiling_->nSize + bubNOffset, bubNLen, groupIdx);
        }
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    if (bubKLen >= 128) {
        for (uint32_t i = 0; i < CeilDiv(bubKLen, 128); i++) {
            Cast(weight16Ping[bubNLen * 128 * i], wLocalPing[bubNLen * 128 * i], RoundMode::CAST_NONE, 128,
                 static_cast<uint8_t>(bubNLen), {static_cast<uint16_t>(bubNLen), static_cast<uint16_t>(bubNLen), 1, 1});
        }
    } else {
        for (uint32_t i = 0; i < CeilDiv(bubKLen, 32); i++) {
            Cast(weight16Ping[bubNLen * 32 * i], wLocalPing[bubNLen * 32 * i], RoundMode::CAST_NONE, 32,
                 static_cast<uint8_t>(bubNLen), {static_cast<uint16_t>(bubNLen), 1, 1, 1});
        }
    }

    event_t eventIdVToMte2Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);

    AntiQuantCompute(bubNLen, bubKLen, bubNLoopIdx * nBubSize_ * BLOCK_CUBE, weight16Ping, bubKOffset);

    event_t eventIdMte1ToMte2Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Ping);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Ping);

    event_t eventIdVToMte3Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping);

    int32_t l1Offset = nBL1Size_ * bubKLoopIdx * kBubSize_ + bubNLoopIdx * nBubSize_ * BLOCK_CUBE;
    CopyVecOut2L1(bL1Local[l1Offset], weight16Ping, CeilDiv(kBubSize_, BLOCK_CUBE), bubNLen, nBL1Size_ - bubNLen);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::PostProcess(int32_t mL0Len,
                                                                                          int32_t nL0Len,
                                                                                          int32_t mAL1Offset,
                                                                                          int32_t nBL1Offset)
{
    LocalTensor<xType> resCNz = apiTmpBuf_.template GetWithOffset<xType>(resCNzElem_, resCNzOffset_);
    TransNz2Nd(resCNz);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CopyVec2Out(int64_t yGmOffset,
                                                                                          int64_t mReal, int64_t nReal,
                                                                                          LocalTensor<yType> resCNd)
{
    DataCopyParams dmaParam;
    dmaParam.blockCount = mReal;
    dmaParam.blockLen = nReal / BLOCK_CUBE;
    dmaParam.srcStride = (nCubSize_ - nReal) / BLOCK_CUBE;
    dmaParam.dstStride = CeilDiv(tiling_->nSize, BLOCK_CUBE) - dmaParam.blockLen;

    if (likely(tiling_->nSize % BLOCK_CUBE == 0)) {
        DataCopy(yGlobal_[yGmOffset], resCNd, dmaParam);
        return;
    }

    // 对于nSize非对齐场景，需要单行搬出，且尾块需向前拼接为一个BLOCK后搬出
    dmaParam.blockCount = 1;
    dmaParam.dstStride = 0;
    int nTail = nReal % BLOCK_CUBE;
    if (nReal >= BLOCK_CUBE) {
        for (int mIdx = 0; mIdx < mReal; mIdx++) {
            DataCopy(yGlobal_[yGmOffset + mIdx * tiling_->nSize], resCNd[mIdx * nCubSize_], dmaParam);
        }

        if (unlikely(nTail > 0)) {
            event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

            DataCopyParams dmaTailParam;
            dmaTailParam.blockCount = 1;
            dmaTailParam.blockLen = 1;
            dmaTailParam.srcStride = 0;
            dmaTailParam.dstStride = 0;

            int tailOffset = nReal / BLOCK_CUBE * BLOCK_CUBE - BLOCK_CUBE + nTail;
            LocalTensor<yType> resCNz = apiTmpBuf_.GetWithOffset<yType>(resCNzElem_, resCNzOffset_);

            for (int mIdx = 0; mIdx < mReal; mIdx++) {
                for (int elemIdx = 0; elemIdx < BLOCK_CUBE; elemIdx++) {
                    resCNz.SetValue(mIdx * nCubSize_ + elemIdx,
                                    resCNd.GetValue(mIdx * nCubSize_ + tailOffset + elemIdx));
                }

                event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

                DataCopy(yGlobal_[yGmOffset + mIdx * tiling_->nSize + tailOffset], resCNz[mIdx * nCubSize_],
                         dmaTailParam);
            }
        }
    } else {
        for (int mIdx = 0; mIdx < mReal; ++mIdx) {
            DataCopy(yGlobal_[yGmOffset + mIdx * tiling_->nSize], resCNd[mIdx * nCubSize_], BLOCK_CUBE);
            PipeBarrier<PIPE_MTE3>();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
    hasAntiQuantOffset>::CubeProcess(LocalTensor<xType> aL1Local, LocalTensor<xType> bL1Local,
                                     LocalTensor<float> biasFp32Local, int32_t mL0Len, int32_t nL0Len, int32_t kL1Len,
                                     int32_t kFactorIdx)
{
    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetBias = 0;
    mmObj_.SetTensorA(aL1Local[offsetA], aTrans);
    mmObj_.SetTensorB(bL1Local[offsetB], bTrans);
    if (biasFlag_) {
        mmObj_.SetBias(biasFp32Local[offsetBias]);
        if (kFactorIdx != 0) {
            mmObj_.ClearBias();
        }
    }

    this->mmObj_.SetTail(this->mL0Size_, this->nL0Size_, kL1Len);

    mmObj_.Iterate(kFactorIdx != 0);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::UpdateGlobalAddr(GM_ADDR x,
                                                                                               GM_ADDR weight,
                                                                                               GM_ADDR antiquantScale,
                                                                                               GM_ADDR antiquantOffset,
                                                                                               GM_ADDR quantScale,
                                                                                               GM_ADDR quantOffset,
                                                                                               GM_ADDR bias, GM_ADDR y,
                                                                                               GM_ADDR workspace)
{
    InitInputs(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline int64_t
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline int32_t
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline int32_t
Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset>::min(int32_t a, int32_t b)
{
    return a > b ? b : a;
}

} // namespace Mc2WeightQuantBatchMatmulV2

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_BASE_H
