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
 * \file weight_quant_batch_matmul_v2_weight_nz_performance.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_PERFORMANCE_H

#include "weight_quant_batch_matmul_v2_weight_nz_performance_base.h"

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
class Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel
    : public Mc2WeightQuantBatchMatmulV2WeightNzBasePerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                      antiQuantType, hasAntiQuantOffset> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel(){};
    __aicore__ inline void Process();
    __aicore__ inline void OneProcess();
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                     hasAntiQuantOffset>::Process()
{
    OneProcess();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                     hasAntiQuantOffset>::OneProcess()
{
    LocalTensor<xType> bL1Local = this->InBufBL1_.template Get<xType>();
    LocalTensor<xType> aL1Local = this->InBufAL1_.template Get<xType>();

    LocalTensor<float> biasFp32Local =
        this->apiTmpBuf_.template GetWithOffset<float>(this->elemsBiasFp32_, this->offsetBiasFp32_);
    LocalTensor<xType> resCNz = this->apiTmpBuf_.template GetWithOffset<xType>(this->resCNzElem_, this->resCNzOffset_);
    LocalTensor<yType> resCNd = this->apiTmpBuf_.template GetWithOffset<yType>(this->resCNdElem_, this->resCNdOffset_);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte3ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    for (int32_t mCoreFactorIdx = 0; mCoreFactorIdx < this->mCoreLoopNum_; mCoreFactorIdx++) {
        int64_t mBlockOffset = this->mBlockOffset_ + mCoreFactorIdx * this->mAL1Size_;
        for (int32_t nCoreFactorIdx = 0; nCoreFactorIdx < this->nCoreLoopNum_; nCoreFactorIdx++) {
            int64_t nBlockOffset = this->nBlockOffset_ + nCoreFactorIdx * this->nBL1Size_;

            int32_t realNLen = this->min(this->nBL1Size_, this->tiling_->nSize - nBlockOffset);
            if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
                this->CopyInAddMul(nBlockOffset, realNLen, 0);
            }

            int64_t mAL1Offset = mBlockOffset;
            int64_t nBL1Offset = nBlockOffset;
            int32_t mL0Len = this->min(this->tiling_->mSize - mAL1Offset, this->mL0Size_);
            int32_t nL0Len = this->min(this->tiling_->nSize - nBL1Offset, this->nL0Size_);
            for (int32_t kFactorIdx = 0; kFactorIdx < this->kSingleCore_; kFactorIdx++) {
                int64_t kBlockOffset = kFactorIdx * this->kL1Size_;
                int32_t kL1Len = this->min(this->tiling_->kSize - kBlockOffset, this->kL1Size_);
                this->BL1Process(bL1Local, nBL1Offset, kBlockOffset, kL1Len, nL0Len);
                this->AL1Process(aL1Local, mAL1Offset, kBlockOffset, kL1Len, mL0Len);

                if (this->biasFlag_) {
                    this->BiasProcess(biasFp32Local, nBlockOffset);
                }

                SetFlag<HardEvent::MTE3_MTE1>(eventIdMte3ToMte1);
                WaitFlag<HardEvent::MTE3_MTE1>(eventIdMte3ToMte1);

                this->CubeProcess(aL1Local, bL1Local, biasFp32Local, mL0Len, nL0Len, kL1Len, kFactorIdx);

                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            this->mmObj_.GetTensorC(resCNz);

            PipeBarrier<PIPE_V>();

            this->PostProcess(mL0Len, nL0Len, mAL1Offset, nBL1Offset);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            int64_t yGmOffset = mAL1Offset * this->tiling_->nSize + nBL1Offset;
            this->CopyVec2Out(yGmOffset, mL0Len, nL0Len, resCNd);

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
class Mc2WeightQuantBatchMatmulV2WeightNzKernel
    : public Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                  antiQuantType, hasAntiQuantOffset> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2WeightNzKernel(){};
    __aicore__ inline void Process();

protected:
    static constexpr int32_t NZ_K = 16;
    static constexpr int32_t NZ_N = 32;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2WeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset>::Process()
{
    uint64_t xSize = this->tiling_->mSize * this->tiling_->kSize;
    uint64_t weightSize =
        this->CeilDiv(this->tiling_->kSize, NZ_N) * this->CeilDiv(this->tiling_->nSize, NZ_K) * NZ_K * NZ_N;
    uint64_t ySize = this->tiling_->mSize * this->tiling_->nSize;
    uint64_t offsetX = xSize * sizeof(xType);
    uint64_t offsetWeight = weightSize * sizeof(wType);
    uint64_t offsetY = ySize * sizeof(yType);
    uint64_t offsetBias = this->tiling_->nSize * sizeof(biasType);

    bool isbatchX0One = (this->tiling_->batchX0 == 1);
    bool isbatchWeight0One = (this->tiling_->batchWeight0 == 1);
    bool isbatchX1One = (this->tiling_->batchX1 == 1);
    bool isbatchWeight1One = (this->tiling_->batchWeight1 == 1);
    bool isbatchX2One = (this->tiling_->batchX2 == 1);
    bool isbatchWeight2One = (this->tiling_->batchWeight2 == 1);
    bool isbatchX3One = (this->tiling_->batchX3 == 1);
    bool isbatchWeight3One = (this->tiling_->batchWeight3 == 1);

    for (uint64_t batchY0Index = 0; batchY0Index < this->tiling_->batchY0; batchY0Index++) {
        uint64_t indexX0 =
            isbatchX0One ? 0 :
                           (batchY0Index * this->tiling_->batchX1 * this->tiling_->batchX2 * this->tiling_->batchX3);
        uint64_t indexWeight0 = isbatchWeight0One ? 0 :
                                                    (batchY0Index * this->tiling_->batchWeight1 *
                                                     this->tiling_->batchWeight2 * this->tiling_->batchWeight3);
        uint64_t indexY0 = batchY0Index * this->tiling_->batchY1 * this->tiling_->batchY2 * this->tiling_->batchY3;
        for (uint64_t batchY1Index = 0; batchY1Index < this->tiling_->batchY1; batchY1Index++) {
            uint64_t indexX1 = isbatchX1One ? 0 : (batchY1Index * this->tiling_->batchX2 * this->tiling_->batchX3);
            uint64_t indexWeight1 =
                isbatchWeight1One ? 0 : (batchY1Index * this->tiling_->batchWeight2 * this->tiling_->batchWeight3);
            uint64_t indexY1 = batchY1Index * this->tiling_->batchY2 * this->tiling_->batchY3;
            for (uint64_t batchY2Index = 0; batchY2Index < this->tiling_->batchY2; batchY2Index++) {
                uint64_t indexX2 = isbatchX2One ? 0 : (batchY2Index * this->tiling_->batchX3);
                uint64_t indexWeight2 = isbatchWeight2One ? 0 : (batchY2Index * this->tiling_->batchWeight3);
                uint64_t indexY2 = batchY2Index * this->tiling_->batchY3;
                for (uint64_t batchY3Index = 0; batchY3Index < this->tiling_->batchY3; batchY3Index++) {
                    uint64_t indexX3 = isbatchX3One ? 0 : batchY3Index;
                    uint64_t indexWeight3 = isbatchWeight3One ? 0 : batchY3Index;
                    uint64_t xIndex = indexX0 + indexX1 + indexX2 + indexX3;
                    uint64_t weightIndex = indexWeight0 + indexWeight1 + indexWeight2 + indexWeight3;
                    uint64_t yIndex = indexY0 + indexY1 + indexY2 + batchY3Index;
                    this->xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(this->xGM_ + offsetX * xIndex),
                                                   xSize);
                    this->wGlobal_.SetGlobalBuffer(
                        reinterpret_cast<__gm__ wType *>(this->weightGM_ + offsetWeight * weightIndex), weightSize);
                    this->yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(this->yGM_ + offsetY * yIndex),
                                                   ySize);
                    if (this->tiling_->biasWithBatch && this->biasFlag_) {
                        this->biasGlobal_.SetGlobalBuffer(
                            reinterpret_cast<__gm__ biasType *>(this->biasGM_ + offsetBias * yIndex),
                            this->tiling_->nSize);
                    }
                    this->OneProcess();
                }
            }
        }
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_WEIGHT_NZ_H
