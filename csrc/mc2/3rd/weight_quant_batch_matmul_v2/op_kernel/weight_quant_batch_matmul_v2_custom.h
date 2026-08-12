/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_custom.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_common.h"
#include "tool.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::InitDump;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
class Mc2WeightQuantBatchMatmulV2CustomKernel
    : public Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                               hasAntiQuantOffset, quantType> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2CustomKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void ComputeWeightOffsetInfo(uint64_t nLoopIdx, uint64_t nBaseOffset, uint64_t kLoopIdx,
                                                   WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputeWeightSplitInfo(uint64_t nLoopIdx, uint64_t nLoopLimit, uint64_t nRealSize,
                                                  uint64_t kLoopIdx, WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void Process();

    using inputXType = MatmulType<TPosition::GM, CubeFormat::ND, xType, aTrans>;
    using inputWType = MatmulType<TPosition::GM, CubeFormat::ND, xType, bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj;

private:
    __aicore__ inline void ProcessCube();
    __aicore__ inline void SetMatmulParams(int32_t cubeNLoopIdx, uint64_t aOffset, uint64_t bOffset, uint64_t nOffset);
    __aicore__ inline void ProcessVector();
    __aicore__ inline void AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset, uint64_t nRealSize,
                                           uint64_t nLoopLimit, WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void NotifyCube();
    uint64_t quantScaleValue = 0;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                     GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                     const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe)
{
    this->BaseInit(tilingData, tPipe);
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->ComputeConstexpr();
    this->InitBuffer();
    this->InitWorkSpace(workspace);

    if ASCEND_IS_AIC {
        mmObj.SetSubBlockIdx(0);
        mmObj.Init(&this->tiling_->matmulTiling, this->pipe_);

        if constexpr (IsSameType<yType, int8_t>::value && quantType == Mc2QuantType::PER_TENSOR) {
            quantScaleValue = this->quantScaleGlobal_.GetValue(0);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                        hasAntiQuantOffset, quantType>::UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight,
                                                                                         GM_ADDR antiquantScale,
                                                                                         GM_ADDR antiquantOffset,
                                                                                         GM_ADDR quantScale,
                                                                                         GM_ADDR quantOffset,
                                                                                         GM_ADDR bias, GM_ADDR y,
                                                                                         GM_ADDR workspace)
{
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->InitWorkSpace(workspace);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                        hasAntiQuantOffset, quantType>::ProcessVector()
{
    uint64_t nBaseSize = this->tiling_->matmulTiling.singleCoreN * this->tiling_->cubeBlockDimN;
    uint64_t nRealSize = nBaseSize;

    // 初始化weight的切分信息
    WeightSplitInfo weightSplitInfo;
    for (int32_t cubeNLoopIdx = 0; cubeNLoopIdx < this->tiling_->cubeSingleNLoop; cubeNLoopIdx++) {
        uint64_t nBaseOffset = cubeNLoopIdx * nBaseSize;
        uint64_t nLoopLimit = this->tiling_->vecSingleNLoop;
        if (unlikely(cubeNLoopIdx + 1 == this->tiling_->cubeSingleNLoop)) {
            nLoopLimit = this->tiling_->vecSingleNTailLoop;
            nRealSize = this->tiling_->nSize - nBaseOffset;
        }
        // 为节约workspace空间，weight缓存空间整体分成WEIGHT_CACHE_COUNT块轮询使用，需要根据cubeNLoopIdx确定当前使用哪块地址
        this->weightCacheIdx_ = cubeNLoopIdx % this->WEIGHT_CACHE_COUNT;

        this->AntiquantWeight(cubeNLoopIdx, nBaseOffset, nRealSize, nLoopLimit, weightSplitInfo);

        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(this->SYNC_AIV_ONLY_ALL_FLAG);
    }
    NotifyCube();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset, uint64_t nRealSize, uint64_t nLoopLimit,
                                WeightSplitInfo &weightSplitInfo)
{
    // 求解当次循环n方向的起点和终点
    uint64_t loopLimit = nLoopLimit * this->tiling_->vecSingleKLoop;
    uint64_t singleCoreLoop = this->CeilDiv(loopLimit, this->tiling_->vecBlockDimN);
    uint64_t singleCoreLoopStart = this->curBlockIdx_ * singleCoreLoop;
    uint64_t singleCoreLoopLimit = singleCoreLoopStart + singleCoreLoop;
    if (singleCoreLoopLimit > loopLimit) {
        singleCoreLoopLimit = loopLimit;
    }
    weightSplitInfo.vecSingleN = this->tiling_->vecSingleN;
    int64_t lastNLoopIdx = -1;
    for (uint64_t loopIdx = singleCoreLoopStart; loopIdx < singleCoreLoopLimit; loopIdx++) {
        uint64_t nLoopIdx = loopIdx / this->tiling_->vecSingleKLoop;
        uint64_t kLoopIdx = loopIdx % this->tiling_->vecSingleKLoop;
        ComputeWeightOffsetInfo(nLoopIdx, nBaseOffset, kLoopIdx, weightSplitInfo);
        ComputeWeightSplitInfo(nLoopIdx, nLoopLimit, nRealSize, kLoopIdx, weightSplitInfo);

        if (likely(this->vecKDimIdx_ < this->tiling_->vecBlockDimK)) {
            if ((antiQuantType == Mc2QuantType::PER_CHANNEL && lastNLoopIdx != nLoopIdx) ||
                antiQuantType == Mc2QuantType::PER_GROUP) {
                this->CopyInAntiquantParams(weightSplitInfo);
                lastNLoopIdx = nLoopIdx;
            }
            this->WeightCopyInAndCast(weightSplitInfo);
            this->AntiQuantCompute(weightSplitInfo);
        }
        if (likely(cubeNLoopIdx > 0 && loopIdx == singleCoreLoopStart)) {
            NotifyCube();
        }
        if (likely(cubeNLoopIdx > 1 && loopIdx == singleCoreLoopStart)) {
            // vector计算需要领先cube一拍
            this->WaitForCube();
        }
        if (likely(this->vecKDimIdx_ < this->tiling_->vecBlockDimK)) {
            this->WeightCopyOut(weightSplitInfo);
        }
    }

    // 尾核场景，此时不会进入上面的循环，因此需要额外补充同步
    if (unlikely(cubeNLoopIdx > 0 && singleCoreLoopStart >= singleCoreLoopLimit)) {
        NotifyCube();
    }
    if (unlikely(cubeNLoopIdx > 1 && singleCoreLoopStart >= singleCoreLoopLimit)) {
        this->WaitForCube();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputeWeightOffsetInfo(uint64_t nLoopIdx, uint64_t nBaseOffset, uint64_t kLoopIdx,
                                        WeightSplitInfo &weightSplitInfo)
{
    weightSplitInfo.splitNOffset = nLoopIdx * this->tiling_->vecSingleN;
    weightSplitInfo.originNOffset = nBaseOffset + weightSplitInfo.splitNOffset;
    weightSplitInfo.kOffset = kLoopIdx * this->tiling_->vecSingleK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputeWeightSplitInfo(uint64_t nLoopIdx, uint64_t nLoopLimit, uint64_t nRealSize, uint64_t kLoopIdx,
                                       WeightSplitInfo &weightSplitInfo)
{
    if (unlikely(nLoopIdx == nLoopLimit - 1)) {
        // 当计算到最后一块时，需要重新计算尾块实际的n是多少
        weightSplitInfo.vecSingleN = nRealSize - weightSplitInfo.splitNOffset;
    }
    weightSplitInfo.vecSingleK = this->tiling_->vecSingleK;
    // 当计算到最后一块时，需要重新计算尾块实际的k是多少
    if (unlikely(kLoopIdx == this->tiling_->vecSingleKLoop - 1)) {
        weightSplitInfo.vecSingleK = this->tiling_->kSize - weightSplitInfo.kOffset;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                        hasAntiQuantOffset, quantType>::NotifyCube()
{
    this->WaitFlagDevLocal(this->SYNC_AIV_ONLY_ALL_FLAG);

    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                        hasAntiQuantOffset, quantType>::ProcessCube()
{
    uint64_t mOffset = this->cubeMDimIdx_ * this->tiling_->matmulTiling.singleCoreM;
    uint64_t aOffset = mOffset;

    if constexpr (!aTrans) {
        aOffset *= this->tiling_->matmulTiling.Ka;
    }
    uint64_t cubeNOffset = this->cubeNDimIdx_ * this->tiling_->matmulTiling.singleCoreN;
    uint64_t bOffset = cubeNOffset;
    if constexpr (bTrans) {
        bOffset *= this->tiling_->matmulTiling.Kb;
    }
    if constexpr (!bTrans) {
        mmObj.SetOrgShape(this->tiling_->matmulTiling.M, this->cubeBaseN_, this->tiling_->matmulTiling.Ka,
                          this->tiling_->matmulTiling.Kb, this->tiling_->matmulTiling.N);
    }

    for (int32_t cubeNLoopIdx = 0; cubeNLoopIdx < this->tiling_->cubeSingleNLoop; cubeNLoopIdx++) {
        this->WaitForVector();
        uint64_t nOffset = cubeNOffset + cubeNLoopIdx * this->cubeBaseN_;
        uint64_t cOffset = mOffset * this->tiling_->matmulTiling.N + nOffset;
        if (likely(nOffset < this->tiling_->nSize &&
                   this->curBlockIdx_ < this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimN)) {
            SetMatmulParams(cubeNLoopIdx, aOffset, bOffset, nOffset);
            mmObj.IterateAll(this->yGlobal_[cOffset]);
            mmObj.End();
        }
        if (cubeNLoopIdx > 0) {
            this->WaitFlagDevLocal(this->SYNC_AIC_ONLY_ALL_FLAG);
        }

        if (cubeNLoopIdx + 1 < this->tiling_->cubeSingleNLoop) {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(this->SYNC_AIC_ONLY_ALL_FLAG);
        }

        if (cubeNLoopIdx + 2 < this->tiling_->cubeSingleNLoop) {
            this->NotifyVector();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                        hasAntiQuantOffset, quantType>::SetMatmulParams(int32_t cubeNLoopIdx,
                                                                                        uint64_t aOffset,
                                                                                        uint64_t bOffset,
                                                                                        uint64_t nOffset)
{
    mmObj.SetTensorA(this->xGlobal_[aOffset], aTrans);
    mmObj.SetTensorB(
        this->weightCache_[(cubeNLoopIdx % this->WEIGHT_CACHE_COUNT) * this->weightCacheSizeAlign_ + bOffset], bTrans);

    if (this->biasFlag_) {
        mmObj.SetBias(this->biasGlobal_[nOffset]);
    }

    int64_t mmSingleN = this->tiling_->matmulTiling.singleCoreN;
    int64_t mmSingleM = this->tiling_->matmulTiling.singleCoreM;

    if (unlikely(nOffset + mmSingleN > this->tiling_->nSize)) {
        mmSingleN = this->tiling_->nSize - nOffset;
    }

    if (unlikely(this->cubeMDimIdx_ == this->tiling_->cubeBlockDimM - 1)) {
        mmSingleM = this->tiling_->cubeTailM;
    }

    mmObj.SetTail(mmSingleM, mmSingleN);

    if constexpr (IsSameType<yType, int8_t>::value) {
        if constexpr (quantType == Mc2QuantType::PER_TENSOR) {
            mmObj.SetQuantScalar(quantScaleValue);
        } else {
            mmObj.SetQuantVector(this->quantScaleGlobal_[nOffset]);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                               antiQuantType, hasAntiQuantOffset, quantType>::Process()
{
    if ASCEND_IS_AIV {
        ProcessVector();
    } else {
        ProcessCube();
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_H
