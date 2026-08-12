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
 * \file weight_quant_batch_matmul_v2_basic_block_controller.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONTROLLER_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONTROLLER_H
#include "basic_block_config.h"
#include "weight_quant_batch_matmul_v2_basic_block.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename antiQuantScaleType, typename biasType, typename yType,
          const WqmmConfig &wqmmCfg, const VecAntiQuantConfig &vecCfg>
class WeightQuantBatchMatmulV2BasicBlockController {
public:
    __aicore__ inline WeightQuantBatchMatmulV2BasicBlockController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    uint64_t curBlockIdx_;
    uint64_t mDimIdx_;
    uint64_t nDimIdx_;
    const Mc2WeightQuantBatchMatmulV2ASTilingData *tiling_;
    WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, biasType, yType, wqmmCfg, vecCfg> wqmmBasicBlock_;
    BasicBlockOffsetParam basicBlockOffsetParam_;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename biasType, typename yType,
          const WqmmConfig &wqmmCfg, const VecAntiQuantConfig &vecCfg>
__aicore__ inline void
WeightQuantBatchMatmulV2BasicBlockController<xType, wType, antiQuantScaleType, biasType, yType, wqmmCfg, vecCfg>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const Mc2WeightQuantBatchMatmulV2ASTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    curBlockIdx_ = GetBlockIdx();
    if ASCEND_IS_AIV {
        curBlockIdx_ = curBlockIdx_ / AscendC::GetTaskRation();
    }
    wqmmBasicBlock_.UpdateGlobalAddr(reinterpret_cast<__gm__ xType *>(x), reinterpret_cast<__gm__ wType *>(weight),
                                     reinterpret_cast<__gm__ antiQuantScaleType *>(antiquantScale),
                                     reinterpret_cast<__gm__ xType *>(antiquantOffset),
                                     reinterpret_cast<__gm__ uint64_t *>(quantScale),
                                     reinterpret_cast<__gm__ biasType *>(bias), reinterpret_cast<__gm__ yType *>(y),
                                     tilingData->hasBias, tilingData->weightL2Cacheable);
    wqmmBasicBlock_.Init(tilingData->aPreloadSize, &(tilingData->matmulTiling), tPipe);
    basicBlockOffsetParam_.kaL1Size = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKa;
    basicBlockOffsetParam_.kbL1Size = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
    basicBlockOffsetParam_.mSize = tiling_->mSize;
    basicBlockOffsetParam_.kSize = tiling_->kSize;
    basicBlockOffsetParam_.kAlign = CeilAlign(tiling_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    basicBlockOffsetParam_.nSize = tiling_->nSize;

    mDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimN;
    nDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimN;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename biasType, typename yType,
          const WqmmConfig &wqmmCfg, const VecAntiQuantConfig &vecCfg>
__aicore__ inline void WeightQuantBatchMatmulV2BasicBlockController<xType, wType, antiQuantScaleType, biasType, yType,
                                                                    wqmmCfg, vecCfg>::Process()
{
    if (unlikely(curBlockIdx_ >= tiling_->cubeBlockDimM * tiling_->cubeBlockDimN)) {
        return;
    }
    uint64_t mDimOffset = mDimIdx_ * tiling_->matmulTiling.singleCoreM;
    // 尾块/尾核的实际计算量的计算逻辑：判断是不是最后一次计算，如果是，就用原始大小减去当前偏移作为实际计算量，
    // 如果不是，就用当前循环的步长作为实际计算量
    uint64_t singleCoreRealM = (mDimIdx_ == tiling_->cubeBlockDimM - 1) ?
                                   tiling_->mSize - mDimIdx_ * tiling_->matmulTiling.singleCoreM :
                                   tiling_->matmulTiling.singleCoreM;
    // 遍历核内的M维度,计算出GM上M方向的偏移以及本次循环要计算的M（mL1）的实际大小
    for (uint64_t mInnerOffset = 0; mInnerOffset < singleCoreRealM; mInnerOffset += tiling_->matmulTiling.baseM) {
        basicBlockOffsetParam_.mOffset = mInnerOffset + mDimOffset;
        basicBlockOffsetParam_.mL1Size = (mInnerOffset + tiling_->matmulTiling.baseM) >= singleCoreRealM ?
                                             singleCoreRealM - mInnerOffset :
                                             tiling_->matmulTiling.baseM;
        for (uint64_t mainBasicBlockIdx = nDimIdx_; mainBasicBlockIdx < tiling_->mainBlockCount;
             mainBasicBlockIdx += tiling_->cubeBlockDimN) {
            basicBlockOffsetParam_.nL1Size = tiling_->mainBlockL1Size;
            basicBlockOffsetParam_.nOffset = mainBasicBlockIdx * tiling_->mainBlockL1Size;
            wqmmBasicBlock_.ComputeBasicBlock(basicBlockOffsetParam_);
        }
        uint64_t mainBlockNOffset = tiling_->mainBlockCount * tiling_->mainBlockL1Size;
        uint64_t totalTailBlockCount = tiling_->firstTailBlockCount + tiling_->secondTailBlockCount;
        uint64_t tailBasicBlockIdx = nDimIdx_;
        for (; tailBasicBlockIdx < tiling_->firstTailBlockCount; tailBasicBlockIdx += tiling_->cubeBlockDimN) {
            basicBlockOffsetParam_.nOffset = mainBlockNOffset + tailBasicBlockIdx * tiling_->firstTailBlockL1Size;
            basicBlockOffsetParam_.nL1Size =
                (basicBlockOffsetParam_.nOffset + tiling_->firstTailBlockL1Size) > tiling_->nSize ?
                    tiling_->nSize - basicBlockOffsetParam_.nOffset :
                    tiling_->firstTailBlockL1Size;
            wqmmBasicBlock_.ComputeBasicBlock(basicBlockOffsetParam_);
        }
        // 第一段尾块的最后位置，也是第二段尾块的起点
        uint64_t firstTailBlockNOffset =
            mainBlockNOffset + tiling_->firstTailBlockCount * tiling_->firstTailBlockL1Size;
        for (; tailBasicBlockIdx < totalTailBlockCount; tailBasicBlockIdx += tiling_->cubeBlockDimN) {
            basicBlockOffsetParam_.nOffset =
                firstTailBlockNOffset +
                (tailBasicBlockIdx - tiling_->firstTailBlockCount) * tiling_->secondTailBlockL1Size;
            basicBlockOffsetParam_.nL1Size =
                (basicBlockOffsetParam_.nOffset + tiling_->secondTailBlockL1Size) > tiling_->nSize ?
                    tiling_->nSize - basicBlockOffsetParam_.nOffset :
                    tiling_->secondTailBlockL1Size;
            wqmmBasicBlock_.ComputeBasicBlock(basicBlockOffsetParam_);
        }
    }
    wqmmBasicBlock_.End();
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONTROLLER_H
