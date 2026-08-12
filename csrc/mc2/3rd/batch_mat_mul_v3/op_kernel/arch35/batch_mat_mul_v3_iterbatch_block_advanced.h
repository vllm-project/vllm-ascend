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
 * \file batch_mat_mul_v3_iterbatch_block_advanced.h
 * \brief
 */
#ifndef BATCH_MATMUL_V3_ITERBATCH_BLOCK_ADVANCED_H
#define BATCH_MATMUL_V3_ITERBATCH_BLOCK_ADVANCED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

namespace Mc2BatchMatMulV3Advanced {
using namespace AscendC;
using namespace matmul;

struct Mc2MatMulMultiBatchBaseBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

struct Mc2MatMulMultiBatchBaseBlockArgs {
    uint64_t singleASize;
    uint64_t singleBSize;
    uint64_t singleCSize;
    uint64_t mainLoopPreCoreBatchNum;
    uint64_t lastLoopAllBatchNum;
    uint64_t lastLoopPreCoreBatchNum;
    uint64_t lastLoopBlockNum;
    uint64_t LoopTimes;
    uint64_t batchIndex;
    uint64_t batchANum;
    uint64_t batchBNum;
    uint64_t useCoreNum;
    bool isHf32;
};

class Mc2BatchMatMulMultiBatchBaseBlock {
public:
    __aicore__ inline Mc2BatchMatMulMultiBatchBaseBlock()
    {
    }
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void GetMultiBatchInfo(uint64_t loopIndex);
    __aicore__ inline void CalcGMOffset();

public:
    Mc2MatMulMultiBatchBaseBlockOffset offset_;
    Mc2MatMulMultiBatchBaseBlockArgs params_;
    const BatchMatMulV3TilingData *batchMatmulTilingData_;
};

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::Init(const void *tilingData)
{
    batchMatmulTilingData_ = static_cast<const BatchMatMulV3TilingData *>(tilingData);
    params_.isHf32 = batchMatmulTilingData_->matMulTilingData.isHf32;

    params_.singleASize = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
                          static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka);
    params_.singleBSize = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.N) *
                          static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb);
    params_.singleCSize = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
                          static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.N);

    params_.useCoreNum = batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    params_.mainLoopPreCoreBatchNum = batchMatmulTilingData_->iterBatch;

    params_.LoopTimes =
        MMV3DivCeil(batchMatmulTilingData_->cBatchDimAll, params_.mainLoopPreCoreBatchNum * params_.useCoreNum);

    params_.lastLoopAllBatchNum =
        batchMatmulTilingData_->cBatchDimAll % (params_.mainLoopPreCoreBatchNum * params_.useCoreNum);
    params_.lastLoopAllBatchNum = params_.lastLoopAllBatchNum == 0 ?
                                      params_.mainLoopPreCoreBatchNum * params_.useCoreNum :
                                      params_.lastLoopAllBatchNum;

    params_.lastLoopPreCoreBatchNum = MMV3DivFloor(params_.lastLoopAllBatchNum, params_.useCoreNum);
    params_.lastLoopBlockNum = params_.lastLoopAllBatchNum % params_.useCoreNum;
    params_.batchIndex = 0;
    params_.batchANum = 1;
    params_.batchBNum = 1;
}

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::GetMultiBatchInfo(uint64_t loopIndex)
{
    // main loop
    if (loopIndex + 1 < params_.LoopTimes) {
        params_.batchANum = params_.mainLoopPreCoreBatchNum;
        params_.batchBNum = params_.mainLoopPreCoreBatchNum;
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             GetCurrentBlockIdx() * params_.mainLoopPreCoreBatchNum;
        return;
    }

    // last loop
    if (GetCurrentBlockIdx() < params_.lastLoopBlockNum) {
        params_.batchANum = params_.lastLoopPreCoreBatchNum + 1;
        params_.batchBNum = params_.lastLoopPreCoreBatchNum + 1;
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             GetCurrentBlockIdx() * (params_.lastLoopPreCoreBatchNum + 1);
    } else {
        params_.batchANum = params_.lastLoopPreCoreBatchNum;
        params_.batchBNum = params_.lastLoopPreCoreBatchNum;
        params_.batchIndex = loopIndex * params_.mainLoopPreCoreBatchNum * params_.useCoreNum +
                             params_.lastLoopBlockNum * (params_.lastLoopPreCoreBatchNum + 1) +
                             (GetCurrentBlockIdx() - params_.lastLoopBlockNum) * params_.lastLoopPreCoreBatchNum;
    }
}

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::CalcGMOffset()
{
    offset_.offsetA = params_.batchIndex * params_.singleASize;
    offset_.offsetB = params_.batchIndex * params_.singleBSize;
    offset_.offsetC = params_.batchIndex * params_.singleCSize;
    offset_.offsetBias = (params_.batchIndex % batchMatmulTilingData_->biasBatchDimAll) *
                         static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.N);
}
} // namespace Mc2BatchMatMulV3Advanced

#endif // BATCH_MATMUL_V3_ITERBATCH_BLOCK_ADVANCED_H
