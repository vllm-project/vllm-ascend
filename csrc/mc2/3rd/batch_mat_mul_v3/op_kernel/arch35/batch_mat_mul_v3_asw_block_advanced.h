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
 * \file batch_mat_mul_v3_asw_block_advanced.h
 * \brief
 */
#ifndef BATCH_MATMUL_V3_ASW_BLOCK_ADVANCED_H
#define BATCH_MATMUL_V3_ASW_BLOCK_ADVANCED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../../../mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../../mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

namespace Mc2BatchMatMulV3Advanced {

using namespace AscendC;
using namespace matmul;

struct Mc2BmmAswBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct Mc2BmmAswBlockArgs {
    uint64_t batchA1 = 1;
    uint64_t batchA2 = 1;
    uint64_t batchA3 = 1;
    uint64_t batchA4 = 1;
    uint64_t batchB1 = 1;
    uint64_t batchB2 = 1;
    uint64_t batchB3 = 1;
    uint64_t batchB4 = 1;
    uint64_t batchC1 = 1;
    uint64_t batchC2 = 1;
    uint64_t batchC3 = 1;
    uint64_t batchC4 = 1;
    uint64_t aBatchDimAll = 1;
    uint64_t bBatchDimAll = 1;
    uint64_t cBatchDimAll = 1;

    uint64_t batchCnt = 0;

    uint64_t index = 0;
    uint64_t mCntIndex = 0;
    uint64_t nCntIndex = 0;
    uint64_t mCnt = 0;
    uint64_t nCnt = 0;
    uint64_t totalCnt = 0;
    uint64_t blockBaseM = 0;
    uint64_t blockBaseN = 0;
    uint64_t nBaseTail = 0;
    uint64_t mBaseTail = 0;
    uint64_t mBaseSplitCnt = 1;
    uint64_t nBaseSplitCnt = 1;
    uint64_t totalSplitCnt = 1;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreN = 0;
    uint64_t round = 0;
    uint64_t roundToReverse = 0;
    uint64_t mainRow = 0;
    uint64_t mainWindow = 0;
    uint64_t tailWindow = 0;
    uint64_t kbAlignSize = 0;
    uint64_t nAlignSize = 0;
};


class Mc2BatchMatMulAswBlock {
public:
    __aicore__ inline Mc2BatchMatMulAswBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();

public:
    Mc2BmmAswBlockOffset offset_;
    Mc2BmmAswBlockArgs params_;
    const BatchMatMulV3TilingData *batchMatmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2BatchMatMulAswBlock::Init(const void *tilingData)
{
    batchMatmulTilingData_ = static_cast<const BatchMatMulV3TilingData *>(tilingData);

    params_.batchA1 = batchMatmulTilingData_->aBatchDim0;
    params_.batchA2 = batchMatmulTilingData_->aBatchDim1;
    params_.batchA3 = batchMatmulTilingData_->aBatchDim2;
    params_.batchA4 = batchMatmulTilingData_->aBatchDim3;
    params_.batchB1 = batchMatmulTilingData_->bBatchDim0;
    params_.batchB2 = batchMatmulTilingData_->bBatchDim1;
    params_.batchB3 = batchMatmulTilingData_->bBatchDim2;
    params_.batchB4 = batchMatmulTilingData_->bBatchDim3;
    params_.batchC1 = batchMatmulTilingData_->cBatchDim0;
    params_.batchC2 = batchMatmulTilingData_->cBatchDim1;
    params_.batchC3 = batchMatmulTilingData_->cBatchDim2;
    params_.batchC4 = batchMatmulTilingData_->cBatchDim3;
    params_.aBatchDimAll = batchMatmulTilingData_->aBatchDimAll;
    params_.bBatchDimAll = batchMatmulTilingData_->bBatchDimAll;
    params_.cBatchDimAll = batchMatmulTilingData_->cBatchDimAll;

    params_.batchCnt = params_.batchC1 * params_.batchC2 * params_.batchC3 * params_.batchC4;

    params_.blockBaseM = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.baseN);
    params_.mCnt = (batchMatmulTilingData_->matMulTilingData.tCubeTiling.M + params_.blockBaseM - 1) /
                   params_.blockBaseM; // 总的m方向base块个数
    params_.nCnt = (batchMatmulTilingData_->matMulTilingData.tCubeTiling.N + params_.blockBaseN - 1) /
                   params_.blockBaseN; // 总的n方向base块个数
    params_.nBaseTail = batchMatmulTilingData_->matMulTilingData.tCubeTiling.N -
                        (params_.nCnt - 1) * params_.blockBaseN; // n方向上的base尾块
    params_.mBaseTail = batchMatmulTilingData_->matMulTilingData.tCubeTiling.M -
                        (params_.mCnt - 1) * params_.blockBaseM; // m方向上的base尾块
    params_.totalCnt = params_.batchCnt * params_.mCnt * params_.nCnt;
    params_.round = (params_.totalCnt + batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum - 1) /
                    batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    params_.mainWindow = AscendC::Std::min(static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.aswWindowLen),
                                           params_.mCnt);                     // 主划窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mainWindow - 1;                  // 主划窗数量
    params_.tailWindow = params_.mCnt - params_.mainRow * params_.mainWindow; // 尾划窗m方向的块个数
    using B_T = typename B_TYPE::T;
    params_.kbAlignSize = (B_TYPE::isTrans) ? BLOCK_BYTE_SIZE / sizeof(B_T) : BLOCK_SIZE;
    params_.nAlignSize = (B_TYPE::isTrans) ? BLOCK_SIZE : BLOCK_BYTE_SIZE / sizeof(B_T);
}

__aicore__ inline void Mc2BatchMatMulAswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    uint64_t matIndex = params_.index % (params_.mCnt * params_.nCnt);
    uint64_t rowIdx = matIndex / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + matIndex % params_.mainWindow;
        params_.nCntIndex = (matIndex / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = matIndex - params_.mainRow * params_.mainWindow * params_.nCnt;
        params_.mCntIndex = params_.mainRow * params_.mainWindow + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % 2 != 0) {
        params_.nCntIndex = params_.nCnt - 1 - params_.nCntIndex;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2BatchMatMulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mCntIndex != (params_.mCnt - 1) ? params_.blockBaseM : params_.mBaseTail;
    params_.singleCoreN = params_.nCntIndex != (params_.nCnt - 1) ? params_.blockBaseN : params_.nBaseTail;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2BatchMatMulAswBlock::CalcGMOffset()
{
    uint64_t batchC1Index =
        params_.index / (params_.batchC2 * params_.batchC3 * params_.batchC4 * params_.mCnt * params_.nCnt);
    uint64_t batchC2Index = params_.index %
                            (params_.batchC2 * params_.batchC3 * params_.batchC4 * params_.mCnt * params_.nCnt) /
                            (params_.batchC3 * params_.batchC4 * params_.mCnt * params_.nCnt);
    uint64_t batchC3Index = params_.index % (params_.batchC3 * params_.batchC4 * params_.mCnt * params_.nCnt) /
                            (params_.batchC4 * params_.mCnt * params_.nCnt);
    uint64_t batchC4Index =
        params_.index % (params_.batchC4 * params_.mCnt * params_.nCnt) / (params_.mCnt * params_.nCnt);
    uint64_t batchCIndex = params_.index / (params_.mCnt * params_.nCnt);
    uint64_t batchA1Index = batchC1Index % params_.batchA1;
    uint64_t batchA2Index = batchC2Index % params_.batchA2;
    uint64_t batchA3Index = batchC3Index % params_.batchA3;
    uint64_t batchA4Index = batchC4Index % params_.batchA4;
    uint64_t batchAIndex = batchA1Index * (params_.batchA2 * params_.batchA3 * params_.batchA4) +
                           batchA2Index * (params_.batchA3 * params_.batchA4) + batchA3Index * params_.batchA4 +
                           batchA4Index;
    uint64_t offsetABatch = batchAIndex * batchMatmulTilingData_->matMulTilingData.tCubeTiling.M *
                            static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka);
    uint64_t batchB1Index = batchC1Index % params_.batchB1;
    uint64_t batchB2Index = batchC2Index % params_.batchB2;
    uint64_t batchB3Index = batchC3Index % params_.batchB3;
    uint64_t batchB4Index = batchC4Index % params_.batchB4;
    uint64_t batchBIndex = batchB1Index * (params_.batchB2 * params_.batchB3 * params_.batchB4) +
                           batchB2Index * (params_.batchB3 * params_.batchB4) + batchB3Index * params_.batchB4 +
                           batchB4Index;
    uint64_t offsetBBatch = batchBIndex * batchMatmulTilingData_->matMulTilingData.tCubeTiling.N *
                            static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb);

    if constexpr (A_TYPE::isTrans) {
        offset_.offsetA = offsetABatch + params_.mCntIndex * params_.blockBaseM;
    } else {
        offset_.offsetA =
            offsetABatch + (params_.mCntIndex * params_.blockBaseM) *
                               static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka);
    }
    if constexpr (B_TYPE::isTrans) {
        offset_.offsetB =
            offsetBBatch + (params_.nCntIndex * params_.blockBaseN) *
                               static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb);
    } else {
        offset_.offsetB = offsetBBatch + params_.nCntIndex * params_.blockBaseN;
    }
    offset_.offsetC = batchCIndex * static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
                          batchMatmulTilingData_->matMulTilingData.tCubeTiling.N +
                      (params_.nCntIndex * params_.blockBaseN) +
                      (params_.mCntIndex * params_.blockBaseM) * batchMatmulTilingData_->matMulTilingData.tCubeTiling.N;
    if (batchMatmulTilingData_->matMulTilingData.tCubeTiling.isBias) {
        offset_.offsetBias = (batchCIndex % batchMatmulTilingData_->biasBatchDimAll) *
                                 static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.N) +
                             params_.nCntIndex * params_.blockBaseN;
    }
}

} // namespace Mc2BatchMatMulV3Advanced

#endif // BATCH_MAT_MUL_V3_ASW_BLOCK_ADVANCED_H
