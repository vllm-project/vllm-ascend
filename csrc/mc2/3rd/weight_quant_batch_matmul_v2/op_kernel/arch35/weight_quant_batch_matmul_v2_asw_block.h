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
 * \file weight_quant_batch_matmul_v2_asw_block.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BMMV2_ASW_BLOCK_H
#define WEIGHT_QUANT_BMMV2_ASW_BLOCK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../tool.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {
struct Mc2ASWTilingParam {
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t mCoreNum;
    uint64_t mTailCoreNum;
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t mIndex;
    uint64_t nIndex;
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t totalTailTile;
    uint64_t mainRow;
    uint64_t round;
    uint64_t index;
};

struct Mc2ASWOffsetParam {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetScale;
    uint64_t scaleScalar;
};

class Mc2WeightQuantBmmAswBlock {
public:
    __aicore__ inline Mc2WeightQuantBmmAswBlock()
    {
    }
    __aicore__ inline void Init(const WeightQuantBatchMatmulV2ASWTilingData *tilingData, uint32_t blockIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <bool aTrans, bool bTrans, CubeFormat formatX2 = CubeFormat::ND>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void ResetAddressOffsets();

public:
    Mc2ASWTilingParam params_;
    Mc2ASWOffsetParam offset_;
    const WeightQuantBatchMatmulV2ASWTilingData *tilingData_;

private:
    const uint64_t WINDOW_LEN = 4;
    uint32_t blockIdx_;
};

__aicore__ inline void Mc2WeightQuantBmmAswBlock::Init(const WeightQuantBatchMatmulV2ASWTilingData *tilingData,
                                                       uint32_t blockIdx)
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.mCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matmulTiling.M),
                           static_cast<uint64_t>(tilingData_->matmulTiling.baseM));
    params_.nCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matmulTiling.N),
                           static_cast<uint64_t>(tilingData_->matmulTiling.baseN));
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTail = tilingData_->matmulTiling.M - (params_.mCnt - 1) * tilingData_->matmulTiling.baseM;
    params_.nBaseTail = tilingData_->matmulTiling.N - (params_.nCnt - 1) * tilingData_->matmulTiling.baseN;
    params_.totalTailTile = tilingData_->mTailTile * tilingData_->nTailTile;
    params_.round = CeilDiv(params_.totalCnt, static_cast<uint64_t>(tilingData_->matmulTiling.usedCoreNum));
    params_.mCoreNum = Min(WINDOW_LEN, params_.mCnt);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;
}

__aicore__ inline void Mc2WeightQuantBmmAswBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    uint64_t newBlockIdx = (roundIdx == params_.round - 1) ? (blockIdx_ / params_.totalTailTile) : blockIdx_;
    params_.index = newBlockIdx + roundIdx * tilingData_->matmulTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mCoreNum;
    if (rowIdx < params_.mainRow) {
        params_.mIndex = rowIdx * params_.mCoreNum + params_.index % params_.mCoreNum;
        params_.nIndex = (params_.index / params_.mCoreNum) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mCoreNum * params_.nCnt;
        params_.mIndex = params_.mainRow * params_.mCoreNum + tailIndex % params_.mTailCoreNum;
        params_.nIndex = (tailIndex / params_.mTailCoreNum) % params_.nCnt;
    }

    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

__aicore__ inline void Mc2WeightQuantBmmAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? tilingData_->matmulTiling.baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->matmulTiling.baseN : params_.nBaseTail;
    if (tilingData_->mTailTile == 1 && tilingData_->nTailTile == 1) {
        return;
    }
    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + tilingData_->mTailTile - 1) / tilingData_->mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + tilingData_->nTailTile - 1) / tilingData_->nTailTile;
        uint64_t mSplitIdx = (blockIdx_ % params_.totalTailTile) % tilingData_->mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % params_.totalTailTile) / tilingData_->mTailTile;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (params_.mSplitAddrOffset + singleCoreMSplit > params_.singleCoreM) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (params_.nSplitAddrOffset + singleCoreNSplit > params_.singleCoreN) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

__aicore__ inline void Mc2WeightQuantBmmAswBlock::ResetAddressOffsets()
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
}

template <bool aTrans, bool bTrans, CubeFormat formatX2>
__aicore__ inline void Mc2WeightQuantBmmAswBlock::CalcGMOffset()
{
    uint64_t mOffset = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;
    if constexpr (aTrans) {
        offset_.offsetA = mOffset;
    } else {
        offset_.offsetA = mOffset * tilingData_->matmulTiling.Ka;
    }

    if constexpr (bTrans) {
        offset_.offsetB = nOffset * tilingData_->matmulTiling.Kb;
    } else {
        offset_.offsetB = nOffset;
    }

    offset_.offsetC = mOffset * tilingData_->matmulTiling.N + nOffset;
    offset_.offsetScale = nOffset;
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35

#endif // WEIGHT_QUANT_BMMV2_ASW_BLOCK_H
