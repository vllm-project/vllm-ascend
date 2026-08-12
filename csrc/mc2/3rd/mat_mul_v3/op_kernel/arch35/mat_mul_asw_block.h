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
 * \file mat_mul_asw_block.h
 * \brief
 */
#ifndef MMV3_MATMUL_ASW_BLOCK_H
#define MMV3_MATMUL_ASW_BLOCK_H

#include "../mat_mul_v3_common.h"
#include "mat_mul_tiling_data.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

constexpr uint64_t FP32_SPLIT_K_THRESHOLD = 8192UL;

struct AswBlockOffset {
    uint64_t offsetA = 0UL;
    uint64_t offsetB = 0UL;
    uint64_t offsetC = 0UL;
    uint64_t offsetBias = 0UL;
};

struct AswBlockArgs {
    uint64_t index = 0UL;
    uint64_t mCntIndex = 0UL;
    uint64_t nCntIndex = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t totalCnt = 0UL;
    uint64_t blockBaseM = 0UL;
    uint64_t blockBaseN = 0UL;
    uint64_t mBaseTailSplitCnt = 0UL;
    uint64_t nBaseTailSplitCnt = 0UL;
    uint64_t mBaseNormCnt = 0UL;
    uint64_t nBaseNormCnt = 0UL;
    uint64_t mBaseTail = 0UL;
    uint64_t nBaseTail = 0UL;
    uint64_t mBaseTailMain = 0UL;
    uint64_t mBaseTailLast = 0UL;
    uint64_t nBaseTailMain = 0UL;
    uint64_t nBaseTailLast = 0UL;
    uint64_t mBaseSplitCnt = 0UL;
    uint64_t nBaseSplitCnt = 0UL;
    uint64_t totalSplitCnt = 0UL;
    uint64_t mSplitAddrOffset = 0UL;
    uint64_t nSplitAddrOffset = 0UL;
    uint64_t singleCoreM = 0UL;
    uint64_t singleCoreN = 0UL;
    uint64_t round = 0UL;
    uint64_t roundToReverse = 0UL;
    uint64_t mainRow = 0UL;
    uint64_t mainWindow = 0UL;
    uint64_t tailWindow = 0UL;
    uint64_t kbAlignSize = 0UL;
    uint64_t nAlignSize = 0UL;

    uint64_t splitKRound = 1UL;
    uint64_t singleCoreSplitK = 0UL;
    uint64_t singleShapeKTail = 0UL;
};

class Mc2MatmulAswBlock {
public:
    __aicore__ inline Mc2MatmulAswBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    template <class A_TYPE, class B_TYPE>
    __aicore__ inline void LoadBalanceInit();
    __aicore__ inline uint64_t GetNewBlockIdx(uint64_t roundIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();
    template <class A_TYPE, class B_TYPE>
    __aicore__ inline void CalcSplitKGMOffset(uint64_t splitKIndex);
    template <typename IndexType, typename BaseNormCntType, typename BlockBaseType, typename BaseTailMainType,
              typename SplitAddrOffsetType>
    __aicore__ inline uint64_t CalculateOffset(IndexType cntIndex, BaseNormCntType baseNormCnt, BlockBaseType blockBase,
                                               BaseTailMainType baseTailMain, SplitAddrOffsetType splitAddrOffset);

public:
    AswBlockOffset offset_;
    AswBlockArgs params_;
    const Mc2MatMulV3TilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE>
__aicore__ inline void Mc2MatmulAswBlock::LoadBalanceInit()
{
    params_.mBaseTailMain =
        params_.mBaseTailSplitCnt == 1UL ? params_.mBaseTail : static_cast<uint64_t>(matmulTilingData_->mTailMain);
    params_.mBaseTailLast = params_.mBaseTail - (params_.mBaseTailSplitCnt - 1UL) * params_.mBaseTailMain;
    params_.nBaseTailMain =
        params_.nBaseTailSplitCnt == 1UL ? params_.nBaseTail : static_cast<uint64_t>(matmulTilingData_->nTailMain);
    params_.nBaseTailLast = params_.nBaseTail - (params_.nBaseTailSplitCnt - 1UL) * params_.nBaseTailMain;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulAswBlock::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const Mc2MatMulV3TilingData *>(tilingData);
    params_.index = 0UL;
    params_.singleCoreM = 0UL;
    params_.singleCoreN = 0UL;
    params_.mSplitAddrOffset = 0UL;
    params_.nSplitAddrOffset = 0UL;
    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseN);
    params_.mCnt = MMV3DivCeil(matmulTilingData_->tCubeTiling.M, params_.blockBaseM); // m方向base块数
    params_.nCnt = MMV3DivCeil(matmulTilingData_->tCubeTiling.N, params_.blockBaseN); // n方向base块数
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTailSplitCnt = static_cast<uint64_t>(matmulTilingData_->mBaseTailSplitCnt);
    params_.nBaseTailSplitCnt = static_cast<uint64_t>(matmulTilingData_->nBaseTailSplitCnt);
    params_.mBaseNormCnt = params_.mCnt - params_.mBaseTailSplitCnt;
    params_.nBaseNormCnt = params_.nCnt - params_.nBaseTailSplitCnt;

    // m方向上的base尾块
    params_.mBaseTail = matmulTilingData_->tCubeTiling.M - params_.mBaseNormCnt * params_.blockBaseM;
    // n方向上的base尾块
    params_.nBaseTail = matmulTilingData_->tCubeTiling.N - params_.nBaseNormCnt * params_.blockBaseN;
    LoadBalanceInit<A_TYPE, B_TYPE>();
    params_.round = (params_.totalCnt + matmulTilingData_->tCubeTiling.usedCoreNum - 1UL) /
                    matmulTilingData_->tCubeTiling.usedCoreNum;
    params_.mainWindow =
        AscendC::Std::min(static_cast<uint64_t>(matmulTilingData_->aswWindowLen), params_.mCnt); // 主划窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mainWindow - 1UL;                                   // 主划窗数量
    params_.tailWindow = params_.mCnt - params_.mainRow * params_.mainWindow; // 尾划窗m方向的块个数

    params_.mBaseSplitCnt = matmulTilingData_->mTailCnt;
    params_.nBaseSplitCnt = matmulTilingData_->nTailCnt;
    params_.totalSplitCnt = params_.mBaseSplitCnt * params_.nBaseSplitCnt;
    using B_T = typename B_TYPE::T;
    params_.kbAlignSize = (B_TYPE::isTrans) ? BLOCK_BYTE_SIZE / sizeof(B_T) : BLOCK_SIZE;
    params_.nAlignSize = (B_TYPE::isTrans) ? BLOCK_SIZE : BLOCK_BYTE_SIZE / sizeof(B_T);

    params_.splitKRound = 1UL;
    params_.singleCoreSplitK = matmulTilingData_->tCubeTiling.singleCoreK;
    params_.singleShapeKTail = matmulTilingData_->tCubeTiling.singleCoreK;
    constexpr bool isFp32 = std::is_same_v<typename C_TYPE::T, float>;
    // 如果是fp32且singleCoreK大于8192且B矩阵不是NZ格式，需要单核切K保精度，否则不需要切K
    if (isFp32 && !matmulTilingData_->isHf32 && B_TYPE::format == CubeFormat::ND &&
        matmulTilingData_->tCubeTiling.singleCoreK > FP32_SPLIT_K_THRESHOLD) {
        params_.splitKRound = MMV3DivCeil(matmulTilingData_->tCubeTiling.singleCoreK, FP32_SPLIT_K_THRESHOLD);
        params_.singleCoreSplitK = MMV3CeilAlign(
            MMV3DivCeil(matmulTilingData_->tCubeTiling.singleCoreK, params_.splitKRound), ALIGN_BYTE / DATA_SIZE_FP32);
        params_.singleShapeKTail = matmulTilingData_->tCubeTiling.singleCoreK % params_.singleCoreSplitK;
        if (params_.singleShapeKTail == 0UL) {
            params_.singleShapeKTail = params_.singleCoreSplitK;
        }
    }
}

__aicore__ inline uint64_t Mc2MatmulAswBlock::GetNewBlockIdx(uint64_t roundIdx)
{
    uint64_t newBlockIdx = GetBlockIdx();
    newBlockIdx = (roundIdx == params_.round - 1UL) ? (newBlockIdx / params_.totalSplitCnt) : newBlockIdx;
    return newBlockIdx;
}

__aicore__ inline void Mc2MatmulAswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * matmulTilingData_->tCubeTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + params_.index % params_.mainWindow;
        params_.nCntIndex = (params_.index / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mainWindow * params_.nCnt;
        params_.mCntIndex = params_.mainRow * params_.mainWindow + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % NUM_TWO != 0UL) {
        params_.nCntIndex = params_.nCnt - 1UL - params_.nCntIndex;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.blockBaseM;
    if (params_.mCntIndex >= params_.mBaseNormCnt) {
        params_.singleCoreM =
            (params_.mCntIndex >= (params_.mCnt - 1UL)) ? params_.mBaseTailLast : params_.mBaseTailMain;
    }

    params_.singleCoreN = params_.blockBaseN;
    if (params_.nCntIndex >= params_.nBaseNormCnt) {
        params_.singleCoreN =
            (params_.nCntIndex >= (params_.nCnt - 1UL)) ? params_.nBaseTailLast : params_.nBaseTailMain;
    }

    if (roundIdx == params_.round - 1UL && (params_.mBaseSplitCnt != 1UL || params_.nBaseSplitCnt != 1UL)) {
        uint64_t singleCoreMSplit = MMV3DivCeil(params_.singleCoreM, params_.mBaseSplitCnt);
        uint64_t singleCoreNSplit = MMV3DivCeil(params_.singleCoreN, params_.nBaseSplitCnt);
        if constexpr (B_TYPE::format != CubeFormat::ND) {
            singleCoreNSplit = MMV3CeilAlign(singleCoreNSplit, params_.nAlignSize);
        }
        params_.mBaseSplitCnt = MMV3DivCeil(params_.singleCoreM, singleCoreMSplit);
        params_.nBaseSplitCnt = MMV3DivCeil(params_.singleCoreN, singleCoreNSplit);
        uint64_t curBlockIdx = GetCurrentBlockIdx();
        uint64_t mSplitIdx = (curBlockIdx % params_.totalSplitCnt) % params_.mBaseSplitCnt;
        uint64_t nSplitIdx = (curBlockIdx % params_.totalSplitCnt) / params_.mBaseSplitCnt;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0UL;
            params_.singleCoreN = 0UL;
            return;
        }
        if (mSplitIdx + 1UL == params_.mBaseSplitCnt) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (nSplitIdx + 1UL == params_.nBaseSplitCnt) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

template <typename IndexType, typename BaseNormCntType, typename BlockBaseType, typename BaseTailMainType,
          typename SplitAddrOffsetType>
__aicore__ inline uint64_t Mc2MatmulAswBlock::CalculateOffset(IndexType cntIndex, BaseNormCntType baseNormCnt,
                                                              BlockBaseType blockBase, BaseTailMainType baseTailMain,
                                                              SplitAddrOffsetType splitAddrOffset)
{
    if (cntIndex > baseNormCnt) {
        return baseNormCnt * blockBase + (cntIndex - baseNormCnt) * baseTailMain + splitAddrOffset;
    }
    return cntIndex * blockBase + splitAddrOffset;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulAswBlock::CalcGMOffset()
{
    uint64_t mOffset = CalculateOffset(params_.mCntIndex, params_.mBaseNormCnt, params_.blockBaseM,
                                       params_.mBaseTailMain, params_.mSplitAddrOffset);
    uint64_t nOffset = CalculateOffset(params_.nCntIndex, params_.nBaseNormCnt, params_.blockBaseN,
                                       params_.nBaseTailMain, params_.nSplitAddrOffset);
    if constexpr (A_TYPE::isTrans) {
        offset_.offsetA = mOffset;
    } else {
        offset_.offsetA = mOffset * matmulTilingData_->tCubeTiling.Ka;
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if constexpr (B_TYPE::isTrans) {
            offset_.offsetB = nOffset * matmulTilingData_->tCubeTiling.Kb;
        } else {
            offset_.offsetB = nOffset;
        }
    } else {
        if constexpr (B_TYPE::isTrans) {
            offset_.offsetB = nOffset * params_.kbAlignSize;
        } else {
            offset_.offsetB = nOffset * MMV3CeilAlign(matmulTilingData_->tCubeTiling.Kb, params_.kbAlignSize);
        }
    }
    offset_.offsetC = nOffset + mOffset * matmulTilingData_->tCubeTiling.N;
    if (matmulTilingData_->tCubeTiling.isBias) {
        offset_.offsetBias = nOffset;
    }
}

template <class A_TYPE, class B_TYPE>
__aicore__ inline void Mc2MatmulAswBlock::CalcSplitKGMOffset(uint64_t splitKIndex)
{
    if (params_.splitKRound == 1) {
        return;
    }
    uint64_t mOffset = CalculateOffset(params_.mCntIndex, params_.mBaseNormCnt, params_.blockBaseM,
                                       params_.mBaseTailMain, params_.mSplitAddrOffset);
    uint64_t nOffset = CalculateOffset(params_.nCntIndex, params_.nBaseNormCnt, params_.blockBaseN,
                                       params_.nBaseTailMain, params_.nSplitAddrOffset);
    if constexpr (A_TYPE::isTrans) {
        offset_.offsetA = mOffset + splitKIndex * params_.singleCoreSplitK * matmulTilingData_->tCubeTiling.M;
    } else {
        offset_.offsetA = mOffset * matmulTilingData_->tCubeTiling.Ka + splitKIndex * params_.singleCoreSplitK;
    }
    if constexpr (B_TYPE::isTrans) {
        offset_.offsetB = nOffset * matmulTilingData_->tCubeTiling.Kb + splitKIndex * params_.singleCoreSplitK;
    } else {
        offset_.offsetB = nOffset + splitKIndex * params_.singleCoreSplitK * matmulTilingData_->tCubeTiling.N;
    }
}

} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_ASW_BLOCK_H
