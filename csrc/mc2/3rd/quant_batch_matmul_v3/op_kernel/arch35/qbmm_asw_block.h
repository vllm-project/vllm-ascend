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
 * \file qbmm_asw_block.h
 * \brief
 */
#ifndef MC2_QBMM_ASW_BLOCK_H
#define MC2_QBMM_ASW_BLOCK_H

#include "quant_batch_matmul_v3_tiling_data.h"
#include "../quant_batch_matmul_v3_base.h"

namespace Mc2QuantBatchMatmulV3 {
constexpr uint64_t PER_BLOCK_SIZE = 128;
struct ASWTilingParam {
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
    uint64_t groupSizeM;
    uint64_t groupSizeK;
    uint64_t groupSizeN;
};

struct ASWOffsetParam {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetScale;
    uint64_t offsetBias;
    uint64_t offsetPerTokenScale;
    uint64_t scaleScalar;
    uint64_t batchAOffset;
    uint64_t batchBOffset;
    uint64_t batchCOffset;
};

struct PerBlockUBParam {
    bool fixpipeSplitN = false;
    uint64_t offsetM;
    uint64_t offsetN;
    uint64_t singleM;
    uint64_t singleN;
    uint64_t validM;
    uint64_t validN;
    uint64_t ubCountM;
    uint64_t ubCountN;
    uint64_t mainPartM;
    uint64_t mainPartN;
    uint64_t offsetC;
    uint64_t offsetScaleM;
    uint64_t offsetScaleN;
};

struct PerBlockMmParam {
    bool fixpipeSplitN = false;
    uint64_t fixpipeM;
    uint64_t fixpipeN;
    uint64_t fixpipeD;
    uint64_t fixSrcStride;
};

__aicore__ inline constexpr uint32_t GetVectorRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

class Mc2QuantBmmAswBlock {
public:
    __aicore__ inline Mc2QuantBmmAswBlock()
    {
    }
    __aicore__ inline void Init(const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData, uint32_t blockIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline void UpdateBasicIndex4AL1FullLoad(uint64_t roundIdx);
    template <bool bTrans, CubeFormat formatX2 = CubeFormat::ND>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <bool bTrans, CubeFormat formatX2 = CubeFormat::ND>
    __aicore__ inline void UpdateBlockParams4AL1FullLoad(uint64_t roundIdx);
    template <bool aTrans, bool bTrans, class x1Type, class scaleType = float, CubeFormat formatX2 = CubeFormat::ND>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void ResetAddressOffsets();
    __aicore__ inline void UpdatePerBlockUBValidMN();
    __aicore__ inline void UpdatePerBlockUBParam();
    template <bool aTrans, bool bTrans>
    __aicore__ inline void UpdatePerBlockMmParam();

public:
    ASWTilingParam params_;
    ASWOffsetParam offset_;
    PerBlockUBParam ubParams_;
    PerBlockMmParam mmParams_;
    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData_;

private:
    const uint64_t WINDOW_LEN = 4;
    static constexpr uint64_t CUBE_BLOCK = 16UL;
    uint32_t blockIdx_;
};

__aicore__ inline void Mc2QuantBmmAswBlock::Init(const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData,
                                                 uint32_t blockIdx)
{
    params_.mSplitAddrOffset = 0UL;
    params_.nSplitAddrOffset = 0UL;
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.mCnt = DequantBmm::CeilDiv(tilingData_->matmulTiling.M, tilingData_->matmulTiling.baseM);
    params_.nCnt = DequantBmm::CeilDiv(tilingData_->matmulTiling.N, tilingData_->matmulTiling.baseN);
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTail = tilingData_->matmulTiling.M - (params_.mCnt - 1UL) * tilingData_->matmulTiling.baseM;
    params_.nBaseTail = tilingData_->matmulTiling.N - (params_.nCnt - 1UL) * tilingData_->matmulTiling.baseN;
    params_.totalTailTile = static_cast<uint64_t>(tilingData_->adaptiveSlidingWin.mTailTile) *
                            static_cast<uint64_t>(tilingData_->adaptiveSlidingWin.nTailTile);
    params_.round = DequantBmm::CeilDiv(params_.totalCnt, tilingData_->matmulTiling.usedCoreNum);
    params_.mCoreNum = DequantBmm::Min(WINDOW_LEN, params_.mCnt);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1UL;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;
    params_.groupSizeM = tilingData_->params.groupSizeM;
    params_.groupSizeK = tilingData_->params.groupSizeK;
    params_.groupSizeN = tilingData_->params.groupSizeN;

    offset_.offsetBias = 0UL;
}

__aicore__ inline void Mc2QuantBmmAswBlock::UpdateBasicIndex(uint64_t roundIdx)
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

__aicore__ inline void Mc2QuantBmmAswBlock::UpdateBasicIndex4AL1FullLoad(uint64_t roundIdx)
{
    params_.index = blockIdx_ / params_.totalTailTile + roundIdx * tilingData_->matmulTiling.usedCoreNum;
    params_.mIndex = blockIdx_ % params_.mCnt;
    params_.nIndex = roundIdx * tilingData_->matmulTiling.usedCoreNum / params_.mCnt % params_.nCnt +
                     blockIdx_ / params_.mCnt / tilingData_->adaptiveSlidingWin.nTailTile;
}

template <bool bTrans, CubeFormat formatX2>
__aicore__ inline void Mc2QuantBmmAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? tilingData_->matmulTiling.baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->matmulTiling.baseN : params_.nBaseTail;
    if (tilingData_->adaptiveSlidingWin.mTailTile == 1 && tilingData_->adaptiveSlidingWin.nTailTile == 1) {
        return;
    }

    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + tilingData_->adaptiveSlidingWin.mTailTile - 1) /
                                    tilingData_->adaptiveSlidingWin.mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + tilingData_->adaptiveSlidingWin.nTailTile - 1) /
                                    tilingData_->adaptiveSlidingWin.nTailTile;
        if constexpr (formatX2 != CubeFormat::ND) {
            if constexpr (bTrans) {
                singleCoreNSplit = DequantBmm::Align(singleCoreNSplit, CUBE_BLOCK);
            } else {
                singleCoreNSplit = DequantBmm::Align(singleCoreNSplit, static_cast<uint64_t>(K0_INT8));
            }
        }
        uint64_t mSplitIdx = (blockIdx_ % params_.totalTailTile) % tilingData_->adaptiveSlidingWin.mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % params_.totalTailTile) / tilingData_->adaptiveSlidingWin.mTailTile;
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

__aicore__ inline void Mc2QuantBmmAswBlock::ResetAddressOffsets()
{
    params_.mSplitAddrOffset = 0UL;
    params_.nSplitAddrOffset = 0UL;
}

template <bool bTrans, CubeFormat formatX2>
__aicore__ inline void Mc2QuantBmmAswBlock::UpdateBlockParams4AL1FullLoad(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? tilingData_->matmulTiling.baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->matmulTiling.baseN : params_.nBaseTail;
    if (tilingData_->adaptiveSlidingWin.mTailTile == 1 && tilingData_->adaptiveSlidingWin.nTailTile == 1) {
        return;
    }

    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = tilingData_->matmulTiling.baseM;
        uint64_t singleCoreNSplit = (params_.singleCoreN + tilingData_->adaptiveSlidingWin.nTailTile - 1) /
                                    tilingData_->adaptiveSlidingWin.nTailTile;
        if constexpr (formatX2 != CubeFormat::ND) {
            if constexpr (bTrans) {
                singleCoreNSplit = DequantBmm::Align(singleCoreNSplit, CUBE_BLOCK);
            } else {
                singleCoreNSplit = DequantBmm::Align(singleCoreNSplit, static_cast<uint64_t>(K0_INT8));
            }
        }
        uint64_t mSplitIdx = blockIdx_ % params_.mCnt;
        uint64_t nSplitIdx = blockIdx_ / params_.mCnt % tilingData_->adaptiveSlidingWin.nTailTile;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= tilingData_->matmulTiling.M ||
            params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        params_.mSplitAddrOffset = 0;
        if (params_.nSplitAddrOffset + singleCoreNSplit > params_.singleCoreN) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

template <bool aTrans, bool bTrans, class x1Type, class scaleType, CubeFormat formatX2>
__aicore__ inline void Mc2QuantBmmAswBlock::CalcGMOffset()
{
    uint64_t mOffset = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;
    if constexpr (aTrans) {
        offset_.offsetA = mOffset;
    } else {
        offset_.offsetA = mOffset * tilingData_->matmulTiling.Ka;
    }
    offset_.offsetA += offset_.batchAOffset * tilingData_->matmulTiling.M * tilingData_->matmulTiling.Ka;

    if constexpr (formatX2 == CubeFormat::ND) {
        if constexpr (bTrans) {
            offset_.offsetB = nOffset * tilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = nOffset;
        }
        offset_.offsetB += offset_.batchBOffset * tilingData_->matmulTiling.N * tilingData_->matmulTiling.Kb;
    } else {
        if constexpr (bTrans) {
            // (k1, n1, n0, k0)
            if constexpr (DequantBmm::IsFp4<x1Type>()) {
                offset_.offsetB = nOffset << 6; // k0是64，通过左移6位表达乘64
                offset_.offsetB += offset_.batchBOffset * DequantBmm::Align16(tilingData_->matmulTiling.N) *
                                   DequantBmm::Align64(tilingData_->matmulTiling.Kb);
            } else {
                offset_.offsetB = nOffset << 5; // k0是32，通过左移5位表达乘32
                offset_.offsetB += offset_.batchBOffset * DequantBmm::Align16(tilingData_->matmulTiling.N) *
                                   DequantBmm::Align32(tilingData_->matmulTiling.Kb);
            }
        } else {
            // (n1, k1, k0, n0)
            uint64_t kAlign = DequantBmm::Align16(tilingData_->matmulTiling.Kb);
            offset_.offsetB = nOffset * kAlign; // tiling保证切到n1
            offset_.offsetB += offset_.batchBOffset * DequantBmm::Align32(tilingData_->matmulTiling.N) * kAlign;
        }
    }

    offset_.offsetC = mOffset * tilingData_->matmulTiling.N + nOffset;
    offset_.offsetC += offset_.batchCOffset * tilingData_->matmulTiling.N * tilingData_->matmulTiling.M;

    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        int32_t pertokenScaleK = MXFP_MULTI_BASE_SIZE;
        int32_t scaleK = MXFP_MULTI_BASE_SIZE;
        if constexpr (!aTrans) {
            pertokenScaleK *= DequantBmm::CeilDiv(tilingData_->matmulTiling.Ka, MXFP_DIVISOR_SIZE);
        }
        if constexpr (bTrans) {
            scaleK *= DequantBmm::CeilDiv(tilingData_->matmulTiling.Ka, MXFP_DIVISOR_SIZE);
        }
        offset_.offsetPerTokenScale = mOffset * pertokenScaleK;
        offset_.offsetScale = nOffset * scaleK;
    } else {
        offset_.offsetPerTokenScale = mOffset;
        offset_.offsetScale = nOffset;
    }

    if (tilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nOffset;
        if (static_cast<bool>(tilingData_->params.biasThreeDim)) {
            offset_.offsetBias += offset_.batchCOffset * tilingData_->matmulTiling.N;
        }
    }
}

__aicore__ inline void Mc2QuantBmmAswBlock::UpdatePerBlockUBValidMN()
{
    if (ubParams_.fixpipeSplitN) {
        ubParams_.validM = ubParams_.singleM;
        uint64_t dirtyN = mmParams_.fixpipeN - params_.singleCoreN;
        if (ubParams_.singleN > dirtyN) {
            if (AscendC::GetSubBlockIdx() == 0) {
                ubParams_.validN = ubParams_.singleN;
            } else {
                ubParams_.validN = ubParams_.singleN - dirtyN;
            }
        } else {
            if (AscendC::GetSubBlockIdx() == 0) {
                ubParams_.validN = params_.singleCoreN;
            } else {
                ubParams_.validN = 0;
            }
        }
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            ubParams_.validM = ubParams_.singleM;
        } else {
            ubParams_.validM = params_.singleCoreM - ubParams_.singleM;
        }
        ubParams_.validN = params_.singleCoreN;
    }
}

__aicore__ inline void Mc2QuantBmmAswBlock::UpdatePerBlockUBParam()
{
    ubParams_.fixpipeSplitN = params_.singleCoreN > PER_BLOCK_SIZE || params_.singleCoreM == 1;
    ubParams_.offsetM = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    ubParams_.offsetN = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;
    ubParams_.singleM =
        ubParams_.fixpipeSplitN ?
            params_.singleCoreM :
            DequantBmm::CeilDiv(params_.singleCoreM, static_cast<uint64_t>(DequantBmm::GetTaskRation()));
    ubParams_.singleN = mmParams_.fixpipeD;
    if (AscendC::GetSubBlockIdx() == 1) {
        auto ubParamsRes = ubParams_.fixpipeSplitN ? 0 : ubParams_.singleM;
        ubParams_.offsetM = ubParams_.offsetM + ubParamsRes;
        if (ubParams_.fixpipeSplitN) {
            uint64_t dirtyN = mmParams_.fixpipeN - params_.singleCoreN;
            if (ubParams_.singleN > dirtyN) {
                ubParams_.offsetN = ubParams_.offsetN + ubParams_.singleN;
            }
        }
    }
    UpdatePerBlockUBValidMN();
    ubParams_.offsetScaleM = ubParams_.offsetM / params_.groupSizeM;
    ubParams_.offsetScaleN = ubParams_.offsetN / PER_BLOCK_SIZE;
    ubParams_.offsetC = ubParams_.offsetM * tilingData_->matmulTiling.N + ubParams_.offsetN;
    ubParams_.offsetC += offset_.batchCOffset * tilingData_->matmulTiling.N * tilingData_->matmulTiling.M;
}

template <bool aTrans, bool bTrans>
__aicore__ inline void Mc2QuantBmmAswBlock::UpdatePerBlockMmParam()
{
    mmParams_.fixpipeSplitN = params_.singleCoreN > PER_BLOCK_SIZE || params_.singleCoreM == 1;
    if ASCEND_IS_AIC {
        if constexpr (aTrans) {
            mmParams_.fixSrcStride = DequantBmm::Align(params_.singleCoreM, static_cast<uint64_t>(DATA_BLOCK));
        } else {
            mmParams_.fixSrcStride = DequantBmm::Align(params_.singleCoreM, static_cast<uint64_t>(BMM_BLOCK_NUM));
        }
        mmParams_.fixpipeM =
            mmParams_.fixpipeSplitN ?
                params_.singleCoreM :
                DequantBmm::Align(params_.singleCoreM, static_cast<uint64_t>(DequantBmm::GetTaskRation()));
    }
    if (mmParams_.fixpipeSplitN) {
        mmParams_.fixpipeN = DequantBmm::Align(params_.singleCoreN, static_cast<uint64_t>(DATA_BLOCK));
        mmParams_.fixpipeD = mmParams_.fixpipeN / static_cast<uint64_t>(DequantBmm::GetTaskRation());
    } else {
        mmParams_.fixpipeN = DequantBmm::Align(params_.singleCoreN, static_cast<uint64_t>(BMM_BLOCK_NUM));
        mmParams_.fixpipeD = mmParams_.fixpipeN;
    }
}
} // namespace Mc2QuantBatchMatmulV3

#endif // QBMM_ASW_BLOCK_H
