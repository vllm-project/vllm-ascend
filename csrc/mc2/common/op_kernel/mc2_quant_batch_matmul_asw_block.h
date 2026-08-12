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
 * \file mc2_quant_batch_matmul_asw_block.h
 * \brief
 */
#ifndef NEW_MC2_QUANT_BMMV3_ASW_BLOCK_H
#define NEW_MC2_QUANT_BMMV3_ASW_BLOCK_H
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_base.h"
#include "mc2_tiling_struct.h"

namespace Mc2MatmulV3 {
constexpr uint64_t PER_BLOCK_SIZE = 128;
constexpr uint64_t DEVICE_NUM = 64;        // group 内卡数， 目前定义为64，后续根据情况扩展
constexpr uint64_t SLIDING_WINDOW_LEN = 4; // 滑窗m方向大小

struct ASWTilingParam {
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t mCoreNum;     // m方向窗口长度
    uint64_t mTailCoreNum; // m方向尾块长度
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t mIndex;
    uint64_t nIndex;
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t totalTailTile;
    uint64_t mainRow; // 主窗口个数
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

__aicore__ inline uint64_t MMV3DivCeil(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

class QuantBatchMatmulAswBlock {
public:
    __aicore__ inline QuantBatchMatmulAswBlock()
    {
    }
    __aicore__ inline void Init(const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData, uint32_t blockIdx);
    __aicore__ inline bool UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx, bool isLast);
    template <bool aTrans, bool bTrans, class x1Type, class scaleType = float, CubeFormat formatX2 = CubeFormat::ND>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void ResetAddressOffsets();
    __aicore__ inline void Update();
    template <class scaleType = float>
    __aicore__ inline void UpdateOffset(uint32_t idx, bool isTail);
    __aicore__ inline void InitForMC2(const Mc2Tiling::RCSTiling &cfg, bool isTail, bool isGather, uint64_t preCoreNum);

public:
    ASWOffsetParam offset_;
    ASWTilingParam params_;
    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData_;
    uint64_t offsetsA_[DEVICE_NUM] = {0};
    uint64_t offsetsC_[DEVICE_NUM] = {0};
    uint64_t offsetsScale_[DEVICE_NUM] = {0};
    uint64_t curSliceM_ = 0;
    uint64_t headSliceM_ = 0;
    uint64_t mSliceCnt_ = 0;
    uint64_t rankM_ = 0;
    uint64_t rankK_ = 0;
    uint64_t rankN_ = 0;
    Mc2Tiling::RCSTiling cfg_;
    bool isGather_ = false;
    // 以下变量保存部分scalar计算值，避免重复计算
    uint64_t preCoreNum_ = 0;         // 前一轮计算除不尽基本快数
    uint64_t rankMN_ = 0;             // rankMN_ = rankM_ * rankN_
    uint64_t rankMK_ = 0;             // rankMK_ = rankM_ * rankK_
    uint64_t rankKN_ = 0;             // rankKN_ = rankK_ * rankN_
    uint64_t windowParams1_ = 0;      // 主滑窗的m方向基本块数
    uint64_t windowParams2_ = 0;      // 主滑窗的m方向基本块数 * nCnt
    uint64_t blockBaseMDotRankK_ = 0; // baseM * rankK_
    uint64_t totalSplitCnt_ = 1;      // tiling要修改
    bool doTailTile_ = false;         // 是否进行尾轮切分

private:
    const uint64_t WINDOW_LEN = 4;
    uint32_t blockIdx_ = 0;
    uint32_t taskRation_ = AscendC::GetTaskRation();
};

__aicore__ inline void
QuantBatchMatmulAswBlock::Init(const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData, uint32_t blockIdx)
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.totalTailTile = tilingData_->adaptiveSlidingWin.mTailTile * tilingData_->adaptiveSlidingWin.nTailTile;
    offset_.offsetBias = 0;
}

__aicore__ inline void QuantBatchMatmulAswBlock::InitForMC2(const Mc2Tiling::RCSTiling &cfg, bool isTail, bool isGather,
                                                            uint64_t preCoreNum)
{
    cfg_ = cfg;
    rankM_ = isGather ? cfg.rankM : cfg.rankM / cfg.rankDim;
    isGather_ = isGather;
    rankN_ = cfg.rankN;
    rankK_ = cfg.rankK;
    rankMN_ = rankM_ * rankN_;
    rankMK_ = rankM_ * rankK_;
    rankKN_ = rankK_ * rankN_;
    preCoreNum_ = preCoreNum;

    headSliceM_ = (rankM_ - cfg.tailM * cfg.tailCnt) / cfg.tileCnt; // 头快的shapeM
    curSliceM_ = isTail ? cfg.tailM : headSliceM_;                  // 当前计算的shapeM
    mSliceCnt_ = MMV3DivCeil(curSliceM_, tilingData_->matmulTiling.baseM);

    uint64_t calRankNum = isGather_ ? (cfg.rankDim - 1) : cfg.rankDim;
    params_.mCnt = mSliceCnt_ * calRankNum;
    params_.nCnt = DequantBmm::CeilDiv(tilingData_->matmulTiling.N, tilingData_->matmulTiling.baseN);
    params_.totalCnt = params_.mCnt * params_.nCnt;

    // 重新计算
    params_.mBaseTail = curSliceM_ - (mSliceCnt_ - 1) * tilingData_->matmulTiling.baseM; // m方向的 base 尾块
    params_.nBaseTail =
        tilingData_->matmulTiling.N - (params_.nCnt - 1) * tilingData_->matmulTiling.baseN; // n方向上的base尾块

    params_.round =
        (params_.totalCnt + tilingData_->matmulTiling.usedCoreNum - 1) / tilingData_->matmulTiling.usedCoreNum;

    params_.mCoreNum = SLIDING_WINDOW_LEN < params_.mCnt ? SLIDING_WINDOW_LEN : params_.mCnt; // 主滑窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1;                                    // 主滑窗数量
    windowParams1_ = params_.mainRow * params_.mCoreNum;
    windowParams2_ = windowParams1_ * params_.nCnt;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow; // 尾滑窗m方向的块个数
    blockBaseMDotRankK_ = tilingData_->matmulTiling.baseM * rankK_;
}

__aicore__ inline void QuantBatchMatmulAswBlock::Update()
{
    params_.totalCnt = params_.mCnt * params_.nCnt + preCoreNum_;
    params_.round =
        (params_.totalCnt + tilingData_->matmulTiling.usedCoreNum - 1) / tilingData_->matmulTiling.usedCoreNum;
}

template <class scaleType>
__aicore__ inline void QuantBatchMatmulAswBlock::UpdateOffset(uint32_t idx, bool isTail)
{
    uint64_t offsetHeadSliceM = 0;
    uint64_t offsetTailSliceM = 0;

    if (isTail) {
        offsetHeadSliceM = cfg_.tileCnt * headSliceM_;
        offsetTailSliceM = idx * cfg_.tailM;
    } else {
        offsetHeadSliceM = idx * headSliceM_;
    }
    uint64_t scaleK =
        DequantBmm::Align(DequantBmm::CeilDiv(tilingData_->matmulTiling.Ka, MXFP_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t offsetHeadAndTailK = (offsetHeadSliceM + offsetTailSliceM) * rankK_;
    uint64_t offserHeadAndTailN = (offsetHeadSliceM + offsetTailSliceM) * rankN_;
    uint64_t scaleOffsetHeadAndTailK = (offsetHeadSliceM + offsetTailSliceM) * scaleK;

    uint64_t cnt = 0;
    for (uint32_t i = 0; i < cfg_.rankDim; i++) {
        if (isGather_ && (i == cfg_.rankID)) {
            offsetsA_[i] = 0;
            offsetsC_[i] = 0;
            offsetsScale_[i] = 0;
            continue;
        }

        offsetsA_[cnt] = i * rankMK_ + offsetHeadAndTailK;
        offsetsC_[cnt] = i * rankMN_ + offserHeadAndTailN;
        if constexpr (DequantBmm::IsMxType<scaleType>()) {
            offsetsScale_[cnt] = i * rankM_ * scaleK + scaleOffsetHeadAndTailK;
        }
        cnt += 1;
    }
}

__aicore__ inline bool QuantBatchMatmulAswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * tilingData_->matmulTiling.usedCoreNum - preCoreNum_;
    if (params_.index >= params_.totalCnt - preCoreNum_) {
        return false;
    }
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
    return true;
}


__aicore__ inline void QuantBatchMatmulAswBlock::UpdateBlockParams(uint64_t roundIdx, bool isLast)
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;

    // 判断mCntIndex是否时gather块尾块
    uint32_t mIndexInGatherBlock = params_.mIndex % mSliceCnt_;
    params_.singleCoreM = mIndexInGatherBlock != (mSliceCnt_ - 1) ? tilingData_->matmulTiling.baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->matmulTiling.baseN : params_.nBaseTail;

    if ((roundIdx == params_.round - 1) &&
        ((tilingData_->adaptiveSlidingWin.mTailTile != 1) || (tilingData_->adaptiveSlidingWin.nTailTile != 1)) &&
        isLast && doTailTile_) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + tilingData_->adaptiveSlidingWin.mTailTile - 1) /
                                    tilingData_->adaptiveSlidingWin.mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + tilingData_->adaptiveSlidingWin.nTailTile - 1) /
                                    tilingData_->adaptiveSlidingWin.nTailTile;
        uint64_t mBaseSplitCnt = MMV3DivCeil(params_.singleCoreM, singleCoreMSplit);
        uint64_t nBaseSplitCnt = MMV3DivCeil(params_.singleCoreN, singleCoreNSplit);
        totalSplitCnt_ = mBaseSplitCnt * nBaseSplitCnt;
        uint64_t mSplitIdx = (blockIdx_ % params_.totalTailTile) % mBaseSplitCnt;
        uint64_t nSplitIdx = (blockIdx_ % params_.totalTailTile) / mBaseSplitCnt;
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

__aicore__ inline void QuantBatchMatmulAswBlock::ResetAddressOffsets()
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
}

template <bool aTrans, bool bTrans, class x1Type, class scaleType, CubeFormat formatX2>
__aicore__ inline void QuantBatchMatmulAswBlock::CalcGMOffset()
{
    uint32_t mc2MIdx = params_.mIndex / mSliceCnt_;
    uint32_t mc2MRest = params_.mIndex % mSliceCnt_;

    uint64_t mOffset = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;
    uint64_t tmp1 = mc2MRest * blockBaseMDotRankK_ + params_.mSplitAddrOffset * rankK_;
    // 非连续A矩阵不支持转置
    offset_.offsetA = offsetsA_[mc2MIdx] + tmp1;

    if constexpr (formatX2 == CubeFormat::ND) {
        if constexpr (bTrans) {
            offset_.offsetB = nOffset * tilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = nOffset;
        }
        offset_.offsetB += offset_.batchBOffset * tilingData_->matmulTiling.N * tilingData_->matmulTiling.Kb;
    }

    uint64_t tmp2 =
        nOffset + (mc2MRest * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset) * tilingData_->matmulTiling.N;
    offset_.offsetC = offsetsC_[mc2MIdx] + tmp2;

    uint64_t scaleK =
        DequantBmm::Align(DequantBmm::CeilDiv(tilingData_->matmulTiling.Ka, MXFP_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t blockBaseMDotScaleK = tilingData_->matmulTiling.baseM * scaleK;
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        offset_.offsetPerTokenScale = offsetsScale_[mc2MIdx] + /* scale 在当前 tile块落在 mc2MIdx 卡的起始偏移 */
                                      mc2MRest * blockBaseMDotScaleK +   /* 整块的M偏移 */
                                      params_.mSplitAddrOffset * scaleK; /* 最后一次计算的M偏移 */
        if constexpr (bTrans) {
            offset_.offsetScale = nOffset * scaleK;
        } else {
            offset_.offsetScale = nOffset * MXFP_MULTI_BASE_SIZE;
        }
    } else {
        offset_.offsetPerTokenScale = mOffset;
        offset_.offsetScale = nOffset;
    }
    if (tilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nOffset;
    }
}
} // namespace Mc2MatmulV3

#endif // MC2_QUANT_BMMV3_ASW_BLOCK_H
