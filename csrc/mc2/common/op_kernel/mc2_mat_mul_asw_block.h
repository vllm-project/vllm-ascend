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
 * \file mc2_mat_mul_asw_block.h
 * \brief
 */

#ifndef NEW_MC2_MAT_MUL_ASW_BLOCK_H
#define NEW_MC2_MAT_MUL_ASW_BLOCK_H

#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_block.h"
#include "mc2_tiling_struct.h"

namespace MC2MatmulV3 {

constexpr uint64_t DEVICE_NUM = 64;        // group 内卡数， 目前定义为64，后续根据情况扩展
constexpr uint64_t SLIDING_WINDOW_LEN = 4; // 滑窗m方向大小

using namespace AscendC;
using namespace matmul;
using namespace Mc2MatmulV3Advanced;

class MC2MatmulAswBlockDerive : public Mc2MatmulAswBlock {
public:
    // constructor
    __aicore__ inline MC2MatmulAswBlockDerive()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void InitForMC2(const void *tilingData, const Mc2Tiling::RCSTiling &cfg, bool isTail,
                                      bool isGather = false);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void UpdateOffset(uint32_t idx, bool isTail);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx, bool isLast);
    __aicore__ inline void Update();
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);

public:
    uint64_t curSliceM_ = 0;
    uint64_t headSliceM_ = 0;
    uint64_t mSliceCnt_ = 0;
    uint64_t offsetsA_[DEVICE_NUM] = {0};
    uint64_t offsetsC_[DEVICE_NUM] = {0};
    Mc2Tiling::RCSTiling cfg_;
    bool isGather_ = false;
    uint64_t rankM_ = 0;
    uint64_t rankK_ = 0;
    uint64_t rankN_ = 0;
    // 以下变量保存部分scalar计算值，避免重复计算
    uint64_t preCoreNum_ = 0;         // 前一轮计算除不尽基本快数
    uint64_t rankMN_ = 0;             // rankMN_ = rankM_ * rankN_
    uint64_t rankMK_ = 0;             // rankMK_ = rankM_ * rankK_
    uint64_t rankKN_ = 0;             // rankKN_ = rankK_ * rankN_
    uint64_t windowParams1_ = 0;      // 主滑窗的m方向基本块数
    uint64_t windowParams2_ = 0;      // 主滑窗的m方向基本块数 * nCnt
    uint64_t blockBaseMDotRankK_ = 0; // blockBaseM * rankK_
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MC2MatmulAswBlockDerive::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const Mc2MatMulV3TilingData *>(tilingData);

    params_.index = 0;
    params_.singleCoreM = 0;
    params_.singleCoreN = 0;
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseN);
    params_.nCnt = (matmulTilingData_->tCubeTiling.N + params_.blockBaseN - 1) / params_.blockBaseN;
    params_.mBaseSplitCnt = matmulTilingData_->mTailCnt;
    params_.nBaseSplitCnt = matmulTilingData_->nTailCnt;
    params_.totalSplitCnt = params_.mBaseSplitCnt * params_.nBaseSplitCnt;
}

__aicore__ inline void MC2MatmulAswBlockDerive::InitForMC2(const void *tilingData, const Mc2Tiling::RCSTiling &cfg,
                                                           bool isTail, bool isGather)
{
    cfg_ = cfg;
    rankM_ = isGather ? cfg.rankM : cfg.rankM / cfg.rankDim;
    isGather_ = isGather;
    rankN_ = cfg.rankN;
    rankK_ = cfg.rankK;
    rankMN_ = rankM_ * rankN_;
    rankMK_ = rankM_ * rankK_;
    rankKN_ = rankK_ * rankN_;

    headSliceM_ = (rankM_ - cfg.tailM * cfg.tailCnt) / cfg.tileCnt; // 头块的shapeM
    curSliceM_ = isTail ? cfg.tailM : headSliceM_;                  // 当前计算的shapeM
    mSliceCnt_ = MMV3DivCeil(curSliceM_, params_.blockBaseM);

    uint64_t calRankNum = isGather_ ? (cfg.rankDim - 1) : cfg.rankDim;
    params_.mCnt = mSliceCnt_ * calRankNum;
    params_.totalCnt = params_.mCnt * params_.nCnt;

    // 重新计算
    params_.mBaseTail = curSliceM_ - (mSliceCnt_ - 1) * params_.blockBaseM; // m方向的 base 尾块
    params_.nBaseTail = matmulTilingData_->tCubeTiling.N - (params_.nCnt - 1) * params_.blockBaseN; // n方向上的base尾块

    params_.round = (params_.totalCnt + matmulTilingData_->tCubeTiling.usedCoreNum - 1) /
                    matmulTilingData_->tCubeTiling.usedCoreNum;

    params_.mainWindow = SLIDING_WINDOW_LEN < params_.mCnt ? SLIDING_WINDOW_LEN : params_.mCnt; // 主滑窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mainWindow - 1;                                    // 主滑窗数量
    windowParams1_ = params_.mainRow * params_.mainWindow;
    windowParams2_ = windowParams1_ * params_.nCnt;
    params_.tailWindow = params_.mCnt - windowParams1_; // 尾滑窗m方向的块个数
    blockBaseMDotRankK_ = params_.blockBaseM * rankK_;
}

__aicore__ inline void MC2MatmulAswBlockDerive::Update()
{
    params_.totalCnt = params_.mCnt * params_.nCnt + preCoreNum_;
    params_.round = (params_.totalCnt + matmulTilingData_->tCubeTiling.usedCoreNum - 1) /
                    matmulTilingData_->tCubeTiling.usedCoreNum;
}

__aicore__ inline void MC2MatmulAswBlockDerive::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * matmulTilingData_->tCubeTiling.usedCoreNum - preCoreNum_;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + params_.index % params_.mainWindow;
        params_.nCntIndex = (params_.index / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - windowParams2_;
        params_.mCntIndex = windowParams1_ + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % 2 != 0) {
        params_.nCntIndex = params_.nCnt - 1 - params_.nCntIndex;
    }
}

__aicore__ inline void MC2MatmulAswBlockDerive::UpdateBlockParams(uint64_t roundIdx, bool isLast)
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;

    // 判断mCntIndex是否时gather块尾块
    uint32_t mIndexInGatherBlock = params_.mCntIndex % mSliceCnt_;
    params_.singleCoreM = mIndexInGatherBlock != (mSliceCnt_ - 1) ? params_.blockBaseM : params_.mBaseTail;
    params_.singleCoreN = params_.nCntIndex != (params_.nCnt - 1) ? params_.blockBaseN : params_.nBaseTail;

    if ((roundIdx == params_.round - 1) && ((params_.mBaseSplitCnt != 1) || (params_.nBaseSplitCnt != 1)) && isLast) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + params_.mBaseSplitCnt - 1) / params_.mBaseSplitCnt;
        uint64_t singleCoreNSplit = (params_.singleCoreN + params_.nBaseSplitCnt - 1) / params_.nBaseSplitCnt;
        params_.mBaseSplitCnt = MMV3DivCeil(params_.singleCoreM, singleCoreMSplit);
        params_.nBaseSplitCnt = MMV3DivCeil(params_.singleCoreN, singleCoreNSplit);
        uint64_t mSplitIdx = (block_idx % params_.totalSplitCnt) % params_.mBaseSplitCnt;
        uint64_t nSplitIdx = (block_idx % params_.totalSplitCnt) / params_.mBaseSplitCnt;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (mSplitIdx + 1 == params_.mBaseSplitCnt) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (nSplitIdx + 1 == params_.nBaseSplitCnt) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MC2MatmulAswBlockDerive::CalcGMOffset()
{
    uint32_t mc2MIdx = params_.mCntIndex / mSliceCnt_;
    uint32_t mc2MRest = params_.mCntIndex % mSliceCnt_;

    offset_.offsetA = offsetsA_[mc2MIdx] + mc2MRest * blockBaseMDotRankK_ + params_.mSplitAddrOffset * rankK_;

    if (B_TYPE::isTrans) {
        offset_.offsetB =
            (params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset) * matmulTilingData_->tCubeTiling.Ka;
    } else {
        offset_.offsetB = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    }

    offset_.offsetC = offsetsC_[mc2MIdx] + (params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset) +
                      (mc2MRest * params_.blockBaseM + params_.mSplitAddrOffset) * matmulTilingData_->tCubeTiling.N;

    if (matmulTilingData_->tCubeTiling.isBias) {
        offset_.offsetBias = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    }
}

__aicore__ inline void MC2MatmulAswBlockDerive::UpdateOffset(uint32_t idx, bool isTail)
{
    uint64_t offsetHeadSliceM = 0;
    uint64_t offsetTailSliceM = 0;

    if (isTail) {
        offsetHeadSliceM = cfg_.tileCnt * headSliceM_;
        offsetTailSliceM = idx * cfg_.tailM;
    } else {
        offsetHeadSliceM = idx * headSliceM_;
    }

    uint64_t offsetHeadAndTailK = (offsetHeadSliceM + offsetTailSliceM) * rankK_;
    uint64_t offserHeadAndTailN = (offsetHeadSliceM + offsetTailSliceM) * rankN_;

    uint64_t cnt = 0;
    for (size_t i = 0; i < cfg_.rankDim; i++) {
        if (isGather_ && (i == cfg_.rankID)) {
            offsetsA_[i] = 0;
            offsetsC_[i] = 0;
            continue;
        }

        offsetsA_[cnt] = i * rankMK_ + offsetHeadAndTailK;
        offsetsC_[cnt] = i * rankMN_ + offserHeadAndTailN;
        cnt += 1;
    }
}

} // namespace MC2MatmulV3

#endif // MC2_MAT_MUL_ASW_BLOCK_H
