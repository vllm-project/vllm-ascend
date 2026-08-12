/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file gqmm_cube_on_the_fly.h
 * \brief
 */
#ifndef MC2_GQMM_CUBE_ON_THE_FLY_H
#define MC2_GQMM_CUBE_ON_THE_FLY_H

#include "quant_block_sch.h"
#include "quant_utils.h"
#include "grouped_matmul_utils.h"
#include "grouped_matmul_tiling_data_apt.h"
using GMMQuantParams = Mc2GroupedMatmulTilingData::GMMQuantParams;

namespace Mc2GroupedMatmul {
constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;

LOCAL_TEMPLATE_CLASS_PARAMS
class Mc2GmmASWKernel {
public:
    __aicore__ inline Mc2GmmASWKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
        GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, const GMMQuantParams *__restrict gmmBaseParamsIn,
        const TCubeTiling *__restrict mmTilingDataIn, TILING_TYPE *gmmArrayAddrIn, TPipe *que);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
        GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, const GMMQuantParams *__restrict gmmBaseParamsIn,
        const TCubeTiling *__restrict mmTilingDataIn, TILING_TYPE *gmmArrayAddrIn, TPipe *que,
        uint32_t rankDim, const __gm__ uint64_t *cGroupOffsetTableGm,
        const __gm__ uint64_t *aGroupOffsetTableGm = nullptr,
        const __gm__ uint64_t *xScaleGroupOffsetTableGm = nullptr);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitAddrAndParams(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                             GM_ADDR perTokenScale, GM_ADDR y, TILING_TYPE *gmmArrayAddrIn);
    __aicore__ inline void UpdateMMGlobalAddr(uint32_t groupIdx);
    __aicore__ inline void SetMNK(uint32_t groupIdx, int32_t &mSize, int32_t &nSize, int32_t &kSize);
    __aicore__ inline void CalcTailTile(uint64_t mTail, uint64_t nTail);
    __aicore__ inline void SetMMParaAndCompute();
    __aicore__ inline bool IsLastGroupAndNeedSplit(uint32_t groupIdx);
    __aicore__ inline bool IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx);

    uint32_t blockIdx_;
    uint32_t groupNum_;
    uint32_t rankDim_ = 1U;
    int8_t groupType_;
    uint8_t groupListType_;
    int32_t preOffset_;
    uint64_t scaleScalar_;
    const TCubeTiling *__restrict mmTilingData_;
    const GMMQuantParams *__restrict gmmQuantParams_;

    TILING_TYPE *mListGm_;
    TILING_TYPE *kListGm_;
    TILING_TYPE *nListGm_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<int64_t> groupListGlobal_;
    GlobalTensor<fp8_e8m0_t> scaleAGlobal_;
    GlobalTensor<fp8_e8m0_t> mxScaleBGlobal_;
    GlobalTensor<uint64_t> scaleBGlobal_;

    // dynamic输入输出需要地址，再按照group依次读取二级指针
    GM_ADDR xTensorPtr_;
    GM_ADDR weightTensorPtr_;
    GM_ADDR biasTensorPtr_;
    GM_ADDR scaleTensorPtr_;
    GM_ADDR groupListPtr_;
    GM_ADDR perTokenScalePtr_;
    GM_ADDR yTensorPtr_;
    Mc2GroupedMatmul::QuantASWBlockSch block_;

    // mxType定义
    using aType = typename AscendC::Conditional<
        Mc2QuantUtils::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, xType, aTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, aTrans>>::type;
    using bType = typename AscendC::Conditional<
        Mc2QuantUtils::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, wType, bTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, wFormat, wType, bTrans>>::type;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    using MmType = typename AscendC::Conditional<
        Mc2QuantUtils::IsMxType<scaleType>(),
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, Mc2QuantUtils::MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG,
                           MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy>,
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, Mc2QuantUtils::MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>>::type;
    MmType mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
    GM_ADDR workspace, const GMMQuantParams *__restrict gmmBaseParamsIn, const TCubeTiling *__restrict mmTilingDataIn,
    TILING_TYPE *gmmArrayAddrIn, TPipe *que)
{
    if ASCEND_IS_AIV {
        return;
    }
    mmTilingData_ = mmTilingDataIn;
    gmmQuantParams_ = gmmBaseParamsIn;
    blockIdx_ = GetBlockIdx();
    mm_.SetSubBlockIdx(0);
    mm_.Init(mmTilingData_, que);
    InitAddrAndParams(x, weight, bias, scale, groupList, perTokenScale, y, gmmArrayAddrIn);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR scale, GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
    const GMMQuantParams *__restrict gmmBaseParamsIn, const TCubeTiling *__restrict mmTilingDataIn,
    TILING_TYPE *gmmArrayAddrIn, TPipe *que, uint32_t rankDim, const __gm__ uint64_t *cGroupOffsetTableGm,
    const __gm__ uint64_t *aGroupOffsetTableGm, const __gm__ uint64_t *xScaleGroupOffsetTableGm)
{
    if ASCEND_IS_AIV {
        return;
    }
    rankDim_ = rankDim;
    mmTilingData_ = mmTilingDataIn;
    gmmQuantParams_ = gmmBaseParamsIn;
    blockIdx_ = GetBlockIdx();
    mm_.SetSubBlockIdx(0);
    mm_.Init(mmTilingData_, que);
    InitAddrAndParams(x, weight, bias, scale, groupList, perTokenScale, y, gmmArrayAddrIn);
    block_.SetRankFirstLayout(rankDim, cGroupOffsetTableGm);
    if (aGroupOffsetTableGm != nullptr) {
        block_.SetAGroupOffsetTable(aGroupOffsetTableGm);
    }
    if (xScaleGroupOffsetTableGm != nullptr) {
        block_.SetXScaleGroupOffsetTable(xScaleGroupOffsetTableGm);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitAddrAndParams(GM_ADDR x, GM_ADDR weight,
                                                                                      GM_ADDR bias, GM_ADDR scale,
                                                                                      GM_ADDR groupList,
                                                                                      GM_ADDR perTokenScale, GM_ADDR y,
                                                                                      TILING_TYPE *gmmArrayAddrIn)
{
    groupNum_ = gmmQuantParams_->groupNum;
    groupType_ = gmmQuantParams_->groupType;
    groupListType_ = gmmQuantParams_->groupListType;
    block_.template Init<true>(mmTilingData_, blockIdx_);
    xTensorPtr_ = x;
    weightTensorPtr_ = weight;
    biasTensorPtr_ = bias;
    scaleTensorPtr_ = scale;
    groupListPtr_ = groupList;
    perTokenScalePtr_ = perTokenScale;
    yTensorPtr_ = y;
    if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
        scaleAGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)perTokenScale);
    }
    if (groupList != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    mListGm_ = gmmArrayAddrIn;
    kListGm_ = gmmArrayAddrIn + MC2_GROUPED_MATMUL::MKN_LIST_LEN;
    nListGm_ = gmmArrayAddrIn + MC2_GROUPED_MATMUL::MKN_LIST_LEN * 2; // 2: mListGm_ + kListGm_
}

// 更新每个group的global地址
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateMMGlobalAddr(uint32_t groupIdx)
{
    if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
        mxScaleBGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<fp8_e8m0_t>(0, scaleTensorPtr_) +
                                        block_.params_.wScaleGroupAddrOffset);
    } else {
        if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(Mc2QuantUtils::QuantMode::DEFAULT) &&
            gmmQuantParams_->bQuantMode == static_cast<uint32_t>(Mc2QuantUtils::QuantMode::DEFAULT)) {
        } else if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(Mc2QuantUtils::QuantMode::PERTENSOR_MODE) &&
            gmmQuantParams_->bQuantMode ==
            static_cast<uint32_t>(Mc2QuantUtils::QuantMode::PERTENSOR_MODE)) { // doubleScale, M_SPLIT
            uint32_t scaleIdx = (rankDim_ > 1U) ? (groupIdx / rankDim_) : groupIdx;
            __gm__ scaleType *scaleB = MC2_GROUPED_MATMUL::GetTensorAddr<scaleType>(0, scaleTensorPtr_) + scaleIdx;
            float scaleBValue = *((__gm__ float *)scaleB);
            float scaleAValue = *((__gm__ float *)perTokenScalePtr_ + scaleIdx);
            float deqScale = scaleBValue * scaleAValue;
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t *>(&deqScale));
            scaleScalar_ = uint32Scale & DEQ_SCALE_MUL; // fixpipe只能取高19位
        } else if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(Mc2QuantUtils::QuantMode::DEFAULT) &&
                   gmmQuantParams_->bQuantMode ==
                       static_cast<uint32_t>(Mc2QuantUtils::QuantMode::PERTENSOR_MODE)) { // pertensor, M_SPLIT
            if constexpr (!IsSameType<scaleType, uint64_t>::value && !IsSameType<scaleType, int64_t>::value) {
                uint32_t scaleIdx = (rankDim_ > 1U) ? (groupIdx / rankDim_) : groupIdx;
                __gm__ scaleType *scaleB = MC2_GROUPED_MATMUL::GetTensorAddr<scaleType>(0, scaleTensorPtr_) + scaleIdx;
                uint32_t uint32Scale = 0;
                if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                    uint16_t uint16Scale = *((__gm__ uint16_t *)scaleB);
                    // 16为将uint16的数据前移16位，依据硬件逻辑，将scale变为uint32类型
                    uint32Scale = uint16Scale << 16;
                } else {
                    uint32Scale = *((__gm__ uint32_t *)scaleB);
                }
                scaleScalar_ = uint32Scale & DEQ_SCALE_MUL;
            } else {
                uint32_t scaleIdx = (rankDim_ > 1U) ? (groupIdx / rankDim_) : groupIdx;
                __gm__ scaleType *scaleB = MC2_GROUPED_MATMUL::GetTensorAddr<scaleType>(0, scaleTensorPtr_) + scaleIdx;
                scaleScalar_ = *((__gm__ uint64_t *)scaleB);
            }
        } else if (gmmQuantParams_->bQuantMode ==
                   static_cast<uint32_t>(Mc2QuantUtils::QuantMode::PERCHANNEL_MODE)) { // perChannel, M_SPLIT
            scaleBGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<uint64_t>(0, scaleTensorPtr_) +
                                          block_.params_.wScaleGroupAddrOffset);
        }
    }
    xGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<xType>(0, xTensorPtr_) +
                             block_.params_.aGroupAddrOffset);
    wGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<wType>(0, weightTensorPtr_) +
                             block_.params_.bGroupAddrOffset);
    yGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<yType>(0, yTensorPtr_) +
                             block_.params_.cGroupAddrOffset);
    if (gmmQuantParams_->hasBias) {
        biasGlobal_.SetGlobalBuffer(MC2_GROUPED_MATMUL::GetTensorAddr<biasType>(0, biasTensorPtr_) +
                                    block_.params_.biasGroupAddrOffset);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMNK(uint32_t groupIdx, int32_t &mSize,
                                                                           int32_t &nSize, int32_t &kSize)
{
    int32_t splitValue =
        Mc2QuantUtils::GetSplitValueFromGroupList(groupIdx, preOffset_, groupType_, groupListType_, groupListGlobal_);
    switch (groupType_) {
        case (Mc2QuantUtils::SPLIT_M):
            {
                mSize = splitValue;
                uint32_t valueIdx = gmmQuantParams_->singleW == 1 ? 0 : groupIdx;
                kSize = kListGm_[valueIdx];
                nSize = nListGm_[valueIdx];
                break;
            }
        case (Mc2QuantUtils::SPLIT_K):
            {
                mSize = gmmQuantParams_->singleX == 1 ? mListGm_[0] : mListGm_[groupIdx];
                kSize = splitValue;
                nSize = gmmQuantParams_->singleW == 1 ? nListGm_[0] : nListGm_[groupIdx];
                break;
            }
        default:
            {
                mSize = mListGm_[groupIdx];
                kSize = kListGm_[groupIdx];
                nSize = nListGm_[groupIdx];
            }
    }
    return;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CalcTailTile(uint64_t mTail, uint64_t nTail)
{
    // 计算实际 base 块数
    uint64_t tailCnt = block_.GetEndBlockIdx() + 1;
    // 计算可切分数
    uint64_t remainTile = mmTilingData_->usedCoreNum / tailCnt;
    if (remainTile <= 1) {
        return;
    }

    // 初始化最小 tile 大小
    uint64_t mMin = Mc2QuantUtils::CUBE_BLOCK;
    uint64_t nMin = Mc2QuantUtils::CUBE_BLOCK;

    // 根据矩阵是否转置调整最小 tile 大小
    if constexpr (aTrans) {
        mMin = Mc2QuantUtils::INNER_AXIS_MIN_SPLIT_VAL;
    }
    if constexpr (!bTrans) {
        nMin = Mc2QuantUtils::INNER_AXIS_MIN_SPLIT_VAL;
    }

    // 计算 mTile 和 nTile，尽可能让m,n方向切分数一致
    uint64_t mTile = MC2_GROUPED_MATMUL::Min(Mc2QuantUtils::CeilDiv(mTail, mMin), remainTile);
    uint64_t nTile = MC2_GROUPED_MATMUL::Min(Mc2QuantUtils::CeilDiv(nTail, nMin), remainTile);
    while (mTile * nTile > remainTile) {
        if (mTile >= nTile) {
            mTile -= 1;
        } else {
            nTile -= 1;
        }
    }
    block_.params_.mTailTile = mTile;
    block_.params_.nTailTile = nTile;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndNeedSplit(uint32_t groupIdx)
{
    // 2: 剩一半及以上核数时才考虑尾块切分
    return groupIdx == groupNum_ - 1 && (block_.GetEndBlockIdx() + 1) <= mmTilingData_->usedCoreNum / 2;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndRound(uint32_t groupIdx,
                                                                                        uint64_t roundIdx)
{
    return groupIdx == groupNum_ - 1 && roundIdx == block_.params_.round - 1 && blockIdx_ <= block_.GetEndBlockIdx();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (groupType_ != -1) { // -1: no split
        if (unlikely(groupListPtr_ == nullptr)) {
            return;
        }
        preOffset_ = 0;
    }

    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        int32_t mSize;
        int32_t nSize;
        int32_t kSize;
        // 更新group内的输入参数M,N,K
        SetMNK(groupIdx, mSize, nSize, kSize);
        block_.template UpdateGroupOffset<aTrans, bTrans, xType, scaleType, wFormat>(mSize, nSize, kSize, groupIdx);
        if (mSize <= 0 || kSize <= 0 || nSize <= 0) {
            continue;
        }
        block_.template UpdateGroupParams<true>();
        // 最后一个group最后一轮是否进一步切分以使用更多的核数
        if (IsLastGroupAndNeedSplit(groupIdx)) {
            CalcTailTile(block_.params_.mBaseTail, block_.params_.nBaseTail);
            block_.UpdateTailTile();
        }

        UpdateMMGlobalAddr(groupIdx);
        for (uint64_t roundIdx = 0; roundIdx < block_.params_.round; ++roundIdx) {
            bool isLastGroupRound = IsLastGroupAndRound(groupIdx, roundIdx);
            block_.template UpdateBasicIndex<true>(roundIdx, isLastGroupRound);
            // 1. Set single core param
            block_.template UpdateBlockParams<aTrans, bTrans, wFormat>(roundIdx, isLastGroupRound);
            if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
                continue;
            }
            // 2. compute offset
            block_.template CalcGMOffset<aTrans, bTrans, scaleType, wFormat>();
            // 3. set offset and compute
            SetMMParaAndCompute();
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2GmmASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if ASCEND_IS_AIV {
        return;
    }
    mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, block_.params_.k);
    if (gmmQuantParams_->aQuantMode != static_cast<uint32_t>(Mc2QuantUtils::QuantMode::DEFAULT) ||
        gmmQuantParams_->bQuantMode != static_cast<uint32_t>(Mc2QuantUtils::QuantMode::DEFAULT)) {
        if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
            mm_.SetTensorScaleA(scaleAGlobal_[block_.offset_.offsetPerTokenScale], aTrans);
            mm_.SetTensorScaleB(mxScaleBGlobal_[block_.offset_.offsetScale], bTrans);
        } else {
            if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(Mc2QuantUtils::QuantMode::PERTENSOR_MODE)) {
                mm_.SetQuantScalar(scaleScalar_);
            } else {
                mm_.SetQuantVector(scaleBGlobal_[block_.offset_.offsetScale]);
            }
        }
        if (gmmQuantParams_->hasBias) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        }
    }
    mm_.SetTensorA(xGlobal_[block_.offset_.offsetA], aTrans);
    mm_.SetTensorB(wGlobal_[block_.offset_.offsetB], bTrans);
    mm_.Iterate();
    mm_.GetTensorC(yGlobal_[block_.offset_.offsetC]);
}
} // namespace Mc2GroupedMatmul
#endif // MC2_GQMM_CUBE_ON_THE_FLY_H
