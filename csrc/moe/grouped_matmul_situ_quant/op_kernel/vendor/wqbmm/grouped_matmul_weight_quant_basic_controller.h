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
 * \file grouped_matmul_weight_quant_basic_controller.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_CONTROLLER_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_CONTROLLER_H

#include "weight_quant_basic_block.h"
#include "../grouped_matmul_tiling_data_apt.h"


using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::DOUBLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::QUADRUPLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace GROUPED_MATMUL {
template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
class GMMWeightQuantBasicController {
public:
    __aicore__ inline GMMWeightQuantBasicController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR bias, GM_ADDR groupList, GM_ADDR y,
                                const GMMWeightQuantParam *__restrict baseTiling,
                                const TCubeTiling *__restrict mmTiling, GM_ADDR tiling, TILING_TYPE *gmmArrayAddrIn,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetMKN(uint64_t groupIdx, uint64_t &preOffset, BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetGmAddr(uint64_t groupIdx, uint64_t &xBaseOffset, uint64_t &weightBaseOffset,
                                     uint64_t &yBaseOffset, uint64_t &antiquantParamsBaseOffset,
                                     const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx, uint64_t &preOffset);

    const GMMWeightQuantParam *gmmBaseTiling_;
    const TCubeTiling *mmTiling_;

    GM_ADDR xGm_;
    GM_ADDR weightGm_;
    GM_ADDR antiquantScaleGm_;
    GM_ADDR antiquantOffsetGm_;
    GM_ADDR biasGm_;
    GM_ADDR yGm_;
    GlobalTensor<int64_t> groupListGm_;
    TILING_TYPE *mListGm_;
    TILING_TYPE *nListGm_;
    TILING_TYPE *kListGm_;

    WeightQuantMatmulBasicBlock<xType, wType, xType, uint64_t, float, biasType, yType, wqmmConfig, vecConfig>
        wqmmBasicBlock_;
};

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig, vecConfig>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList,
    GM_ADDR y, const GMMWeightQuantParam *__restrict baseTiling, const TCubeTiling *__restrict mmTiling, GM_ADDR tiling,
    TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe)
{
    gmmBaseTiling_ = baseTiling;
    mmTiling_ = mmTiling;

    mListGm_ = gmmArrayAddrIn;
    kListGm_ = gmmArrayAddrIn + MKN_LIST_LEN;
    nListGm_ = gmmArrayAddrIn + MKN_LIST_LEN * 2;

    xGm_ = x;
    weightGm_ = weight;
    antiquantScaleGm_ = antiquantScale;
    antiquantOffsetGm_ = antiquantOffset;
    biasGm_ = bias;
    yGm_ = y;
    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    wqmmBasicBlock_.Init(gmmBaseTiling_->hasBias, gmmBaseTiling_->groupSize, 0, mmTiling_,
                         tPipe);  // gmm场景不确定group是否激活，prefetch size设定为0
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig, vecConfig>::Process()
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    uint64_t preOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t weightBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t antiquantParamsBaseOffset = 0;

    BasicBlockOffsetParam offsetParam;
    InitOffsetParam(offsetParam);
    for (uint32_t groupIdx = 0, count = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        SetMKN(groupIdx, preOffset, offsetParam);  // 每个核都必须知道之前group的信息，在单场景下作为baseOffset
        SetGmAddr(groupIdx, xBaseOffset, weightBaseOffset, yBaseOffset, antiquantParamsBaseOffset, offsetParam);
        if (offsetParam.mSize == 0 || offsetParam.nSize == 0) {
            continue;
        }
        /*
         * 1.在mSize小于mmTiling_.baseM或者mSize大于mmTiling_.baseM * 4时，baseM设置为mmTiling_.baseM
         * 2.在mSize大于mmTiling_.baseM且小于等于mmTiling_.baseM * 2时，baseM使用mSize / 2向上取整
         * 3.在mSize大于mmTiling_.baseM * 2且小于等于mmTiling_.baseM * 4时，baseM使用mSize / 4向上取整
         */
        uint64_t baseM =
            offsetParam.mSize <= mmTiling_->baseM || offsetParam.mSize >= mmTiling_->baseM * QUADRUPLE_BUFFER_NUM
                ? mmTiling_->baseM
                : (offsetParam.mSize <= DOUBLE_BUFFER_NUM * mmTiling_->baseM
                       ? CeilDivide(offsetParam.mSize, (uint64_t)DOUBLE_BUFFER_NUM)
                       : CeilDivide(offsetParam.mSize, (uint64_t)QUADRUPLE_BUFFER_NUM));
        uint64_t baseN = offsetParam.nSize >= mmTiling_->baseN * gmmBaseTiling_->coreNum ? mmTiling_->baseN
                                                                                         : (mmTiling_->baseN >> 1);

        uint32_t mBlockNum = CeilDivide(offsetParam.mSize, baseM);
        uint32_t nBlockNum = CeilDivide(offsetParam.nSize, baseN);

        uint32_t curCount = count + mBlockNum * nBlockNum;
        uint32_t curBlock = cubeBlockIdx >= count ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;

        while (curBlock < curCount) {
            offsetParam.mOffset = ((curBlock - count) / nBlockNum) * baseM;
            offsetParam.nOffset = ((curBlock - count) % nBlockNum) * baseN;
            offsetParam.mL1Size =
                offsetParam.mOffset + baseM > offsetParam.mSize ? offsetParam.mSize - offsetParam.mOffset : baseM;
            offsetParam.nL1Size =
                offsetParam.nOffset + baseN > offsetParam.nSize ? offsetParam.nSize - offsetParam.nOffset : baseN;

            wqmmBasicBlock_.ComputeBasicBlock(offsetParam, offsetParam);
            curBlock += gmmBaseTiling_->coreNum;
        }
        count = curCount % gmmBaseTiling_->coreNum;
    }
    wqmmBasicBlock_.End(offsetParam);
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig,
                                                     vecConfig>::InitOffsetParam(BasicBlockOffsetParam &offsetParam)
{
    // 单的场景不需要频繁取值，n/k轴在开始的时候获取一次即可
    if (gmmBaseTiling_->singleWeight == 1) {
        offsetParam.kSize = kListGm_[0];
        offsetParam.nSize = nListGm_[0];
    }

    offsetParam.kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    offsetParam.kaL1Size = offsetParam.kbL1Size;  // 当前实现a矩阵切分保持b矩阵一致
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig, vecConfig>::SetMKN(
    uint64_t groupIdx, uint64_t &preOffset, BasicBlockOffsetParam &offsetParam)
{
    uint64_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset);
    if (gmmBaseTiling_->groupType == 0) {
        offsetParam.mSize = splitValue;
        offsetParam.kSize = gmmBaseTiling_->singleWeight == 1 ? offsetParam.kSize : kListGm_[groupIdx];
        offsetParam.kAlign =
            WeightQuantBatchMatmulV2::Arch35::CeilAlign(offsetParam.kSize, static_cast<uint64_t>(BLOCK_CUBE));
        offsetParam.nSize = gmmBaseTiling_->singleWeight == 1 ? offsetParam.nSize : nListGm_[groupIdx];
        return;
    }

    offsetParam.mSize = mListGm_[groupIdx];
    offsetParam.kSize = kListGm_[groupIdx];
    offsetParam.kAlign =
        WeightQuantBatchMatmulV2::Arch35::CeilAlign(offsetParam.kSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam.nSize = nListGm_[groupIdx];
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig, vecConfig>::SetGmAddr(
    uint64_t groupIdx, uint64_t &xBaseOffset, uint64_t &weightBaseOffset, uint64_t &yBaseOffset,
    uint64_t &antiquantParamsBaseOffset, const BasicBlockOffsetParam &offsetParam)
{
    __gm__ xType *xGm;
    __gm__ wType *weightGm;
    __gm__ xType *antiquantScaleGm;
    __gm__ xType *antiquantOffsetGm;
    __gm__ biasType *biasGm;
    __gm__ yType *yGm;
    if (gmmBaseTiling_->singleX == 0) {
        xGm = GetTensorAddr<xType>(groupIdx, xGm_);
    } else {
        xGm = GetTensorAddr<xType>(0, xGm_) + xBaseOffset;
    }

    if (gmmBaseTiling_->singleY == 0) {
        yGm = GetTensorAddr<yType>(groupIdx, yGm_);
    } else {
        yGm = GetTensorAddr<yType>(0, yGm_) + yBaseOffset;
    }

    if (gmmBaseTiling_->singleWeight == 0) {
        weightGm = GetTensorAddr<wType>(groupIdx, weightGm_);
        antiquantScaleGm = GetTensorAddr<xType>(groupIdx, antiquantScaleGm_);
        antiquantOffsetGm = GetTensorAddr<xType>(groupIdx, antiquantOffsetGm_);
        biasGm = GetTensorAddr<biasType>(groupIdx, biasGm_);
    } else {
        weightGm = GetTensorAddr<wType>(0, weightGm_) + weightBaseOffset;
        antiquantScaleGm = GetTensorAddr<xType>(0, antiquantScaleGm_) + antiquantParamsBaseOffset;
        antiquantOffsetGm = GetTensorAddr<xType>(0, antiquantOffsetGm_) + antiquantParamsBaseOffset;
        biasGm = GetTensorAddr<biasType>(0, biasGm_) + antiquantParamsBaseOffset;
    }
    wqmmBasicBlock_.UpdateGlobalAddr(xGm, weightGm, antiquantScaleGm, antiquantOffsetGm, nullptr, nullptr, biasGm,
                                     yGm, mmTiling_->isBias, true);
    xBaseOffset += offsetParam.mSize * offsetParam.kSize;
    if constexpr (IsSameType<wType, int4b_t>::value) {
        weightBaseOffset += (offsetParam.nSize * offsetParam.kSize) >> 1;
    } else {
        weightBaseOffset += offsetParam.nSize * offsetParam.kSize;
    }
    antiquantParamsBaseOffset += offsetParam.nSize;
    yBaseOffset += offsetParam.mSize * offsetParam.nSize;
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline uint64_t GMMWeightQuantBasicController<xType, wType, biasType, yType, wqmmConfig,
                                                         vecConfig>::GetSplitValueFromGroupList(uint64_t groupIdx,
                                                                                                uint64_t &preOffset)
{
    uint64_t splitValue = 0;
    if (likely(gmmBaseTiling_->groupType != -1)) {
        if (gmmBaseTiling_->groupListType == 0) {
            uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

}  // namespace GROUPED_MATMUL

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_CONTROLLER_H
