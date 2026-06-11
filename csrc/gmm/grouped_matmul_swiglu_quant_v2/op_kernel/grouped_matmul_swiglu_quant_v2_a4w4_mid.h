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
 * \file grouped_matmul_swiglu_quant_v2_a4w4_mid.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_MID_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_MID_H

#include "grouped_matmul_swiglu_quant_v2_utils.h"

#ifdef GMM_SWIGLU_QUANT_V2_A4W4

namespace GroupedMatmulDequantSwigluQuant {
using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

template <class mmType>
class GMMA4W4MidProcess {
public:
    using bT = typename mmType::BT;

public:
    __aicore__ inline GMMA4W4MidProcess(typename mmType::MT &matmul) : mm(matmul)
    {
    }
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN);
    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx);

private:
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig &mnConfig, WorkSpaceSplitConfig &workspaceSplitConfig);
    __aicore__ inline void SetMNConfig(const int32_t splitValue, MNConfig &mnConfig);
    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig, bool resetOutputOffset);

private:
    typename mmType::MT &mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<int4b_t> xGM;
    GlobalTensor<int4b_t> weightGM;

    GlobalTensor<half> mmOutGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<uint64_t> weightScaleGM;

    GM_ADDR weightTensorPtr;
    GM_ADDR weightScaleTensorPtr;

    MNConfig mnConfig;

    // define the que
    uint32_t subBlockIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t quantGroupSize = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams = nullptr;
};

template <typename mmType>
__aicore__ inline void
GMMA4W4MidProcess<mmType>::Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN)
{
    if ASCEND_IS_AIC {
        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;
        xRowSumCount = gmmSwigluQuantV2BaseParams->M;
        xGM.SetGlobalBuffer((__gm__ int4b_t *)gmAddrParams.xGM);
        weightGM.SetGlobalBuffer(GetTensorAddr<int4b_t>(0, gmAddrParams.weightGM));
        weightScaleGM.SetGlobalBuffer(GetTensorAddr<uint64_t>(0, gmAddrParams.weightScaleGM));
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM);
        mmOutGM1.SetGlobalBuffer((__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
        quantGroupSize = gmmSwigluQuantV2BaseParams->K / gmmSwigluQuantV2BaseParams->quantGroupNum; // 约束为整除关系
        subBlockIdx = GetSubBlockIdx();
        coreIdx = GetBlockIdx();
        weightTensorPtr = gmAddrParams.weightGM;
        weightScaleTensorPtr = gmAddrParams.weightScaleGM;
    }
}

template <typename mmType>
__aicore__ inline void GMMA4W4MidProcess<mmType>::UpdateMnConfig(MNConfig &mnConfig, bool resetOutputOffset)
{
    if constexpr (bT::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    if (resetOutputOffset) {
        mnConfig.yBaseOffset = 0;
    } else {
        mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
    }
}

template <typename mmType>
__aicore__ inline void GMMA4W4MidProcess<mmType>::SetMNConfig(const int32_t splitValue, MNConfig &mnConfig)
{
    mnConfig.m = static_cast<int64_t>(splitValue);
    mnConfig.baseM = gmmSwigluQuantV2BaseParams->baseM;
    mnConfig.baseN = gmmSwigluQuantV2BaseParams->baseN;
    mnConfig.singleM = gmmSwigluQuantV2BaseParams->baseM;
    mnConfig.singleN = gmmSwigluQuantV2BaseParams->singleN != 0 && gmmSwigluQuantV2BaseParams->quantGroupNum == 1?
                         gmmSwigluQuantV2BaseParams->singleN : gmmSwigluQuantV2BaseParams->baseN;
}

template <typename mmType>
__aicore__ inline void GMMA4W4MidProcess<mmType>::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                          int64_t workspaceSplitLoopIdx)
{
    if ASCEND_IS_AIC {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        mmOutGM = (workspaceSplitLoopIdx % NUM_2 == 0 ? mmOutGM1 : mmOutGM2);
        mnConfig.baseM = gmmSwigluQuantV2BaseParams->baseM;
        mnConfig.baseN = gmmSwigluQuantV2BaseParams->baseN;
        mnConfig.singleM = gmmSwigluQuantV2BaseParams->baseM;
        mnConfig.singleN = gmmSwigluQuantV2BaseParams->singleN != 0 && gmmSwigluQuantV2BaseParams->quantGroupNum == 1?
                            gmmSwigluQuantV2BaseParams->singleN : gmmSwigluQuantV2BaseParams->baseN;
        mnConfig.k = gmmSwigluQuantV2BaseParams->K; // tilingData
        mnConfig.n = gmmSwigluQuantV2BaseParams->N; // tilingData
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
        int32_t prevSplitValue = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
        int32_t totalTmp = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
            for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
                totalTmp += groupListGM.GetValue(i);
            }
        }
        // 当workspace切换时,需要将输出的地址偏移初始化为0,使用resetOutputOffset控制
        bool resetOutputOffset = true;
        for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex, preCount = 0;
             groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; ++groupIdx) {
            UpdateMnConfig(mnConfig, resetOutputOffset);
            resetOutputOffset = false;
            int32_t currSplitValue = 0;
            if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
                currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
            } else {
                totalTmp += static_cast<int32_t>(groupListGM.GetValue(groupIdx));
                currSplitValue = totalTmp;
            }
            currSplitValue = currSplitValue > (workspaceSplitLoopIdx + 1) * gmmSwigluQuantV2BaseParams->mLimit ?
                                 (workspaceSplitLoopIdx + 1) * gmmSwigluQuantV2BaseParams->mLimit :
                                 currSplitValue;

            int32_t splitValue = (currSplitValue - prevSplitValue);
            prevSplitValue = currSplitValue;

            SetMNConfig(splitValue, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
            uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
            uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + gmmSwigluQuantV2BaseParams->coreNum;
            while (curBlock < curCount) {
                mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
                mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
                MMCompute(groupIdx, mnConfig, workspaceSplitConfig);
                curBlock += gmmSwigluQuantV2BaseParams->coreNum;
            }
            preCount = curCount % gmmSwigluQuantV2BaseParams->coreNum;
        }
    }
}

template <typename mmType>
__aicore__ inline void GMMA4W4MidProcess<mmType>::MMCompute(uint32_t groupIdx, MNConfig &mnConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
        curSingleN = gmmSwigluQuantV2BaseParams->N - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t weightOffset = 0;
    mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize);
    GlobalTensor<int4b_t> weightSlice;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    mnConfig.workspaceOffset = outOffset + mnConfig.yBaseOffset;
    for (uint32_t loopK = 0; loopK < gmmSwigluQuantV2BaseParams->quantGroupNum; loopK++) {
        mm.SetTensorA(
            xGM[mnConfig.xBaseOffset + mnConfig.mIdx * mnConfig.k * mnConfig.singleM + loopK * quantGroupSize]);
        if (gmmSwigluQuantV2BaseParams->isSingleTensor == 0) {
            weightGM.SetGlobalBuffer(GetTensorAddr<int4b_t>(groupIdx, weightTensorPtr));
            if constexpr (mmType::BT::format == CubeFormat::NZ && mmType::BT::isTrans == true) {
                weightOffset = tailN * 64;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSwigluQuantV2BaseParams->N];
            } else if constexpr (mmType::BT::format == CubeFormat::NZ && mmType::BT::isTrans == false) {
                weightOffset = tailN * gmmSwigluQuantV2BaseParams->K;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightOffset = tailN;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSwigluQuantV2BaseParams->N];
            }
        } else {
            if constexpr (mmType::BT::format == CubeFormat::NZ && mmType::BT::isTrans == true) {
                weightOffset = static_cast<uint64_t>(groupIdx) * gmmSwigluQuantV2BaseParams->N * gmmSwigluQuantV2BaseParams->K +
                               tailN * 64;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSwigluQuantV2BaseParams->N];
            } else if constexpr (mmType::BT::format == CubeFormat::NZ && mmType::BT::isTrans == false) {
                weightOffset =
                    static_cast<uint64_t>(groupIdx) * gmmSwigluQuantV2BaseParams->N * gmmSwigluQuantV2BaseParams->K +
                    tailN * gmmSwigluQuantV2BaseParams->K;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightOffset =
                    static_cast<uint64_t>(groupIdx) * gmmSwigluQuantV2BaseParams->N * gmmSwigluQuantV2BaseParams->K +
                    tailN;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSwigluQuantV2BaseParams->N];
            }
        }
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice, mmType::BT::isTrans);
        if (gmmSwigluQuantV2BaseParams->isSingleTensor == 0) {
            weightScaleGM.SetGlobalBuffer(GetTensorAddr<uint64_t>(groupIdx, weightScaleTensorPtr));
            mm.SetQuantVector(weightScaleGM[loopK * gmmSwigluQuantV2BaseParams->N + tailN]);
        } else {
            mm.SetQuantVector(
                weightScaleGM[groupIdx * gmmSwigluQuantV2BaseParams->N * gmmSwigluQuantV2BaseParams->quantGroupNum +
                              loopK * gmmSwigluQuantV2BaseParams->N + tailN]);
        }
        mm.IterateAll(mmOutGM[mnConfig.workspaceOffset], loopK == 0 ? 0 : 1);
    }
}
} // namespace GroupedMatmulDequantSwigluQuant
#endif // GMM_SWIGLU_QUANT_V2_A4W4
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_MID_H