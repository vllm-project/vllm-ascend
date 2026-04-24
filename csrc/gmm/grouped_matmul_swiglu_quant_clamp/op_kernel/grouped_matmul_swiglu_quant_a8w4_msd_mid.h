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
 * \file grouped_matmul_antiquant_a8w4_msd.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_MID_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_MID_H

#include "grouped_matmul_swiglu_quant_utils.h"
#include "grouped_matmul_swiglu_quant_clamp.h"

#ifdef GMM_SWIGLU_QUANT_A8W4_MSD
namespace GROUPED_MATMUL_SWIGLU_QUANT {
using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                         uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4ND(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                           uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1,
                                         uint32_t dim0, uint32_t srcDim0, uint32_t dstDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <class mmType>
class GMMA8W4MidProcess {
public:
    using bT = typename mmType::BT;

public:
    __aicore__ inline GMMA8W4MidProcess(typename mmType::MT &matmul) : mm(matmul)
    {
    }
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN);
    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx);

private:
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig &mnConfig, WorkSpaceSplitConfig &workspaceSplitConfig);
    __aicore__ inline void SetMNConfig(const int32_t splitValue, MNConfig &mnConfig);
    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

private:
    typename mmType::MT &mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<int4b_t> xGM;
    GlobalTensor<int4b_t> xGM1;
    GlobalTensor<int4b_t> xGM2;
    GlobalTensor<int4b_t> weightGM;

    GlobalTensor<half> mmOutGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<uint64_t> weightScaleGM;
    // define the que
    uint32_t subBlockIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t quantGroupSize = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    const GMMSwigluBaseParams *__restrict gmmBaseParams;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::Init(const GMAddrParams gmAddrParams,
                                                       const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN)
{
    if ASCEND_IS_AIC {
        gmmBaseParams = gmmSwigluBaseParamsIN;
        xRowSumCount = gmmBaseParams->M;
        xGM1.SetGlobalBuffer((__gm__ int4b_t *)gmAddrParams.workSpaceGM); // 从前处理中获得的结果
        xGM2.SetGlobalBuffer(
            (__gm__ int4b_t *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
        weightGM.SetGlobalBuffer((__gm__ int4b_t *)gmAddrParams.weightGM);
        weightScaleGM.SetGlobalBuffer((__gm__ uint64_t *)gmAddrParams.weightScaleGM);
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM);
        mmOutGM1.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset2));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset3));
        quantGroupSize = gmmBaseParams->K / gmmBaseParams->quantGroupNum; // 约束为整除关系
        subBlockIdx = GetSubBlockIdx();
        coreIdx = GetBlockIdx();
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::UpdateMnConfig(MNConfig &mnConfig)
{
    if constexpr (bT::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::SetMNConfig(const int32_t splitValue, MNConfig &mnConfig)
{
    mnConfig.m = static_cast<int64_t>(splitValue);
    mnConfig.baseM = gmmBaseParams->baseM;
    mnConfig.baseN = gmmBaseParams->baseN;
    mnConfig.singleM = gmmBaseParams->baseM;
    mnConfig.singleN = gmmBaseParams->baseN;
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                          int64_t workspaceSplitLoopIdx)
{
    if ASCEND_IS_AIC {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        xGM = (workspaceSplitLoopIdx % 2 == 0 ? xGM1 : xGM2);
        mmOutGM = (workspaceSplitLoopIdx % 2 == 0 ? mmOutGM1 : mmOutGM2);
        MNConfig mnConfig;
        mnConfig.baseM = gmmBaseParams->baseM;
        mnConfig.baseN = gmmBaseParams->baseN;
        mnConfig.singleM = gmmBaseParams->baseM;
        mnConfig.singleN = gmmBaseParams->baseN;
        mnConfig.k = gmmBaseParams->K; // tilingData
        mnConfig.n = gmmBaseParams->N; // tilingData
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
        int32_t prevSplitValue = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
        for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex, preCount = 0;
             groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; ++groupIdx) {
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
            currSplitValue = currSplitValue > (workspaceSplitLoopIdx + 1) * gmmBaseParams->mLimit ?
                                 (workspaceSplitLoopIdx + 1) * gmmBaseParams->mLimit :
                                 currSplitValue;

            int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4
            prevSplitValue = currSplitValue;

            SetMNConfig(splitValue, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
            uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
            uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + gmmBaseParams->coreNum;
            while (curBlock < curCount) {
                mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
                mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
                MMCompute(groupIdx, mnConfig, workspaceSplitConfig);
                curBlock += gmmBaseParams->coreNum;
            }
            preCount = curCount % gmmBaseParams->coreNum;
        }
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::MMCompute(uint32_t groupIdx, MNConfig &mnConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
        curSingleN = gmmBaseParams->N - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t weightOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        weightOffset = static_cast<uint64_t>(groupIdx) * gmmBaseParams->N * gmmBaseParams->K + tailN * gmmBaseParams->K;
    } else {
        weightOffset = static_cast<uint64_t>(groupIdx) * gmmBaseParams->N * gmmBaseParams->K + tailN;
    }
    mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
    GlobalTensor<int4b_t> weightSlice;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    for (uint32_t loopK = 0; loopK < gmmBaseParams->quantGroupNum; loopK++) {
        mm.SetTensorA(
            xGM[mnConfig.xBaseOffset + mnConfig.mIdx * mnConfig.k * mnConfig.singleM + loopK * quantGroupSize]);
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
        } else {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmBaseParams->N];
        }
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        mm.SetQuantVector(weightScaleGM[groupIdx * gmmBaseParams->N * gmmBaseParams->quantGroupNum +
                                        loopK * gmmBaseParams->N + tailN]);
        mm.Iterate();
        mm.GetTensorC(mmOutGM[mnConfig.workSpaceOffset], loopK == 0 ? 0 : 1);
    }
}
} // namespace GROUPED_MATMUL_SWIGLU_QUANT
#endif // GMM_SWIGLU_QUANT_A8W4_MSD
#endif // ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_MID_H