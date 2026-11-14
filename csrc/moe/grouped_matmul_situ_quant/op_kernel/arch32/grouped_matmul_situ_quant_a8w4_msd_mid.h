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
 * \file grouped_matmul_situ_quant_a8w4_msd_mid.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_MID_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_MID_H

#include "grouped_matmul_situ_quant_utils_kernel.h"

#ifdef GMM_SITU_QUANT_A8W4_MSD

namespace GroupedMatmulDequantSituQuant {
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
    __aicore__ inline GMMA8W4MidProcess(typename mmType::MT& matmul)
        : mm(matmul)
    {}
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN);
    __aicore__ inline void Process(WorkSpaceSplitConfig& workspaceSplitConfig, int64_t workspaceSplitLoopIdx);

private:
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, WorkSpaceSplitConfig& workspaceSplitConfig);
    __aicore__ inline void SetMNConfig(const int32_t splitValue, MNConfig& mnConfig);
    __aicore__ inline void UpdateMnConfig(MNConfig& mnConfig);

private:
    typename mmType::MT& mm;
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

    GM_ADDR weightTensorPtr;
    GM_ADDR weightScaleTensorPtr;

    // define the que
    uint32_t subBlockIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t quantGroupSize = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParams;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::Init(
    const GMAddrParams gmAddrParams, const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN)
{
    if ASCEND_IS_AIC {
        gmmSituQuantBaseParams = gmmSituQuantBaseParamsIN;
        xRowSumCount = gmmSituQuantBaseParams->M;
        xGM1.SetGlobalBuffer((__gm__ int4b_t*)gmAddrParams.workSpaceGM); // 从前处理中获得的结果
        xGM2.SetGlobalBuffer(
            (__gm__ int4b_t*)((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
        weightGM.SetGlobalBuffer(GetTensorAddr<int4b_t>(0, gmAddrParams.weightGM));
        weightScaleGM.SetGlobalBuffer(GetTensorAddr<uint64_t>(0, gmAddrParams.weightScaleGM));
        groupListGM.SetGlobalBuffer((__gm__ int64_t*)gmAddrParams.groupListGM);
        mmOutGM1.SetGlobalBuffer(
            (__gm__ half*)((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset2));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half*)((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset3));
        quantGroupSize = gmmSituQuantBaseParams->K / gmmSituQuantBaseParams->quantGroupNum; // 约束为整除关系
        subBlockIdx = GetSubBlockIdx();
        coreIdx = GetBlockIdx();
        weightTensorPtr = gmAddrParams.weightGM;
        weightScaleTensorPtr = gmAddrParams.weightScaleGM;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::UpdateMnConfig(MNConfig& mnConfig)
{
    if constexpr (bT::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset +=
            static_cast<uint64_t>(AlignUp<16>(mnConfig.k)) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += static_cast<uint64_t>(mnConfig.k) * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.n;
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::SetMNConfig(const int32_t splitValue, MNConfig& mnConfig)
{
    mnConfig.m = static_cast<int64_t>(splitValue);
    mnConfig.baseM = gmmSituQuantBaseParams->baseM;
    mnConfig.baseN = gmmSituQuantBaseParams->baseN;
    mnConfig.singleM = gmmSituQuantBaseParams->baseM;
    mnConfig.singleN = gmmSituQuantBaseParams->baseN;
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::Process(WorkSpaceSplitConfig& workspaceSplitConfig,
                                                          int64_t workspaceSplitLoopIdx)
{
    if ASCEND_IS_AIC {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        xGM = (workspaceSplitLoopIdx % 2 == 0 ? xGM1 : xGM2);
        mmOutGM = (workspaceSplitLoopIdx % 2 == 0 ? mmOutGM1 : mmOutGM2);
        MNConfig mnConfig;
        mnConfig.baseM = gmmSituQuantBaseParams->baseM;
        mnConfig.baseN = gmmSituQuantBaseParams->baseN;
        mnConfig.singleM = gmmSituQuantBaseParams->baseM;
        mnConfig.singleN = gmmSituQuantBaseParams->baseN;
        mnConfig.k = gmmSituQuantBaseParams->K; // tilingData
        mnConfig.n = gmmSituQuantBaseParams->N; // tilingData
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
        int32_t prevSplitValue = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
        int32_t totalTmp = 0;
        if (gmmSituQuantBaseParams->groupListType == 1) {
            for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
                totalTmp += groupListGM.GetValue(i);
            }
        }
        for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex, preCount = 0;
             groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; ++groupIdx) {
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = 0;
            if (gmmSituQuantBaseParams->groupListType == 0) {
                currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
            } else {
                totalTmp += static_cast<int32_t>(groupListGM.GetValue(groupIdx));
                currSplitValue = totalTmp;
            }
            currSplitValue = currSplitValue > (workspaceSplitLoopIdx + 1) * gmmSituQuantBaseParams->mLimit ?
                                 (workspaceSplitLoopIdx + 1) * gmmSituQuantBaseParams->mLimit :
                                 currSplitValue;

            int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4
            // 自算补偿矩阵时，每个非空专家块在有效行之后再占一行：A 里是全 -8 的 int4 行，
            // cube 顺手算出 cAux = -8*sum_k(w)*scale。挂在块尾是为了落在同一个 M tile 里，
            // 让这一行和其它行共用 B，不必单独把该专家的权重再流一遍。
            if (gmmSituQuantBaseParams->assistRowsPerExpert != 0 && splitValue > 0) {
                splitValue += 1;
            }
            prevSplitValue = currSplitValue;

            SetMNConfig(splitValue, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
            uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
            uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + gmmSituQuantBaseParams->coreNum;
            while (curBlock < curCount) {
                mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
                mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
                MMCompute(groupIdx, mnConfig, workspaceSplitConfig);
                curBlock += gmmSituQuantBaseParams->coreNum;
            }
            preCount = curCount % gmmSituQuantBaseParams->coreNum;
        }
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig,
                                                            WorkSpaceSplitConfig& workspaceSplitConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
        curSingleN = gmmSituQuantBaseParams->N - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t weightOffset = 0;
    mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
    GlobalTensor<int4b_t> weightSlice;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    mnConfig.workspaceOffset = outOffset + mnConfig.yBaseOffset;
    for (uint32_t loopK = 0; loopK < gmmSituQuantBaseParams->quantGroupNum; loopK++) {
        mm.SetTensorA(
            xGM[mnConfig.xBaseOffset + mnConfig.mIdx * mnConfig.k * mnConfig.singleM + loopK * quantGroupSize]);
        if (gmmSituQuantBaseParams->isSingleTensor == 0) {
            weightGM.SetGlobalBuffer(GetTensorAddr<int4b_t>(groupIdx, weightTensorPtr));
            if constexpr (mmType::BT::format == CubeFormat::NZ) {
                weightOffset = tailN * gmmSituQuantBaseParams->K;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightOffset = tailN;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSituQuantBaseParams->N];
            }
        } else {
            if constexpr (mmType::BT::format == CubeFormat::NZ) {
                weightOffset =
                    static_cast<uint64_t>(groupIdx) * gmmSituQuantBaseParams->N * gmmSituQuantBaseParams->K +
                    tailN * gmmSituQuantBaseParams->K;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightOffset =
                    static_cast<uint64_t>(groupIdx) * gmmSituQuantBaseParams->N * gmmSituQuantBaseParams->K +
                    tailN;
                weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmSituQuantBaseParams->N];
            }
        }
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        if (gmmSituQuantBaseParams->isSingleTensor == 0) {
            weightScaleGM.SetGlobalBuffer(GetTensorAddr<uint64_t>(groupIdx, weightScaleTensorPtr));
            mm.SetQuantVector(weightScaleGM[loopK * gmmSituQuantBaseParams->N + tailN]);
        } else {
            mm.SetQuantVector(
                weightScaleGM[groupIdx * gmmSituQuantBaseParams->N * gmmSituQuantBaseParams->quantGroupNum +
                              loopK * gmmSituQuantBaseParams->N + tailN]);
        }
        mm.Iterate();
        mm.GetTensorC(mmOutGM[mnConfig.workspaceOffset], loopK == 0 ? 0 : 1);
    }
}
} // namespace GroupedMatmulDequantSituQuant
#endif // GMM_SITU_QUANT_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_MID_H
