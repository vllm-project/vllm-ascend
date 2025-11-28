/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_weight_nz_tensor_list_split_ws.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_SPLIT_WS_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_SPLIT_WS_H

#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list_utils.h"
namespace GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST {
/** @brief internal computation class
*/

template <class mmType, bool sync = false, typename CHANNELDTYPE = float>
class GMMSwigluSplitWorkSpaceCompute{
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = int8_t;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMSwigluSplitWorkSpaceCompute(typename mmType::MT& mm_): mm(mm_)  {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale,
                                GM_ADDR groupList, GM_ADDR quantOutput, GM_ADDR quantScaleOutput,
                                GM_ADDR workspace,
                                const GMMSwigluBaseParams* __restrict gmmBaseParamsIN,
                                const TCubeTiling* __restrict mmTilingDataIN, 
                                const GMMSwiglu* __restrict gmmSwigluIN, TPipe* tPipeIN);
    __aicore__ inline void Process();

 private:
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx, GlobalTensor<int32_t> &mmOutGM);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

    __aicore__ inline void SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline uint64_t GetWOffset(uint32_t tailN, uint32_t k);

    __aicore__ inline void MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
                                             const uint32_t count, const uint32_t thresholdM_dimN);

    template <typename DTYPE_CS>
    __aicore__ inline void UpdateChannelScale(uint32_t loopidx, VecConfig& vecConfig);

    __aicore__ inline void VectorCompute(uint32_t loopidx, VecConfig& vecConfig);

    template <typename DTYPE_CS>
    __aicore__ inline void PreLoadTokenAndChannel(LocalTensor<float>& channelScaleLocal, VecConfig& vecConfig);

    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig);

    __aicore__ inline void UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig, int32_t workspaceSplitLoopIdx);

    __aicore__ inline void InitWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig);
    
    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<int32_t> &mmOutGM, VecConfig& vecConfig);

    __aicore__ inline void customDataCopyOut(VecConfig& vecConfig);   

    __aicore__ inline void Dequant(uint32_t loopidx, VecConfig& vecConfig);  

    __aicore__ inline void Quant(uint32_t loopidx, VecConfig& vecConfig); 

    __aicore__ inline void Swiglu(uint32_t loopidx, VecConfig& vecConfig);  

 private: 
    typename mmType::MT& mm;
    const GMMSwigluBaseParams* __restrict gmmBaseParams;
    const GMMSwiglu* __restrict gmmSwiglu;
    const TCubeTiling* __restrict mmTilingData;
    uint32_t blockIdx;
    WorkSpaceSplitConfig workspaceSplitConfig;
    TPipe* pipe;
    GlobalTensor<int8_t> xGM;
    GlobalTensor<int8_t> weightGM;
    GlobalTensor<CHANNELDTYPE> perChannelScaleGM;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<int32_t> mmOutGM1;
    GlobalTensor<int32_t> mmOutGM2;
    // define the que
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECIN, 1> perChannelScaleInQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace; 
    TBuf<TPosition::VECCALC> castWorkspace; 
    bool sequentialWrite = true;
    uint32_t cubeNum;  // Matmul completions on the kernel
    uint32_t groupNum; // Matmul completions on the kernel
    int64_t aicCoreNum;
    int64_t aivCoreNum;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR perChannelScalePtr;
};

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::Init(GM_ADDR x, GM_ADDR weight,
                                                            GM_ADDR perChannelScale, GM_ADDR perTokenScale,
                                                            GM_ADDR groupList, GM_ADDR quantOutput,
                                                            GM_ADDR quantScaleOutput, GM_ADDR workspace,
                                                            const GMMSwigluBaseParams* __restrict gmmSwigluBaseParamsIn,
                                                            const TCubeTiling* __restrict mmTilingDataIN, 
                                                            const GMMSwiglu* __restrict gmmSwigluIN, TPipe* tPipeIN)
{   
    aicCoreNum = GetBlockNum();
    aivCoreNum = aicCoreNum * 2;
    blockIdx = GetBlockIdx();
    pipe = tPipeIN;
    xTensorPtr = x;
    weightTensorPtr = weight;
    perChannelScalePtr = perChannelScale;
    mmTilingData = mmTilingDataIN;
    gmmBaseParams = gmmSwigluBaseParamsIn;
    gmmSwiglu = gmmSwigluIN;
    groupNum = gmmSwiglu->groupListLen;
    if ASCEND_IS_AIC {
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)groupList, gmmSwiglu->groupListLen);
        mmOutGM1.SetGlobalBuffer((__gm__ int32_t *)workspace, gmmBaseParams->mLimit * gmmSwiglu->tokenLen);
        mmOutGM2.SetGlobalBuffer((__gm__ int32_t *)workspace + gmmBaseParams->mLimit * gmmSwiglu->tokenLen,
                                 gmmBaseParams->mLimit * gmmSwiglu->tokenLen);
    }
    if ASCEND_IS_AIV {
        mmOutGM1.SetGlobalBuffer((__gm__ int32_t *)workspace, gmmBaseParams->mLimit * gmmSwiglu->tokenLen);
        mmOutGM2.SetGlobalBuffer((__gm__ int32_t *)workspace + gmmBaseParams->mLimit * gmmSwiglu->tokenLen,
                                 gmmBaseParams->mLimit * gmmSwiglu->tokenLen);
        perChannelScaleGM.SetGlobalBuffer((__gm__ CHANNELDTYPE *)perChannelScale,
                                          gmmSwiglu->groupListLen * gmmSwiglu->tokenLen);
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)perTokenScale, gmmBaseParams->M);
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)groupList, gmmSwiglu->groupListLen);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)quantOutput, gmmBaseParams->M * gmmSwiglu->tokenLen / 2);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)quantScaleOutput, gmmBaseParams->M);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::InitWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig) 
{
    workspaceSplitConfig.M = groupListGM.GetValue(gmmSwiglu->groupListLen - 1);
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmBaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmBaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize = workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmBaseParams->mLimit;
    workspaceSplitConfig.leftMatrixStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertNextStartIndex = 0;
    workspaceSplitConfig.isLastLoop = false;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig, int32_t workspaceSplitLoopIdx) 
{   
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmBaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // Calculate the right expert matrix end index (rightMatrixExpertEndIndex) and the next start index (rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    while(workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwiglu->groupListLen)
    {
        curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) - workspaceSplitConfig.leftMatrixStartIndex;
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwiglu->groupListLen - 1 \
                                ? gmmSwiglu->groupListLen - 1 \
                                : workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        if (curTaskNum > gmmBaseParams->mLimit){
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmBaseParams->mLimit && nextTaskNum > gmmBaseParams->mLimit){
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmBaseParams->mLimit){
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;

    if (workspaceSplitConfig.isLastLoop) {
        workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwiglu->groupListLen \
                                                         ? gmmSwiglu->groupListLen - 1 \
                                                         : workspaceSplitConfig.rightMatrixExpertEndIndex;
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::Process() {
    InitWorkSpaceSplitConfig(workspaceSplitConfig);
    int32_t parallelNum = gmmBaseParams->isPreFill ? 2 : 1; // 2: double workspace buffer
    for (int32_t workspaceSplitLoopIdx = 0; workspaceSplitLoopIdx < workspaceSplitConfig.loopCount; workspaceSplitLoopIdx++) {
        UpdateWorkSpaceSplitConfig(workspaceSplitConfig, workspaceSplitLoopIdx);
        GlobalTensor<int32_t> mmOutGM = (workspaceSplitLoopIdx % 2 == 0 ) ? mmOutGM1 : mmOutGM2;
        
        if ASCEND_IS_AIC {
            if (workspaceSplitLoopIdx >= parallelNum){ // first parallelNum core no need to wait
                SyncAll<false>();
            }
            MNConfig mnConfig;
            int32_t prevSplitValue = workspaceSplitConfig.leftMatrixStartIndex;
            for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex, count = 0; groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; ++groupIdx) {
                UpdateMnConfig(mnConfig);
                int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
                currSplitValue = currSplitValue > (workspaceSplitLoopIdx + 1) * gmmBaseParams->mLimit \
                                    ? (workspaceSplitLoopIdx + 1) * gmmBaseParams->mLimit \
                                    : currSplitValue;
                int32_t splitValue = currSplitValue - prevSplitValue;
                prevSplitValue = currSplitValue;
                SetMNConfig(splitValue, groupIdx, mnConfig);
                if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                    continue;
                }
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

                uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
                uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + gmmBaseParams->coreNum;
                uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;
                
                while (curBlock < curCount) {
                    MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
                    MMCompute(groupIdx, mnConfig, blockIdx, mmOutGM);
                    curBlock += aicCoreNum;
                }
                count = curCount % gmmBaseParams->coreNum;
            }
            SyncAll<false>();
        }
        
        if ASCEND_IS_AIV {
            VecConfig vecConfig;
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.AllocTensor<float>();
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                PreLoadTokenAndChannel<CHANNELDTYPE>(channelScaleLocal, vecConfig);
            }
            SyncAll<false>();
            if (blockIdx < vecConfig.usedCoreNum) {
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwiglu->maxProcessRowNum;
                    PipeBarrier<PIPE_ALL>();
                    customDataCopyIn(outLoopIdx, mmOutGM, vecConfig);
                    PipeBarrier<PIPE_ALL>();
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        UpdateChannelScale<CHANNELDTYPE>(innerLoopIdx, vecConfig);
                        VectorCompute(innerLoopIdx, vecConfig);
                    }
                    PipeBarrier<PIPE_ALL>();
                    customDataCopyOut(vecConfig);
                    PipeBarrier<PIPE_ALL>();
                }

                LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.DeQue<float>();
                LocalTensor<int32_t> mmLocal = mmOutQueue.DeQue<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                perChannelScaleInQueue.FreeTensor(channelScaleLocal);
                mmOutQueue.FreeTensor(mmLocal);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
            if (workspaceSplitLoopIdx < workspaceSplitConfig.loopCount - parallelNum){
                SyncAll<false>();
            }
        }
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::PreLoadTokenAndChannel(LocalTensor<float>& channelScaleLocal, VecConfig& vecConfig)
{
    GlobalTensor<CHANNELDTYPE> perChannelScaleTensor;
    perChannelScaleTensor.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(vecConfig.curGroupIdx, perChannelScalePtr));

    DataCopyExtParams copyChannelParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_CS> padParams{false, 0 ,0, 0};
    if constexpr(!IsSameType<DTYPE_CS, float>::value) {
        LocalTensor<DTYPE_CS> dstLocalT = channelScaleLocal.template ReinterpretCast<DTYPE_CS>();
        DataCopyPad(dstLocalT[gmmSwiglu->tokenLen], perChannelScaleTensor, copyChannelParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(channelScaleLocal, dstLocalT[gmmSwiglu->tokenLen], RoundMode::CAST_NONE, gmmSwiglu->tokenLen);
    } else {
        DataCopyPad(channelScaleLocal, perChannelScaleTensor, copyChannelParams, padParams);
    }
    perChannelScaleInQueue.EnQue(channelScaleLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx, GlobalTensor<int32_t> &mmOutGM)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr + mnConfig.xBaseOffset + workspaceSplitConfig.leftMatrixStartIndex * mnConfig.k);
    weightGM.SetGlobalBuffer(GetTensorAddr<int8_t>(groupIdx, weightTensorPtr) + GetWOffset(tailN, mnConfig.k));
    if (mnConfig.blockDimM == 1){
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    } else {
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    }
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGM[xOffset], transposeX);
    mm.SetTensorB(weightGM, transposeW);
    mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::UpdateMnConfig(MNConfig &mnConfig) {
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n);  // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig) {
    SetMKN(splitValue, groupIdx, mnConfig);
    mnConfig.baseM = BASIC_M;
    mnConfig.baseN = BASIC_N;
    mnConfig.singleM = SINGLE_CORE_M;
    mnConfig.singleN = SINGLE_CORE_N;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig)
{
	mnConfig.m = static_cast<int64_t>(splitValue);
    mnConfig.k = gmmBaseParams->K; // tilingData
    mnConfig.n = gmmBaseParams->N; // tilingData
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline uint64_t GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::GetWOffset(uint32_t tailN, uint32_t k) {
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k);  // 16: nz format last two dim size
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
    const uint32_t count, const uint32_t thresholdM_dimN) {
	mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
    mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig)
{
    // Step 1: Read grouplist reduceSum to calculate total data count
    vecConfig.M = workspaceSplitConfig.isLastLoop \
                    ? workspaceSplitConfig.lastLoopTaskSize\
                    : workspaceSplitConfig.notLastTaskSize;
    // Step 2: Calculate core allocation
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx = blockIdx < tailCoreIdx 
                            ? eachCoreTaskNum * blockIdx 
                            :((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t  prevM = workspaceSplitConfig.leftMatrixStartIndex;
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex; groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++){
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
            vecConfig.curGroupIdx = groupIdx;
            vecConfig.nextUpadteInterVal = tempM - curStartIdx;
        }
        curStartIdx -= tempM;
    }
    // Step 3: Calculate total data volume
    vecConfig.outLoopNum = (vecConfig.taskNum + gmmSwiglu->maxProcessRowNum - 1) / gmmSwiglu->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwiglu->maxProcessRowNum 
                            ? vecConfig.taskNum % gmmSwiglu->maxProcessRowNum 
                            : gmmSwiglu->maxProcessRowNum;
    pipe->Reset();
    // Step 4: Allocate space
    pipe->InitBuffer(mmOutQueue, 1, gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen * sizeof(int32_t));
    pipe->InitBuffer(perChannelScaleInQueue, 1, gmmSwiglu->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1, gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen / 2 * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<int32_t>(gmmSwiglu->maxProcessRowNum, 8) * sizeof(float));
    pipe->InitBuffer(reduceWorkspace, 1024 * sizeof(float));
    pipe->InitBuffer(castWorkspace, 32 * sizeof(int8_t));
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<int32_t> &mmOutGM, VecConfig& vecConfig) 
{
    LocalTensor<int32_t> _inMMLocal_0 = mmOutQueue.DeQue<int32_t>();
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(vecConfig.innerLoopNum * gmmSwiglu->tokenLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams_0{false, 0 ,0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(_inMMLocal_0, mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);
    mmOutQueue.EnQue(_inMMLocal_0);
    
    LocalTensor<int32_t> _inMMLocal_1 = mmOutQueue.DeQue<int32_t>();
    
    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1, RoundMode::CAST_NONE, vecConfig.innerLoopNum * gmmSwiglu->tokenLen);
    
    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++){
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        float scale = perTokenScaleGM.GetValue(vecConfig.curIdx + workspaceSplitConfig.leftMatrixStartIndex);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Muls(_inMMLocal_2[i * gmmSwiglu->tokenLen], _inMMLocal_2[i * gmmSwiglu->tokenLen], scale, gmmSwiglu->tokenLen);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vecConfig.curIdx++;
    }
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    vecConfig.curOffset = vecConfig.curIdx * gmmSwiglu->tokenLen;
    mmOutQueue.EnQue(_inMMLocal_2);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::UpdateChannelScale(uint32_t loopIdx, VecConfig& vecConfig){
    // Update perChannel
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        int64_t loop = gmmSwiglu->groupListLen - vecConfig.curGroupIdx;
        while (loop--) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if(nextTemp != curTemp){
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
        LocalTensor<float> _inChannel = perChannelScaleInQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_CS> padParams{false, 0 ,0, 0};

        GlobalTensor<CHANNELDTYPE> perChannelScaleTensor;
        perChannelScaleTensor.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(vecConfig.curGroupIdx, perChannelScalePtr));

        if constexpr(!IsSameType<DTYPE_CS, float>::value) {
            LocalTensor<DTYPE_CS> dstLocalT = _inChannel.template ReinterpretCast<DTYPE_CS>();
            DataCopyPad(dstLocalT[gmmSwiglu->tokenLen], perChannelScaleTensor, copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(_inChannel, dstLocalT[gmmSwiglu->tokenLen], RoundMode::CAST_NONE, gmmSwiglu->tokenLen);
        } else {
            DataCopyPad(_inChannel, perChannelScaleTensor, copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        
        perChannelScaleInQueue.EnQue(_inChannel);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::VectorCompute(uint32_t loopIdx, VecConfig& vecConfig) {
        Dequant(loopIdx, vecConfig);
        Swiglu(loopIdx, vecConfig);
        Quant(loopIdx, vecConfig);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::Dequant(uint32_t loopIdx, VecConfig& vecConfig) {
    // perChanelScale * perTokenScale
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    LocalTensor<float> perChannelLocal = perChannelScaleInQueue.DeQue<float>();
    Mul(mmLocal[loopIdx * gmmSwiglu->tokenLen], mmLocal[loopIdx * gmmSwiglu->tokenLen], perChannelLocal, gmmSwiglu->tokenLen);
    vecConfig.nextUpadteInterVal--;
    mmOutQueue.EnQue(mmLocal);
    perChannelScaleInQueue.EnQue(perChannelLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::Swiglu(uint32_t loopIdx, VecConfig& vecConfig) {
    // High-level API swiglu
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal= reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local = _inMMLocal[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / 2];
    LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwiglu->tokenLen];
    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwiglu->tokenLen / 2);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams repeatParams{1, static_cast<uint16_t>((gmmSwiglu->tokenLen / 2) / 8), 0, 0};
    DataCopy(_inMMLocal[loopIdx * gmmSwiglu->tokenLen], workspaceLocal, repeatParams);
    mmOutQueue.EnQue(_inMMLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::Quant(uint32_t loopIdx, VecConfig& vecConfig) {
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    Abs(_inMMLocal[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / BISECT],
        _inMMLocal[loopIdx * gmmSwiglu->tokenLen],
        gmmSwiglu->tokenLen / BISECT);
    LocalTensor<float> workspaceLocal= reduceWorkspace.Get<float>();
    PipeBarrier<PIPE_ALL>();
    ReduceMaxTemplate(workspaceLocal, 
        _inMMLocal, loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / BISECT, gmmSwiglu->tokenLen / BISECT);
    PipeBarrier<PIPE_ALL>();
    float quantScale = workspaceLocal.GetValue(0) / QUANT_SCALE_INT8;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    PipeBarrier<PIPE_ALL>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    PipeBarrier<PIPE_ALL>();
    quantScale = 1 / quantScale;
    PipeBarrier<PIPE_ALL>();
    Muls(_inMMLocal[loopIdx * gmmSwiglu->tokenLen], _inMMLocal[loopIdx * gmmSwiglu->tokenLen],
         quantScale, gmmSwiglu->tokenLen / BISECT);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(loopIdx * gmmSwiglu->tokenLen / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(loopIdx * gmmSwiglu->tokenLen);
    int32_t tempCount = static_cast<int32_t>(gmmSwiglu->tokenLen / BISECT);
    LocalTensor<int8_t> castSpace = castWorkspace.Get<int8_t>();
    CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
    mmOutQueue.EnQue(_inMMLocal);
    quantOutQueue.EnQue(quantLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluSplitWorkSpaceCompute<mmType, sync, CHANNELDTYPE>::customDataCopyOut(VecConfig& vecConfig) {
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParams_0{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(quantScaleOutputGM[workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx], quantScaleLocal, copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{1, (uint16_t)(vecConfig.innerLoopNum * gmmSwiglu->tokenLen / 2 * sizeof(int8_t)), 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) * gmmSwiglu->tokenLen / 2], quantLocal, copyParams_1);
    PipeBarrier<PIPE_ALL>();
    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

}  // namespace GROUPED_MATMUL
#endif  // ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_SPLIT_WS_H
