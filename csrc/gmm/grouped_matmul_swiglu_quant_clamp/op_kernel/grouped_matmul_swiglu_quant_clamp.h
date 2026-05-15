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
 * \file grouped_matmul_swiglu_quant_clamp.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_H

#include "grouped_matmul_swiglu_quant_utils.h"
namespace GROUPED_MATMUL_SWIGLU_QUANT {
/** @brief intenal computation class
 */
template <class mmType, bool sync = false, typename CHANNELDTYPE = float>
class GMMSwigluCompute {
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = int8_t;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;
    static constexpr float FLOAT_INF = 3e+99;
    /** @brief constructor */
    __aicore__ inline GMMSwigluCompute(typename mmType::MT &mm_) : mm(mm_)
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale,
                                GM_ADDR groupList, GM_ADDR quantOutput, GM_ADDR quantScaleOutput, GM_ADDR workspace,
                                const GMMSwigluBaseParams *__restrict gmmBaseParamsIN,
                                const TCubeTiling *__restrict mmTilingDataIN, const GMMSwiglu *__restrict gmmSwigluIN,
                                TPipe *tPipeIN);
    __aicore__ inline void Process();

private:
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig &mnConfig, uint32_t coreIdx);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

    __aicore__ inline void SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline uint64_t GetWOffset(uint32_t tailN, uint32_t k);

    __aicore__ inline void CubeProcess(MNConfig &mnConfig);
    __aicore__ inline void VecProcess(VecConfig &vecConfig);
    __aicore__ inline void MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock, const uint32_t count,
                                             const uint32_t thresholdM_dimN);
    template <typename DTYPE_CS>
    __aicore__ inline void UpdateChannelScale(uint32_t loopidx, VecConfig &vecConfig);
    __aicore__ inline void VectorCompute(uint32_t loopidx, VecConfig &vecConfig);
    template <typename DTYPE_CS>
    __aicore__ inline void PreLoadTokenAndChannel(LocalTensor<float> &channelScaleLocal, VecConfig &vecConfig);
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig);
    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, VecConfig &vecConfig);
    __aicore__ inline void customDataCopyOut(VecConfig &vecConfig);
    __aicore__ inline void Dequant(uint32_t loopidx, VecConfig &vecConfig);
    __aicore__ inline void Quant(uint32_t loopidx);
    __aicore__ inline void Swiglu(uint32_t loopidx);

private:
    typename mmType::MT &mm;
    const GMMSwigluBaseParams *__restrict gmmBaseParams;
    const GMMSwiglu *__restrict gmmSwiglu;
    const TCubeTiling *__restrict mmTilingData;
    TPipe *pipe;
    GlobalTensor<int8_t> xGM;
    GlobalTensor<int8_t> weightGM;
    GlobalTensor<CHANNELDTYPE> perChannelScaleGM;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<int32_t> mmOutGM;
    // define the que
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECIN, 1> perChannelScaleInQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int32_t preOffset = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
    float limited = FLOAT_INF;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
};

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale, GM_ADDR groupList, GM_ADDR quantOutput,
    GM_ADDR quantScaleOutput, GM_ADDR workspace, const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIn,
    const TCubeTiling *__restrict mmTilingDataIN, const GMMSwiglu *__restrict gmmSwigluIN, TPipe *tPipeIN)
{
    aicCoreNum = GetBlockNum();
    aivCoreNum = aicCoreNum * 2;
    blockIdx = GetBlockIdx();
    mmTilingData = mmTilingDataIN;
    gmmBaseParams = gmmSwigluBaseParamsIn;
    gmmSwiglu = gmmSwigluIN;
    pipe = tPipeIN;
    xTensorPtr = x;
    limited = gmmBaseParams->limited;
    weightTensorPtr = weight;
    groupListGM.SetGlobalBuffer((__gm__ int64_t *)groupList, gmmSwiglu->groupListLen);
    mmOutGM.SetGlobalBuffer((__gm__ int32_t *)workspace, gmmBaseParams->M * gmmSwiglu->tokenLen);
    if ASCEND_IS_AIV {
        perChannelScaleGM.SetGlobalBuffer((__gm__ CHANNELDTYPE *)perChannelScale,
                                          gmmSwiglu->groupListLen * gmmSwiglu->tokenLen);
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)perTokenScale, gmmBaseParams->M);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)quantOutput,
                                      gmmBaseParams->M * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)quantScaleOutput, gmmBaseParams->M);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Process()
{
    MNConfig mnConfig;
    VecConfig vecConfig;
    CubeProcess(mnConfig);
    VecProcess(vecConfig);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void
GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PreLoadTokenAndChannel(LocalTensor<float> &channelScaleLocal,
                                                                     VecConfig &vecConfig)
{
    DataCopyExtParams copyChannelParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_CS> padParams{false, 0, 0, 0};
    if constexpr (!IsSameType<DTYPE_CS, float>::value) {
        LocalTensor<DTYPE_CS> dstLocalT = channelScaleLocal.template ReinterpretCast<DTYPE_CS>();
        DataCopyPad(dstLocalT[gmmSwiglu->tokenLen], perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                    copyChannelParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(channelScaleLocal, dstLocalT[gmmSwiglu->tokenLen], RoundMode::CAST_NONE, gmmSwiglu->tokenLen);
    } else {
        DataCopyPad(channelScaleLocal, perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                    copyChannelParams, padParams);
    }
    perChannelScaleInQueue.EnQue(channelScaleLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMCompute(uint32_t groupIdx, MNConfig &mnConfig,
                                                                               uint32_t coreIdx)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr + mnConfig.xBaseOffset);
    weightGM.SetGlobalBuffer((__gm__ int8_t *)weightTensorPtr + mnConfig.wBaseOffset + GetWOffset(tailN, mnConfig.k));
    if (mnConfig.blockDimM == 1) {
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGM[xOffset], transposeX);
    mm.SetTensorB(weightGM, transposeW);
    mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateMnConfig(MNConfig &mnConfig)
{
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMNConfig(const int32_t splitValue,
                                                                                 const uint32_t groupIdx,
                                                                                 MNConfig &mnConfig)
{
    SetMKN(splitValue, groupIdx, mnConfig);
    mnConfig.baseM = BASIC_M;
    mnConfig.baseN = BASIC_N;
    mnConfig.singleM = SINGLE_CORE_M;
    mnConfig.singleN = SINGLE_CORE_N;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMKN(const int32_t splitValue,
                                                                            const uint32_t groupIdx, MNConfig &mnConfig)
{
    mnConfig.m = static_cast<uint32_t>(splitValue);
    mnConfig.k = gmmBaseParams->K; // tilingData
    mnConfig.n = gmmBaseParams->N; // tilingData
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline uint64_t GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::GetWOffset(uint32_t tailN, uint32_t k)
{
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k); // 16: nz format last two dim size
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::CubeProcess(MNConfig &mnConfig)
{
    if ASCEND_IS_AIC {
        preOffset = 0;
        int32_t prevSplitValue = 0;
        for (uint32_t groupIdx = 0, count = 0; groupIdx < gmmSwiglu->groupListLen; ++groupIdx) {
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
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
                MMCompute(groupIdx, mnConfig, blockIdx);
                curBlock += aicCoreNum;
            }
            count = curCount % gmmBaseParams->coreNum;
        }
        SyncAll<false>();
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VecProcess(VecConfig &vecConfig)
{
    if ASCEND_IS_AIV {
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
                vecConfig.innerLoopNum =
                    outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum : gmmSwiglu->maxProcessRowNum;
                customDataCopyIn(outLoopIdx, vecConfig);
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    UpdateChannelScale<CHANNELDTYPE>(innerLoopIdx, vecConfig);
                    VectorCompute(innerLoopIdx, vecConfig);
                }
                customDataCopyOut(vecConfig);
            }

            LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.DeQue<float>();
            LocalTensor<int32_t> mmLocal = mmOutQueue.DeQue<int32_t>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            perChannelScaleInQueue.FreeTensor(channelScaleLocal);
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        } else {
            return;
        }
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void
GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
                                                                const uint32_t count, const uint32_t thresholdM_dimN)
{
    mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
    mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateVecConfig(uint32_t blockIdx,
                                                                                     VecConfig &vecConfig)
{
    // 第一步 读取grouplist reduceSum 计算总数据个数
    int64_t prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwiglu->groupListLen; groupIdx++) {
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        vecConfig.M += tempM;
    }
    // 第二步 计算分核
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx =
        blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwiglu->groupListLen; groupIdx++) {
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
            vecConfig.curGroupIdx = groupIdx;
            vecConfig.nextUpadteInterVal = tempM - curStartIdx;
        }
        curStartIdx -= tempM;
    }
    // 第三步 计算总数据量
    vecConfig.outLoopNum = (vecConfig.taskNum + gmmSwiglu->maxProcessRowNum - 1) / gmmSwiglu->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwiglu->maxProcessRowNum ?
                                vecConfig.taskNum % gmmSwiglu->maxProcessRowNum :
                                gmmSwiglu->maxProcessRowNum;
    pipe->Reset();
    // 第四步 申请空间
    pipe->InitBuffer(mmOutQueue, DOUBLE_BUFFER, gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen * sizeof(int32_t));
    pipe->InitBuffer(perChannelScaleInQueue, DOUBLE_BUFFER, gmmSwiglu->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, DOUBLE_BUFFER,
                     gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, DOUBLE_BUFFER,
                     AlignUp<int32_t>(gmmSwiglu->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) + UB_BLOCK_UNIT_SIZE +
                                          UB_BLOCK_UNIT_SIZE);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyIn(uint32_t outLoopIdx,
                                                                                      VecConfig &vecConfig)
{
    LocalTensor<int32_t> _inMMLocal_0 = mmOutQueue.DeQue<int32_t>();
    DataCopyExtParams copyParams_0{
        1, static_cast<uint32_t>(vecConfig.innerLoopNum * gmmSwiglu->tokenLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams_0{false, 0, 0, 0};
    DataCopyPad(_inMMLocal_0, mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);

    mmOutQueue.EnQue(_inMMLocal_0);

    LocalTensor<int32_t> _inMMLocal_1 = mmOutQueue.DeQue<int32_t>();

    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1, RoundMode::CAST_NONE,
         vecConfig.innerLoopNum * gmmSwiglu->tokenLen);

    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        float scale = perTokenScaleGM.GetValue(vecConfig.curIdx);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(_inMMLocal_2[i * gmmSwiglu->tokenLen], _inMMLocal_2[i * gmmSwiglu->tokenLen], scale, gmmSwiglu->tokenLen);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        vecConfig.curIdx++;
    }
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    vecConfig.curOffset = vecConfig.curIdx * gmmSwiglu->tokenLen;
    mmOutQueue.EnQue(_inMMLocal_2);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateChannelScale(uint32_t loopIdx,
                                                                                        VecConfig &vecConfig)
{
    // 更新perChannel
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        int64_t loop = gmmSwiglu->groupListLen - vecConfig.curGroupIdx;
        while (loop--) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if (nextTemp != curTemp) {
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
        LocalTensor<float> _inChannel = perChannelScaleInQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_CS> padParams{false, 0, 0, 0};
        if constexpr (!IsSameType<DTYPE_CS, float>::value) {
            LocalTensor<DTYPE_CS> dstLocalT = _inChannel.template ReinterpretCast<DTYPE_CS>();
            DataCopyPad(dstLocalT[gmmSwiglu->tokenLen], perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                        copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(_inChannel, dstLocalT[gmmSwiglu->tokenLen], RoundMode::CAST_NONE, gmmSwiglu->tokenLen);
        } else {
            DataCopyPad(_inChannel, perChannelScaleGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen], copyParams,
                        padParams);
        }
        PipeBarrier<PIPE_ALL>();
        perChannelScaleInQueue.EnQue(_inChannel);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VectorCompute(uint32_t loopIdx,
                                                                                   VecConfig &vecConfig)
{
    Dequant(loopIdx, vecConfig);
    Swiglu(loopIdx);
    Quant(loopIdx);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Dequant(uint32_t loopIdx, VecConfig &vecConfig)
{
    // perChanelScale * perTokenScale
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    LocalTensor<float> perChannelLocal = perChannelScaleInQueue.DeQue<float>();
    Mul(mmLocal[loopIdx * gmmSwiglu->tokenLen], mmLocal[loopIdx * gmmSwiglu->tokenLen], perChannelLocal,
        gmmSwiglu->tokenLen);
    vecConfig.nextUpadteInterVal--;
    mmOutQueue.EnQue(mmLocal);
    perChannelScaleInQueue.EnQue(perChannelLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Swiglu(uint32_t loopIdx)
{
    // 高阶API swiglu
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        _inMMLocal[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwiglu->tokenLen];
    Mins(src0Local, src0Local, limited, gmmSwiglu->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    Maxs(src0Local, src0Local, (-1.0f * limited), gmmSwiglu->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    Mins(src1Local, src1Local, limited, gmmSwiglu->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams repeatParams{1, static_cast<uint16_t>((gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0,
                                0};
    DataCopy(_inMMLocal[loopIdx * gmmSwiglu->tokenLen], workspaceLocal, repeatParams);
    mmOutQueue.EnQue(_inMMLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Quant(uint32_t loopIdx)
{
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    uint64_t preOffset = loopIdx * gmmSwiglu->tokenLen;
    uint64_t halfTokenLen = gmmSwiglu->tokenLen / BISECT;
    Abs(_inMMLocal[preOffset + gmmSwiglu->tokenLen / BISECT], _inMMLocal[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, _inMMLocal[preOffset + gmmSwiglu->tokenLen / BISECT], reduceTmpLocal,
                      static_cast<uint32_t>(halfTokenLen));

    int32_t eventIdVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    quantScale = 1 / quantScale;
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(_inMMLocal[preOffset], _inMMLocal[preOffset], quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(preOffset);
    int32_t tempCount = static_cast<int32_t>(halfTokenLen);
    LocalTensor<int8_t> castSpace = reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
    CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
    mmOutQueue.EnQue(_inMMLocal);
    quantOutQueue.EnQue(quantLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyOut(VecConfig &vecConfig)
{
    // perChanelScale * perTokenScale
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParams_0{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(quantScaleOutputGM[vecConfig.startIdx], quantScaleLocal, copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(quantOutputGM[vecConfig.startIdx * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR], quantLocal,
                copyParams_1);
    PipeBarrier<PIPE_ALL>();
    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

} // namespace GROUPED_MATMUL_SWIGLU_QUANT
#endif // ASCENDC_GROUPED_MATMUL_QUANT_MIXCORE_H
