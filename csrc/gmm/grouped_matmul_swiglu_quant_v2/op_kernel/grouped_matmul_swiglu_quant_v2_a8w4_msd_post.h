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
 * \file grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_POST_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_POST_H

#include "grouped_matmul_swiglu_quant_v2_utils.h"
#include "kernel_operator.h"

#ifdef GMM_SWIGLU_QUANT_V2_A8W4_MSD

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA8W4PostProcess {
public:
    __aicore__ inline GMMA8W4PostProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
                                const GMMSwigluQuantV2 *__restrict gmmSwigluIN);

    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe *pipe);
    static constexpr float FLOAT_INF = 3e+99;
private:
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                           WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                           TPipe *pipe);

    __aicore__ inline void UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                         WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyOut(VecConfig &vecConfig, WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void PreLoadAuxiliaryMatrix(VecConfig &vecConfig);

    __aicore__ inline void Quant(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void Swiglu(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);
    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> weightAuxiliaryMatrixGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    TQue<QuePosition::VECIN, 1> weightAuxiliaryMatrixInQueue;
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
    GM_ADDR weightAuxiliaryMatrixTensorPtr;
    float limited = FLOAT_INF;
};

__aicore__ inline void
GMMA8W4PostProcess::Init(const GMAddrParams gmAddrParams,
                         const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
                         const GMMSwigluQuantV2 *__restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        aicCoreNum = GetBlockNum();
        aivCoreNum = aicCoreNum * 2;
        blockIdx = GetBlockIdx();
        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;
        gmmSwigluQuantV2 = gmmSwigluIN;
        weightAuxiliaryMatrixGM.SetGlobalBuffer(GetTensorAddr<float>(0, gmAddrParams.weightAuxiliaryMatrixGM));
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM, gmmSwigluQuantV2->groupListLen);
        mmOutGM1.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset2));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset3));
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.xScaleGM, gmmSwigluQuantV2BaseParams->M);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.yGM, gmmSwigluQuantV2BaseParams->M *
                                                                             gmmSwigluQuantV2->tokenLen /
                                                                             SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.yScaleGM, gmmSwigluQuantV2BaseParams->M);
        weightAuxiliaryMatrixTensorPtr = gmAddrParams.weightAuxiliaryMatrixGM;
        limited = gmmSwigluQuantV2BaseParams->swigluLimit;
    }
}

__aicore__ inline void GMMA8W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM,
                                                            VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<half> _inMMLocal_0 = mmOutQueue.DeQue<half>();
    const int64_t processNum = 2 * vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(_inMMLocal_0[processNum], mmOutGM[vecConfig.curOffset * DOUBLE_ROW], copyParams_0, padParams_0);

    mmOutQueue.EnQue(_inMMLocal_0);

    LocalTensor<half> _inMMLocal_1 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1[processNum], RoundMode::CAST_NONE, processNum);

    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    // 2. high_4bit * 16 + low_4bit
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        Muls(_inMMLocal_2[(DOUBLE_ROW * i) * gmmSwigluQuantV2->tokenLen],
             _inMMLocal_2[(DOUBLE_ROW * i) * gmmSwigluQuantV2->tokenLen], DEFAULT_MUL_SCALE,
             gmmSwigluQuantV2->tokenLen);
        PipeBarrier<PIPE_V>();
        Add(_inMMLocal_2[i * gmmSwigluQuantV2->tokenLen], _inMMLocal_2[(DOUBLE_ROW * i) * gmmSwigluQuantV2->tokenLen],
            _inMMLocal_2[(DOUBLE_ROW * i + 1) * gmmSwigluQuantV2->tokenLen], gmmSwigluQuantV2->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
    }
    vecConfig.curOffset = vecConfig.curIdx * gmmSwigluQuantV2->tokenLen;
    mmOutQueue.EnQue(_inMMLocal_2);
}

__aicore__ inline void GMMA8W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                                         WorkSpaceSplitConfig &workspaceSplitConfig)
{
    // 1.辅助矩阵加回
    MergeAuxiliaryMatrix(loopIdx, vecConfig);
    // 2.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 3.Swiglu
    Swiglu(loopIdx, vecConfig);
    // 4.Quant
    Quant(loopIdx, vecConfig);
}

__aicore__ inline void GMMA8W4PostProcess::MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig &vecConfig)
{
    // perChanelScale * perTokenScale
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    Add(mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen], mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen],
        weightAuxiliaryMatrixLocal, gmmSwigluQuantV2->tokenLen);
    vecConfig.nextUpdateInterVal--;
    mmOutQueue.EnQue(mmLocal);
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen], mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen], scale,
         gmmSwigluQuantV2->tokenLen);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void GMMA8W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 高阶API swiglu
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        _inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen + gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen];

    Mins(src0Local, src0Local, limited, gmmSwigluQuantV2->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    Maxs(src0Local, src0Local, (-1.0f * limited), gmmSwigluQuantV2->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    Mins(src1Local, src1Local, limited, gmmSwigluQuantV2->tokenLen / 2);
    PipeBarrier<PIPE_V>();
    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_V>();
    DataCopyParams repeatParams{
        1, static_cast<uint16_t>((gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0, 0};
    DataCopy(_inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen], workspaceLocal, repeatParams);

    mmOutQueue.EnQue(_inMMLocal);
}

__aicore__ inline void GMMA8W4PostProcess::Quant(uint32_t loopIdx, VecConfig &vecConfig)
{
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;
    uint64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / BISECT;
    Abs(_inMMLocal[preOffset + gmmSwigluQuantV2->tokenLen / BISECT], _inMMLocal[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, _inMMLocal[preOffset + gmmSwigluQuantV2->tokenLen / BISECT],
                      reduceTmpLocal, static_cast<uint32_t>(halfTokenLen));
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

__aicore__ inline void GMMA8W4PostProcess::UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                                           WorkSpaceSplitConfig &workspaceSplitConfig,
                                                           int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    // 第一步 读取grouplist reduceSum 计算总数据个数
    vecConfig.M = workspaceSplitLoopIdx < workspaceSplitConfig.loopCount - 1 ? workspaceSplitConfig.notLastTaskSize :
                                                                               workspaceSplitConfig.lastLoopTaskSize;
    // 第二步 计算分核
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx =
        blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t prevM = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
    int64_t totalTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex;
         groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++) {
        int64_t currM = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            currM = groupListGM.GetValue(groupIdx);
        } else {
            totalTmp += groupListGM.GetValue(groupIdx);
            currM = totalTmp;
        }
        int64_t tempM = currM - prevM;
        prevM = currM;
        if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
            vecConfig.curGroupIdx = groupIdx;
            vecConfig.nextUpdateInterVal = tempM - curStartIdx;
        }
        curStartIdx -= tempM;
    }
    // 第三步 计算总数据量
    vecConfig.outLoopNum =
        (vecConfig.taskNum + gmmSwigluQuantV2->maxProcessRowNum - 1) / gmmSwigluQuantV2->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum ?
                                vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum :
                                gmmSwigluQuantV2->maxProcessRowNum;

    // 第四步 申请空间
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n / 2 *sizeof(float) + 64 < 191 * 1024
    pipe->InitBuffer(mmOutQueue, 1,
                     2 * gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1,
                     gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR *
                         sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1,
                     AlignUp<int32_t>(gmmSwigluQuantV2->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    pipe->InitBuffer(weightAuxiliaryMatrixInQueue, 1, gmmSwigluQuantV2->tokenLen * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) +
                                          UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA8W4PostProcess::PreLoadAuxiliaryMatrix(VecConfig &vecConfig)
{
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    DataCopyExtParams copyAuxiliaryMatrixParams{1, static_cast<uint32_t>(gmmSwigluQuantV2->tokenLen * sizeof(float)), 0,
                                                0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (gmmSwigluQuantV2BaseParams->isSingleTensor == 0) {
        weightAuxiliaryMatrixGM.SetGlobalBuffer(
            GetTensorAddr<float>(vecConfig.curGroupIdx, weightAuxiliaryMatrixTensorPtr));
        DataCopyPad(weightAuxiliaryMatrixLocal, weightAuxiliaryMatrixGM, copyAuxiliaryMatrixParams, padParams);
    } else {
        DataCopyPad(weightAuxiliaryMatrixLocal,
                    weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSwigluQuantV2->tokenLen],
                    copyAuxiliaryMatrixParams, padParams);
    }
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 更新weightAuxiliaryMatrix
    if (unlikely(vecConfig.nextUpdateInterVal == 0)) {
        int64_t loop = gmmSwigluQuantV2->groupListLen - vecConfig.curGroupIdx;
        while (loop--) {
            if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
                int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                vecConfig.curGroupIdx++;
                int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                if (nextTemp != curTemp) {
                    vecConfig.nextUpdateInterVal = nextTemp - curTemp;
                    break;
                }
            } else {
                vecConfig.curGroupIdx++;
                int64_t nextUpdateInterValTmp = groupListGM.GetValue(vecConfig.curGroupIdx);
                if (nextUpdateInterValTmp != 0) {
                    vecConfig.nextUpdateInterVal = nextUpdateInterValTmp;
                    break;
                }
            }
        }
        LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSwigluQuantV2->tokenLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(weightAuxiliaryMatrixLocal,
                    weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSwigluQuantV2->tokenLen], copyParams, padParams);
        weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
    }
}

__aicore__ inline void GMMA8W4PostProcess::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                   int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        VecConfig vecConfig;
        UpdateVecConfig(blockIdx, vecConfig, workspaceSplitConfig, workspaceSplitLoopIdx, pipe);

        if (blockIdx < vecConfig.usedCoreNum) {
            mmOutGM = (workspaceSplitLoopIdx % 2 == 0 ? mmOutGM1 : mmOutGM2);
            LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.AllocTensor<float>();
            LocalTensor<half> mmLocal = mmOutQueue.AllocTensor<half>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();

            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
            PreLoadAuxiliaryMatrix(vecConfig);
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum :
                                                                                    gmmSwigluQuantV2->maxProcessRowNum;
                int32_t eventIdMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                // 1.matmul中间结果搬入 + 高四位与低四位合并
                customDataCopyIn(outLoopIdx, mmOutGM, vecConfig, workspaceSplitConfig);

                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    // 2.如果涉及group切换，更新辅助矩阵
                    UpdateAuxiliaryMatrix(innerLoopIdx, vecConfig);
                    // 3. 四步vector计算（辅助矩阵加回、perToken反量化、Swiglu、Quant）
                    VectorCompute(innerLoopIdx, vecConfig, workspaceSplitConfig);
                }
                int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                customDataCopyOut(vecConfig, workspaceSplitConfig);
            }
            weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
            mmLocal = mmOutQueue.DeQue<half>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();

            weightAuxiliaryMatrixInQueue.FreeTensor(weightAuxiliaryMatrixLocal);
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        }
    }
}

__aicore__ inline void GMMA8W4PostProcess::customDataCopyOut(VecConfig &vecConfig,
                                                             WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParams_0{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    DataCopyPad(quantScaleOutputGM[workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx], quantScaleLocal,
                copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0,
        0};
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) *
                              gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR],
                quantLocal, copyParams_1);

    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

} // namespace GroupedMatmulDequantSwigluQuant
#endif // GMM_SWIGLU_QUANT_V2_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_POST_H