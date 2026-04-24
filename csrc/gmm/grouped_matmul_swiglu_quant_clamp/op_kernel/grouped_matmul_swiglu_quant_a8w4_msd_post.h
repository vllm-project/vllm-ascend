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
 * \file grouped_matmul_swiglu_quant_a8w4_msd_post.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_POST_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_POST_H
#include "grouped_matmul_swiglu_quant_utils.h"
#include "kernel_operator.h"
#ifdef GMM_SWIGLU_QUANT_A8W4_MSD
namespace GROUPED_MATMUL_SWIGLU_QUANT {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA8W4PostProcess {
public:
    __aicore__ inline GMMA8W4PostProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN,
                                const GMMSwiglu *__restrict gmmSwigluIN);

    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe *pipe);

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
    const GMMSwiglu *__restrict gmmSwiglu;
    const GMMSwigluBaseParams *__restrict gmmBaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> weightAuxiliaryMatrixGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    LocalTensor<float> mmLocal_fp32;
    LocalTensor<half> mmLocal_fp16;
    TQue<QuePosition::VECIN, 1> weightAuxiliaryMatrixInQueue;
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
};

__aicore__ inline void GMMA8W4PostProcess::Init(const GMAddrParams gmAddrParams,
                                                const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN,
                                                const GMMSwiglu *__restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        aicCoreNum = GetBlockNum();
        aivCoreNum = aicCoreNum * 2;
        blockIdx = GetBlockIdx();
        gmmBaseParams = gmmSwigluBaseParamsIN;
        gmmSwiglu = gmmSwigluIN;
        weightAuxiliaryMatrixGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.weightAuxiliaryMatrixGM); // E, N
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM, gmmSwiglu->groupListLen);
        mmOutGM1.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset2));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset3));
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.xScaleGM, gmmBaseParams->M);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.yGM,
                                      gmmBaseParams->M * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.yScaleGM, gmmBaseParams->M);
    }
}

__aicore__ inline void GMMA8W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM,
                                                            VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    mmLocal_fp32 = mmLocal_fp16.ReinterpretCast<float>();
    const int64_t processNum = 2 * vecConfig.innerLoopNum * gmmSwiglu->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(mmLocal_fp16[processNum], mmOutGM[vecConfig.curOffset * DOUBLE_ROW], copyParams_0, padParams_0);

    mmOutQueue.EnQue(mmLocal_fp16);
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(mmLocal_fp32, mmLocal_fp16[processNum], RoundMode::CAST_NONE, processNum);
    PipeBarrier<PIPE_V>();
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    // 2. high_4bit * 16 + low_4bit
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        Muls(mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
             DEFAULT_MUL_SCALE, gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        Add(mmLocal_fp32[i * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
            mmLocal_fp32[(DOUBLE_ROW * i + 1) * gmmSwiglu->tokenLen], gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
    }
    vecConfig.curOffset = vecConfig.curIdx * gmmSwiglu->tokenLen;
    PipeBarrier<PIPE_V>();
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
    mmLocal_fp32 = mmOutQueue.DeQue<float>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    Add(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], weightAuxiliaryMatrixLocal,
        gmmSwiglu->tokenLen);
    vecConfig.nextUpadteInterVal--;
    PipeBarrier<PIPE_V>();
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], scale, gmmSwiglu->tokenLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA8W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 高阶API swiglu
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen];

    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_V>();
    DataCopyParams repeatParams{1, static_cast<uint16_t>((gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0,
                                0};
    DataCopy(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], workspaceLocal, repeatParams);

    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA8W4PostProcess::Quant(uint32_t loopIdx, VecConfig &vecConfig)
{
    uint64_t preOffset = loopIdx * gmmSwiglu->tokenLen;
    uint64_t halfTokenLen = gmmSwiglu->tokenLen / BISECT;
    Abs(mmLocal_fp32[preOffset + gmmSwiglu->tokenLen / BISECT], mmLocal_fp32[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, mmLocal_fp32[preOffset + gmmSwiglu->tokenLen / BISECT], reduceTmpLocal,
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
    Muls(mmLocal_fp32[preOffset], mmLocal_fp32[preOffset], quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(preOffset);
    int32_t tempCount = static_cast<int32_t>(halfTokenLen);
    LocalTensor<int8_t> castSpace = reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
    CastFp32ToInt8Template(quantLocal, mmLocal_fp32, castSpace, dstTempOffset, srcTempOffset, tempCount);
    mmOutQueue.EnQue(mmLocal_fp32);
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
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t prevM = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex;
         groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++) {
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

    // 第四步 申请空间
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n / 2 *sizeof(float) + 64 < 191 * 1024
    pipe->InitBuffer(mmOutQueue, 1, 2 * gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1,
                     gmmSwiglu->maxProcessRowNum * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<int32_t>(gmmSwiglu->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    pipe->InitBuffer(weightAuxiliaryMatrixInQueue, 1, gmmSwiglu->tokenLen * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) + UB_BLOCK_UNIT_SIZE +
                                          UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA8W4PostProcess::PreLoadAuxiliaryMatrix(VecConfig &vecConfig)
{
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    DataCopyExtParams copyAuxiliaryMatrixParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(weightAuxiliaryMatrixLocal, weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                copyAuxiliaryMatrixParams, padParams);
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 更新weightAuxiliaryMatrix
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
        LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSwiglu->tokenLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(weightAuxiliaryMatrixLocal, weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSwiglu->tokenLen],
                    copyParams, padParams);
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
            LocalTensor<half> mmLocal_fp32 = mmOutQueue.AllocTensor<half>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();

            mmOutQueue.EnQue(mmLocal_fp32);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
            PreLoadAuxiliaryMatrix(vecConfig);
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum =
                    outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum : gmmSwiglu->maxProcessRowNum;
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
            mmLocal_fp32 = mmOutQueue.DeQue<half>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();

            weightAuxiliaryMatrixInQueue.FreeTensor(weightAuxiliaryMatrixLocal);
            mmOutQueue.FreeTensor(mmLocal_fp32);
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
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0, 0};
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) * gmmSwiglu->tokenLen /
                              SWIGLU_REDUCE_FACTOR],
                quantLocal, copyParams_1);

    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwiglu->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

} // namespace GROUPED_MATMUL_SWIGLU_QUANT
#endif // GMM_SWIGLU_QUANT_A8W4_MSD
#endif // ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_AFTER_H