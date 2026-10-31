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
 * \file grouped_matmul_swiglu_quant_v2_a4w4_post.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H

#include "grouped_matmul_swiglu_quant_v2_utils.h"
#include "kernel_operator.h"

#ifdef GMM_SWIGLU_QUANT_V2_A4W4

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA4W4PostProcess {
public:
    __aicore__ inline GMMA4W4PostProcess(){};
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

    __aicore__ inline void VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                         WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyOut(VecConfig &vecConfig, WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void Quant(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void Swiglu(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<float> smoothScaleGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    LocalTensor<float> mmLocal_fp32;
    LocalTensor<half> mmLocal_fp16;
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
    float limited = FLOAT_INF;
};

__aicore__ inline void GMMA4W4PostProcess::Init(const GMAddrParams gmAddrParams,
                         const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
                         const GMMSwigluQuantV2 *__restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        aicCoreNum = GetBlockNum();
        aivCoreNum = aicCoreNum * NUM_2;
        blockIdx = GetBlockIdx();
        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;
        gmmSwigluQuantV2 = gmmSwigluIN;
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM, gmmSwigluQuantV2->groupListLen);
        mmOutGM1.SetGlobalBuffer((__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.xScaleGM, gmmSwigluQuantV2BaseParams->M);
        smoothScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.smoothScaleGM);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.yGM, gmmSwigluQuantV2BaseParams->M *
                                                                             gmmSwigluQuantV2->tokenLen /
                                                                             SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.yScaleGM, gmmSwigluQuantV2BaseParams->M);
        limited = gmmSwigluQuantV2BaseParams->swigluLimit;
    }
}

__aicore__ inline void GMMA4W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM,
                                                            VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    mmLocal_fp32 = mmLocal_fp16.ReinterpretCast<float>();
    const int64_t processNum = vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(mmLocal_fp16[processNum], mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);

    mmOutQueue.EnQue(mmLocal_fp16);
    mmLocal_fp16 = mmOutQueue.DeQue<half>();

    // 1. fp16 -> fp32
    Cast(mmLocal_fp32, mmLocal_fp16[processNum], RoundMode::CAST_NONE, processNum);
    PipeBarrier<PIPE_V>();

    vecConfig.curIdx += vecConfig.innerLoopNum;
    vecConfig.curOffset = vecConfig.curIdx * gmmSwigluQuantV2->tokenLen;
}

__aicore__ inline void GMMA4W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                                         WorkSpaceSplitConfig &workspaceSplitConfig)
{
    // 1.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 2.Swiglu
    Swiglu(loopIdx, vecConfig);
    // 3.ApplySmoothScale（smoothScaleDimNum为0时跳过，表示smoothScale为空指针）
    if (gmmSwigluQuantV2BaseParams->smoothScaleDimNum != 0) {
        ApplySmoothScale(loopIdx, vecConfig, workspaceSplitConfig);
    }
    // 4.Quant
    Quant(loopIdx, vecConfig);
}

__aicore__ inline void GMMA4W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    if (loopIdx != 0) {
        mmLocal_fp32 = mmOutQueue.DeQue<float>();
    }
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    PipeBarrier<PIPE_V>();
    Muls(mmLocal_fp32[loopIdx * gmmSwigluQuantV2->tokenLen], mmLocal_fp32[loopIdx * gmmSwigluQuantV2->tokenLen], scale,
         gmmSwigluQuantV2->tokenLen);
}

__aicore__ inline void GMMA4W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 高阶API swiglu
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        mmLocal_fp32[loopIdx * gmmSwigluQuantV2->tokenLen + gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = mmLocal_fp32[loopIdx * gmmSwigluQuantV2->tokenLen];
    if (limited > 0.0f) {
        Mins(src0Local, src0Local, limited, gmmSwigluQuantV2->tokenLen / 2);
        PipeBarrier<PIPE_V>();
        Maxs(src0Local, src0Local, (-1.0f * limited), gmmSwigluQuantV2->tokenLen / 2);
        PipeBarrier<PIPE_V>();
        Mins(src1Local, src1Local, limited, gmmSwigluQuantV2->tokenLen / 2);
        PipeBarrier<PIPE_V>();
    }
    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_V>();
    DataCopyParams repeatParams{
        1, static_cast<uint16_t>((gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0, 0};
    DataCopy(mmLocal_fp32[loopIdx * gmmSwigluQuantV2->tokenLen], workspaceLocal, repeatParams);
}

__aicore__ inline void GMMA4W4PostProcess::ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                                WorkSpaceSplitConfig &workspaceSplitConfig)
{
    int64_t smoothScaleDimNum = gmmSwigluQuantV2BaseParams->smoothScaleDimNum;
    int64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR;
    int64_t currentTokenIdx = workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx + loopIdx;

    // 找到当前token所属的group
    uint32_t groupIdx = 0;
    int64_t prevM = 0;
    int64_t totalTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    for (uint32_t i = workspaceSplitConfig.rightMatrixExpertStartIndex;
         i <= workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
        int64_t currM = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            currM = groupListGM.GetValue(i);
        } else {
            totalTmp += groupListGM.GetValue(i);
            currM = totalTmp;
        }
        if (currentTokenIdx < currM) {
            groupIdx = i;
            break;
        }
        prevM = currM;
    }

    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;

    if (smoothScaleDimNum == NUM_2) {
        // smoothScale形状为 (E, N/2)，只需要当前group的那一行
        for (uint32_t j = 0; j < halfTokenLen; j++) {
            float scale = smoothScaleGM.GetValue(groupIdx * halfTokenLen + j);
            float val = mmLocal_fp32.GetValue(preOffset + j);
            mmLocal_fp32.SetValue(preOffset + j, val * scale);
        }
    } else if (smoothScaleDimNum == 1) {
        // smoothScale形状为 (E,)，需要广播到 (N/2)
        float scale = smoothScaleGM.GetValue(groupIdx);
        PipeBarrier<PIPE_V>();
        Muls(mmLocal_fp32[preOffset], mmLocal_fp32[preOffset], scale, halfTokenLen);
    }
}

__aicore__ inline void GMMA4W4PostProcess::Quant(uint32_t loopIdx, VecConfig &vecConfig)
{
    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;
    uint64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / BISECT;
    PipeBarrier<PIPE_V>();
    Abs(mmLocal_fp32[preOffset + gmmSwigluQuantV2->tokenLen / BISECT], mmLocal_fp32[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, mmLocal_fp32[preOffset + gmmSwigluQuantV2->tokenLen / BISECT],
                      reduceTmpLocal, static_cast<uint32_t>(halfTokenLen));
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    quantScale = QUANT_SCALE_INT8 / reduceResLocal.GetValue(0);
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
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

__aicore__ inline void GMMA4W4PostProcess::UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
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
                     gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1,
                     gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR *
                         sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1,
                     AlignUp<int32_t>(gmmSwigluQuantV2->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) +
                                          UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA4W4PostProcess::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                   int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        VecConfig vecConfig;
        UpdateVecConfig(blockIdx, vecConfig, workspaceSplitConfig, workspaceSplitLoopIdx, pipe);

        if (blockIdx < vecConfig.usedCoreNum) {
            mmOutGM = (workspaceSplitLoopIdx % NUM_2 == 0 ? mmOutGM1 : mmOutGM2);
            LocalTensor<half> mmLocal = mmOutQueue.AllocTensor<half>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();

            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum :
                                                                                    gmmSwigluQuantV2->maxProcessRowNum;
                int32_t eventIdMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                // 1.matmul中间结果搬入
                customDataCopyIn(outLoopIdx, mmOutGM, vecConfig, workspaceSplitConfig);

                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    // 2. 四步vector计算（perToken反量化、Swiglu、SmoothScale、Quant）
                    VectorCompute(innerLoopIdx, vecConfig, workspaceSplitConfig);
                }
                int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                customDataCopyOut(vecConfig, workspaceSplitConfig);
            }
            mmLocal = mmOutQueue.DeQue<half>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();

            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        }
    }
}

__aicore__ inline void GMMA4W4PostProcess::customDataCopyOut(VecConfig &vecConfig,
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
#endif // GMM_SWIGLU_QUANT_V2_A4W4
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H
