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
 * \file grouped_matmul_situ_quant_a8w4_msd_post.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_POST_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_POST_H

#include "grouped_matmul_situ_quant_utils_kernel.h"
#include "grouped_matmul_situ_quant_activation.h"
#include "kernel_operator.h"

#ifdef GMM_SITU_QUANT_A8W4_MSD

namespace GroupedMatmulDequantSituQuant {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA8W4PostProcess {
public:
    __aicore__ inline GMMA8W4PostProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN,
                                const GMMSituQuant* __restrict gmmSwigluIN);

    __aicore__ inline void Process(WorkSpaceSplitConfig& workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe* pipe);

private:
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig,
                                           WorkSpaceSplitConfig& workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                           TPipe* pipe);

    __aicore__ inline void UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig);

    // 自算通路用：把第 curGroupIdx 个专家块的 cAux（C 区末尾那一行）load 进辅助矩阵缓冲。
    __aicore__ inline void LoadAssistRowFromWorkspace(VecConfig& vecConfig);

    // 自算通路用：一个专家块的 token 处理完，跳过它的补偿行并切到下一个非空专家块。
    __aicore__ inline void AdvanceGroupForAssist(VecConfig& vecConfig);

    // 自算通路的后处理主循环：块不跨专家，块内行号连续。
    __aicore__ inline void ProcessWithAssistRows(GlobalTensor<half>& mmOutGM, VecConfig& vecConfig,
                                                 WorkSpaceSplitConfig& workspaceSplitConfig);

    __aicore__ inline void VectorCompute(uint32_t loopIdx, VecConfig& vecConfig,
                                         WorkSpaceSplitConfig& workspaceSplitConfig);

    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half>& mmOutGM, VecConfig& vecConfig,
                                            WorkSpaceSplitConfig& workspaceSplitConfig);

    __aicore__ inline void customDataCopyOut(VecConfig& vecConfig, WorkSpaceSplitConfig& workspaceSplitConfig);

    __aicore__ inline void PreLoadAuxiliaryMatrix(VecConfig& vecConfig);

    __aicore__ inline void Quant(uint32_t loopIdx, VecConfig& vecConfig);

    __aicore__ inline void Situ(uint32_t loopIdx, VecConfig& vecConfig);

    __aicore__ inline void MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig);

    __aicore__ inline void MulPertokenScale(uint32_t loopIdx, VecConfig& vecConfig,
                                            WorkSpaceSplitConfig& workspaceSplitConfig);
    const GMMSituQuant* __restrict gmmSituQuant;
    const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> weightAuxiliaryMatrixGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    TQue<QuePosition::VECIN, 1> weightAuxiliaryMatrixInQueue;
    // 自算补偿矩阵时，cAux 直接从 C 区该专家块末尾那一行读（fp16），再 cast 进
    // weightAuxiliaryMatrixInQueue 的 float 缓冲 —— 两条通路的消费端因此完全一样。
    TQue<QuePosition::VECIN, 1> assistRowInQueue;
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
    GM_ADDR weightAuxiliaryMatrixTensorPtr;
    // 补偿矩阵自算开关（tiling 的 assistRowsPerExpert）。
    int64_t assistRows = 0;
};

__aicore__ inline void GMMA8W4PostProcess::Init(
    const GMAddrParams gmAddrParams, const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN,
    const GMMSituQuant* __restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        aicCoreNum = GetBlockNum();
        aivCoreNum = aicCoreNum * 2;
        blockIdx = GetBlockIdx();
        gmmSituQuantBaseParams = gmmSituQuantBaseParamsIN;
        gmmSituQuant = gmmSwigluIN;
        assistRows = gmmSituQuantBaseParams->assistRowsPerExpert;
        weightAuxiliaryMatrixGM.SetGlobalBuffer(GetTensorAddr<float>(0, gmAddrParams.weightAuxiliaryMatrixGM));
        groupListGM.SetGlobalBuffer((__gm__ int64_t*)gmAddrParams.groupListGM, gmmSituQuant->groupListLen);
        mmOutGM1.SetGlobalBuffer(
            (__gm__ half*)((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset2));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half*)((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset3));
        perTokenScaleGM.SetGlobalBuffer((__gm__ float*)gmAddrParams.xScaleGM, gmmSituQuantBaseParams->M);
        quantOutputGM.SetGlobalBuffer(
            (__gm__ int8_t*)gmAddrParams.yGM,
            gmmSituQuantBaseParams->M * gmmSituQuant->tokenLen / SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float*)gmAddrParams.yScaleGM, gmmSituQuantBaseParams->M);
        weightAuxiliaryMatrixTensorPtr = gmAddrParams.weightAuxiliaryMatrixGM;
    }
}

__aicore__ inline void GMMA8W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half>& mmOutGM,
                                                            VecConfig& vecConfig,
                                                            WorkSpaceSplitConfig& workspaceSplitConfig)
{
    LocalTensor<half> _inMMLocal_0 = mmOutQueue.DeQue<half>();
    const int64_t processNum = 2 * vecConfig.innerLoopNum * gmmSituQuant->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(_inMMLocal_0[processNum], mmOutGM[vecConfig.curRowOffset * gmmSituQuant->tokenLen], copyParams_0,
                padParams_0);
    GmmsqSetWaitFlag<HardEvent::MTE2_V>();

    mmOutQueue.EnQue(_inMMLocal_0);

    LocalTensor<half> _inMMLocal_1 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1[processNum], RoundMode::CAST_NONE, processNum);
    PipeBarrier<PIPE_V>();

    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();
    // 2. high_4bit * 16 + low_4bit
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        Muls(_inMMLocal_2[(DOUBLE_ROW * i) * gmmSituQuant->tokenLen],
             _inMMLocal_2[(DOUBLE_ROW * i) * gmmSituQuant->tokenLen], DEFAULT_MUL_SCALE,
             gmmSituQuant->tokenLen);
        PipeBarrier<PIPE_V>();
        Add(_inMMLocal_2[i * gmmSituQuant->tokenLen], _inMMLocal_2[(DOUBLE_ROW * i) * gmmSituQuant->tokenLen],
            _inMMLocal_2[(DOUBLE_ROW * i + 1) * gmmSituQuant->tokenLen], gmmSituQuant->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
    }
    vecConfig.curOffset = vecConfig.curIdx * gmmSituQuant->tokenLen;
    // 行号跟着块走：自算通路里块不跨专家，块内行连续，块尾那一行 cAux 由调用方跳过。
    vecConfig.curRowOffset += DOUBLE_ROW * vecConfig.innerLoopNum;
    mmOutQueue.EnQue(_inMMLocal_2);
}

__aicore__ inline void GMMA8W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig& vecConfig,
                                                         WorkSpaceSplitConfig& workspaceSplitConfig)
{
    // 1.辅助矩阵加回
    MergeAuxiliaryMatrix(loopIdx, vecConfig);
    // 2.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 3.SiTU-GLU
    Situ(loopIdx, vecConfig);
    // 4.Quant
    Quant(loopIdx, vecConfig);
}

__aicore__ inline void GMMA8W4PostProcess::MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // perChanelScale * perTokenScale
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    if (assistRows == 0) {
        // 外部补偿矩阵：值就是 8*sum_k(w)*scale，加回去。
        Add(mmLocal[loopIdx * gmmSituQuant->tokenLen], mmLocal[loopIdx * gmmSituQuant->tokenLen],
            weightAuxiliaryMatrixLocal, gmmSituQuant->tokenLen);
    } else {
        // 自算补偿行：cube 算出来的是 -8*sum_k(w)*scale（A 里那一行是 -8），减掉即可。
        Sub(mmLocal[loopIdx * gmmSituQuant->tokenLen], mmLocal[loopIdx * gmmSituQuant->tokenLen],
            weightAuxiliaryMatrixLocal, gmmSituQuant->tokenLen);
    }
    PipeBarrier<PIPE_V>();
    vecConfig.nextUpdateInterVal--;
    mmOutQueue.EnQue(mmLocal);
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig& vecConfig,
                                                            WorkSpaceSplitConfig& workspaceSplitConfig)
{
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(mmLocal[loopIdx * gmmSituQuant->tokenLen], mmLocal[loopIdx * gmmSituQuant->tokenLen], scale,
         gmmSituQuant->tokenLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA8W4PostProcess::Situ(uint32_t loopIdx, VecConfig& vecConfig)
{
    // SiTU-GLU 原地计算：结果直接写回行块前半区，省掉 SwiGLU 版本里
    // workspace -> _inMMLocal 的一次 DataCopy。
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    const uint32_t halfTokenLen =
        static_cast<uint32_t>(gmmSituQuant->tokenLen) / SWIGLU_REDUCE_FACTOR;

    // reduceWorkspace 布局：[0, halfTokenLen) 为 SiTU 暂存，其后为 Tanh 的 tmp。
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<uint8_t> tanhTmpLocal = reduceWorkspace.GetWithOffset<uint8_t>(
        GroupedMatmulSituQuantActivation::SituTanhTmpElems(halfTokenLen) * sizeof(float),
        halfTokenLen * sizeof(float));

    GroupedMatmulSituQuantActivation::SituParams situParams;
    situParams.beta = gmmSituQuant->beta;
    situParams.linearBeta = gmmSituQuant->linearBeta;
    situParams.activateLeft = gmmSituQuant->activateLeft;

    GroupedMatmulSituQuantActivation::SituGluCompute(
        _inMMLocal[loopIdx * gmmSituQuant->tokenLen], workLocal, tanhTmpLocal, halfTokenLen, situParams);

    mmOutQueue.EnQue(_inMMLocal);
}

__aicore__ inline void GMMA8W4PostProcess::Quant(uint32_t loopIdx, VecConfig& vecConfig)
{
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    uint64_t preOffset = loopIdx * gmmSituQuant->tokenLen;
    uint64_t halfTokenLen = gmmSituQuant->tokenLen / BISECT;
    Abs(_inMMLocal[preOffset + gmmSituQuant->tokenLen / BISECT], _inMMLocal[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, _inMMLocal[preOffset + gmmSituQuant->tokenLen / BISECT],
                      reduceTmpLocal, static_cast<uint32_t>(halfTokenLen));
    GmmsqSetWaitFlag<HardEvent::V_S>();
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    // An all-zero row has scale zero and must quantize to zero, not 0 * inf.
    quantScale = quantScale > 0.0f ? 1.0f / quantScale : 0.0f;
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(_inMMLocal[preOffset], _inMMLocal[preOffset], quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(preOffset);
    int32_t tempCount = static_cast<int32_t>(halfTokenLen);
    LocalTensor<int8_t> castSpace = reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
    CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
    PipeBarrier<PIPE_V>();
    mmOutQueue.EnQue(_inMMLocal);
    quantOutQueue.EnQue(quantLocal);
}

__aicore__ inline void GMMA8W4PostProcess::UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig,
                                                           WorkSpaceSplitConfig& workspaceSplitConfig,
                                                           int64_t workspaceSplitLoopIdx, TPipe* pipe)
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
    vecConfig.startOffset = vecConfig.startIdx * gmmSituQuant->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t prevM = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
    int64_t totalTmp = 0;
    if (gmmSituQuantBaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    // 自算补偿矩阵时 C 区每个非空专家块末尾多一行，起始行号因此要加上本核第一行之前
    // 已经插进去的补偿行数。
    int64_t splitPos = 0;
    int64_t auxBefore = 0;
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex;
         groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++) {
        int64_t currM = 0;
        if (gmmSituQuantBaseParams->groupListType == 0) {
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
            // 自算通路拿它算「本组 cAux 在 C 区的行号」，必须按本 split 的剩余量截断：
            // 最后一组可能被 split 边界切开，只有落在本 split 里的那部分才有 cAux。
            if (assistRows != 0 && vecConfig.nextUpdateInterVal > vecConfig.M - vecConfig.startIdx) {
                vecConfig.nextUpdateInterVal = vecConfig.M - vecConfig.startIdx;
            }
        }
        if (tempM > 0) {
            if (splitPos + tempM <= vecConfig.startIdx) {
                auxBefore++;
            }
            splitPos += tempM;
        }
        curStartIdx -= tempM;
    }
    vecConfig.curRowOffset = DOUBLE_ROW * vecConfig.startIdx + (assistRows != 0 ? auxBefore : 0);
    // 第三步 计算总数据量
    vecConfig.outLoopNum =
        (vecConfig.taskNum + gmmSituQuant->maxProcessRowNum - 1) / gmmSituQuant->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSituQuant->maxProcessRowNum ?
                                vecConfig.taskNum % gmmSituQuant->maxProcessRowNum :
                                gmmSituQuant->maxProcessRowNum;

    // 第四步 申请空间
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n * sizeof(float) + 64 < 191 * 1024
    // (reduceWorkspace 由 n/2 扩到 n：SiTU 暂存 n/2 个 float + Tanh tmp n/2 个 float)
    pipe->InitBuffer(mmOutQueue, 1,
                     2 * gmmSituQuant->maxProcessRowNum * gmmSituQuant->tokenLen * sizeof(float));
    pipe->InitBuffer(
        quantOutQueue, 1,
        gmmSituQuant->maxProcessRowNum * gmmSituQuant->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1,
                     AlignUp<int32_t>(gmmSituQuant->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    pipe->InitBuffer(weightAuxiliaryMatrixInQueue, 1, gmmSituQuant->tokenLen * sizeof(float));
    if (assistRows != 0) {
        // 自算通路：cAux 在 C 区里是 fp16 的一行，先落到 UB 再 cast 成 float。
        pipe->InitBuffer(assistRowInQueue, 1, gmmSituQuant->tokenLen * SIZE_OF_HALF_2);
    }
    // SiTU 暂存 + Tanh tmp；尾部两个 32 byte 块留给 Quant 的 reduceMax。
    const uint32_t halfTokenLen = static_cast<uint32_t>(gmmSituQuant->tokenLen) / SWIGLU_REDUCE_FACTOR;
    pipe->InitBuffer(reduceWorkspace,
                     (halfTokenLen + GroupedMatmulSituQuantActivation::SituTanhTmpElems(halfTokenLen)) *
                             sizeof(float) +
                         UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA8W4PostProcess::PreLoadAuxiliaryMatrix(VecConfig& vecConfig)
{
    if (assistRows != 0) {
        LoadAssistRowFromWorkspace(vecConfig);
        return;
    }
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    DataCopyExtParams copyAuxiliaryMatrixParams{1, static_cast<uint32_t>(gmmSituQuant->tokenLen * sizeof(float)), 0,
                                                0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (gmmSituQuantBaseParams->isSingleTensor == 0) {
        weightAuxiliaryMatrixGM.SetGlobalBuffer(
            GetTensorAddr<float>(vecConfig.curGroupIdx, weightAuxiliaryMatrixTensorPtr));
        DataCopyPad(weightAuxiliaryMatrixLocal, weightAuxiliaryMatrixGM, copyAuxiliaryMatrixParams, padParams);
    } else {
        DataCopyPad(weightAuxiliaryMatrixLocal,
                    weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSituQuant->tokenLen],
                    copyAuxiliaryMatrixParams, padParams);
    }
    GmmsqSetWaitFlag<HardEvent::MTE2_V>();
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

// 自算补偿矩阵：cAux 就在 C 区该专家块有效行之后的那一行（cube 用它跑了一行全 -8 的 A）。
// 这里读成 fp16 再 cast 成 float，后续 MergeAuxiliaryMatrix 与外部补偿矩阵完全共用。
__aicore__ inline void GMMA8W4PostProcess::LoadAssistRowFromWorkspace(VecConfig& vecConfig)
{
    LocalTensor<half> auxHalfLocal = assistRowInQueue.DeQue<half>();
    const int64_t auxRow = vecConfig.curRowOffset + DOUBLE_ROW * vecConfig.nextUpdateInterVal;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSituQuant->tokenLen * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    DataCopyPad(auxHalfLocal, mmOutGM[auxRow * gmmSituQuant->tokenLen], copyParams, padParams);
    GmmsqSetWaitFlag<HardEvent::MTE2_V>();
    assistRowInQueue.EnQue(auxHalfLocal);

    LocalTensor<half> auxHalf = assistRowInQueue.DeQue<half>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    Cast(weightAuxiliaryMatrixLocal, auxHalf, RoundMode::CAST_NONE, gmmSituQuant->tokenLen);
    PipeBarrier<PIPE_V>();
    assistRowInQueue.EnQue(auxHalf);
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

// 一个专家块的 token 处理完：跳过它的补偿行，再切到下一个非空专家块并把它那行的 cAux 读进来。
__aicore__ inline void GMMA8W4PostProcess::AdvanceGroupForAssist(VecConfig& vecConfig)
{
    vecConfig.curRowOffset += 1; // 刚结束那一组末尾的 cAux 行，不参与后处理
    int64_t loop = gmmSituQuant->groupListLen - vecConfig.curGroupIdx - 1;
    while (loop--) {
        if (gmmSituQuantBaseParams->groupListType == 0) {
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
    // 同理按本 split 的剩余量截断（最后一组可能被 split 切开）。
    const int64_t splitRemain = vecConfig.M - vecConfig.startIdx;
    if (vecConfig.nextUpdateInterVal > splitRemain) {
        vecConfig.nextUpdateInterVal = splitRemain;
    }
    LoadAssistRowFromWorkspace(vecConfig);
}

// 自算通路的主循环：块绝不跨专家块，因此 C 区里这一段行号连续，一次 DataCopy 就能取完；
// 换组时下一次搬入的起点由 curRowOffset 跳过补偿行给出。
__aicore__ inline void GMMA8W4PostProcess::ProcessWithAssistRows(GlobalTensor<half>& mmOutGM,
                                                                 VecConfig& vecConfig,
                                                                 WorkSpaceSplitConfig& workspaceSplitConfig)
{
    int64_t remaining = vecConfig.taskNum;
    while (remaining > 0) {
        int64_t rowNum = remaining < gmmSituQuant->maxProcessRowNum ? remaining : gmmSituQuant->maxProcessRowNum;
        if (rowNum > vecConfig.nextUpdateInterVal) {
            rowNum = vecConfig.nextUpdateInterVal;
        }
        if (rowNum <= 0) {
            break; // 正常不可达：剩余 token 必属于某个非空专家块
        }
        vecConfig.innerLoopNum = rowNum;
        int32_t eventIdMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        customDataCopyIn(0, mmOutGM, vecConfig, workspaceSplitConfig);
        for (int64_t innerLoopIdx = 0; innerLoopIdx < rowNum; innerLoopIdx++) {
            VectorCompute(static_cast<uint32_t>(innerLoopIdx), vecConfig, workspaceSplitConfig);
        }
        int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        customDataCopyOut(vecConfig, workspaceSplitConfig);
        remaining -= rowNum;
        if (remaining > 0 && vecConfig.nextUpdateInterVal == 0) {
            AdvanceGroupForAssist(vecConfig);
        }
    }
}

__aicore__ inline void GMMA8W4PostProcess::UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // 更新weightAuxiliaryMatrix
    if (unlikely(vecConfig.nextUpdateInterVal == 0)) {
        int64_t loop = gmmSituQuant->groupListLen - vecConfig.curGroupIdx - 1;
        while (loop--) {
            if (gmmSituQuantBaseParams->groupListType == 0) {
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
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSituQuant->tokenLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        // 单张量形态：weightAuxiliaryMatrixGM 指向 [E, N] 首地址，按组号偏移即可。
        // 多张量形态：每个专家是一个 [N] 张量，必须先按组号取基址再拷贝 —— 此时
        // weightAuxiliaryMatrixGM 仍停在 PreLoadAuxiliaryMatrix 设的那个专家的行首，
        // 直接叠加 curGroupIdx * tokenLen 会读到该张量之外（越界）。
        if (gmmSituQuantBaseParams->isSingleTensor == 0) {
            weightAuxiliaryMatrixGM.SetGlobalBuffer(
                GetTensorAddr<float>(vecConfig.curGroupIdx, weightAuxiliaryMatrixTensorPtr));
            DataCopyPad(weightAuxiliaryMatrixLocal, weightAuxiliaryMatrixGM, copyParams, padParams);
        } else {
            DataCopyPad(weightAuxiliaryMatrixLocal,
                        weightAuxiliaryMatrixGM[vecConfig.curGroupIdx * gmmSituQuant->tokenLen], copyParams,
                        padParams);
        }
        GmmsqSetWaitFlag<HardEvent::MTE2_V>();
        weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
    }
}

__aicore__ inline void GMMA8W4PostProcess::Process(WorkSpaceSplitConfig& workspaceSplitConfig,
                                                   int64_t workspaceSplitLoopIdx, TPipe* pipe)
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
            if (assistRows != 0) {
                LocalTensor<half> auxHalfLocal = assistRowInQueue.AllocTensor<half>();
                assistRowInQueue.EnQue(auxHalfLocal);
            }
            PreLoadAuxiliaryMatrix(vecConfig);
            if (assistRows != 0) {
                ProcessWithAssistRows(mmOutGM, vecConfig, workspaceSplitConfig);
            } else {
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum :
                                                                                    gmmSituQuant->maxProcessRowNum;
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
            }
            weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
            mmLocal = mmOutQueue.DeQue<half>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();

            if (assistRows != 0) {
                LocalTensor<half> auxHalfLocal = assistRowInQueue.DeQue<half>();
                assistRowInQueue.FreeTensor(auxHalfLocal);
            }
            weightAuxiliaryMatrixInQueue.FreeTensor(weightAuxiliaryMatrixLocal);
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        }
    }
}

__aicore__ inline void GMMA8W4PostProcess::customDataCopyOut(VecConfig& vecConfig,
                                                             WorkSpaceSplitConfig& workspaceSplitConfig)
{
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParams_0{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    GmmsqSetWaitFlag<HardEvent::S_MTE3>();
    DataCopyPad(quantScaleOutputGM[workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx], quantScaleLocal,
                copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSituQuant->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0,
        0};
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) *
                              gmmSituQuant->tokenLen / SWIGLU_REDUCE_FACTOR],
                quantLocal, copyParams_1);
    GmmsqSetWaitFlag<HardEvent::MTE3_V>();

    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSituQuant->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

} // namespace GroupedMatmulDequantSituQuant
#endif // GMM_SITU_QUANT_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_POST_H
