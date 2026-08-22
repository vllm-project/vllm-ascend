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
 * \file moe_custom_sort_multi_core.h
 * \brief
 */
#ifndef MOE_CUSTOM_VBS_ONE_CORE_H
#define MOE_CUSTOM_VBS_ONE_CORE_H

#include "moe_custom_sort_base.h"
#include "moe_custom_mrgsort.h"
#include "moe_custom_mrgsort_out.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

class MoeSortMultiCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortMultiCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset);
    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);

private:
    GlobalTensor<float> workspaceGms[2];
    // GlobalTensor<int64_t> expertTokensCountGm_;

    const MoeCustomVBSComputeTilingData *vbsTilingData;
    const MoeCustomVMSMiddleComputeTilingData *vmsTilingData;
    const MoeCustomSortOutComputeTilingData *sortOutTilingData;

    // for MoeMrgsort
    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t srcWsIndex = 0;

    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void MoeSortMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm[inOffset], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];
    int64_t startValue = this->blockIdx * this->vbsTilingData->perCoreElements + inOffset;
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    ArithProgression<int32_t>(rowIdxLocal, startValue, 1, size);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeSortMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32;

    expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, sortNum);

    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, sortNum);

    if (ep_) {
        LocalTensor<uint8_t> maskLocalTensor = sortedBuffer.Get<uint8_t>();
        AscendC::CompareScalar(
            maskLocalTensor, expertForSourceRowLocalFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::GT,
            (sortNum + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
        LocalTensor<float> floatMinLocalTensor = tempBuffer.Get<float>();
        Duplicate(floatMinLocalTensor, MIN_FP32, sortNum);
        Select(expertForSourceRowLocalFp32, maskLocalTensor, floatMinLocalTensor, expertForSourceRowLocalFp32,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, sortNum);
    }

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(outLocal, concatLocal, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
    DataCopy(workspaceGms[0][this->blockIdx * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                             GetSortLen<float>(progress * sortCoreLoopElements)],
             outLocal, Align(GetSortLen<float>(size), sizeof(float)));
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                                        int64_t loopOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset + loopOffset];
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(oneLoopMaxElements_) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOffset + loopOffset];
    sorter->SetOutput(dstWsGm, outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(oneLoopMaxElements_) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[oneLoopMaxElements_ * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdxGm, this->expendedRowIdxGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer = sortedBuffer.Get<float>(GetSortLen<float>(oneLoopMaxElements_) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                           int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
    mrgsortParam.oneLoopMaxElements = oneLoopMaxElements_;

    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = perListElements;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        listNum = loops;
        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
        if (loops == 1) {
            break;
        }
    }
}

__aicore__ inline void MoeSortMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void MoeSortMultiCore::VBSProcess()
{
    if (this->blockIdx < this->vbsTilingData->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        if (sortCoreLoops > 1) {
            OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
        }
    }
    SyncAll();
}

__aicore__ inline void MoeSortMultiCore::VMSProcess()
{
    int64_t currentStageNeedCoreNum = this->vmsTilingData->needCoreNum;
    perListElements = this->vbsTilingData->perCoreElements;
    lastListElements = this->vbsTilingData->lastCoreElements;
    listNum = this->vbsTilingData->needCoreNum;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx < currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;
            mrgsortParam.oneLoopMaxElements = oneLoopMaxElements_;
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = oneLoopMaxElements_;
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;

        SyncAll();
    }
}

__aicore__ inline void MoeSortMultiCore::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = oneLoopMaxElements_;

        MoeMrgsortOut sorter;
        InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
    SyncAll();
}

__aicore__ inline void MoeSortMultiCore::Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                              const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);

    this->blockIdx = GetBlockIdx();
    this->tileLength = this->vbsTilingData->perCorePerLoopElements;
    this->sortTotalLength = this->vbsTilingData->perCoreElements;
    if (this->blockIdx == tilingData->vbsComputeParamsOp.needCoreNum - 1) {
        this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->lastCoreElements;
    }
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->ep_ = tilingData->ep;
    this->oneLoopMaxElements_ = ep_ ? this->sortOutTilingData->oneLoopMaxElements : MRGSORT_LIST_MAX_ELEMENT;

    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    // VBS param init
    if (this->blockIdx == this->vbsTilingData->needCoreNum - 1) {
        sortCoreLoops = this->vbsTilingData->lastCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->lastCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->lastCoreLastLoopElements;
    } else {
        sortCoreLoops = this->vbsTilingData->perCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;
    }

    this->pipe = tPipe;
    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx +
                                    this->blockIdx * tilingData->vbsComputeParamsOp.perCoreElements,
                                this->sortTotalLength);
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                      Align(this->totalLength, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expendedRowIdx, Align(this->totalLength, sizeof(int32_t)));
    } else {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)),
                                         Align(this->totalLength, sizeof(int32_t)));
    }

    if (GetBlockIdx() == 0) {
        expertCountTempGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                              Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2,
                                          tilingData->actualExpertNum);
        InitGlobalMemory(expertCountTempGm, tilingData->actualExpertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    // key and value
    int64_t kvFactor = 2;
    workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2 +
                                        tilingData->actualExpertNum,
                                    Align(this->totalLength, sizeof(int32_t)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace +
                                        Align(this->totalLength, sizeof(int32_t)) * (kvFactor + 2) +
                                        tilingData->actualExpertNum,
                                    Align(this->totalLength, sizeof(int32_t)) * kvFactor);

    int64_t bufferSize = Ceil(Max(oneLoopMaxElements_ * MAX_MRGSORT_LIST, sortCoreLoopElements), ONE_REPEAT_SORT_NUM) *
                         ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    if (ep_) {
        pipe->InitBuffer(tempBuffer, bufferSize);
    }
}

__aicore__ inline void MoeSortMultiCore::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_VBS_ONE_CORE_H