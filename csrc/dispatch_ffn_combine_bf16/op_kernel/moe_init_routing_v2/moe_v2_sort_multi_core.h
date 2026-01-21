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
 * \file moe_v2_sort_multi_core.h
 * \brief
 */
#ifndef MOE_V2_VBS_ONE_CORE_H
#define MOE_V2_VBS_ONE_CORE_H

#include "moe_v2_sort_base.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_mrgsort_out.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;
using namespace optiling;

class MoeV2SortMultiCore : public MoeV2SortBase {
public:
    __aicore__ inline MoeV2SortMultiCore(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace, const TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    __aicore__ inline void ResetIO(GM_ADDR expandedRowIdx, GM_ADDR workspace);
#endif
private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitMoeMrgSort(MoeV2Mrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                          int64_t loopOffset);
    __aicore__ inline void InitMoeMrgSortOut(MoeV2MrgsortOut *sorter, int64_t listNum, int64_t coreOffset);
    __aicore__ inline void InitExpertTokensGlobalMemory();

private:
    GlobalTensor<float> workspaceGms[2];

    const MoeV2VBSComputeTilingData *vbsTilingData;
    const MoeV2VMSMiddleComputeTilingData *vmsTilingData;
    const MoeV2SortOutComputeTilingData *sortOutTilingData;

    // for MoeMrgsort
    MoeV2Mrgsort mrgsorter;
    MoeV2MrgsortParam mrgsortParam;

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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    int64_t perCoreOffset;
#endif
    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void MoeV2SortMultiCore::InitExpertTokensGlobalMemory()
{
    if (this->blockIdx < this->needInitExpertCore) {
        if (this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            InitGlobalMemory(expertTokensCountOrCumsumGm, Align(this->currentCoreExpert, sizeof(int32_t)), 0);
#else
            InitGlobalMemory(expertTokensCountOrCumsumGm, currentCoreExpert, 0);
#endif
        }
        if (this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            InitGlobalMemory(expertTokensBeforeCapacityGm, Align(this->currentCoreExpert, sizeof(int32_t)), 0);
#else
            InitGlobalMemory(expertTokensBeforeCapacityGm, currentCoreExpert, 0);
#endif
        }
    }
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
__aicore__ inline void MoeV2SortMultiCore::ResetIO(GM_ADDR expandedRowIdx, GM_ADDR workspace)
{
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandedRowIdx),
        Align(this->totalLength, sizeof(int32_t)));
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandedRowIdx),
        Align(this->totalLength, sizeof(int32_t)));
    expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) +
                                Align(this->totalLength, sizeof(int32_t)) + this->blockIdx * perCoreOffset,
                                this->sortTotalLength);
    this->srcWsIndex = 0;
    this->needInitExpertCore = 0;
}
#endif

__aicore__ inline void MoeV2SortMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyPadCustom(inLocal[0], expertIdxGm[inOffset], dataCopyParams, dataCopyPadParams);
#else
    DataCopyPad(inLocal[0], expertIdxGm[inOffset], dataCopyParams, dataCopyPadParams);
#endif
    LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];
    int64_t startValue = this->blockIdx * this->vbsTilingData->perCoreElements + inOffset;
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    ArithProgression<int32_t>(rowIdxLocal, startValue, 1, size);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2SortMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32;

    expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_NONE, sortNum);
    #else
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, sortNum);
    #endif
    PipeBarrier<PIPE_V>();
    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, sortNum);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> concatLocal;
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    LocalTensor<float> outLocal = tempBuffer.Get<float>(GetSortLen<float>(sortNum));

    Concat(concatLocal, expertForSourceRowLocalFp32, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

    PipeBarrier<PIPE_V>();

    Sort<float, true>(outLocal, concatLocal, sourceRowLocal, concatLocal, sortNum / ONE_REPEAT_SORT_NUM);

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
#else
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;

    Sort<float, true>(outLocal, concatLocal, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
#endif
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2SortMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> outLocal = tempBuffer.Get<float>(GetSortLen<float>(sortNum));
#else
    LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
#endif
    DataCopy(workspaceGms[0][this->blockIdx * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                             GetSortLen<float>(progress * sortCoreLoopElements)],
             outLocal, Align(GetSortLen<float>(size), sizeof(float)));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
#else
    sortDataCopyOutQueue.FreeTensor(outLocal);
#endif       
}

__aicore__ inline void MoeV2SortMultiCore::InitMoeMrgSort(MoeV2Mrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                                          int64_t loopOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset + loopOffset];
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> inLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    LocalTensor<float> outLocal = tempBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
#else
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
#endif
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOffset + loopOffset];
    sorter->SetOutput(dstWsGm, outLocal);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 200
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
#endif
}

__aicore__ inline void MoeV2SortMultiCore::InitMoeMrgSortOut(MoeV2MrgsortOut *sorter, int64_t listNum,
                                                             int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> inLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
#else
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
#endif
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdxGm, this->expandDstToSrcRowGm, outLocal, outLocalV);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> tempLocal = tempBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    // buffer for Extract
    sorter->SetBuffer(tempLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
#else
    LocalTensor<float> tempLocal =
        sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    // buffer for Extract
    sorter->SetBuffer(tempLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
#endif
}

__aicore__ inline void MoeV2SortMultiCore::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                             int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
    mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

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

__aicore__ inline void MoeV2SortMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void MoeV2SortMultiCore::VBSProcess()
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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
    AscendC::SyncAll(syncTmpSpaceGm_, syncLocal, GetBlockNum());
#else
    SyncAll();
#endif
}

__aicore__ inline void MoeV2SortMultiCore::VMSProcess()
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
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
        AscendC::SyncAll(syncTmpSpaceGm_, syncLocal, GetBlockNum());
#else
        SyncAll();
#endif
    }
}

__aicore__ inline void MoeV2SortMultiCore::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

        MoeV2MrgsortOut sorter;
        InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
    AscendC::SyncAll(syncTmpSpaceGm_, syncLocal, GetBlockNum());
#else
    SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2SortMultiCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                                GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                                const TilingData *tilingData, TPipe *tPipe)
{
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    this->perCoreOffset = this->vbsTilingData->perCoreElements;
#endif
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->tileLength = this->vbsTilingData->perCorePerLoopElements;
    this->sortTotalLength = this->vbsTilingData->perCoreElements;
    if (this->blockIdx == tilingData->vbsComputeParamsOp.needCoreNum - 1) {
        this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->lastCoreElements;
    }
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

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
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) +
                                            Align(this->totalLength, sizeof(int32_t)),
                                        Align(this->totalLength, sizeof(int32_t)));

    this->perCoreExpert = Align((this->expertNum + this->coreNum - 1) / this->coreNum, sizeof(int32_t));
    this->needInitExpertCore = (this->expertNum + this->perCoreExpert - 1) / this->perCoreExpert;
    this->currentCoreExpert = this->perCoreExpert;
    if (this->blockIdx == needInitExpertCore - 1) {
        this->currentCoreExpert = this->expertNum - (this->needInitExpertCore - 1) * this->perCoreExpert;
    }
    if (this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
        expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum +
                                                        this->blockIdx * this->perCoreExpert,
                                                    this->currentCoreExpert);
    }
    if (this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensBeforeCapacity +
                                                         this->blockIdx * this->perCoreExpert,
                                                     this->currentCoreExpert);
    }
    // key and value
    int64_t kvFactor = 2;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    int64_t workspaceLen = GetSortLen<float>(Align(this->totalLength, sizeof(int32_t)));
    workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2,
                                    workspaceLen);
    workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace +
                                        Align(this->totalLength, sizeof(int32_t)) * 2 + workspaceLen,
                                    workspaceLen);
                                    
#else
    workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2,
                                    Align(this->totalLength, sizeof(int32_t)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace +
                                        Align(this->totalLength, sizeof(int32_t)) * (kvFactor + 2),
                                    Align(this->totalLength, sizeof(int32_t)) * kvFactor);
#endif

    int64_t bufferSize = Ceil(Max(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
                              ONE_REPEAT_SORT_NUM) *
                         ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                        Align(this->totalLength, sizeof(int32_t)) * 2 + 2 * workspaceLen,
                                    INT32_ONE_BLOCK_NUM * GetBlockNum() * BLOCK_BYTES);
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(tempBuffer, bufferSize * REGIONP_ROPOSAL_KV_RATIO);
    pipe->InitBuffer(sortedBuffer, bufferSize* REGIONP_ROPOSAL_KV_RATIO);
    LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
    Duplicate<int32_t>(syncLocal, 0, SYNC_LEN);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopy(syncTmpSpaceGm_, syncLocal, SYNC_LEN);
#else
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
#endif
}

__aicore__ inline void MoeV2SortMultiCore::Process()
{
    InitExpertTokensGlobalMemory();
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeInitRoutingV2
#endif // MOE_V2_VBS_ONE_CORE_H