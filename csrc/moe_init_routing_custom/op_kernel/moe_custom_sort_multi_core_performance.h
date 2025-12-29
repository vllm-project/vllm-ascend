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
 * \file moe_custom_sort_multi_core_performance.h
 * \brief
 */
#ifndef MOE_CUSTOM_VBS_ONE_CORE_PERFORMANCE_H
#define MOE_CUSTOM_VBS_ONE_CORE_PERFORMANCE_H

#include "moe_custom_sort_base.h"
#include "moe_custom_mrgsort_performance.h"
#include "moe_custom_mrgsort_out_performance.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

class MoeSortMultiCorePerformance : public MoeSortBase {
public:
    __aicore__ inline MoeSortMultiCorePerformance(){};
    __aicore__ inline void Init(GM_ADDR expendedRowIdx, GM_ADDR workspace, const MoeInitRoutingCustomTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void InitMoeMrgSort(MoeMrgsortPerformance *sorter, int64_t coreOffset);
    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOutPerformance *sorter);

private:
    GlobalTensor<float> workspaceGms[2];
    GlobalTensor<int32_t> workspaceGatheredSortNumGm_;

    const MoeCustomSortOutComputeTilingData *sortOutTilingData;
    const MoeCustomVBSComputeTilingData *vbsTilingData;

    // for MoeMrgsortPerformance
    MoeMrgsortPerformance mrgsorter;
    MoeMrgsortPerformanceParam mrgsortParam;

    int64_t blockIdx;

    int64_t perListElements;
    int64_t maxPerListElements;
};

__aicore__ inline void MoeSortMultiCorePerformance::InitMoeMrgSort(MoeMrgsortPerformance *sorter, int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[0][this->blockIdx * coreOffset]; // 0-3
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    GlobalTensor<int32_t> sortNumGm = workspaceGatheredSortNumGm_[this->blockIdx * MAX_MRGSORT_LIST];
    for (int64_t i = 0; i < MAX_MRGSORT_LIST; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(maxPerListElements) * i];
        sorter->SetInput(srcWsGm, inLocalT, sortNumGm);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1][this->blockIdx * coreOffset];
    sorter->SetOutput(dstWsGm, outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCorePerformance::InitMoeMrgSortOut(MoeMrgsortOutPerformance *sorter)
{
    GlobalTensor<float> srcWsGm = workspaceGms[1];
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    GlobalTensor<int32_t> sortNumGm = workspaceGatheredSortNumGm_;
    for (int64_t i = 0; i < MAX_MRGSORT_LIST; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(maxPerListElements) * i];
        sorter->SetInput(srcWsGm, inLocalT, sortNumGm);
    }

    LocalTensor<float> outLocalV = outLocal[maxPerListElements * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdxGm, this->expendedRowIdxGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer = sortedBuffer.Get<float>(GetSortLen<float>(maxPerListElements) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortMultiCorePerformance::VMSProcess()
{
    int64_t currentStageNeedCoreNum = MAX_MRGSORT_LIST;
    int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
    if (this->blockIdx <= currentStageNeedCoreNum - 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.oneLoopMaxElements = maxPerListElements;
        InitMoeMrgSort(&mrgsorter, coreOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();
    }
    SyncAll();
}

__aicore__ inline void MoeSortMultiCorePerformance::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.oneLoopMaxElements = maxPerListElements;
        MoeMrgsortOutPerformance sorter;
        InitMoeMrgSortOut(&sorter);
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
        InitGlobalMemory(expertCountTempGm, expertEnd_ - expertStart_, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SyncAll();
}

__aicore__ inline void MoeSortMultiCorePerformance::Init(GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                                         const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    this->totalLength = tilingData->n * tilingData->k;
    this->blockIdx = GetBlockIdx();
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);
    this->perListElements = Ceil(this->totalLength, MAX_MRGSORT_LIST_TOTAL);
    this->maxPerListElements = this->sortOutTilingData->oneLoopMaxElements;

    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    this->pipe = tPipe;
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                      Align(this->totalLength, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expendedRowIdx, Align(this->totalLength, sizeof(int32_t)));
    } else {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)),
                                         Align(this->totalLength, sizeof(int32_t)));
    }

    // key and value
    int64_t kvFactor = 2;
    workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace, Align(this->totalLength, sizeof(float)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace + Align(this->totalLength, sizeof(float)) * kvFactor,
                                    Align(this->totalLength, sizeof(float)) * kvFactor);
    workspaceGatheredSortNumGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                                    Align(this->totalLength, sizeof(int32_t)) * kvFactor * kvFactor,
                                                MAX_MRGSORT_LIST_TOTAL);
    expertCountTempGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2,
                                      expertEnd_ - expertStart_);

    int64_t bufferSize = Ceil(maxPerListElements * MAX_MRGSORT_LIST, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM *
                         sizeof(float) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    pipe->InitBuffer(tempBuffer, bufferSize);
}

__aicore__ inline void MoeSortMultiCorePerformance::Process()
{
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_VBS_ONE_CORE_PERFORMANCE_H