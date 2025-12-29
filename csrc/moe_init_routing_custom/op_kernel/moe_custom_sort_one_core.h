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
 * \file moe_custom_sort_one_core.h
 * \brief
 */
#ifndef MOE_CUSTOM_SORT_ONE_CORE_H
#define MOE_CUSTOM_SORT_ONE_CORE_H

#include "moe_custom_sort_base.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

class MoeSortOneCore : public MoeSortBase {
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdx = inLocal[0];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength);
    Muls(expertIdxFp32, expertIdxFp32, (float)-1, this->tileLength);

    if (ep_) {
        LocalTensor<uint8_t> maskLocalTensor = sortedBuffer.Get<uint8_t>();
        AscendC::CompareScalar(maskLocalTensor, expertIdxFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::GT,
                               (this->totalLength + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM *
                                   ONE_REPEAT_COMPARE_NUM);
        LocalTensor<float> floatMinLocalTensor = tempBuffer.Get<float>();
        Duplicate(floatMinLocalTensor, MIN_FP32, this->tileLength);
        Select(expertIdxFp32, maskLocalTensor, floatMinLocalTensor, expertIdxFp32, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               this->totalLength);
    }

    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal;
    expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);

    LocalTensor<int32_t> expertForSourceRowLocalInt32;
    expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(sortedexpertIdxGm, outLocal[0], intriParams);
    DataCopyPad(expendedRowIdxGm, outLocal[this->sortNum], intriParams);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                            const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->ep_ = tilingData->ep;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;

    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
                                      Align(this->totalLength, sizeof(int32_t)));
    if (rowIdxType_ == SCATTER) {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expendedRowIdx, this->tileLength);
    } else {
        expendedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->tileLength, sizeof(int32_t)),
                                         Align(this->tileLength, sizeof(int32_t)));
    }

    if (GetBlockIdx() == 0) {
        expertCountTempGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                              Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2,
                                          tilingData->actualExpertNum);
        InitGlobalMemory(expertCountTempGm, tilingData->actualExpertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    int64_t coreNum = GetBlockNum();

    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_SORT_ONE_CORE_H