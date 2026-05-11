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
 * \file moe_v2_sort_one_core.h
 * \brief
 */
#ifndef MOE_V2_SORT_ONE_CORE_H
#define MOE_V2_SORT_ONE_CORE_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_base.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;
using namespace optiling;

class MoeV2SortOneCore : public MoeV2SortBase {
public:
    __aicore__ inline MoeV2SortOneCore(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace, const TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    __aicore__ inline void ResetIO(GM_ADDR expandedRowIdx, GM_ADDR workspace);
    bool needCopy = true;
#endif

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
    int64_t blockIdx;
    int64_t needCoreNum;
};

__aicore__ inline void MoeV2SortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyPadCustom(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
#else
    DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
#endif
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_NONE, this->tileLength);
#else
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
#endif
    PipeBarrier<PIPE_V>();
    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, this->tileLength);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertForSourceRowLocalFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal;
    expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> expertForSourceRowLocalInt32;
    expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyCustom<int32_t,true,false>(expandDstToSrcRowGm, outLocal[this->sortNum], intriParams.blockCount, intriParams.blockLen);
    if (this->needCopy) {
        DataCopyCustom<int32_t,true,false>(sortedexpertIdxGm, outLocal[0], intriParams.blockCount, intriParams.blockLen);
    }
#else
    DataCopyPad(sortedexpertIdxGm, outLocal[0], intriParams);
    DataCopyPad(expandDstToSrcRowGm, outLocal[this->sortNum], intriParams);
#endif
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
__aicore__ inline void MoeV2SortOneCore::ResetIO(GM_ADDR expandedRowIdx, GM_ADDR workspace)
{
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandedRowIdx), this->tileLength);
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandedRowIdx), this->tileLength);
    expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + this->tileLength, this->tileLength);
    this->expertTokensCountOrCumsumFlag = 0;
    this->expertTokensBeforeCapacityFlag = 0;
    this->needCopy = false;
}
#endif

template <typename TilingData>
__aicore__ inline void MoeV2SortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                              GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                              const TilingData *tilingData, TPipe *tPipe)
{
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->pipe = tPipe;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;
    this->needCoreNum = tilingData->vbsComputeParamsOp.needCoreNum;

    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);
    sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), this->tileLength);
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + this->tileLength,
                                        this->tileLength);

    if (this->blockIdx == this->coreNum - 1) {
        if (this->expertTokensCountOrCumsumFlag > 0) {
            expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum,
                                                        Align(this->expertNum, sizeof(int32_t)));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            InitGlobalMemory(expertTokensCountOrCumsumGm, Align(this->expertNum, sizeof(int32_t)), 0);
#else
            InitGlobalMemory(expertTokensCountOrCumsumGm, this->expertNum, 0);
#endif
        }
        if (this->expertTokensBeforeCapacityFlag == 1) {
            expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensBeforeCapacity,
                                                         Align(this->expertNum, sizeof(int32_t)));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            InitGlobalMemory(expertTokensBeforeCapacityGm, Align(this->expertNum, sizeof(int32_t)), 0);
#else
            InitGlobalMemory(expertTokensBeforeCapacityGm, this->expertNum, 0);
#endif
        }
    }
    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + 2 * this->tileLength, SYNC_LEN);
    buffSize = GetSortLen<float>(this->sortNum) * sizeof(int32_t);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
    LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
    Duplicate<int32_t>(syncLocal, 0, SYNC_LEN);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopy(syncTmpSpaceGm_, syncLocal, SYNC_LEN);
    PipeBarrier<PIPE_ALL>();
#else
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
#endif
}

__aicore__ inline void MoeV2SortOneCore::Process()
{
    if (GetBlockIdx() < this->needCoreNum) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<int32_t> syncLocal = tempBuffer.Get<int32_t>();
    AscendC::SyncAll(syncTmpSpaceGm_, syncLocal, GetBlockNum());
#else
    this->SyncAll();
#endif
}
} // namespace MoeInitRoutingV2
#endif // MOE_V2_SORT_ONE_CORE_H