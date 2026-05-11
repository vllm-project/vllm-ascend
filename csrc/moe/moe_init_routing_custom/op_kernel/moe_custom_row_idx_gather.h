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
 * \file moe_custom_row_idx_gather.h
 * \brief
 */
#ifndef MOE_CUSTOM_ROW_IDX_GATHER_H
#define MOE_CUSTOM_ROW_IDX_GATHER_H

#include "moe_custom_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

class RowIdxGather {
public:
    __aicore__ inline RowIdxGather(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR workspace, const MoeInitRoutingCustomTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t elements);
    __aicore__ inline void Compute(int64_t loop, int64_t elements);
    __aicore__ inline void CopyOut(int64_t loop, int64_t elements, GlobalTensor<int32_t> &RowIdxDstGm_);
    __aicore__ inline void AssistInit();

private:
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndicesGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> assistGm_;
    GlobalTensor<int32_t> gatherIndicesGm_;

    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIndicesInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;
    TBuf<TPosition::VECCALC> assistBuffer_;

    const MoeCustomSrcToDstComputeTilingData *srcToDstComputeTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t actualExpertNum_ = 0;
    int64_t ep_ = 0;
    int64_t rowIdxType_ = 0;
    int64_t expertTotalCount_ = 0;

    int64_t loops_ = 0;
    int64_t perLoopElements_ = 0;
    int64_t lastLoopElements_ = 0;
};

__aicore__ inline void RowIdxGather::AssistInit()
{
    LocalTensor<int32_t> assistTensor = assistBuffer_.Get<int32_t>(ASSIST_NUM);
    DataCopy(assistTensor, assistGm_, ASSIST_NUM);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Adds(assistTensor, assistTensor, (int32_t)(blockIdx_ * perCoreElements_), ASSIST_NUM);
}

__aicore__ inline void RowIdxGather::Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                          const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    srcToDstComputeTilingData_ = &(tilingData->srcToDstComputeParamsOp);
    blockIdx_ = GetBlockIdx();
    actualExpertNum_ = tilingData->actualExpertNum;
    ep_ = tilingData->ep;
    rowIdxType_ = tilingData->rowIdxType;

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, actualExpertNum_);

    if (ep_) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                                Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            actualExpertNum_);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
        expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    } else {
        expertTotalCount_ = tilingData->n * tilingData->k;
    }
    assistGm_.SetGlobalBuffer((__gm__ int32_t *)assist, ASSIST_NUM);
    perCoreElements_ = Ceil(expertTotalCount_, srcToDstComputeTilingData_->needCoreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreElements_);

    int64_t lastCoreElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreElements_;
    int64_t perCoreLoops = Ceil(perCoreElements_, srcToDstComputeTilingData_->perCorePerLoopElements);
    int64_t perCorePerLoopElements = Ceil(perCoreElements_, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements_ - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = Ceil(lastCoreElements, srcToDstComputeTilingData_->perCorePerLoopElements);
    int64_t lastCorePerLoopElements = Ceil(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopELements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    loops_ = perCoreLoops;
    if (blockIdx_ == needCoreNum_ - 1) {
        loops_ = lastCoreLoops;
        perLoopElements_ = lastCorePerLoopElements;
        lastLoopElements_ = lastCoreLastLoopELements;
    } else {
        loops_ = perCoreLoops;
        perLoopElements_ = perCorePerLoopElements;
        lastLoopElements_ = perCoreLastLoopElements;
    }

    if (rowIdxType_ == SCATTER) {
        sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_,
                                               actualExpertNum_);
    } else {
        sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                                   Align(tilingData->n * tilingData->k, sizeof(int32_t)) +
                                                   blockIdx_ * perCoreElements_,
                                               actualExpertNum_);
    }

    if ((ep_ == 0 && rowIdxType_ == SCATTER) && (blockIdx_ < needCoreNum_)) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                          Align(tilingData->n * tilingData->k, sizeof(int32_t)));
    }
    pipe_->InitBuffer(sortedExpertIndicesInQueue_, 1, AlignBytes(perLoopElements_, sizeof(int32_t)));
    pipe_->InitBuffer(copyOutQueue_, 1, Ceil(perLoopElements_, ASSIST_NUM) * ASSIST_NUM * BLOCK_BYTES);
    pipe_->InitBuffer(assistBuffer_, ASSIST_NUM * sizeof(int32_t));
}

__aicore__ inline void RowIdxGather::Process()
{
    if (ep_ == 1 && rowIdxType_ == SCATTER) {
        return;
    } else {
        if (blockIdx_ < needCoreNum_) {
            AssistInit();
            for (int64_t loop = 0; loop < loops_; loop++) {
                int64_t elements = perLoopElements_;
                if (loop == loops_ - 1) {
                    elements = lastLoopElements_;
                }
                CopyIn(loop, elements);
                Compute(loop, elements);
                CopyOut(loop, elements, expandedRowIdxGm_);
            }
        }
    }
    AscendC::SyncAll();
}

__aicore__ inline void RowIdxGather::CopyIn(int64_t loop, int64_t elements)
{
    LocalTensor<int32_t> sortedExpertIndicesInLocal = sortedExpertIndicesInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(elements * sizeof(int32_t)), 0, 0,
                                     0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(sortedExpertIndicesInLocal, sortedExpertIndicesGm_[loop * perLoopElements_], dataCopyParams,
                dataCopyPadParams);
    sortedExpertIndicesInQueue_.EnQue(sortedExpertIndicesInLocal);
}

__aicore__ inline void RowIdxGather::Compute(int64_t loop, int64_t elements)
{
    LocalTensor<int32_t> outLocal = copyOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> assistTensor = assistBuffer_.Get<int32_t>(ASSIST_NUM);
    PipeBarrier<PIPE_V>();
    int64_t loops = Ceil(elements, ASSIST_INDEX_NUM);
    for (int64_t i = 0; i < loops; i++) {
        Adds(outLocal[i * ASSIST_NUM], assistTensor,
             static_cast<int32_t>(perLoopElements_ * loop + i * ASSIST_INDEX_NUM), ASSIST_NUM);
    }
    PipeBarrier<PIPE_V>();
    copyOutQueue_.EnQue<int32_t>(outLocal);
}

__aicore__ inline void RowIdxGather::CopyOut(int64_t loop, int64_t elements, GlobalTensor<int32_t> &RowIdxDstGm_)
{
    LocalTensor<int32_t> inLocal = sortedExpertIndicesInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> outLocal = copyOutQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = sizeof(int32_t);
    uint32_t outOffset;
    for (int64_t idx = 0; idx < elements; idx++) {
        outOffset = inLocal.GetValue(idx);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(RowIdxDstGm_[outOffset], outLocal[idx * INT32_ONE_BLOCK_NUM], intriParams);
    }

    sortedExpertIndicesInQueue_.FreeTensor(inLocal);
    copyOutQueue_.FreeTensor(outLocal);
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_ROW_IDX_GATHER_H