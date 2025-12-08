/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_sort_multi_core.h
 * \brief
 */
#ifndef MOE_V3_GATHER_SORT_MULTI_CORE_H
#define MOE_V3_GATHER_SORT_MULTI_CORE_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t SORT32_ALIGN_ELEMENT = 32;
constexpr int64_t PARALLEL_GATHERED_SORT_NEED_CORE_NUM = 16;
constexpr int64_t MULTI_GATHERED_MAX_NUM = 4096; // 8192 * 8 / 16

class MoeGatherSortMultiCore {
public:
    __aicore__ inline MoeGatherSortMultiCore(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

private:
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buffer_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<int32_t> expendedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> sortedExpertIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndexGm_;
    GlobalTensor<int32_t> sortedNumGm_;

    TQue<QuePosition::VECOUT, 1> sortedNumCopyOutQueue_;

    int64_t expertIdxOffset_ = 0;
    int64_t expertIndexOffset_ = 0;
    int64_t compareScalarMask0Offset_ = 0;
    int64_t compareScalarMask1Offset_ = 0;
    int64_t gatherMaskOffset_ = 0;

    int64_t totalLength_;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actual_expert_num_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t perCoreElements_ = 0;
    int64_t blockIdx_;
    int64_t currentCoreElements_ = 0;
    int64_t needSortNum_ = 0;
    int64_t kvFactor = 2;

    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;
};

__aicore__ inline void MoeGatherSortMultiCore::CopyIn()
{
    LocalTensor<int32_t> expertIdx = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];

    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(currentCoreElements_ * sizeof(int32_t)), 0, 0, 0};

    DataCopyPad(expertIdx, expertIdxGm_[blockIdx_ * perCoreElements_], dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
}

__aicore__ inline void MoeGatherSortMultiCore::Compute()
{
    LocalTensor<int32_t> expertIdx = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    LocalTensor<int32_t> gatheredExpertIdx = buffer_.Get<int32_t>();
    LocalTensor<float> gatheredExpertIdxFp32 = gatheredExpertIdx.ReinterpretCast<float>();

    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, currentCoreElements_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxFp32, expertIdxFp32, (float)-1, currentCoreElements_);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> compareScalarMaskLocalTensor0 = buffer_.Get<uint8_t>()[compareScalarMask0Offset_];
    LocalTensor<uint8_t> compareScalarMaskLocalTensor1 = buffer_.Get<uint8_t>()[compareScalarMask1Offset_];
    LocalTensor<uint8_t> gatherMaskLocalTensor = buffer_.Get<uint8_t>()[gatherMaskOffset_];

    // Find elements >= expertStart_, which means -elements <= -expertStart_
    AscendC::CompareScalar(
        compareScalarMaskLocalTensor0, expertIdxFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::LE,
        (currentCoreElements_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();

    // Find elements < expertEnd_, which means -elements > -expertEnd_
    AscendC::CompareScalar(
        compareScalarMaskLocalTensor1, expertIdxFp32, static_cast<float>(-expertEnd_), AscendC::CMPMODE::GT,
        (currentCoreElements_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();

    // Get experts between [expert_start, expert_end)
    And(gatherMaskLocalTensor.ReinterpretCast<uint16_t>(), compareScalarMaskLocalTensor0.ReinterpretCast<uint16_t>(),
        compareScalarMaskLocalTensor1.ReinterpretCast<uint16_t>(),
        Ceil(currentCoreElements_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / kvFactor);
    PipeBarrier<PIPE_V>();

    uint64_t sortedNum = 0; // 用于保存筛选后保留下来的元素个数
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
    gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
    GatherMask(gatheredExpertIdxFp32, expertIdxFp32, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(currentCoreElements_), gatherMaskParams, sortedNum);
    PipeBarrier<PIPE_V>();
    actual_expert_num_ = sortedNum;
    int64_t needSortNum = Ceil(static_cast<int64_t>(sortedNum), ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    needSortNum_ = needSortNum;

    // Handle actual_expert_num_ == 0
    if (actual_expert_num_ < 1) {
        return;
    }

    LocalTensor<int32_t> expertIndex = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredExpertIndex = buffer_.Get<int32_t>()[needSortNum];
    ArithProgression<int32_t>(expertIndex, blockIdx_ * perCoreElements_, 1, currentCoreElements_);
    GatherMask(gatheredExpertIndex, expertIndex, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(currentCoreElements_), gatherMaskParams, sortedNum);
    PipeBarrier<PIPE_V>();
    int64_t duplicateNum = sortedNum % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = sortedNum - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(gatheredExpertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> concatLocal;
    LocalTensor<float> sortTempTensor = buffer_.Get<float>()[needSortNum * kvFactor];
    Concat(concatLocal, gatheredExpertIdxFp32, sortTempTensor, needSortNum / ONE_REPEAT_SORT_NUM);
    LocalTensor<float> sortedLocal = buffer_.Get<float>()[needSortNum * kvFactor + needSortNum * kvFactor * kvFactor];
    Sort<float, true>(sortedLocal, concatLocal, gatheredExpertIndex.ReinterpretCast<uint32_t>(), sortTempTensor,
                      needSortNum / ONE_REPEAT_SORT_NUM);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
}

__aicore__ inline void MoeGatherSortMultiCore::CopyOut()
{
    // Copy out sortedLocal for MergeSort
    if (actual_expert_num_ > 0) {
        LocalTensor<float> sortedLocal =
            buffer_.Get<float>()[needSortNum_ * kvFactor + needSortNum_ * kvFactor * kvFactor];
        DataCopyExtParams extParams{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(2 * actual_expert_num_ * sizeof(float)), 0, 0, 0};
        int64_t curCoreStartIndex = 2 * GetBlockIdx() * perCoreElements_;
        DataCopyPad(sortedExpertIdxGm_[curCoreStartIndex], sortedLocal, extParams);
    }

    // Copyout actual_expert_num_
    LocalTensor<int32_t> sortedNumOutLocal = sortedNumCopyOutQueue_.AllocTensor<int32_t>();
    sortedNumOutLocal.SetValue(0, actual_expert_num_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams3{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPad(sortedNumGm_[GetBlockIdx()], sortedNumOutLocal, copyParams3);

    sortedNumCopyOutQueue_.FreeTensor(sortedNumOutLocal);
}

__aicore__ inline void MoeGatherSortMultiCore::Init(GM_ADDR expertIdx, GM_ADDR expendedRowIdx, GM_ADDR workspace,
                                                    const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    totalLength_ = tilingData->n * tilingData->k;

    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);

    expendedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expendedRowIdx);

    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ float *)workspace);
    sortedExpertIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)));

    // key and value
    sortedNumGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                 Align(totalLength_, sizeof(int32_t)) * kvFactor * kvFactor);

    needCoreNum_ = PARALLEL_GATHERED_SORT_NEED_CORE_NUM;
    perCoreElements_ = Ceil(totalLength_, needCoreNum_);

    int32_t lastCoreElements = totalLength_ - (needCoreNum_ - 1) * perCoreElements_;
    if (blockIdx_ == (needCoreNum_ - 1)) {
        currentCoreElements_ = lastCoreElements;
    } else {
        currentCoreElements_ = perCoreElements_;
    }

    // expertIdxOffset_
    expertIdxOffset_ = AlignBytes(currentCoreElements_, sizeof(int32_t));
    expertIndexOffset_ = expertIdxOffset_;

    gatherMaskOffset_ = expertIdxOffset_ * kvFactor;
    int64_t maskOffset =
        AlignBytes(Ceil(currentCoreElements_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE, sizeof(int8_t));
    compareScalarMask0Offset_ = gatherMaskOffset_ + maskOffset;
    compareScalarMask1Offset_ = compareScalarMask0Offset_ + maskOffset;
    int64_t bufferSize = MULTI_GATHERED_MAX_NUM * kvFactor * kvFactor * kvFactor * sizeof(int32_t);
    pipe_->InitBuffer(sortedNumCopyOutQueue_, 1, AlignBytes(1, sizeof(int32_t)));
    pipe_->InitBuffer(buffer_, bufferSize); // 73728 Bytes
}

__aicore__ inline void MoeGatherSortMultiCore::Process()
{
    if (blockIdx_ < PARALLEL_GATHERED_SORT_NEED_CORE_NUM) {
        CopyIn();
        Compute();
        CopyOut();
    }
    SyncAll();
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_SORT_MULTI_CORE_H