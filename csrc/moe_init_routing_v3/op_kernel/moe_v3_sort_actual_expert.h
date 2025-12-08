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
 * \file moe_v3_sort_actual_expert.h
 * \brief
 */
#ifndef MOE_V3_SORT_ACTUAL_EXPERT_H
#define MOE_V3_SORT_ACTUAL_EXPERT_H

namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t MULTI_GATHERED_SORT_CORE_NUM = 16;
constexpr int64_t MULTI_GATHERED_SORT_THRSHOLD = 5632;
constexpr int64_t SINGLE_GATHERED_BUFFER_NUM = 2;
constexpr int64_t SINGLE_GATHERED_MAX_NUM = 21845;

template <typename T>
class MoeSortActualExpert {
public:
    __aicore__ inline MoeSortActualExpert(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expendedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline bool Process();
    __aicore__ inline void multiCoreGatheredSort();
    __aicore__ inline void CopyOutExpandRowIdx();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void TilingInKernel();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void CopyOutExpertCount();

private:
    TPipe *pipe;
    TBuf<TPosition::VECCALC> buffer_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, SINGLE_GATHERED_BUFFER_NUM> scaleCopyInQueue_;
    TQue<TPosition::VECOUT, 1> sortedNumCopyOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int64_t> expertTokensCountOrCumsumGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> expendedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<float> workspaceExpertIdxGm_;
    GlobalTensor<int32_t> workspaceGatheredSortNumGm_;
    GlobalTensor<float> workspaceGatheredExpertIdxGm_;
    GlobalTensor<int32_t> workspaceGatheredExpertIndexGm_;

    int64_t expertIdxOffset_ = 0;
    int64_t expertIndexOffset_ = 0;
    int64_t compareScalarMaskOffset_ = 0;
    int64_t compareScalarMask0Offset_ = 0;
    int64_t compareScalarMask1Offset_ = 0;
    int64_t gatherMaskOffset_ = 0;

    int64_t totalLength_;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actual_expert_num_ = 0;
    int64_t cols_ = 0;
    int64_t rowIdxType_ = 0;
    int64_t isInputScale_ = 0;
    int64_t k_ = 0;

    int64_t needSortNum_ = 0;

    int64_t needCoreNum_ = 0;
    int64_t perCoreElements_ = 0;
    int64_t lastCoreElements_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t curCoreStartIndex_ = 0;

    bool needMultiSort = false;

    int64_t kvFactor = 2;

    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;
};

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::CopyIn()
{
    LocalTensor<int32_t> expertIdx = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertIdx, expertIdxGm_, dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::SortCompute()
{
    LocalTensor<int32_t> expertIdx = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    LocalTensor<int32_t> gatheredExpertIdx = buffer_.Get<int32_t>();
    LocalTensor<float> gatheredExpertIdxFp32 = gatheredExpertIdx.ReinterpretCast<float>();

    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->totalLength_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxFp32, expertIdxFp32, (float)-1, this->totalLength_);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> compareScalarMaskLocalTensor0 = buffer_.Get<uint8_t>()[compareScalarMask0Offset_];
    LocalTensor<uint8_t> compareScalarMaskLocalTensor1 = buffer_.Get<uint8_t>()[compareScalarMask1Offset_];
    LocalTensor<uint8_t> gatherMaskLocalTensor = buffer_.Get<uint8_t>()[gatherMaskOffset_];

    // 找到大于等于exper_start的元素
    AscendC::CompareScalar(
        compareScalarMaskLocalTensor0, expertIdxFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::LE,
        (this->totalLength_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();

    // 找到小于exper_end的元素
    AscendC::CompareScalar(
        compareScalarMaskLocalTensor1, expertIdxFp32, static_cast<float>(-expertEnd_), AscendC::CMPMODE::GT,
        (this->totalLength_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
    PipeBarrier<PIPE_V>();
    // 按位与操作，筛选出[expert_start, expert_end)
    And(gatherMaskLocalTensor.ReinterpretCast<uint16_t>(), compareScalarMaskLocalTensor0.ReinterpretCast<uint16_t>(),
        compareScalarMaskLocalTensor1.ReinterpretCast<uint16_t>(),
        Ceil(this->totalLength_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / kvFactor);
    PipeBarrier<PIPE_V>();

    uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8;
    gatherMaskParams.src1RepeatStride = 8;
    GatherMask(gatheredExpertIdxFp32, expertIdxFp32, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(this->totalLength_), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    actual_expert_num_ = rsvdCnt;
    // Handle actual_expert_num_ == 0
    if (actual_expert_num_ < 1) {
        return;
    }
    int64_t needSortNum = Ceil(static_cast<int64_t>(rsvdCnt), ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    needSortNum_ = needSortNum;

    LocalTensor<int32_t> expertIndex = buffer_.Get<int32_t>()[expertIdxOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredExpertIndex = buffer_.Get<int32_t>()[needSortNum];
    ArithProgression<int32_t>(expertIndex, 0, 1, this->totalLength_);
    GatherMask(gatheredExpertIndex, expertIndex, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(this->totalLength_), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    if (rsvdCnt > MULTI_GATHERED_SORT_THRSHOLD) {
        if (GetBlockIdx() == 0) {
            // 复制 gatheredExpertIdxFp32 和 gatheredExpertIndex 数据到workspace
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCnt * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(workspaceGatheredExpertIdxGm_, gatheredExpertIdxFp32, copyParams);
            DataCopyPad(workspaceGatheredExpertIndexGm_, gatheredExpertIndex, copyParams);
        }
        needMultiSort = true;
        return;
    }
    int64_t duplicateNum = rsvdCnt % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = rsvdCnt - duplicateNum;
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
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sortedExpertIdx = gatheredExpertIdxFp32;
    LocalTensor<int32_t> sortedExpertIndex = gatheredExpertIndex.ReinterpretCast<int32_t>();

    Extract(sortedExpertIdx, sortedExpertIndex.ReinterpretCast<uint32_t>(), sortedLocal,
            needSortNum / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> sortedExpertIdxInt32 = sortedExpertIdx.ReinterpretCast<int32_t>();

    Muls(sortedExpertIdx, sortedExpertIdx, (float)-1, rsvdCnt);
    Cast(sortedExpertIdxInt32, sortedExpertIdx, RoundMode::CAST_ROUND, rsvdCnt);
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::TilingInKernel()
{
    int64_t coreNum = needMultiSort ? MULTI_GATHERED_SORT_CORE_NUM : GetBlockNum();
    perCoreElements_ = Ceil(actual_expert_num_, coreNum);
    needCoreNum_ = Ceil(actual_expert_num_, perCoreElements_);
    lastCoreElements_ = actual_expert_num_ - (needCoreNum_ - 1) * perCoreElements_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        curCoreElements_ = lastCoreElements_;
    } else {
        curCoreElements_ = perCoreElements_;
    }
    curCoreStartIndex_ = GetBlockIdx() * perCoreElements_;
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::multiCoreGatheredSort()
{
    // 重新计算实际核处理数并对齐 32
    needSortNum_ = Ceil(static_cast<int64_t>(curCoreElements_), ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    perCoreElements_ = Ceil(this->totalLength_, MULTI_GATHERED_SORT_CORE_NUM);

    LocalTensor<int32_t> sortedNumOutLocal = sortedNumCopyOutQueue_.AllocTensor<int32_t>();
    LocalTensor<float> gatheredExpertIdxFp32 = buffer_.Get<float>();
    LocalTensor<int32_t> gatheredExpertIndex = buffer_.Get<int32_t>()[needSortNum_];
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curCoreElements_ * sizeof(float)),
                                     0, 0, 0};
    DataCopyPadExtParams<float> expertIdxPadParams{false, 0, 0, 0};
    DataCopyPad(gatheredExpertIdxFp32, workspaceGatheredExpertIdxGm_[curCoreStartIndex_], dataCopyParams,
                expertIdxPadParams);
    DataCopyPadExtParams<int32_t> expertIndexPadParams{false, 0, 0, 0};
    DataCopyPad(gatheredExpertIndex, workspaceGatheredExpertIndexGm_[curCoreStartIndex_], dataCopyParams,
                expertIndexPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    LocalTensor<float> concatLocal;
    LocalTensor<float> sortTempTensor = buffer_.Get<float>()[needSortNum_ * kvFactor];
    // Duplicate MIN_FP32
    int64_t duplicateNum = curCoreElements_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = curCoreElements_ - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(gatheredExpertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    Concat(concatLocal, gatheredExpertIdxFp32, sortTempTensor, needSortNum_ / ONE_REPEAT_SORT_NUM);
    LocalTensor<float> sortedLocal = buffer_.Get<float>()[needSortNum_ * kvFactor + needSortNum_ * kvFactor * kvFactor];
    Sort<float, true>(sortedLocal, concatLocal, gatheredExpertIndex.ReinterpretCast<uint32_t>(), sortTempTensor,
                      needSortNum_ / ONE_REPEAT_SORT_NUM);

    // Copy out sortedLocal for MergeSort
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t curCoreSortedStartIndex = kvFactor * GetBlockIdx() * perCoreElements_;
    dataCopyParams.blockLen = static_cast<uint32_t>(kvFactor * curCoreElements_ * sizeof(float));
    DataCopyPad(workspaceExpertIdxGm_[curCoreSortedStartIndex], sortedLocal, dataCopyParams);
    // Copyout sortedNum
    sortedNumOutLocal.SetValue(0, curCoreElements_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    dataCopyParams.blockLen = static_cast<uint32_t>(sizeof(int32_t));
    DataCopyPad(workspaceGatheredSortNumGm_[GetBlockIdx()], sortedNumOutLocal, dataCopyParams);
    sortedNumCopyOutQueue_.FreeTensor(sortedNumOutLocal);
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::CopyOutExpandRowIdx()
{
    LocalTensor<int32_t> sortedExpertIndex = buffer_.Get<int32_t>()[needSortNum_];
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    if (GetBlockIdx() == 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(actual_expert_num_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expendedRowIdxGm_, sortedExpertIndex, copyParams);
    }
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::ExpertCountCompute()
{
    LocalTensor<int32_t> sortedExpertIdx = buffer_.Get<int32_t>()[curCoreStartIndex_];
    LocalTensor<int32_t> expertCountLocalTensor = buffer_.Get<int32_t>()[needSortNum_ * kvFactor];
    Duplicate(expertCountLocalTensor, 0, expertEnd_ - expertStart_);

    for (int64_t i = 0; i < curCoreElements_; i++) {
        int64_t expertIdx = sortedExpertIdx.GetValue(i) - expertStart_;
        int32_t curExpertCount = expertCountLocalTensor.GetValue(expertIdx);
        expertCountLocalTensor.SetValue(expertIdx, curExpertCount + 1);
    }
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyOutParams1{1, static_cast<uint32_t>((expertEnd_ - expertStart_) * sizeof(int32_t)), 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(workspaceGm_, expertCountLocalTensor, copyOutParams1);
    SetAtomicNone();
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::CopyOut()
{
    LocalTensor<int32_t> sortedExpertIndex = buffer_.Get<int32_t>()[needSortNum_ + curCoreStartIndex_];
    int64_t xLocalOffset = (needSortNum_ * kvFactor + ASSIST_NUM) * sizeof(int32_t) / sizeof(T);
    LocalTensor<T> xLocalTensor = buffer_.Get<T>()[xLocalOffset];

    for (int64_t i = 0; i < curCoreElements_; i++) {
        int64_t srcRow = sortedExpertIndex.GetValue(i) / k_;
        int64_t dstRow = i + curCoreStartIndex_;
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);

        LocalTensor<float> scaleLocalTensor;
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(xLocalTensor, xGm_[srcRow * cols_], dataCopyParams, dataCopyPadParams);
        if (isInputScale_ == 1) {
            scaleLocalTensor = scaleCopyInQueue_.AllocTensor<float>();
            DataCopyExtParams dataCopyParams2{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> dataCopyPadParams2{false, 0, 0, 0};
            DataCopyPad(scaleLocalTensor, scaleGm_[srcRow], dataCopyParams2, dataCopyPadParams2);
            scaleCopyInQueue_.EnQue<float>(scaleLocalTensor);
        }
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyExtParams copyOutParams1{1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(expandedXGm_[dstRow * cols_], xLocalTensor, copyOutParams1);
        if (isInputScale_ == 1) {
            scaleLocalTensor = scaleCopyInQueue_.DeQue<float>();
            DataCopyExtParams copyOutParams2{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[dstRow], scaleLocalTensor, copyOutParams2);
            scaleCopyInQueue_.FreeTensor(scaleLocalTensor);
        }
    }
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::CopyOutExpertCount()
{
    LocalTensor<int32_t> expertCountLocalTensor = buffer_.Get<int32_t>()[needSortNum_ * kvFactor];
    LocalTensor<int64_t> expertCountLocalTensorInt64 =
        buffer_.Get<int32_t>()[needSortNum_ * kvFactor + ASSIST_NUM].ReinterpretCast<int64_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((expertEnd_ - expertStart_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountLocalTensor, workspaceGm_, dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Cast(expertCountLocalTensorInt64, expertCountLocalTensor, RoundMode::CAST_NONE, (expertEnd_ - expertStart_));
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams copyOutParams1{1, static_cast<uint32_t>((expertEnd_ - expertStart_) * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountOrCumsumGm_, expertCountLocalTensorInt64, copyOutParams1);
}

template <typename T>
__aicore__ inline void MoeSortActualExpert<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                                    GM_ADDR expendedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                    GM_ADDR expandedScale, GM_ADDR workspace,
                                                    const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->totalLength_ = tilingData->n * tilingData->k;
    cols_ = tilingData->cols;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    rowIdxType_ = tilingData->rowIdxType;
    isInputScale_ = tilingData->isInputScale;
    k_ = tilingData->k;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);

    expendedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expendedRowIdx);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    expertTokensCountOrCumsumGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace, ASSIST_NUM);
    if (GetBlockIdx() == 0) {
        InitGlobalMemory(workspaceGm_, ASSIST_NUM, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    // 多核sort场景workspace分为5段，分别为ExpertIdx、ExpertIndex、gatheredSortNum、GatheredExpertIdx、GatheredExpertIndex
    workspaceExpertIdxGm_.SetGlobalBuffer((__gm__ float *)workspace);
    int64_t offset = kvFactor * Align(this->totalLength_, sizeof(int32_t));
    workspaceGatheredExpertIdxGm_.SetGlobalBuffer((__gm__ float *)workspace + offset);
    offset += Align(this->totalLength_, sizeof(float));
    workspaceGatheredExpertIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + offset);
    offset += Align(this->totalLength_, sizeof(float));
    workspaceGatheredSortNumGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + offset);

    expertIdxOffset_ = AlignBytes(this->totalLength_, sizeof(int32_t));
    expertIndexOffset_ = expertIdxOffset_;

    gatherMaskOffset_ = expertIdxOffset_ * kvFactor;
    int64_t maskOffset =
        AlignBytes(Ceil(this->totalLength_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE, sizeof(int8_t));
    compareScalarMask0Offset_ = gatherMaskOffset_ + maskOffset;
    compareScalarMask1Offset_ = compareScalarMask0Offset_ + maskOffset;
    int64_t maskOffsetMax = Ceil(SINGLE_GATHERED_MAX_NUM, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE;
    int64_t bufferSize =
        AlignBytes(SINGLE_GATHERED_MAX_NUM, sizeof(int32_t)) * kvFactor + maskOffsetMax + maskOffsetMax + maskOffsetMax;
    pipe->InitBuffer(scaleCopyInQueue_, SINGLE_GATHERED_BUFFER_NUM, 32);
    pipe->InitBuffer(sortedNumCopyOutQueue_, SINGLE_GATHERED_BUFFER_NUM, 32);
    pipe->InitBuffer(buffer_, bufferSize); // 182992 Bytes
}

template <typename T>
__aicore__ inline bool MoeSortActualExpert<T>::Process()
{
    CopyIn();
    SortCompute();
    TilingInKernel(); // 适配多核sort场景tiling
    if (needMultiSort) {
        // 大于5632后，进行多核并行排序，并将结果输出到workspace
        SyncAll();
        if (GetBlockIdx() < needCoreNum_) {
            multiCoreGatheredSort();
        }
        SyncAll();
        return false;
    }

    if (GetBlockIdx() < needCoreNum_) {
        CopyOutExpandRowIdx();
    }
    // 小于5632后，输出expertCount和 gatherout （量化(暂不支持)、scale、数据类型、expertCount支持2维（暂不支持））
    if (GetBlockIdx() < needCoreNum_) {
        ExpertCountCompute();
        CopyOut();
    }
    SyncAll();
    if (GetBlockIdx() == GetBlockNum() - 1) {
        CopyOutExpertCount();
    }
    return true;
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_SORT_ACTUAL_EXPERT_H