/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_TOP_K_TOP_P_OPT_H_KERNEL
#define APPLY_TOP_K_TOP_P_OPT_H_KERNEL

#include "kernel_operator.h"

using namespace AscendC;
namespace ApplyTopKTopPOptOp {
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
constexpr uint16_t BF16_NEG_INF = 0xFF80;
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000;

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t RESERVED_UB = 1024;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t SOFTMAX_UB_NUM = 2;
constexpr uint32_t CMP_ALIGN_BYTES = 256;
constexpr uint32_t SCATTER_PART_LENGTH = 1024;

template <typename inputT, typename calT, typename outputT>
class ApplyTopKTopPOpt {
public:
    __aicore__ inline ApplyTopKTopPOpt(){};
    __aicore__ inline void InitTilingData(const ApplyTopKTopPWithSortedTilingData& __restrict tilingData,
                                          GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR p, GM_ADDR k,
                                          GM_ADDR logits, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void ProcessTopKTopPOpt();
    __aicore__ inline void ProcessTopKOpt();
    __aicore__ inline void ProcessTopPOpt();

private:
    __aicore__ inline void GetMaxValue(int64_t baseGmIdx);
    __aicore__ inline void ComputeSoftmaxSum(int64_t baseGmIdx, float kthVal, uint32_t kthStartIdx);
    __aicore__ inline void WriteSoftmaxToGm(int64_t baseGmIdx, uint32_t kthStartIdx);
    __aicore__ inline void CumsumKoggleStone(int64_t baseGmIdx);
    __aicore__ inline void FindFirstIndex(int64_t baseGmIdx, uint32_t& firstIdx);
    __aicore__ inline void CompareSelectOnLogits(int64_t batchGmBase, inputT thresholdVal, bool useGE);
    __aicore__ inline void ScatterBoundary(int64_t batchGmBase, int64_t sortedBase, uint32_t firstIdx,
                                           inputT thresholdVal);
    __aicore__ inline void CopyOutLast(int64_t batchGmBase, int64_t sortedBase);
    __aicore__ inline void FillNegInfAndScatterTopK(int64_t batchGmBase, int64_t sortedBase, uint32_t kthStartIdx);
    __aicore__ inline uint32_t FindKthStartIdx(int64_t sortedBase, int32_t kValue);
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) { return y == 0 ? x : (x + y - 1) / y; }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(id);
        WaitFlag<HardEvent::S_MTE3>(id);
    }
    __aicore__ inline void VToMTE3Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void MTE2ToVSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void VToMTE2Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
    }
    __aicore__ inline void MTE2ToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(id);
        WaitFlag<HardEvent::MTE2_S>(id);
    }
    __aicore__ inline void MTE3ToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(id);
        WaitFlag<HardEvent::MTE3_S>(id);
    }
    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
    }
    __aicore__ inline void VToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline void SToVSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> calBuf_;

    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t tailBatch_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t loopBatch_ = 0;
    uint32_t batchOffset_ = 0;

    uint32_t softmaxLength = 1;
    uint32_t lineSfLoopTimes = 1;
    uint32_t softmaxLengthTail = 1;
    uint32_t cmpSelTileLength = 0;
    uint32_t scatterLength = 1;
    uint32_t iterateTimes = 0;

    GlobalTensor<inputT> mGmSortedValue_;
    GlobalTensor<int32_t> mGmSortedIndices_;
    GlobalTensor<inputT> mGmP_;
    GlobalTensor<int32_t> mGmK_;
    GlobalTensor<inputT> mGmLogits_;
    GlobalTensor<outputT> mGmOut_;
    GlobalTensor<float> softMaxGm;

    LocalTensor<uint8_t> totalUb;
    LocalTensor<float> sfLocalFp32;
    LocalTensor<inputT> sfLocalInput;
    LocalTensor<float> sfResLocal;
    LocalTensor<float> reduceLocal;

    LocalTensor<inputT> cmpLogitsLocal;
    LocalTensor<inputT> cmpNegInfLocal;
    LocalTensor<float> cmpFp32Local;
    LocalTensor<uint8_t> cmpMaskLocal;

    LocalTensor<inputT> scatterValLocal;
    LocalTensor<int32_t> scatterIdxLocal;
    LocalTensor<inputT> scatterAlignedLocal;
    LocalTensor<float> cumsumLocal;

    float maxValue = 0;
    float pValue = 0;
    float kthValue_ = 0;
    float reduceSumValue = 0;
    float reduceSumValueInvert = 0;
};

// --- InitTilingData ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::InitTilingData(
    const ApplyTopKTopPWithSortedTilingData& __restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR logits, GM_ADDR out, GM_ADDR workspace)
{
    batchSize_ = tilingData.batchSize;
    vocabSize_ = tilingData.vocabSize;
    batchPerCore_ = tilingData.batchPerCore;
    tailBatch_ = tilingData.tailBatch;
    blockNum_ = tilingData.blockNum;
    calUbSize_ = static_cast<uint32_t>(tilingData.calUbSize);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ < tailBatch_) {
        loopBatch_ = batchPerCore_ + 1;
        batchOffset_ = blockIdx_ * loopBatch_;
    } else {
        loopBatch_ = batchPerCore_;
        batchOffset_ = blockIdx_ * batchPerCore_ + tailBatch_;
    }
    uint32_t maxSfLen = (calUbSize_ - RESERVED_UB) / SOFTMAX_UB_NUM / FLOAT_BYTES;
    softmaxLength = maxSfLen < vocabSize_ ? maxSfLen : vocabSize_;
    lineSfLoopTimes = CeilDiv(vocabSize_, softmaxLength);
    softmaxLengthTail = vocabSize_ - (lineSfLoopTimes - 1) * softmaxLength;
    scatterLength = (calUbSize_ - RESERVED_UB - BLOCK_BYTES) /
                    (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT) + BLOCK_BYTES) / SCATTER_PART_LENGTH *
                    SCATTER_PART_LENGTH;

    mGmSortedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(sorted_value));
    mGmSortedIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices));
    mGmP_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(p));
    mGmK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
    mGmLogits_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(logits));
    mGmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ outputT*>(out));
    softMaxGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize_ * vocabSize_);
    iterateTimes = tilingData.iterateTimes;
}

// --- InitBuffer ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::InitBuffer(TPipe* inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(calBuf_, calUbSize_);
    totalUb = calBuf_.Get<uint8_t>();

    uint32_t sfLenAligned = CeilDiv(softmaxLength, BLOCK_BYTES / sizeof(inputT)) * (BLOCK_BYTES / sizeof(inputT));
    sfLocalFp32 = totalUb.ReinterpretCast<float>();
    sfLocalInput = totalUb[sfLenAligned * sizeof(inputT)].ReinterpretCast<inputT>();
    sfResLocal = totalUb[sfLenAligned * sizeof(float)].ReinterpretCast<float>();
    reduceLocal = totalUb[sfLenAligned * sizeof(float) * 2].ReinterpretCast<float>();

    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    scatterLength = (calUbSize_ - RESERVED_UB - BLOCK_BYTES) /
                    (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT) + BLOCK_BYTES) / SCATTER_PART_LENGTH *
                    SCATTER_PART_LENGTH;
    scatterValLocal = totalUb[0].ReinterpretCast<inputT>();
    scatterIdxLocal = totalUb[scatterLength * sizeof(inputT)].ReinterpretCast<int32_t>();
    cumsumLocal = totalUb[scatterLength * (FLOAT_BYTES + sizeof(inputT))].ReinterpretCast<float>();
    uint32_t alignedOff = scatterLength * (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT));
    scatterAlignedLocal = totalUb[alignedOff].ReinterpretCast<inputT>();

    constexpr uint32_t CMP_ALIGN_ELEMS = CMP_ALIGN_BYTES / sizeof(inputT);
    uint32_t cmpAvail = calUbSize_ - RESERVED_UB - BLOCK_BYTES;
    if constexpr (IsSameType<inputT, float>::value || IsSameType<inputT, half>::value) {
        cmpSelTileLength = cmpAvail / (2 * sizeof(inputT) + 1) / CMP_ALIGN_ELEMS * CMP_ALIGN_ELEMS;
        cmpLogitsLocal = totalUb[0].ReinterpretCast<inputT>();
        cmpNegInfLocal = totalUb[cmpSelTileLength * sizeof(inputT)].ReinterpretCast<inputT>();
        cmpMaskLocal = totalUb[cmpSelTileLength * 2 * sizeof(inputT)];
    } else {
        cmpSelTileLength = cmpAvail / (2 * sizeof(inputT) + sizeof(float) + 1) / CMP_ALIGN_ELEMS * CMP_ALIGN_ELEMS;
        cmpLogitsLocal = totalUb[0].ReinterpretCast<inputT>();
        cmpNegInfLocal = totalUb[cmpSelTileLength * sizeof(inputT)].ReinterpretCast<inputT>();
        cmpFp32Local = totalUb[cmpSelTileLength * 2 * sizeof(inputT)].ReinterpretCast<float>();
        cmpMaskLocal = totalUb[cmpSelTileLength * (2 * sizeof(inputT) + sizeof(float))];
    }
}

// --- GetMaxValue: get max from valid elements (>= kthValue, excluding +inf) ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::GetMaxValue(int64_t baseGmIdx)
{
    int64_t idx = baseGmIdx + vocabSize_ - 1;
    if constexpr (IsSameType<inputT, float>::value) {
        maxValue = -mGmSortedValue_[idx].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        maxValue = -static_cast<float>(mGmSortedValue_[idx].GetValue(0));
    } else {
        maxValue = -ToFloat(mGmSortedValue_[idx].GetValue(0));
    }
}

// --- ComputeSoftmaxSum: tile-by-tile exp sum with TopK filtering ---
// For partial tile: only load valid data (from kthStartIdx); UB is pre-cleared by prior Duplicate.
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ComputeSoftmaxSum(int64_t baseGmIdx, float kthVal,
                                                                                  uint32_t kthStartIdx)
{
    reduceSumValue = 0;
    uint32_t loopDataNum = softmaxLength;
    for (uint32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
        uint32_t tileStart = loopInner * softmaxLength;
        int64_t gmIdx = baseGmIdx + tileStart;
        if (loopInner == lineSfLoopTimes - 1)
            loopDataNum = softmaxLengthTail;
        uint32_t tileEnd = tileStart + loopDataNum;

        if (tileEnd <= kthStartIdx) {
            continue;
        }

        if (tileStart >= kthStartIdx) {
            if constexpr (!IsSameType<inputT, float>::value) {
                DataCopyPad(sfLocalInput, mGmSortedValue_[gmIdx],
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, loopDataNum);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopyPad(sfLocalFp32, mGmSortedValue_[gmIdx],
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
            }
        } else {
            uint32_t localOffset = kthStartIdx - tileStart;
            if constexpr (!IsSameType<inputT, float>::value) {
                DataCopyPad(sfLocalInput, mGmSortedValue_[gmIdx],
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, loopDataNum);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopyPad(sfLocalFp32, mGmSortedValue_[gmIdx],
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
            }
            Duplicate(sfLocalFp32, -1.0e30f, localOffset);
            PipeBarrier<PIPE_V>();
        }

        Adds(sfResLocal, sfLocalFp32, maxValue, loopDataNum);
        PipeBarrier<PIPE_V>();
        Exp(sfResLocal, sfResLocal, loopDataNum);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal, sfResLocal, reduceLocal, loopDataNum);
        VToSSync();
        reduceSumValue += reduceLocal.GetValue(0);
        SToVSync();
    }
    reduceSumValueInvert = (reduceSumValue > 0.0f) ? (1.0f / reduceSumValue) : 0.0f;
}

// --- WriteSoftmaxToGm: compute normalized probabilities with TopK masking, write to softMaxGm ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::WriteSoftmaxToGm(int64_t baseGmIdx,
                                                                                 uint32_t kthStartIdx)
{
    uint32_t loopDataNum = softmaxLength;
    for (uint32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
        uint32_t tileStart = loopInner * softmaxLength;
        int64_t gmIdx = baseGmIdx + tileStart;
        if (loopInner == lineSfLoopTimes - 1)
            loopDataNum = softmaxLengthTail;
        uint32_t tileEnd = tileStart + loopDataNum;

        if (tileEnd <= kthStartIdx) {
            Duplicate(sfResLocal, 0.0f, loopDataNum);
            VToMTE3Sync();
            DataCopyPad(softMaxGm[gmIdx], sfResLocal, {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
            MTE3ToMTE2Sync();
            continue;
        }

        if constexpr (!IsSameType<inputT, float>::value) {
            DataCopyPad(sfLocalInput, mGmSortedValue_[gmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(sfLocalFp32, mGmSortedValue_[gmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
        }

        if (tileStart < kthStartIdx) {
            uint32_t localOffset = kthStartIdx - tileStart;
            Duplicate(sfLocalFp32, -1.0e30f, localOffset);
            PipeBarrier<PIPE_V>();
        }

        Adds(sfResLocal, sfLocalFp32, maxValue, loopDataNum);
        PipeBarrier<PIPE_V>();
        Exp(sfResLocal, sfResLocal, loopDataNum);
        PipeBarrier<PIPE_V>();
        Muls(sfResLocal, sfResLocal, reduceSumValueInvert, loopDataNum);
        VToMTE3Sync();
        DataCopyPad(softMaxGm[gmIdx], sfResLocal, {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
        MTE3ToMTE2Sync();
    }
}

// --- CumsumKoggleStone: iterative parallel prefix sum on softMaxGm ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CumsumKoggleStone(int64_t baseGmIdx)
{
    int64_t iteratOffset = 1;
    for (uint32_t iterateTime = 0; iterateTime < iterateTimes; iterateTime++) {
        uint32_t addLength = vocabSize_ - static_cast<uint32_t>(iteratOffset);
        uint32_t innerLoopNum = addLength / softmaxLength;
        uint32_t dataTail = addLength - innerLoopNum * softmaxLength;
        uint32_t loopDataNum = softmaxLength;
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
            int64_t loopInnerOffset = dataTail + (innerLoopNum - 1 - innerLoopIdx) * softmaxLength;
            DataCopyPad(sfLocalFp32, softMaxGm[baseGmIdx + loopInnerOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(sfResLocal, softMaxGm[baseGmIdx + loopInnerOffset + iteratOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Add(sfLocalFp32, sfLocalFp32, sfResLocal, loopDataNum);
            VToMTE3Sync();
            DataCopyPad(softMaxGm[baseGmIdx + loopInnerOffset + iteratOffset], sfLocalFp32,
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
            MTE3ToMTE2Sync();
        }
        if (dataTail > 0) {
            loopDataNum = dataTail;
            DataCopyPad(sfLocalFp32, softMaxGm[baseGmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(sfResLocal, softMaxGm[baseGmIdx + iteratOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Add(sfLocalFp32, sfLocalFp32, sfResLocal, loopDataNum);
            VToMTE3Sync();
            DataCopyPad(softMaxGm[baseGmIdx + iteratOffset], sfLocalFp32,
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
            MTE3ToMTE2Sync();
        }
        iteratOffset *= 2;
    }
}

// --- FindFirstIndex: read cumsum from softMaxGm to find threshold crossing ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::FindFirstIndex(int64_t baseGmIdx, uint32_t& firstIdx)
{
    firstIdx = vocabSize_;
    bool found = false;
    uint32_t searchTimes = CeilDiv(vocabSize_, scatterLength);
    uint32_t searchTail = vocabSize_ - (searchTimes - 1) * scatterLength;
    for (uint32_t sIdx = 0; sIdx < searchTimes && !found; sIdx++) {
        uint32_t curLen = (sIdx == searchTimes - 1) ? searchTail : scatterLength;
        int64_t gmOff = baseGmIdx + sIdx * scatterLength;
        DataCopyPad(cumsumLocal, softMaxGm[gmOff], {1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0});
        MTE2ToSSync();
        if (cumsumLocal.GetValue(curLen - 1) <= pValue) {
            continue;
        }
        uint32_t scatterLoop = CeilDiv(curLen, SCATTER_PART_LENGTH);
        uint32_t scatterNumsTail = curLen - (scatterLoop - 1) * SCATTER_PART_LENGTH;
        for (uint32_t si = 0; si < scatterLoop && !found; si++) {
            uint32_t curNums = (si == scatterLoop - 1) ? scatterNumsTail : SCATTER_PART_LENGTH;
            if (cumsumLocal.GetValue(si * SCATTER_PART_LENGTH + curNums - 1) <= pValue) {
                continue;
            }
            for (uint32_t i = 0; i < curNums; i++) {
                uint32_t off = si * SCATTER_PART_LENGTH + i;
                if (cumsumLocal.GetValue(off) > pValue) {
                    firstIdx = sIdx * scatterLength + off;
                    found = true;
                    break;
                }
            }
        }
    }
}

// --- CompareSelectOnLogits: vectorized compare + select on original logits ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CompareSelectOnLogits(int64_t batchGmBase,
                                                                                      inputT thresholdVal, bool useGE)
{
    constexpr uint32_t CMP_ALIGN_ELEMS = CMP_ALIGN_BYTES / sizeof(inputT);
    uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
    uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;

    for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
        uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
        uint32_t curLenAligned = CeilDiv(curLen, CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
        int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;

        DataCopyPad(cmpLogitsLocal, mGmLogits_[gmOff], {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0},
                    {false, 0, 0, 0});
        MTE2ToVSync();

        if constexpr (IsSameType<inputT, float>::value) {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpLogitsLocal, thresholdVal, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal, cmpMaskLocal, cmpLogitsLocal, cmpNegInfLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   curLenAligned);
        } else if constexpr (IsSameType<inputT, half>::value) {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpLogitsLocal, thresholdVal, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal, cmpMaskLocal, cmpLogitsLocal, cmpNegInfLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   curLenAligned);
        } else {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            Cast(cmpFp32Local, cmpLogitsLocal, RoundMode::CAST_NONE, curLenAligned);
            PipeBarrier<PIPE_V>();
            float threshFp32 = ToFloat(thresholdVal);
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpFp32Local, threshFp32, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal.template ReinterpretCast<half>(), cmpMaskLocal,
                   cmpLogitsLocal.template ReinterpretCast<half>(), cmpNegInfLocal.template ReinterpretCast<half>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, curLenAligned);
        }
        VToMTE3Sync();
        DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal, {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
        MTE3ToMTE2Sync();
    }
}

// --- ScatterBoundary: scatter elements == threshold at sorted positions >= firstIndex ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ScatterBoundary(int64_t batchGmBase, int64_t sortedBase,
                                                                                uint32_t firstIdx, inputT thresholdVal)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ELEM_PER_REP = 8;
    constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
    constexpr uint32_t MAX_BRCB_REPEAT = 255;
    constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

    float threshFloat;
    if constexpr (IsSameType<inputT, float>::value) {
        threshFloat = thresholdVal;
    } else if constexpr (IsSameType<inputT, half>::value) {
        threshFloat = static_cast<float>(thresholdVal);
    } else {
        threshFloat = ToFloat(thresholdVal);
    }

    uint32_t equalCount = 0;
    for (uint32_t ei = firstIdx; ei < vocabSize_; ei++) {
        float val;
        if constexpr (IsSameType<inputT, float>::value) {
            val = mGmSortedValue_[sortedBase + ei].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            val = static_cast<float>(mGmSortedValue_[sortedBase + ei].GetValue(0));
        } else {
            val = ToFloat(mGmSortedValue_[sortedBase + ei].GetValue(0));
        }
        if (val != threshFloat)
            break;
        equalCount++;
    }
    if (equalCount == 0)
        return;

    uint32_t copyTimes = CeilDiv(equalCount, scatterLength);
    uint32_t copyTail = equalCount - (copyTimes - 1) * scatterLength;
    for (uint32_t ci = 0; ci < copyTimes; ci++) {
        uint32_t curLen = (ci == copyTimes - 1) ? copyTail : scatterLength;
        int64_t gmOff = sortedBase + firstIdx + ci * scatterLength;
        DataCopyPad(scatterIdxLocal, mGmSortedIndices_[gmOff],
                    {1, static_cast<uint32_t>(curLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(scatterValLocal, mGmSortedValue_[gmOff],
                    {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
        MTE2ToVSync();
        MTE2ToSSync();

        uint32_t totalRepeat = CeilDiv(curLen, BRCB_ELEM_PER_REP);
        uint32_t brcbDone = 0;
        while (brcbDone < totalRepeat) {
            uint32_t remaining = totalRepeat - brcbDone;
            uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ?
                                     ALIGNED_MAX_REPEAT :
                                     (remaining + BRCB_ALIGN_REPEATS - 1) / BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
            Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                 scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
            brcbDone += curRepeat;
        }
        VToMTE3Sync();

        for (uint32_t idx = 0; idx < curLen; idx++) {
            int32_t lineIndex = scatterIdxLocal.GetValue(idx);
            DataCopyPad(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK],
                        {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
        }
        MTE3ToSSync();
    }
}

// --- CopyOutLast: unconditionally output the last sorted element ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CopyOutLast(int64_t batchGmBase, int64_t sortedBase)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    int64_t lastOff = sortedBase + vocabSize_ - 1;
    DataCopyPad(scatterAlignedLocal, mGmSortedValue_[lastOff], {1, static_cast<uint32_t>(sizeof(inputT)), 0, 0, 0},
                {false, 0, 0, 0});
    int32_t lineIndex = mGmSortedIndices_.GetValue(lastOff);
    MTE2ToSSync();
    SToMTE3Sync();
    DataCopyPad(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal.template ReinterpretCast<outputT>(),
                {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
    MTE3ToSSync();
}

// --- FindKthStartIdx: find actual start index by searching backward for boundary ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline uint32_t ApplyTopKTopPOpt<inputT, calT, outputT>::FindKthStartIdx(int64_t sortedBase, int32_t kValue)
{
    int64_t kthIdx = sortedBase + vocabSize_ - kValue;
    float kthFloat;
    if constexpr (IsSameType<inputT, float>::value) {
        kthFloat = mGmSortedValue_[kthIdx].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        kthFloat = static_cast<float>(mGmSortedValue_[kthIdx].GetValue(0));
    } else {
        kthFloat = ToFloat(mGmSortedValue_[kthIdx].GetValue(0));
    }

    uint32_t kthStartIdx = vocabSize_ - static_cast<uint32_t>(kValue);
    while (kthStartIdx > 0) {
        uint32_t chunkLen = (kthStartIdx > scatterLength) ? scatterLength : kthStartIdx;
        uint32_t chunkStart = kthStartIdx - chunkLen;
        DataCopyPad(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart],
                    {1, static_cast<uint32_t>(chunkLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
        MTE2ToSSync();
        bool foundBoundary = false;
        for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
            float prevVal;
            if constexpr (IsSameType<inputT, float>::value) {
                prevVal = scatterValLocal.GetValue(i);
            } else if constexpr (IsSameType<inputT, half>::value) {
                prevVal = static_cast<float>(scatterValLocal.GetValue(i));
            } else {
                prevVal = ToFloat(scatterValLocal.GetValue(i));
            }
            if (prevVal < kthFloat) {
                kthStartIdx = chunkStart + static_cast<uint32_t>(i) + 1;
                foundBoundary = true;
                break;
            }
        }
        if (foundBoundary)
            break;
        kthStartIdx = chunkStart;
    }
    return kthStartIdx;
}

// --- FillNegInfAndScatterTopK: fill output with -inf, then Brcb scatter elements from kthStartIdx ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::FillNegInfAndScatterTopK(int64_t batchGmBase,
                                                                                         int64_t sortedBase,
                                                                                         uint32_t kthStartIdx)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ELEM_PER_REP = 8;
    constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
    constexpr uint32_t MAX_BRCB_REPEAT = 255;
    constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

    uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
    uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
    for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
        uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
        int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
        if constexpr (IsSameType<inputT, float>::value) {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLen);
        } else if constexpr (IsSameType<inputT, half>::value) {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLen);
        } else {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLen);
        }
        VToMTE3Sync();
        DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal, {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
        MTE3ToMTE2Sync();
    }

    uint32_t topKCount = vocabSize_ - kthStartIdx;
    int64_t topKStart = sortedBase + kthStartIdx;

    uint32_t copyTimes = CeilDiv(topKCount, scatterLength);
    uint32_t copyTail = topKCount - (copyTimes - 1) * scatterLength;
    for (uint32_t ci = 0; ci < copyTimes; ci++) {
        uint32_t curLen = (ci == copyTimes - 1) ? copyTail : scatterLength;
        int64_t gmOff = topKStart + ci * scatterLength;
        DataCopyPad(scatterValLocal, mGmSortedValue_[gmOff],
                    {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(scatterIdxLocal, mGmSortedIndices_[gmOff],
                    {1, static_cast<uint32_t>(curLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        MTE2ToVSync();
        MTE2ToSSync();

        uint32_t totalRepeat = CeilDiv(curLen, BRCB_ELEM_PER_REP);
        uint32_t brcbDone = 0;
        while (brcbDone < totalRepeat) {
            uint32_t remaining = totalRepeat - brcbDone;
            uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ?
                                     ALIGNED_MAX_REPEAT :
                                     (remaining + BRCB_ALIGN_REPEATS - 1) / BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
            Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                 scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
            brcbDone += curRepeat;
        }
        VToMTE3Sync();

        for (uint32_t idx = 0; idx < curLen; idx++) {
            int32_t lineIndex = scatterIdxLocal.GetValue(idx);
            DataCopyPad(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK],
                        {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
        }
        MTE3ToSSync();
    }
}

// --- ProcessTopKTopPOpt: main entry for optimized TopKTopP ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopKTopPOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;

        int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);

        if constexpr (IsSameType<inputT, float>::value) {
            pValue = 1.0f - mGmP_[batchOffset_ + loopBatch].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            pValue = 1.0f - static_cast<float>(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        } else {
            pValue = 1.0f - ToFloat(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        }

        GetMaxValue(sortedBase);

        if (*reinterpret_cast<int32_t*>(&maxValue) == FLOAT32_NEG_INF) {
            if (kValue <= 0 || static_cast<uint32_t>(kValue) >= vocabSize_) {
                uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
                uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
                for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                    uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                    int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                    DataCopyPad(cmpLogitsLocal, mGmLogits_[gmOff],
                                {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                    MTE2ToVSync();
                    VToMTE3Sync();
                    DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal,
                                {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
                    MTE3ToMTE2Sync();
                }
            } else {
                uint32_t kthStart = FindKthStartIdx(sortedBase, kValue);
                uint32_t actualTopKCount = vocabSize_ - kthStart;
                if (actualTopKCount <= scatterLength) {
                    FillNegInfAndScatterTopK(batchGmBase, sortedBase, kthStart);
                } else {
                    int64_t kthIdx = sortedBase + kthStart;
                    inputT kthValRaw = mGmSortedValue_[kthIdx].GetValue(0);
                    CompareSelectOnLogits(batchGmBase, kthValRaw, true);
                    CopyOutLast(batchGmBase, sortedBase);
                }
            }
            continue;
        }

        uint32_t kthStartIdx = 0;
        float kthVal = 0;
        if (kValue > 0 && static_cast<uint32_t>(kValue) < vocabSize_) {
            kthStartIdx = FindKthStartIdx(sortedBase, kValue);
            int64_t kthIdx = sortedBase + kthStartIdx;
            if constexpr (IsSameType<inputT, float>::value) {
                kthVal = mGmSortedValue_[kthIdx].GetValue(0);
            } else if constexpr (IsSameType<inputT, half>::value) {
                kthVal = static_cast<float>(mGmSortedValue_[kthIdx].GetValue(0));
            } else {
                kthVal = ToFloat(mGmSortedValue_[kthIdx].GetValue(0));
            }
            kthValue_ = kthVal;
        }

        if (kValue > 0 && (vocabSize_ - kthStartIdx) <= scatterLength && (vocabSize_ - kthStartIdx) <= softmaxLength) {
            constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ELEM_PER_REP = 8;
            constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
            constexpr uint32_t MAX_BRCB_REPEAT = 255;
            constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

            uint32_t topKActual = vocabSize_ - kthStartIdx;
            int64_t topKStart = sortedBase + kthStartIdx;

            if constexpr (!IsSameType<inputT, float>::value) {
                DataCopyPad(sfLocalInput, mGmSortedValue_[topKStart],
                            {1, static_cast<uint32_t>(topKActual * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, topKActual);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopyPad(sfLocalFp32, mGmSortedValue_[topKStart],
                            {1, static_cast<uint32_t>(topKActual * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
            }
            Adds(sfResLocal, sfLocalFp32, maxValue, topKActual);
            PipeBarrier<PIPE_V>();
            Exp(sfResLocal, sfResLocal, topKActual);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduceLocal, sfResLocal, reduceLocal, topKActual);
            VToSSync();
            float localReduceSum = reduceLocal.GetValue(0);
            float localReduceSumInv = (localReduceSum > 0.0f) ? (1.0f / localReduceSum) : 0.0f;
            SToVSync();

            Muls(sfResLocal, sfResLocal, localReduceSumInv, topKActual);
            VToSSync();

            float cumVal = 0.0f;
            uint32_t firstScatterIdx = topKActual - 1;
            for (uint32_t ci = 0; ci < topKActual; ci++) {
                cumVal += sfResLocal.GetValue(ci);
                if (cumVal > pValue) {
                    firstScatterIdx = ci;
                    break;
                }
            }

            uint32_t scatterCount = topKActual - firstScatterIdx;
            int64_t scatterGmStart = topKStart + firstScatterIdx;

            uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
            for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                if constexpr (IsSameType<inputT, float>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLen);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLen);
                } else {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLen);
                }
                VToMTE3Sync();
                DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal,
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
                MTE3ToMTE2Sync();
            }

            uint32_t scCopyTimes = CeilDiv(scatterCount, scatterLength);
            uint32_t scCopyTail = scatterCount - (scCopyTimes - 1) * scatterLength;
            for (uint32_t sci = 0; sci < scCopyTimes; sci++) {
                uint32_t curScLen = (sci == scCopyTimes - 1) ? scCopyTail : scatterLength;
                int64_t scGmOff = scatterGmStart + sci * scatterLength;

                DataCopyPad(scatterValLocal, mGmSortedValue_[scGmOff],
                            {1, static_cast<uint32_t>(curScLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(scatterIdxLocal, mGmSortedIndices_[scGmOff],
                            {1, static_cast<uint32_t>(curScLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                MTE2ToSSync();

                uint32_t totalRepeat = CeilDiv(curScLen, BRCB_ELEM_PER_REP);
                uint32_t brcbDone = 0;
                while (brcbDone < totalRepeat) {
                    uint32_t remaining = totalRepeat - brcbDone;
                    uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ? ALIGNED_MAX_REPEAT :
                                                                          (remaining + BRCB_ALIGN_REPEATS - 1) /
                                                                              BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
                    Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                         scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
                    brcbDone += curRepeat;
                }
                VToMTE3Sync();

                for (uint32_t idx = 0; idx < curScLen; idx++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(idx);
                    DataCopyPad(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK],
                                {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
                }
                MTE3ToSSync();
            }
            continue;
        }

        ComputeSoftmaxSum(sortedBase, kthVal, kthStartIdx);
        WriteSoftmaxToGm(sortedBase, kthStartIdx);
        CumsumKoggleStone(sortedBase);

        uint32_t firstIdx = vocabSize_;
        FindFirstIndex(sortedBase, firstIdx);
        if (firstIdx >= vocabSize_)
            firstIdx = vocabSize_ - 1;

        inputT thresholdVal = mGmSortedValue_[sortedBase + firstIdx].GetValue(0);

        // Always use GE compare (keeps all elements >= threshold)
        CompareSelectOnLogits(batchGmBase, thresholdVal, true);

        // Exclude elements == threshold at sorted positions BEFORE firstIdx
        // These were incorrectly kept by GE, need to write -inf back
        float threshFloat;
        if constexpr (IsSameType<inputT, float>::value) {
            threshFloat = thresholdVal;
        } else if constexpr (IsSameType<inputT, half>::value) {
            threshFloat = static_cast<float>(thresholdVal);
        } else {
            threshFloat = ToFloat(thresholdVal);
        }

        uint32_t excludeCount = 0;
        if (firstIdx > 0) {
            uint32_t searchPos = firstIdx;
            bool exDone = false;
            while (searchPos > 0 && !exDone) {
                uint32_t chunkLen = (searchPos > scatterLength) ? scatterLength : searchPos;
                uint32_t chunkStart = searchPos - chunkLen;
                DataCopyPad(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart],
                            {1, static_cast<uint32_t>(chunkLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToSSync();
                for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
                    float val;
                    if constexpr (IsSameType<inputT, float>::value) {
                        val = scatterValLocal.GetValue(i);
                    } else if constexpr (IsSameType<inputT, half>::value) {
                        val = static_cast<float>(scatterValLocal.GetValue(i));
                    } else {
                        val = ToFloat(scatterValLocal.GetValue(i));
                    }
                    if (val != threshFloat) {
                        exDone = true;
                        break;
                    }
                    excludeCount++;
                }
                searchPos = chunkStart;
            }
        }

        if (excludeCount > 0) {
            uint32_t excludeStart = firstIdx - excludeCount;
            uint32_t writeTimes = CeilDiv(excludeCount, scatterLength);
            uint32_t writeTail = excludeCount - (writeTimes - 1) * scatterLength;
            for (uint32_t wi = 0; wi < writeTimes; wi++) {
                uint32_t curLen = (wi == writeTimes - 1) ? writeTail : scatterLength;
                int64_t idxGmOff = sortedBase + excludeStart + wi * scatterLength;
                DataCopyPad(scatterIdxLocal, mGmSortedIndices_[idxGmOff],
                            {1, static_cast<uint32_t>(curLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToSSync();
                if constexpr (IsSameType<inputT, float>::value) {
                    scatterAlignedLocal.template ReinterpretCast<int32_t>().SetValue(0, FLOAT32_NEG_INF);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, FLOAT16_NEG_INF);
                } else {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, BF16_NEG_INF);
                }
                SToMTE3Sync();
                for (uint32_t ei = 0; ei < curLen; ei++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(ei);
                    DataCopyPad(mGmOut_[batchGmBase + lineIndex],
                                scatterAlignedLocal.template ReinterpretCast<outputT>(),
                                {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
                }
                MTE3ToSSync();
            }
        }
        CopyOutLast(batchGmBase, sortedBase);
    }
}

// --- ProcessTopKOpt: main entry for optimized TopK-only ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopKOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;

        int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);

        if (kValue <= 0 || static_cast<uint32_t>(kValue) >= vocabSize_) {
            uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
            for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                DataCopyPad(cmpLogitsLocal, mGmLogits_[gmOff],
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                VToMTE3Sync();
                DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal,
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
                MTE3ToMTE2Sync();
            }
            continue;
        }

        uint32_t kthStart = FindKthStartIdx(sortedBase, kValue);
        uint32_t actualTopKCount = vocabSize_ - kthStart;

        if (actualTopKCount <= scatterLength) {
            FillNegInfAndScatterTopK(batchGmBase, sortedBase, kthStart);
            continue;
        }
        int64_t kthIdx = sortedBase + kthStart;
        inputT kthValRaw = mGmSortedValue_[kthIdx].GetValue(0);

        CompareSelectOnLogits(batchGmBase, kthValRaw, true);
        CopyOutLast(batchGmBase, sortedBase);
    }
}

// --- ProcessTopPOpt: main entry for optimized TopP-only (with logits) ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopPOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;

        GetMaxValue(sortedBase);

        if (*reinterpret_cast<int32_t*>(&maxValue) == FLOAT32_NEG_INF) {
            uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
            for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                DataCopyPad(cmpLogitsLocal, mGmLogits_[gmOff],
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                VToMTE3Sync();
                DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal,
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
                MTE3ToMTE2Sync();
            }
            continue;
        }

        ComputeSoftmaxSum(sortedBase, 0, 0);
        WriteSoftmaxToGm(sortedBase, 0);
        CumsumKoggleStone(sortedBase);

        if constexpr (IsSameType<inputT, float>::value) {
            pValue = 1.0f - mGmP_[batchOffset_ + loopBatch].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            pValue = 1.0f - static_cast<float>(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        } else {
            pValue = 1.0f - ToFloat(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        }

        uint32_t firstIdx = vocabSize_;
        FindFirstIndex(sortedBase, firstIdx);
        if (firstIdx >= vocabSize_)
            firstIdx = vocabSize_ - 1;

        uint32_t survivorCount = vocabSize_ - firstIdx;

        if (survivorCount <= scatterLength) {
            constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ELEM_PER_REP = 8;
            constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
            constexpr uint32_t MAX_BRCB_REPEAT = 255;
            constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

            uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
            for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                if constexpr (IsSameType<inputT, float>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLen);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLen);
                } else {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLen);
                }
                VToMTE3Sync();
                DataCopyPad(mGmOut_[gmOff], cmpLogitsLocal,
                            {1, static_cast<uint32_t>(curLen * sizeof(inputT)), 0, 0, 0});
                MTE3ToMTE2Sync();
            }

            int64_t scatterGmStart = sortedBase + firstIdx;
            uint32_t scCopyTimes = CeilDiv(survivorCount, scatterLength);
            uint32_t scCopyTail = survivorCount - (scCopyTimes - 1) * scatterLength;
            for (uint32_t sci = 0; sci < scCopyTimes; sci++) {
                uint32_t curScLen = (sci == scCopyTimes - 1) ? scCopyTail : scatterLength;
                int64_t scGmOff = scatterGmStart + sci * scatterLength;
                DataCopyPad(scatterValLocal, mGmSortedValue_[scGmOff],
                            {1, static_cast<uint32_t>(curScLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(scatterIdxLocal, mGmSortedIndices_[scGmOff],
                            {1, static_cast<uint32_t>(curScLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                MTE2ToVSync();
                MTE2ToSSync();

                uint32_t totalRepeat = CeilDiv(curScLen, BRCB_ELEM_PER_REP);
                uint32_t brcbDone = 0;
                while (brcbDone < totalRepeat) {
                    uint32_t remaining = totalRepeat - brcbDone;
                    uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ? ALIGNED_MAX_REPEAT :
                                                                          (remaining + BRCB_ALIGN_REPEATS - 1) /
                                                                              BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
                    Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                         scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
                    brcbDone += curRepeat;
                }
                VToMTE3Sync();

                for (uint32_t idx = 0; idx < curScLen; idx++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(idx);
                    DataCopyPad(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK],
                                {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
                }
                MTE3ToSSync();
            }
        } else {
            inputT thresholdVal = mGmSortedValue_[sortedBase + firstIdx].GetValue(0);
            CompareSelectOnLogits(batchGmBase, thresholdVal, true);

            float threshFloat;
            if constexpr (IsSameType<inputT, float>::value) {
                threshFloat = thresholdVal;
            } else if constexpr (IsSameType<inputT, half>::value) {
                threshFloat = static_cast<float>(thresholdVal);
            } else {
                threshFloat = ToFloat(thresholdVal);
            }

            uint32_t excludeCount = 0;
            if (firstIdx > 0) {
                uint32_t searchPos = firstIdx;
                bool exDone = false;
                while (searchPos > 0 && !exDone) {
                    uint32_t chunkLen = (searchPos > scatterLength) ? scatterLength : searchPos;
                    uint32_t chunkStart = searchPos - chunkLen;
                    DataCopyPad(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart],
                                {1, static_cast<uint32_t>(chunkLen * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
                    MTE2ToSSync();
                    for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
                        float val;
                        if constexpr (IsSameType<inputT, float>::value) {
                            val = scatterValLocal.GetValue(i);
                        } else if constexpr (IsSameType<inputT, half>::value) {
                            val = static_cast<float>(scatterValLocal.GetValue(i));
                        } else {
                            val = ToFloat(scatterValLocal.GetValue(i));
                        }
                        if (val != threshFloat) {
                            exDone = true;
                            break;
                        }
                        excludeCount++;
                    }
                    searchPos = chunkStart;
                }
            }

            if (excludeCount > 0) {
                uint32_t excludeStart = firstIdx - excludeCount;
                uint32_t writeTimes = CeilDiv(excludeCount, scatterLength);
                uint32_t writeTail = excludeCount - (writeTimes - 1) * scatterLength;
                for (uint32_t wi = 0; wi < writeTimes; wi++) {
                    uint32_t curLen = (wi == writeTimes - 1) ? writeTail : scatterLength;
                    int64_t idxGmOff = sortedBase + excludeStart + wi * scatterLength;
                    DataCopyPad(scatterIdxLocal, mGmSortedIndices_[idxGmOff],
                                {1, static_cast<uint32_t>(curLen * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                    MTE2ToSSync();
                    if constexpr (IsSameType<inputT, float>::value) {
                        scatterAlignedLocal.template ReinterpretCast<int32_t>().SetValue(0, FLOAT32_NEG_INF);
                    } else if constexpr (IsSameType<inputT, half>::value) {
                        scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, FLOAT16_NEG_INF);
                    } else {
                        scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, BF16_NEG_INF);
                    }
                    SToMTE3Sync();
                    for (uint32_t ei = 0; ei < curLen; ei++) {
                        int32_t lineIndex = scatterIdxLocal.GetValue(ei);
                        DataCopyPad(mGmOut_[batchGmBase + lineIndex],
                                    scatterAlignedLocal.template ReinterpretCast<outputT>(),
                                    {1, (uint32_t)(sizeof(outputT)), 0, 0, 0});
                    }
                    MTE3ToSSync();
                }
            }
            CopyOutLast(batchGmBase, sortedBase);
        }
    }
}

} // namespace ApplyTopKTopPOptOp
#endif
