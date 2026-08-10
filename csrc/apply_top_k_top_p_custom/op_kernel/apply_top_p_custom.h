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
 * \file apply_top_p_custom.h
 * \brief
 */
#ifndef APPLY_TOP_P_WITH_SORTED_H_KERNEL
#define APPLY_TOP_P_WITH_SORTED_H_KERNEL

#include "kernel_operator.h"

using namespace AscendC;
namespace ApplyTopPCustomOp {
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;    // -inf 64512
constexpr uint16_t BF16_NEG_INF = 0xFF80;       // -inf 65408
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000; // -inf -2139095040

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
constexpr uint32_t DATA_PER_REPEAT_B32 = 64;
constexpr uint32_t SCATTER_PART_LENGTH = 1024;
constexpr uint32_t RESERVED_UB = 1024;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t SOFTMAX_UB_NUM = 2;

template <typename inputT, typename calT, typename outputT>
class ApplyTopPCustom {
public:
    __aicore__ inline ApplyTopPCustom(){};
    __aicore__ inline void InitTilingData(const ApplyTopKTopPCustomTilingData& __restrict tilingData,
                                          GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR p, GM_ADDR k,
                                          GM_ADDR logits, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void ProcessTopP();

private:
    __aicore__ inline void ReduceSumWithAddsAndExpImpl(uint32_t loopDataNum);
    // topp func
    __aicore__ inline void ProcessPreSingleBatch(uint32_t loopBatch);
    __aicore__ inline void GetSoftmaxSum(uint32_t loopBatch);
    __aicore__ inline void CumsumKoggleStone(uint32_t loopBatch);
    __aicore__ inline void CopyOutLastValue(uint32_t loopBatch);
    __aicore__ inline void GetPValue(uint32_t batchOffset);
    __aicore__ inline void CumsumParamCompute(uint32_t iterateTime);
    __aicore__ inline void GetMaxValue(int64_t baseGmIdx);
    __aicore__ inline void GetSoftMaxRes(uint32_t loopBatch);
    __aicore__ inline void ScatterSingleTask(uint32_t taskIndex);

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }
    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }
    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }
    __aicore__ inline void MTE3ToSSync()
    {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }
    __aicore__ inline void VToSSync()
    {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    __aicore__ inline void SToVSync()
    {
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
    }
    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    __aicore__ inline void MTE2ToSSync()
    {
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) { return y == 0 ? x : (x + y - 1) / y; }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> calBuf_;

    // tilingData
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t tailBatch_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t loopBatch_ = 0;
    uint32_t batchOffset_ = 0;
    uint32_t bufOffsetLoop = 0;
    int64_t baseGmIdx_ = 0;

    // topp scalar
    uint32_t maxSoftmaxLength = 1;
    uint32_t softmaxLength = 1;
    uint32_t lineSfLoopTimes = 1;
    uint32_t softmaxLengthTail = 1;
    uint32_t scatterLength = 1;

    uint32_t vCnt = 0;
    uint32_t bCnt = 0;
    uint32_t singleCoreV = 1;
    uint32_t singleCoreVTail = 1;
    uint32_t iterateTimes = 1;

    GlobalTensor<inputT> mGmSortedValue_;
    GlobalTensor<int32_t> mGmSortedIndices_;
    GlobalTensor<inputT> mGmP_;
    GlobalTensor<int32_t> mGmK_;
    GlobalTensor<outputT> mGmOut_;
    GlobalTensor<float> softMaxGm;

    LocalTensor<uint8_t> totalUb;

    // softmax tensor
    LocalTensor<float> softMaxLocalFp32;
    LocalTensor<inputT> softMaxLocal;
    LocalTensor<float> softMaxResLocal;
    LocalTensor<float> reduceLocal;
    LocalTensor<inputT> outInfLocal;

    // cumsum tensor
    LocalTensor<float> cumSumInput1Local;
    LocalTensor<float> cumSumInput2Local;

    // scatter tensor
    LocalTensor<inputT> sortedValueLocal;
    LocalTensor<int32_t> sortedIndicesLocal;
    LocalTensor<float> cumsumLocal;

    // batch scatter tensor: scatterAlignedLocal stores Brcb output for DataCopyPad
    LocalTensor<inputT> scatterAlignedLocal;

    float pValue = 0;
    float maxValue = 0;
    float reduceSumValueInvert = 0;
    float reduceSumValue = 0;
    BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
};

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::InitTilingData(
    const ApplyTopKTopPCustomTilingData& __restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR logits, GM_ADDR out, GM_ADDR workspace)
{
    batchSize_ = tilingData.batchSize;
    vocabSize_ = tilingData.vocabSize;
    batchPerCore_ = tilingData.batchPerCore;
    tailBatch_ = tilingData.tailBatch;
    blockNum_ = tilingData.blockNum;
    calUbSize_ = tilingData.calUbSize;
    iterateTimes = tilingData.iterateTimes;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ < tailBatch_) {
        loopBatch_ = batchPerCore_ + 1;
        batchOffset_ = blockIdx_ * loopBatch_;
    } else {
        loopBatch_ = batchPerCore_;
        batchOffset_ = blockIdx_ * batchPerCore_ + tailBatch_;
    }
    maxSoftmaxLength = (calUbSize_ - RESERVED_UB) / SOFTMAX_UB_NUM / FLOAT_BYTES;
    softmaxLength = maxSoftmaxLength < vocabSize_ ? maxSoftmaxLength : vocabSize_;

    lineSfLoopTimes = (vocabSize_ + softmaxLength - 1) / softmaxLength;
    softmaxLengthTail = vocabSize_ - (lineSfLoopTimes - 1) * softmaxLength;
    scatterLength = (calUbSize_ - RESERVED_UB - BLOCK_BYTES) / (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT)) /
                    SCATTER_PART_LENGTH * SCATTER_PART_LENGTH;
    vCnt = batchSize_ < blockNum_ ? blockNum_ / batchSize_ : 1;
    bCnt = batchSize_;
    singleCoreV = vocabSize_ / vCnt;
    singleCoreVTail = vocabSize_ - vCnt * singleCoreV;
    mGmSortedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(sorted_value));
    mGmSortedIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices));
    mGmP_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(p));
    mGmK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
    mGmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ outputT*>(out));
    softMaxGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize_ * vocabSize_);
}

// init used buffer
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::InitBuffer(TPipe* inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(calBuf_, calUbSize_);
    totalUb = calBuf_.Get<uint8_t>();
    // softmax ub
    uint32_t softmaxLengthAligned = CeilDiv(softmaxLength, BLOCK_BYTES / sizeof(inputT)) * BLOCK_BYTES / sizeof(inputT);
    softMaxLocalFp32 = totalUb.ReinterpretCast<float>();
    softMaxLocal = totalUb[softmaxLengthAligned * sizeof(inputT)].ReinterpretCast<inputT>();
    softMaxResLocal = totalUb[softmaxLengthAligned * sizeof(float)].ReinterpretCast<float>();
    reduceLocal = totalUb[softmaxLengthAligned * sizeof(float) * 2].ReinterpretCast<float>(); // 32 bytes
    outInfLocal = totalUb.ReinterpretCast<inputT>();                                          // Take softmax ub

    // cumsum ub
    cumSumInput1Local = totalUb.ReinterpretCast<float>();                                       // Take softmax local
    cumSumInput2Local = totalUb[softmaxLengthAligned * sizeof(float)].ReinterpretCast<float>(); // Take softmax res ub

    // scatter ub: layout = sortedValueLocal | sortedIndicesLocal | cumsumLocal | scatterAlignedLocal
    // scatterAlignedLocal stores Brcb output: each value in its own 32B-aligned datablock
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    scatterLength = (calUbSize_ - RESERVED_UB - BLOCK_BYTES) /
                    (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT) + BLOCK_BYTES) / SCATTER_PART_LENGTH *
                    SCATTER_PART_LENGTH;
    sortedValueLocal = totalUb[0].ReinterpretCast<inputT>();
    sortedIndicesLocal = totalUb[scatterLength * sizeof(inputT)].ReinterpretCast<int32_t>();
    cumsumLocal = totalUb[scatterLength * (FLOAT_BYTES + sizeof(inputT))].ReinterpretCast<float>();
    uint32_t alignedAreaOffset = scatterLength * (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT));
    scatterAlignedLocal = totalUb[alignedAreaOffset].ReinterpretCast<inputT>();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::GetMaxValue(int64_t baseGmIdx)
{
    int64_t initGmIdx = baseGmIdx + vocabSize_ - 1;
    if constexpr (IsSameType<inputT, float>::value) {
        maxValue = -mGmSortedValue_[initGmIdx].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        maxValue = -static_cast<float>(mGmSortedValue_[initGmIdx].GetValue(0));
    } else {
        maxValue = -ToFloat(mGmSortedValue_[initGmIdx].GetValue(0));
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::GetPValue(uint32_t batchOffset)
{
    if constexpr (IsSameType<inputT, float>::value) {
        pValue = float(1.0) - mGmP_[batchOffset].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        pValue = float(1.0) - static_cast<float>(mGmP_[batchOffset].GetValue(0));
    } else {
        pValue = float(1.0) - ToFloat(mGmP_[batchOffset].GetValue(0));
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::ProcessPreSingleBatch(uint32_t loopBatch)
{
    reduceSumValue = 0;
    GetSoftmaxSum(loopBatch);
    GetSoftMaxRes(loopBatch);
    CumsumKoggleStone(loopBatch);
    CopyOutLastValue(loopBatch);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::ProcessTopP()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        baseGmIdx_ = batchOffset_ * vocabSize_ + loopBatch * vocabSize_;
        GetMaxValue(baseGmIdx_); // Get max value in softmax.

        if (*reinterpret_cast<int32_t*>(&maxValue) == FLOAT32_NEG_INF) {
            uint32_t loopDataNum = softmaxLength;
            for (uint32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
                int64_t currentGmIdx = baseGmIdx_ + loopInner * softmaxLength;
                if (loopInner == (lineSfLoopTimes - 1))
                    loopDataNum = softmaxLengthTail;
                if constexpr (IsSameType<inputT, float>::value) {
                    Duplicate(outInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, loopDataNum);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    Duplicate(outInfLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, loopDataNum);
                } else {
                    Duplicate(outInfLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, loopDataNum);
                }
                VToMTE3Sync();
                DataCopyPad(mGmOut_[currentGmIdx], outInfLocal,
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0});
                MTE3ToMTE2Sync();
                Duplicate(softMaxResLocal, 2.0f, loopDataNum);
                VToMTE3Sync();
                DataCopyPad(softMaxGm[currentGmIdx], softMaxResLocal,
                            {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
                MTE3ToMTE2Sync();
            }
            CopyOutLastValue(loopBatch);
            continue;
        }

        ProcessPreSingleBatch(loopBatch); // Softmax and cumsum.
    }
    SyncAll();
    for (uint32_t batchIdx = blockIdx_; batchIdx < bCnt; batchIdx += blockNum_) {
        ScatterSingleTask(batchIdx);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::ScatterSingleTask(uint32_t batchIdx)
{
    uint32_t copyTimes = CeilDiv(vocabSize_, scatterLength);
    uint32_t copyLength = scatterLength;
    uint32_t copyLengthTail = vocabSize_ - (copyTimes - 1) * scatterLength;
    GetPValue(batchIdx);
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);

    bool thresholdCrossed = false;
    for (uint32_t cpIndex = 0; cpIndex < copyTimes; cpIndex++) {
        uint32_t curCopyLength = cpIndex == (copyTimes - 1) ? copyLengthTail : copyLength;
        int64_t gmOffset = static_cast<int64_t>(batchIdx) * vocabSize_ + cpIndex * copyLength;
        if (!thresholdCrossed) {
            DataCopyPad(cumsumLocal, softMaxGm[gmOffset],
                        {1, static_cast<uint32_t>(curCopyLength * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToSSync();
            if (cumsumLocal.GetValue(curCopyLength - 1) <= pValue) {
                continue;
            }
        }
        DataCopyPad(sortedIndicesLocal, mGmSortedIndices_[gmOffset],
                    {1, static_cast<uint32_t>(curCopyLength * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sortedValueLocal, mGmSortedValue_[gmOffset],
                    {1, static_cast<uint32_t>(curCopyLength * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
        MTE2ToVSync();
        MTE2ToSSync();

        uint32_t startIdx = 0;
        if (!thresholdCrossed) {
            uint32_t scatterLoop = CeilDiv(curCopyLength, SCATTER_PART_LENGTH);
            uint32_t scatterNumsTail = curCopyLength - (scatterLoop - 1) * SCATTER_PART_LENGTH;
            for (uint32_t scatterLoopIndex = 0; scatterLoopIndex < scatterLoop; scatterLoopIndex++) {
                uint32_t curScatterNums = scatterLoopIndex == (scatterLoop - 1) ? scatterNumsTail : SCATTER_PART_LENGTH;
                uint32_t blockEndIdx = scatterLoopIndex * SCATTER_PART_LENGTH + curScatterNums - 1;
                if (cumsumLocal.GetValue(blockEndIdx) <= pValue) {
                    continue;
                }
                for (uint32_t i = 0; i < curScatterNums; i++) {
                    uint32_t offset = scatterLoopIndex * SCATTER_PART_LENGTH + i;
                    if (cumsumLocal.GetValue(offset) > pValue) {
                        startIdx = offset;
                        thresholdCrossed = true;
                        break;
                    }
                }
                if (thresholdCrossed) {
                    break;
                }
            }
            if (!thresholdCrossed) {
                continue;
            }
        }

        uint32_t collectCount = curCopyLength - startIdx;
        constexpr uint32_t BRCB_ELEM_PER_REP = 8;
        constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
        constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
        constexpr uint32_t MAX_BRCB_REPEAT = 255;
        constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;
        uint32_t brcbAlignedStart = (startIdx / BRCB_SRC_ALIGN) * BRCB_SRC_ALIGN;
        uint32_t brcbSrcCount = collectCount + (startIdx - brcbAlignedStart);
        uint32_t totalRepeat = CeilDiv(brcbSrcCount, BRCB_ELEM_PER_REP);
        MTE3ToVSync();

        uint32_t brcbDone = 0;
        while (brcbDone < totalRepeat) {
            uint32_t remaining = totalRepeat - brcbDone;
            uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ?
                                     ALIGNED_MAX_REPEAT :
                                     (remaining + BRCB_ALIGN_REPEATS - 1) / BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
            Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                 sortedValueLocal[brcbAlignedStart + brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
            brcbDone += curRepeat;
        }
        VToMTE3Sync();

        uint32_t brcbValueOffset = startIdx - brcbAlignedStart;
        for (uint32_t idx = 0; idx < collectCount; idx++) {
            int32_t lineIndex = sortedIndicesLocal.GetValue(startIdx + idx);
            DataCopyPad(mGmOut_[batchIdx * vocabSize_ + lineIndex],
                        scatterAlignedLocal[(brcbValueOffset + idx) * DATA_PER_BLOCK],
                        {1, (uint32_t)(1 * sizeof(outputT)), 0, 0, 0});
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::GetSoftMaxRes(uint32_t loopBatch)
{
    uint32_t loopDataNum = softmaxLength;
    for (int32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * softmaxLength;
        if (loopInner == (lineSfLoopTimes - 1)) {
            loopDataNum = softmaxLengthTail;
        }
        if constexpr (!IsSameType<inputT, float>::value) {
            DataCopyPad(softMaxLocal, mGmSortedValue_[currentGmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Cast(softMaxLocalFp32, softMaxLocal, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(softMaxLocalFp32, mGmSortedValue_[currentGmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
        }
        Adds(softMaxResLocal, softMaxLocalFp32, maxValue, loopDataNum);
        VToMTE2Sync();
        PipeBarrier<PIPE_V>();
        Exp(softMaxResLocal, softMaxResLocal, loopDataNum);
        PipeBarrier<PIPE_V>();
        Muls(softMaxResLocal, softMaxResLocal, reduceSumValueInvert, loopDataNum);
        VToMTE3Sync();
        DataCopyPad(softMaxGm[currentGmIdx], softMaxResLocal,
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
        MTE3ToMTE2Sync();
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::CumsumKoggleStone(uint32_t loopBatch)
{
    int64_t iteratOffset = 1; // 2^0 = 1, doubles each iteration
    for (uint32_t iterateTime = 0; iterateTime < iterateTimes; iterateTime++) {
        uint32_t addLength = vocabSize_ - static_cast<uint32_t>(iteratOffset);
        uint32_t innerLoopNum = addLength / softmaxLength;
        uint32_t dataTail = addLength - innerLoopNum * softmaxLength;
        uint32_t loopDataNum = softmaxLength;
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
            // Copy data from right
            int64_t loopInnerOffset = dataTail + (innerLoopNum - 1 - innerLoopIdx) * softmaxLength;
            DataCopyPad(cumSumInput1Local, softMaxGm[baseGmIdx_ + loopInnerOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(cumSumInput2Local, softMaxGm[baseGmIdx_ + loopInnerOffset + iteratOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Add(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, loopDataNum);
            VToMTE3Sync();
            DataCopyPad(softMaxGm[baseGmIdx_ + loopInnerOffset + iteratOffset], cumSumInput1Local,
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
            MTE3ToMTE2Sync();
        }
        if (dataTail > 0) {
            loopDataNum = dataTail;
            DataCopyPad(cumSumInput1Local, softMaxGm[baseGmIdx_],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(cumSumInput2Local, softMaxGm[baseGmIdx_ + iteratOffset],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Add(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, loopDataNum);
            VToMTE3Sync();
            DataCopyPad(softMaxGm[baseGmIdx_ + iteratOffset], cumSumInput1Local,
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(float)), 0, 0, 0});
            MTE3ToMTE2Sync();
        }
        iteratOffset *= 2; // 2^(iterateTime+1)
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::CopyOutLastValue(uint32_t loopBatch)
{
    int64_t lastValueOffset = (static_cast<int64_t>(batchOffset_) + loopBatch + 1) * vocabSize_ - 1;
    DataCopyPad(scatterAlignedLocal, mGmSortedValue_[lastValueOffset],
                {1, static_cast<uint32_t>(sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
    int32_t lineIndex = mGmSortedIndices_.GetValue(lastValueOffset);
    MTE2ToSSync();
    DataCopyPad(mGmOut_[(static_cast<int64_t>(batchOffset_) + loopBatch) * vocabSize_ + lineIndex],
                scatterAlignedLocal.template ReinterpretCast<outputT>(), {1, (uint32_t)(1 * sizeof(outputT)), 0, 0, 0});
    MTE3ToVSync();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::ReduceSumWithAddsAndExpImpl(uint32_t loopDataNum)
{
    Adds(softMaxResLocal, softMaxLocalFp32, maxValue, loopDataNum);
    PipeBarrier<PIPE_V>();
    Exp(softMaxResLocal, softMaxResLocal, loopDataNum);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceLocal, softMaxResLocal, reduceLocal, loopDataNum);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopPCustom<inputT, calT, outputT>::GetSoftmaxSum(uint32_t loopBatch)
{
    uint32_t loopDataNum = softmaxLength;
    for (int32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * softmaxLength;
        if (loopInner == (lineSfLoopTimes - 1)) {
            loopDataNum = softmaxLengthTail;
        }
        if constexpr (IsSameType<inputT, float>::value) {
            Duplicate(outInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, loopDataNum);
        } else if constexpr (IsSameType<inputT, half>::value) {
            Duplicate(outInfLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, loopDataNum);
        } else {
            Duplicate(outInfLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, loopDataNum);
        }
        VToMTE3Sync();
        DataCopyPad(mGmOut_[currentGmIdx], outInfLocal,
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0});
        MTE3ToMTE2Sync();
        if constexpr (!IsSameType<inputT, float>::value) {
            DataCopyPad(softMaxLocal, mGmSortedValue_[currentGmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
            Cast(softMaxLocalFp32, softMaxLocal, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(softMaxLocalFp32, mGmSortedValue_[currentGmIdx],
                        {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0}, {false, 0, 0, 0});
            MTE2ToVSync();
        }

        ReduceSumWithAddsAndExpImpl(loopDataNum);
        VToSSync();
        reduceSumValue += reduceLocal.GetValue(0);
        SToVSync();
    }
    reduceSumValueInvert = 1 / reduceSumValue;
    SToVSync();
}
} // namespace ApplyTopPCustomOp

#endif // APPLY_TOP_P_WITH_SORTED_H_KERNEL
