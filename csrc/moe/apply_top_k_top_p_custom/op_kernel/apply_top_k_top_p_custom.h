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
 * \file apply_top_k_top_p_custom.h
 * \brief
 */
#ifndef APPLY_TOP_K_TOP_P_CUSTOM_H_KERNEL
#define APPLY_TOP_K_TOP_P_CUSTOM_H_KERNEL

#include "kernel_operator.h"

using namespace AscendC;
namespace ApplyTopKTopPCustomOp {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00; // -inf 64512
constexpr uint16_t BF16_NEG_INF = 0xFF80; // -inf 65408
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000; // -inf -2139095040

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
constexpr uint32_t DATA_PER_REPEAT_B32 = 64;
constexpr uint32_t K_MAX = 1024;
constexpr uint64_t MASK_64 = 64;
constexpr CumSumConfig CUMSUM_CONFIG{true, true, false};

template <typename inputT, typename calT, typename outputT>
class ApplyTopKTopPCustom {
public:
    __aicore__ inline ApplyTopKTopPCustom(){};
    __aicore__ inline void InitTilingData(
        const ApplyTopKTopPCustomTilingData &__restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
        GM_ADDR p, GM_ADDR k, GM_ADDR out);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessTopK();
private:
    __aicore__ inline void InitCopyIn(uint32_t loopBatch, int64_t currentGmIdx);
    __aicore__ inline void InitProcess(uint32_t loopBatch);
    __aicore__ inline void ProcessKLtKMax(uint32_t loopBatch);
    __aicore__ inline void ScatterCumtomImpl(uint32_t loopBatch, uint32_t loopProbNum, uint32_t offset);
    __aicore__ inline void ProcessRemain(uint32_t loopBatch);
    __aicore__ inline void GetKthResult(uint32_t loopBatch, uint32_t offset, uint8_t repeatTimes);
    __aicore__ inline void GetFirstKLoop(uint32_t loopBatch, int32_t &firstKLoop);
    __aicore__ inline void ScatterFromFirstKLoop(uint32_t loopBatch, int32_t firstKLoop, float &cumsumData);
    __aicore__ inline void ReduceSumWithAddsAndExpImpl(uint32_t offset, uint32_t loopDataNum);
    __aicore__ inline void CumSumWithAddsAndExpImpl(
        uint32_t offset, uint32_t loopDataNum, uint32_t cumsumInner, float cumsumData);
    // topk func
    __aicore__ inline void InitProcessTopK(uint32_t loopBatch);
    __aicore__ inline void ProcessKLtKMaxTopK(uint32_t loopBatch);
    __aicore__ inline void ProcessRemainTopK(uint32_t loopBatch);
    __aicore__ inline void GetFirstKLoopTopK(uint32_t loopBatch, int32_t &firstKLoop);
    __aicore__ inline void ScatterFromFirstKLoopTopK(uint32_t loopBatch, int32_t firstKLoop);
    __aicore__ inline void ScatterCumtomImplTopK(uint32_t loopBatch, uint32_t loopProbNum, uint32_t offset);
    __aicore__ inline void SToMTE3Sync() {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }
    __aicore__ inline void MTE3ToSSync() {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }
    __aicore__ inline void VToSSync() {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    __aicore__ inline void MTE2ToVSync() {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    __aicore__ inline void MTE2ToSSync() {
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    }
    __aicore__ inline void MTE3ToVSync() {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    __aicore__ inline void SToMTE2Sync()
    {
        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    }
    __aicore__ inline void VToMTE3Sync() {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }
private:
    TPipe *pipe_;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> sortedValueInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sortedIndicesInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> pInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> kInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> calBuf_;

    // tilingData
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t tailBatch_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t dataNumInit_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t ubFactorElementAligned_ = 0;
    uint32_t tailUbFactorElement_ = 0;
    uint32_t tailUbFactorElementAligned_ = 0;
    uint32_t calUbSize_ = 0;

    uint32_t blockIdx_ = 0;
    uint32_t loopBatch_ = 0;
    uint32_t batchOffset_ = 0;
    uint32_t bufOffsetLoop = 0;
    uint32_t loopInner_ = 0;
    uint32_t loopInnerOnlyP_ = 0;
    int64_t baseGmIdx_ = 0;

    GlobalTensor<inputT> mGmSortedValue_;
    GlobalTensor<int32_t> mGmSortedIndices_;
    GlobalTensor<inputT> mGmP_;
    GlobalTensor<int32_t> mGmK_;
    GlobalTensor<outputT> mGmOut_;

    LocalTensor<int32_t> kLocal;
    LocalTensor<inputT> pLocal;
    LocalTensor<outputT> outTensor;
    LocalTensor<inputT> sortedValueLocal;
    LocalTensor<int32_t> sortedIndicesLocal;

    LocalTensor<float> sortedValueLocalFp32;
    LocalTensor<float> negInfLocal;

    LocalTensor<float> calLocalFp32;
    LocalTensor<float> kthValueLocal;
    LocalTensor<float> tmpLocal;
    LocalTensor<float> cumSumRes;
    LocalTensor<float> cumSumTmp;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> softMaxRes;
    LocalTensor<inputT> scatterTensor;
    LocalTensor<uint8_t> sharedTmpBuffer;

    float kthValue = 0;
    float pValue = 0;
    float maxValue = 0;
    float reduceSumValueInvert = 0;
    float reduceSumValue = 0;
    inputT kthTopKValue = 0;
    BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
    DataCopyExtParams scatterCopyParams{1, (uint32_t)(sizeof(outputT)), 0, 0, 0};
};

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::InitTilingData(
    const ApplyTopKTopPCustomTilingData &__restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out) {
    batchSize_ = tilingData.batchSize;
    vocabSize_ = tilingData.vocabSize;
    batchPerCore_ = tilingData.batchPerCore;
    tailBatch_ = tilingData.tailBatch;
    blockNum_ = tilingData.blockNum;
    dataNumInit_ = tilingData.dataNumInit;
    dataNumInitAligned_ = AscendC::AlignUp(dataNumInit_, DATA_PER_BLOCK_B32);
    ubFactorElement_ = tilingData.ubFactorElement;
    ubFactorElementAligned_ = tilingData.ubFactorElementAligned;
    tailUbFactorElement_ = tilingData.tailUbFactorElement;
    tailUbFactorElementAligned_ = tilingData.tailUbFactorElementAligned;
    calUbSize_ = tilingData.calUbSize;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ < tailBatch_)
    {
        loopBatch_ = batchPerCore_ + 1;
        batchOffset_ = blockIdx_ * loopBatch_;
    }
    else
    {
        loopBatch_ = batchPerCore_;
        batchOffset_ = blockIdx_ * batchPerCore_ + tailBatch_;
    }
    loopInner_ = (vocabSize_ - dataNumInit_ + ubFactorElementAligned_ - 1) / ubFactorElementAligned_;
    loopInnerOnlyP_ = (vocabSize_ + ubFactorElementAligned_ - 1) / ubFactorElementAligned_;
    mGmSortedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(sorted_value));
    mGmSortedIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sorted_indices));
    mGmP_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(p));
    mGmK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(k));
    mGmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ outputT *>(out));
}

// init used buffer
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::InitBuffer(TPipe *inputPipe) {
    pipe_ = inputPipe;
    pipe_->InitBuffer(sortedValueInQueue_, BUFFER_NUM, sizeof(inputT) * (ubFactorElementAligned_ + K_MAX));
    pipe_->InitBuffer(sortedIndicesInQueue_, BUFFER_NUM, sizeof(int32_t) * (ubFactorElementAligned_ + K_MAX));
    pipe_->InitBuffer(pInQueue_, BUFFER_NUM, BLOCK_BYTES);
    pipe_->InitBuffer(kInQueue_, BUFFER_NUM, BLOCK_BYTES);
    pipe_->InitBuffer(outQueue_, BUFFER_NUM, sizeof(outputT) * ubFactorElementAligned_);
    pipe_->InitBuffer(calBuf_, calUbSize_);
    if constexpr (!IsSameType<inputT, float>::value) {
        sortedValueLocalFp32 = calBuf_.GetWithOffset<float>(ubFactorElementAligned_ + K_MAX, bufOffsetLoop);
        bufOffsetLoop = bufOffsetLoop + (ubFactorElementAligned_ + K_MAX) * sizeof(float);
    }
    kthValueLocal = calBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + BLOCK_BYTES;

    negInfLocal = calBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + BLOCK_BYTES;

    tmpLocal = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    cumSumRes = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    cumSumTmp = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    reduceLocal = calBuf_.GetWithOffset<float>(ubFactorElementAligned_ * BLOCK_BYTES, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * BLOCK_BYTES * sizeof(float);

    softMaxRes = tmpLocal.template ReinterpretCast<float>();
    scatterTensor = reduceLocal.template ReinterpretCast<inputT>();
    sharedTmpBuffer = reduceLocal.template ReinterpretCast<uint8_t>();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::Process() {
    kLocal = kInQueue_.AllocTensor<int32_t>();
    pLocal = pInQueue_.AllocTensor<inputT>();
    outTensor = outQueue_.AllocTensor<outputT>();
    sortedValueLocal = sortedValueInQueue_.AllocTensor<inputT>();
    sortedIndicesLocal = sortedIndicesInQueue_.AllocTensor<int32_t>();
    Duplicate(negInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, DATA_PER_BLOCK_B32);
    if constexpr (IsSameType<inputT, float>::value) {
        calLocalFp32 = sortedValueLocal;
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, ubFactorElementAligned_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, ubFactorElementAligned_);
    } else {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, ubFactorElementAligned_);
    }
    VToMTE3Sync();
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        baseGmIdx_ = batchOffset_ * vocabSize_ + loopBatch * vocabSize_;
        InitProcess(loopBatch);
        if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
            ProcessKLtKMax(loopBatch);
        } else {
            ProcessRemain(loopBatch);
        }
    }
    kInQueue_.FreeTensor(kLocal);
    pInQueue_.FreeTensor(pLocal);
    sortedValueInQueue_.FreeTensor(sortedValueLocal);
    sortedIndicesInQueue_.FreeTensor(sortedIndicesLocal);
    outQueue_.FreeTensor(outTensor);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::InitCopyIn(uint32_t loopBatch,
    int64_t currentGmIdx) {
    DataCopyPad(mGmOut_[currentGmIdx], outTensor, {1, (uint32_t)(dataNumInit_ * sizeof(outputT)), 0, 0, 0});
    DataCopyPad(sortedValueLocal[ubFactorElementAligned_], mGmSortedValue_[currentGmIdx],
                {1, static_cast<uint32_t>(dataNumInit_ * sizeof(inputT)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(pLocal, mGmP_[batchOffset_ + loopBatch], {1, static_cast<uint32_t>(sizeof(inputT)), 0, 0, 0},
                {false, 0, 0, 0});
    if constexpr (!IsSameType<inputT, float>::value) {
        MTE2ToVSync();
        Cast(sortedValueLocalFp32[ubFactorElementAligned_], sortedValueLocal[ubFactorElementAligned_],
             RoundMode::CAST_NONE, dataNumInit_);
        Cast(tmpLocal, pLocal, RoundMode::CAST_NONE, DATA_PER_BLOCK_B32);
    }
    DataCopyPad(sortedIndicesLocal[ubFactorElementAligned_], mGmSortedIndices_[currentGmIdx],
                {1, static_cast<uint32_t>(dataNumInit_ * sizeof(int32_t)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(kLocal, mGmK_[batchOffset_ + loopBatch], {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                {false, 0, 0, 0});
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::GetKthResult(uint32_t loopBatch,
    uint32_t offset, uint8_t repeatTimes){
    Compare(tmpLocal.template ReinterpretCast<uint8_t>(), kthValueLocal, calLocalFp32[offset],
            CMPMODE::GT, MASK_64, repeatTimes, repeatParams);
    PipeBarrier<PIPE_V>();
    Select(calLocalFp32[offset], tmpLocal.template ReinterpretCast<uint8_t>(),
           negInfLocal, calLocalFp32[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE, MASK_64,
           repeatTimes, repeatParams);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ReduceSumWithAddsAndExpImpl(uint32_t offset,
    uint32_t loopDataNum) {
    Adds(softMaxRes, calLocalFp32[offset], maxValue, loopDataNum);
    PipeBarrier<PIPE_V>();
    Exp(softMaxRes, softMaxRes, loopDataNum);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceLocal, softMaxRes, reduceLocal, loopDataNum);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::InitProcess(uint32_t loopBatch) {
    int64_t initGmIdx = baseGmIdx_ + vocabSize_ - dataNumInit_;
    InitCopyIn(loopBatch, initGmIdx);
    MTE2ToSSync();

    int32_t kValue = kLocal.GetValue(0);
    if constexpr (IsSameType<inputT, float>::value) {
        pValue = float(1.0) - pLocal.GetValue(0);
    } else {
        pValue = float(1.0) - tmpLocal.GetValue(0);
    }
    maxValue = -calLocalFp32[ubFactorElementAligned_].GetValue(dataNumInit_ - 1);
    if constexpr (IsSameType<inputT, float>::value) {
        kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    } else {
        kthValue = ToFloat(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    }

    Duplicate(kthValueLocal, kthValue, 8);
    PipeBarrier<PIPE_V>();

    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    GetKthResult(loopBatch, ubFactorElementAligned_, repeatTimes);
    PipeBarrier<PIPE_V>();
    DataCopyExtParams copyParams{1, (uint32_t)(ubFactorElementAligned_ * sizeof(outputT)), 0, 0, 0};

    ReduceSumWithAddsAndExpImpl(ubFactorElementAligned_, dataNumInit_);
    VToSSync();
    reduceSumValue = reduceLocal.GetValue(0);
    reduceSumValueInvert = 1 / reduceSumValue;
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ProcessKLtKMax(uint32_t loopBatch) {
    DataCopyExtParams copyParams{1, (uint32_t)(ubFactorElementAligned_ * sizeof(outputT)), 0, 0, 0};
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdxInner = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == loopInner_ - 1) {
            DataCopyPad(mGmOut_[currentGmIdxInner], outTensor,
                        {1, (uint32_t)(tailUbFactorElement_ * sizeof(outputT)), 0, 0, 0});
        } else {
            DataCopyPad(mGmOut_[currentGmIdxInner], outTensor, copyParams);
        }
    }
    Muls(softMaxRes, softMaxRes, reduceSumValueInvert, dataNumInit_);
    PipeBarrier<PIPE_V>();
    const CumSumInfo cumSumInfo{1, dataNumInitAligned_};
    CumSum<float, CUMSUM_CONFIG>(cumSumRes, cumSumTmp, softMaxRes, sharedTmpBuffer, cumSumInfo);
    VToSSync();
    int32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    SToMTE3Sync();
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex], scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
    MTE3ToSSync();
    loopProb = loopProb - 1;
    for (; loopProb >= 0; loopProb--) {
        float cumsumData = cumSumRes.GetValue(loopProb);
        if (cumsumData <= pValue) {
            break;
        }
        scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
        gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
        SToMTE3Sync();
        DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex],
                    scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
        MTE3ToSSync();
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ScatterCumtomImpl(uint32_t loopBatch,
    uint32_t loopProbNum, uint32_t offset) {
    for (int32_t loopProb = 0; loopProb < static_cast<int32_t>(loopProbNum); loopProb++) {
        float cumsumDataTmp = cumSumRes.GetValue(loopProb);
        if (cumsumDataTmp <= pValue) {
            continue;
        }
        scatterTensor.SetValue(0, sortedValueLocal[offset].GetValue(loopProb));
        int32_t gmIndex = sortedIndicesLocal[offset].GetValue(loopProb);
        SToMTE3Sync();
        DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex], scatterTensor.template ReinterpretCast<outputT>(),
                    {1, (uint32_t)(1 * sizeof(outputT)), 0, 0, 0});
        MTE3ToSSync();
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::GetFirstKLoop(uint32_t loopBatch,
    int32_t &firstKLoop) {
    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    uint32_t loopDataNum = ubFactorElementAligned_;
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = ((tailUbFactorElement_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
        }
        DataCopyPad(mGmOut_[currentGmIdx], outTensor, {1, (uint32_t)(loopDataNum * sizeof(outputT)), 0, 0, 0});
        DataCopyPad(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0},
                    {false, 0, 0, 0});
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            VToSSync();
        } else {
            MTE2ToSSync();
        }
        if (calLocalFp32.GetValue(loopDataNum - 1) < kthValue) {
            firstKLoop += 1;
            continue;
        }

        GetKthResult(loopBatch, 0, repeatTimes);
        PipeBarrier<PIPE_V>();

        ReduceSumWithAddsAndExpImpl(0, loopDataNum);
        VToSSync();
        reduceSumValue += reduceLocal.GetValue(0);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::CumSumWithAddsAndExpImpl(uint32_t offset,
    uint32_t loopDataNum, uint32_t cumsumInner, float cumsumData) {
    Adds(softMaxRes, calLocalFp32[offset], maxValue, loopDataNum);
    PipeBarrier<PIPE_V>();
    Exp(softMaxRes, softMaxRes, loopDataNum);
    PipeBarrier<PIPE_V>();
    Muls(softMaxRes, softMaxRes, reduceSumValueInvert, loopDataNum);
    PipeBarrier<PIPE_V>();
    const CumSumInfo cumSumInfo{1, cumsumInner};
    CumSum<float, CUMSUM_CONFIG>(cumSumRes, cumSumTmp, softMaxRes, sharedTmpBuffer, cumSumInfo);
    PipeBarrier<PIPE_V>();
    Adds(cumSumRes, cumSumRes, cumsumData, loopDataNum);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ProcessRemain(uint32_t loopBatch) {
    int32_t firstKLoop = 0;
    GetFirstKLoop(loopBatch, firstKLoop);
    reduceSumValueInvert = 1 / reduceSumValue;
    float cumsumData = 0;
    ScatterFromFirstKLoop(loopBatch, firstKLoop, cumsumData);
    uint32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    SToMTE3Sync();
    DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex],
                scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
    MTE3ToVSync();
    CumSumWithAddsAndExpImpl(ubFactorElementAligned_, dataNumInit_, dataNumInitAligned_, cumsumData);
    VToSSync();
    ScatterCumtomImpl(loopBatch, dataNumInit_ - 1, ubFactorElementAligned_);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ScatterFromFirstKLoop(uint32_t loopBatch,
    int32_t firstKLoop, float &cumsumData) {
    uint32_t loopDataNum = ubFactorElementAligned_;
    uint32_t cumsumInner = ubFactorElementAligned_;
    uint8_t repeatTimes = ((ubFactorElementAligned_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    for (int32_t loopInner = firstKLoop; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = (tailUbFactorElement_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
            cumsumInner = tailUbFactorElementAligned_;
        }
        DataCopyPad(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(sortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(int32_t)), 0, 0, 0},
                    {false, 0, 0, 0});
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            MTE2ToVSync();
        }
        GetKthResult(loopBatch, 0, repeatTimes);
        PipeBarrier<PIPE_V>();
        CumSumWithAddsAndExpImpl(0, loopDataNum, cumsumInner, cumsumData);
        VToSSync();
        float cumsumDataTmp = cumSumRes.GetValue(loopDataNum - 1);
        cumsumData = cumsumDataTmp;
        if (cumsumDataTmp <= pValue) {
            continue;
        }
        ScatterCumtomImpl(loopBatch, loopDataNum, 0);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ProcessTopK() {
    kLocal = kInQueue_.AllocTensor<int32_t>();
    outTensor = outQueue_.AllocTensor<outputT>();
    sortedValueLocal = sortedValueInQueue_.AllocTensor<inputT>();
    sortedIndicesLocal = sortedIndicesInQueue_.AllocTensor<int32_t>();
    Duplicate(negInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, DATA_PER_BLOCK_B32);
    if constexpr (IsSameType<inputT, float>::value) {
        calLocalFp32 = sortedValueLocal;
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, ubFactorElementAligned_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, ubFactorElementAligned_);
    } else {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, ubFactorElementAligned_);
    }
    VToMTE3Sync();
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        baseGmIdx_ = batchOffset_ * vocabSize_ + loopBatch * vocabSize_;
        InitProcessTopK(loopBatch);
        /* The difference lies in that for the max branch, some data is less than the kthvalue,
        so part of the data can be filtered out in advance;
        while for the remain branch, all data must undergo the topk calculation.*/
        if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
            ProcessKLtKMaxTopK(loopBatch);
        } else {
            ProcessRemainTopK(loopBatch);
        }
    }
    kInQueue_.FreeTensor(kLocal);
    sortedValueInQueue_.FreeTensor(sortedValueLocal);
    sortedIndicesInQueue_.FreeTensor(sortedIndicesLocal);
    outQueue_.FreeTensor(outTensor);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ProcessRemainTopK(uint32_t loopBatch) {
    int32_t firstKLoop = 0;
    GetFirstKLoopTopK(loopBatch, firstKLoop);
    // Start the scatter calculation from the first loop in the row where the value is ≥ kthValue.
    ScatterFromFirstKLoopTopK(loopBatch, firstKLoop);
    /* Perform scatter calculation on the maximum number of ubFactorElementAligned_,
       which does not overlap with the previous ones.*/
    uint32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    SToMTE3Sync();
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex],
                scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
    MTE3ToSSync();
    ScatterCumtomImplTopK(loopBatch, dataNumInit_ - 1, ubFactorElementAligned_);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::GetFirstKLoopTopK(uint32_t loopBatch,
    int32_t &firstKLoop) {
    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    uint32_t loopDataNum = ubFactorElementAligned_;
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = ((tailUbFactorElement_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
        }
        DataCopyPad(mGmOut_[currentGmIdx], outTensor, {1, (uint32_t)(loopDataNum * sizeof(outputT)), 0, 0, 0});
        DataCopyPad(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0},
                    {false, 0, 0, 0});
        MTE2ToSSync();
        float rightVlaue = 0;
        // Make a judgment on the rightmost value of each loop to filter the data.
        if constexpr (IsSameType<inputT, bfloat16_t>::value) {
            rightVlaue = ToFloat(sortedValueLocal.GetValue(loopDataNum - 1));
        } else {
            rightVlaue = static_cast<float>(sortedValueLocal.GetValue(loopDataNum - 1));
        }
        SToMTE2Sync();
        if (rightVlaue < kthValue) {
            firstKLoop += 1;
            continue;
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ScatterFromFirstKLoopTopK(uint32_t loopBatch,
    int32_t firstKLoop) {
    uint32_t loopDataNum = ubFactorElementAligned_;
    uint32_t cumsumInner = ubFactorElementAligned_;
    uint8_t repeatTimes = ((ubFactorElementAligned_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    for (int32_t loopInner = firstKLoop; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = (tailUbFactorElement_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
            cumsumInner = tailUbFactorElementAligned_;
        }
        DataCopyPad(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(inputT)), 0, 0, 0},
                    {false, 0, 0, 0});
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            VToSSync();
        }
        DataCopyPad(sortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                    {1, static_cast<uint32_t>(loopDataNum * sizeof(int32_t)), 0, 0, 0},
                    {false, 0, 0, 0});
        MTE2ToSSync();
        ScatterCumtomImplTopK(loopBatch, loopDataNum, 0);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ScatterCumtomImplTopK(uint32_t loopBatch,
    uint32_t loopProbNum, uint32_t offset) {
    // Reverse traversal, returning early to improve performance.
    for (int32_t loopProb = static_cast<int32_t>(loopProbNum) - 1; loopProb >= 0; loopProb--) {
        float curValue = calLocalFp32[offset].GetValue(loopProb);
        if (curValue < kthValue) {
            break;
        }
        scatterTensor.SetValue(0, sortedValueLocal[offset].GetValue(loopProb));
        int32_t gmIndex = sortedIndicesLocal[offset].GetValue(loopProb);
        SToMTE3Sync();
        DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex], scatterTensor.template ReinterpretCast<outputT>(),
                    {1, (uint32_t)(1 * sizeof(outputT)), 0, 0, 0});
        MTE3ToSSync();
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::InitProcessTopK(uint32_t loopBatch) {
    int64_t initGmIdx = baseGmIdx_ + vocabSize_ - dataNumInit_;
    DataCopyPad(mGmOut_[initGmIdx], outTensor, {1, (uint32_t)(dataNumInit_ * sizeof(outputT)), 0, 0, 0});
    DataCopyPad(sortedValueLocal[ubFactorElementAligned_], mGmSortedValue_[initGmIdx],
                {1, static_cast<uint32_t>(dataNumInit_ * sizeof(inputT)), 0, 0, 0},
                {false, 0, 0, 0});
    if constexpr (!IsSameType<inputT, float>::value) {
        MTE2ToVSync();
        Cast(sortedValueLocalFp32[ubFactorElementAligned_], sortedValueLocal[ubFactorElementAligned_],
             RoundMode::CAST_NONE, dataNumInit_);
    }
    DataCopyPad(sortedIndicesLocal[ubFactorElementAligned_], mGmSortedIndices_[initGmIdx],
                {1, static_cast<uint32_t>(dataNumInit_ * sizeof(int32_t)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(kLocal, mGmK_[batchOffset_ + loopBatch], {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                {false, 0, 0, 0});
    MTE2ToSSync();
    int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);
    maxValue = -calLocalFp32[ubFactorElementAligned_].GetValue(dataNumInit_ - 1);
    if constexpr (IsSameType<inputT, float>::value) {
        kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    } else {
        kthValue = ToFloat(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPCustom<inputT, calT, outputT>::ProcessKLtKMaxTopK(uint32_t loopBatch) {
    DataCopyExtParams copyParams{1, (uint32_t)(ubFactorElementAligned_ * sizeof(outputT)), 0, 0, 0};
    // Move out -infinity to fill GM
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdxInner = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == loopInner_ - 1) {
            DataCopyPad(mGmOut_[currentGmIdxInner], outTensor,
                        {1, (uint32_t)(tailUbFactorElement_ * sizeof(outputT)), 0, 0, 0});
        } else {
            DataCopyPad(mGmOut_[currentGmIdxInner], outTensor, copyParams);
        }
    }
    // Scatter calculation
    int32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    SToMTE3Sync();
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex], scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
    loopProb = loopProb - 1;

    for (; loopProb >= 0; loopProb--) {
        float curValue = calLocalFp32[ubFactorElementAligned_].GetValue(loopProb);
        if (curValue < kthValue) {
            break;
        }
        MTE3ToSSync();
        scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
        gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
        SToMTE3Sync();
        DataCopyPad(mGmOut_[baseGmIdx_ + gmIndex],
                    scatterTensor.template ReinterpretCast<outputT>(), scatterCopyParams);
    }
}

} // namespace

#endif // APPLY_TOP_K_TOP_P_CUSTOM_H_KERNEL