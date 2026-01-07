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
 * \file moe_gating_top_k_without_group.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_E_K_WITHOUT_GROUP_H
#define MOE_GATING_TOP_K_E_K_WITHOUT_GROUP_H
#include "kernel_operator.h"
#include "common.h"
#include "kernel_utils.h"
namespace MoeGatingTopK {
using namespace AscendC;

template <typename T>
class MoeGatingTopKWithoutGroup {
public:
    __aicore__ inline MoeGatingTopKWithoutGroup(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBiasAndInitExpertId();
    __aicore__ inline void CopyInX(int64_t progress);
    __aicore__ inline void ComputeX();
    __aicore__ inline void CopuOutXNorm(int64_t row);
    __aicore__ inline void SelectTopKExpertIdx();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CopyOut(int64_t row);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;          // Store input bias
    TBuf<TPosition::VECCALC> expertIdBuf_;      // Expert ID
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_; // Store value after adding bias
    TBuf<TPosition::VECCALC> xNormBuf_;         // Store value after computing sigmoid or softmax
    TBuf<TPosition::VECCALC> topKExpertIdBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_ = 0;
    int64_t perCoreRowCount_ = 0;
    int64_t curCoreRowCount_ = 0;
    int64_t expertCount_ = 0;
    bool addBias_ = false;
    bool outFlag_ = false;
    int64_t k_ = 0;
    int64_t renorm_ = 0;
    int64_t normType_ = 0;
    int64_t expertCountAlign_ = 0;
    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::CopyInBiasAndInitExpertId()
{
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    LocalTensor<int32_t> expertIdTensor = expertIdBuf_.Get<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if (addBias_) {
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            DataCopyPad(biasTensor[expertCountAlign_].ReinterpretCast<T>(), biasGm_, dataCopyParams, dataCopyPadParams);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 expertCountAlign_);
            PipeBarrier<PIPE_V>();
        }
    }
    ArithProgression(expertIdTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), expertCount_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::CopyInX(int64_t row)
{
    LocalTensor<float> xInLocalTensor = xInQueue_.AllocTensor<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
    } else {
        DataCopyPad(xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), xGm_[row * expertCount_], dataCopyParams,
                    dataCopyPadParams);
    }
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::ComputeX()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xInLocalTensor = xInQueue_.DeQue<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(xInLocalTensor, xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
             expertCount_);
        PipeBarrier<PIPE_V>();
    }

    if (normType_ == 1) { // sigmoid
        LocalTensor<uint8_t> calcNormTmpTensor = calcTmpBuf_.Get<uint8_t>();
        Sigmoid(xNormTensor, xInLocalTensor, calcNormTmpTensor, expertCount_);
        PipeBarrier<PIPE_V>();
    } else if (normType_ == 0) { // sigmoid
        LocalTensor<float> reduceValueTensor = calcTmpBuf_.Get<float>();
        LocalTensor<float> calcTmp = calcTmpBuf_.Get<float>()[8];
        ReduceMax(reduceValueTensor, xInLocalTensor, calcTmp, expertCount_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float maxValue = reduceValueTensor.GetValue(0);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Adds(xNormTensor, xInLocalTensor, -maxValue, expertCount_);
        PipeBarrier<PIPE_V>();
        Exp(xNormTensor, xNormTensor, expertCount_);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceValueTensor, xNormTensor, calcTmp, expertCount_);
        eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float sumValue = reduceValueTensor.GetValue(0);
        eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(xNormTensor, xNormTensor, 1.0f / sumValue, expertCount_);
        PipeBarrier<PIPE_V>();
    }
    if (addBias_) {
        Add(xNormWithBiasTensor, xNormTensor, biasTensor, expertCount_);
    } else {
        DataCopy(xNormWithBiasTensor, xNormTensor, expertCountAlign_);
    }

    int64_t duplicateNum = expertCount_ % ONE_REPEAT_SORT_NUM;
    int duplicateIndex = expertCount_ - duplicateNum;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xNormWithBiasTensor.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }
    xInQueue_.FreeTensor(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::CopuOutXNorm(int64_t row)
{
    LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    DataCopy(outOutTensor, xNormTensor, expertCountAlign_);
    outOutQueue_.EnQue<float>(outOutTensor);
    outOutTensor = outOutQueue_.DeQue<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(outGm_[row * expertCount_], outOutTensor, dataCopyParams);
    outOutQueue_.FreeTensor(outOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::SelectTopKExpertIdx()
{
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<uint32_t> expertIdTensor = expertIdBuf_.Get<uint32_t>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<float> sortedScore = calcTmpBuf_.Get<float>();
    LocalTensor<float> sortTmp = calcTmpBuf_.Get<float>()[expertCountAlign_ * CONSTANT_TWO];
    PipeBarrier<PIPE_ALL>();
    Sort<float, true>(sortedScore, xNormWithBiasTensor, expertIdTensor, sortTmp,
                      expertCountAlign_ / ONE_REPEAT_SORT_NUM);

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(k_ * sizeof(float) * CONSTANT_TWO, REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = REPEAT_BLOCKS;
    gatherMaskParams.src1RepeatStride = 0;

    uint64_t rsvdCnt = 0;    // Used to store the number of elements retained after filtering
    uint8_t src1Pattern = 2; // Built-in fixed pattern
    PipeBarrier<PIPE_V>();
    GatherMask(topKExpertId, sortedScore.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);

    DataCopy(expertIdxOut, topKExpertId, expertCountAlign_);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::SelectTopKExpertScore()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> yOutTensor = yOutQueue_.AllocTensor<float>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<int32_t> topKExpertIdWithByte = calcTmpBuf_.Get<int32_t>();
    PipeBarrier<PIPE_V>();
    Muls(topKExpertIdWithByte, topKExpertId, static_cast<int32_t>(sizeof(float)), k_);
    PipeBarrier<PIPE_V>();
    Gather(yOutTensor, xNormTensor, topKExpertIdWithByte.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
           k_);

    bool needRenorm = (normType_ == 1 ) ||  // Case 1: sigmoid + renorm
            (normType_ == 0 && renorm_ == 1);   // Case 3: softmax + renorm
    if (needRenorm == 1) {
        LocalTensor<float> maxValueTensor = calcTmpBuf_.Get<float>();
        LocalTensor<float> tmpTensor = calcTmpBuf_.Get<float>()[BLOCK_BYTES];
        PipeBarrier<PIPE_V>();
        ReduceSum(maxValueTensor, yOutTensor, tmpTensor, k_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float sumValue = maxValueTensor.GetValue(0) + tilingData_->eps;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(tmpTensor, sumValue, k_);
        PipeBarrier<PIPE_V>();
        Div(yOutTensor, yOutTensor, tmpTensor, k_);
    }
    PipeBarrier<PIPE_V>();
    Muls(yOutTensor, yOutTensor, tilingData_->routedScalingFactor, k_);

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(yOutTensor.ReinterpretCast<T>(), yOutTensor, RoundMode::CAST_RINT, k_);
    }

    yOutQueue_.EnQue<float>(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);
    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxOut, dataCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                          GM_ADDR out, GM_ADDR workspace,
                                                          const MoeGatingTopKTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    addBias_ = tilingData_->addBias == 1;
    outFlag_ = tilingData_->outFlag == 1;
    k_ = tilingData_->k;
    renorm_ = tilingData_->renorm;
    normType_ = tilingData_->normType;
    expertCountAlign_ = Ceil(expertCount_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

    // init input gm buf
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    // init que
    pipe_->InitBuffer(xInQueue_, 1, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, Align(k_, sizeof(float)) * sizeof(float));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, Align(k_, sizeof(float)) * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, 1, expertCountAlign_ * sizeof(float));

    // init calc buf
    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(expertIdBuf_, expertCountAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xNormWithBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(topKExpertIdBuf_, Align(k_, sizeof(float)) * sizeof(int32_t));

    // init tmp buf
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * CONSTANT_EIGHT);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroup<T>::Process()
{
    CopyInBiasAndInitExpertId();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        if (outFlag_) {
            CopuOutXNorm(row);
        }
        SelectTopKExpertIdx();
        SelectTopKExpertScore();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_WITHOUT_GROUP_H