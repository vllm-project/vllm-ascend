/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_base.h
 * \brief causal_conv1d_update kernel
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d_update_base.h"

namespace CausalConv1dUpdateOp {
using namespace AscendC;
template <typename T>
class CausalConv1dUpdate : public CausalConv1dUpdateBase<T> {
public:
    __aicore__ inline CausalConv1dUpdate(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convState, GM_ADDR convStateIndices, 
                                GM_ADDR bias, GM_ADDR numAcceptedTokens, GM_ADDR queryStartLoc, 
                                GM_ADDR y, const CausalConv1dUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeUpdate(int64_t xOffset);
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyInWeight(int64_t weightLen, int64_t weightInOffset);
    __aicore__ inline void CopyInState(int64_t stateLen, int64_t stateInOffset);
    __aicore__ inline void CopyInBias(int64_t biasLen, int64_t biasInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void CopyOutState(int64_t stateLen, int64_t stateOutOffset);

private:
    // constexpr static int32_t bufferNum_ = 2;
    TPipe pipe_;
    // TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueWeight_;
    // TQue<QuePosition::VECIN, 1> inQueueState_;
    TQue<QuePosition::VECIN, 1> inQueueBias_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;

    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> castBufInput_;
    TBuf<QuePosition::VECCALC> castBufWeight_;
    TBuf<QuePosition::VECCALC> resultBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<T> convStateGm_;
    GlobalTensor<int32_t> convStateIndicesGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<int32_t> numAcceptGm_;
    GlobalTensor<int32_t> queryLocGm_;
    GlobalTensor<T> yGm_;

    LocalTensor<T> inLocal;
    LocalTensor<float> resultLocal;
    LocalTensor<float> castIn;
    LocalTensor<float> castWeight; 

    CausalConv1dUpdateTilingData tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t gmStateOffset_ = 0;
    int64_t batchNum_ = 1;
    int64_t inStateOffset_ = 0;

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void MTE2ToMTE3Sync()
    {
        event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    }

    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }
};

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convState, GM_ADDR convStateIndices, 
                                                   GM_ADDR bias, GM_ADDR numAcceptedTokens, GM_ADDR queryStartLoc, 
                                                   GM_ADDR y, const CausalConv1dUpdateTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    convStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convState));
    convStateIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(convStateIndices));
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    numAcceptGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numAcceptedTokens));
    queryLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    this->ParseCoreBlocks(tilingData_, blockIdx_, batchNum_);

    // alloc TQue
    // pipe_.InitBuffer(inQueueX_, 1, tilingData_.dim * sizeof(T));
    pipe_.InitBuffer(inQueueWeight_, 1, tilingData_.width * tilingData_.dim * sizeof(T));
    // pipe_.InitBuffer(inQueueState_, 1, tilingData_.dim * sizeof(T));
    pipe_.InitBuffer(inQueueBias_, 1, tilingData_.dim * sizeof(T));
    pipe_.InitBuffer(outQueueY_, 1, tilingData_.dim * sizeof(T));

    // alloc TBuf
    pipe_.InitBuffer(inputBuf_, tilingData_.dim * sizeof(T));
    pipe_.InitBuffer(castBufInput_, tilingData_.dim * sizeof(float));
    pipe_.InitBuffer(castBufWeight_, tilingData_.dim * sizeof(float));
    pipe_.InitBuffer(resultBuf_, tilingData_.dim * sizeof(float));
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::Process()
{
    if (blockIdx_ >= tilingData_.numCore) {
        return;
    }

    if (tilingData_.hasQueryLoc) {
        gmXOffset_ = queryLocGm_.GetValue(blockIdx_ * tilingData_.blockFactor) * tilingData_.dim;
    } else {
        gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.seqLen * tilingData_.dim;
    }
    
    CopyInWeight(tilingData_.width * tilingData_.dim, 0);

    ComputeUpdate(gmXOffset_);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::ComputeUpdate(int64_t xOffset) 
{
    LocalTensor<T> wLocal = inQueueWeight_.DeQue<T>();
    inLocal = inputBuf_.Get<T>();
    resultLocal = resultBuf_.Get<float>();
    castIn = castBufInput_.Get<float>();
    castWeight = castBufWeight_.Get<float>();
    int64_t stateOffset = 0;

    if(!tilingData_.hasIndices) {
        gmStateOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.stateLen * tilingData_.dim;
    }

    for (int64_t i = 0; i < batchNum_; ++i) {
        // 获取 stateOffset
        if (tilingData_.hasIndices) {
            int64_t stateIdx = convStateIndicesGm_.GetValue(blockIdx_ * tilingData_.blockFactor + i);
            if (stateIdx == tilingData_.padSlotId) {
                continue;
            }
            stateOffset = tilingData_.stateLen * tilingData_.dim * stateIdx;
        } else {
            stateOffset = gmStateOffset_ + i * tilingData_.stateLen * tilingData_.dim;
        }
        
        int64_t actSeqLen = tilingData_.seqLen;
        if (tilingData_.hasQueryLoc) {
            int64_t startLoc = queryLocGm_.GetValue(blockIdx_ * tilingData_.blockFactor + i);
            int64_t endLoc = queryLocGm_.GetValue(blockIdx_ * tilingData_.blockFactor + i + 1);
            actSeqLen = endLoc - startLoc;
        }

        int64_t numAccept = actSeqLen;
        if (tilingData_.hasNumAccept) {
            numAccept = numAcceptGm_.GetValue(blockIdx_ * tilingData_.blockFactor + i);
            inStateOffset_ = numAccept - 1;
        }

        int64_t calcSeqLen = numAccept < actSeqLen ? numAccept : actSeqLen;
        for (int64_t j = 0; j < calcSeqLen; ++j) {
            Duplicate<float>(resultLocal, 0, tilingData_.dim);
            LocalTensor<T> outLocal = outQueueY_.AllocTensor<T>();
            for (int64_t k = 0; k < tilingData_.width - 1; ++k) {
                // 循环历史token
                CopyInState(tilingData_.dim, stateOffset + (k + inStateOffset_) * tilingData_.dim);
                MTE2ToMTE3Sync();

                if ((k + inStateOffset_) != 0) {
                    // 搬出覆盖conv state
                    CopyOutState(tilingData_.dim, stateOffset + (k + inStateOffset_ -1) * tilingData_.dim);
                    MTE3ToMTE2Sync();
                }

                MTE2ToVSync();
                Cast(castIn, inLocal, RoundMode::CAST_NONE, tilingData_.dim);
                Cast(castWeight, wLocal[k * tilingData_.dim], RoundMode::CAST_NONE, tilingData_.dim);
                MulAddDst(resultLocal, castIn, castWeight, tilingData_.dim);

                VToMTE2Sync();
            }
            // 搬入 x
            CopyInX(tilingData_.dim, xOffset + j * tilingData_.dim);
            MTE2ToMTE3Sync();
            CopyOutState(tilingData_.dim, stateOffset + (inStateOffset_ + tilingData_.width - 2) * tilingData_.dim);
            MTE3ToMTE2Sync();

            MTE2ToVSync();
            Cast(castIn, inLocal, RoundMode::CAST_NONE, tilingData_.dim);
            Cast(castWeight, wLocal[(tilingData_.width - 1) * tilingData_.dim], RoundMode::CAST_NONE, tilingData_.dim);
            MulAddDst(resultLocal, castIn, castWeight, tilingData_.dim);

            if (tilingData_.hasBias) {
                CopyInBias(tilingData_.dim, 0);
                LocalTensor<T> biasLocal = inQueueBias_.DeQue<T>();
                Cast(castIn, biasLocal, RoundMode::CAST_NONE, tilingData_.dim);
                Add(resultLocal, castIn, resultLocal, tilingData_.dim);
                inQueueBias_.FreeTensor(biasLocal);
            }

            if (tilingData_.activationMode) {
                Muls(castWeight, resultLocal, (float)-1.0, tilingData_.dim);
                Exp(castWeight, castWeight, tilingData_.dim);
                Adds(castWeight, castWeight, (float)1.0, tilingData_.dim);
                Div(resultLocal, resultLocal, castWeight, tilingData_.dim);
            }

            Cast(outLocal, resultLocal, RoundMode::CAST_ROUND, tilingData_.dim);
            
            outQueueY_.EnQue(outLocal);
            CopyOutY(tilingData_.dim, xOffset + j * tilingData_.dim);
        }

        for (int64_t j = 0; j < actSeqLen - calcSeqLen; ++j) {
            Duplicate<float>(resultLocal, 0, tilingData_.dim);
            LocalTensor<T> outLocal = outQueueY_.AllocTensor<T>();
            for (int64_t k = 0; k < tilingData_.width - 1; ++k) {
                CopyInState(tilingData_.dim, stateOffset + (k + inStateOffset_ + j) * tilingData_.dim);

                MTE2ToVSync();
                Cast(castIn, inLocal, RoundMode::CAST_NONE, tilingData_.dim);
                Cast(castWeight, wLocal[k * tilingData_.dim], RoundMode::CAST_NONE, tilingData_.dim);
                MulAddDst(resultLocal, castIn, castWeight, tilingData_.dim);

                VToMTE2Sync();
            }

            CopyInX(tilingData_.dim, xOffset + (calcSeqLen + j) * tilingData_.dim);
            MTE2ToMTE3Sync();
            CopyOutState(tilingData_.dim, stateOffset + (inStateOffset_ + tilingData_.width - 1 + j) * tilingData_.dim);
            MTE3ToMTE2Sync();

            MTE2ToVSync();
            Cast(castIn, inLocal, RoundMode::CAST_NONE, tilingData_.dim);
            Cast(castWeight, wLocal[(tilingData_.width - 1) * tilingData_.dim], RoundMode::CAST_NONE, tilingData_.dim);
            MulAddDst(resultLocal, castIn, castWeight, tilingData_.dim);

            if (tilingData_.hasBias) {
                CopyInBias(tilingData_.dim, 0);
                LocalTensor<T> biasLocal = inQueueBias_.DeQue<T>();
                Cast(castIn, biasLocal, RoundMode::CAST_NONE, tilingData_.dim);
                Add(resultLocal, castIn, resultLocal, tilingData_.dim);
                inQueueBias_.FreeTensor(biasLocal);
            }

            if (tilingData_.activationMode) {
                Muls(castWeight, resultLocal, (float)-1.0, tilingData_.dim);
                Exp(castWeight, castWeight, tilingData_.dim);
                Adds(castWeight, castWeight, (float)1.0, tilingData_.dim);
                Div(resultLocal, resultLocal, castWeight, tilingData_.dim);
            }

            Cast(outLocal, resultLocal, RoundMode::CAST_ROUND, tilingData_.dim);
            
            outQueueY_.EnQue(outLocal);
            CopyOutY(tilingData_.dim, xOffset + (calcSeqLen + j) * tilingData_.dim);
        }

        xOffset = xOffset + actSeqLen * tilingData_.dim;
    }
    inQueueWeight_.FreeTensor(wLocal);
}

// x [batch, seqLen, dim] xLen = dataCount
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyInX(int64_t xLen, int64_t xInOffset)
{
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(xLen, copyParams);
    DataCopyPad(inLocal, xGm_[xInOffset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyInState(int64_t stateLen, int64_t stateInOffset)
{
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetStateInCopyParams(stateLen, copyParams);
    DataCopyPad(inLocal, convStateGm_[stateInOffset], copyParams, padParams);
}

// weight [width, dim] wLen = width * dim
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyInWeight(int64_t weightLen, int64_t weightInOffset)
{
    LocalTensor<T> wLocal = inQueueWeight_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetWeightInCopyParams(weightLen, copyParams);
    DataCopyPad(wLocal, weightGm_[weightInOffset], copyParams, padParams);
    inQueueWeight_.EnQue(wLocal);
}

// bias [dim, ] bLen = dim
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyInBias(int64_t biasLen, int64_t biasInOffset)
{
    LocalTensor<T> bLocal = inQueueBias_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetBiasInCopyParams(biasLen, copyParams);
    DataCopyPad(bLocal, biasGm_[biasInOffset], copyParams, padParams);
    inQueueBias_.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyOutState(int64_t stateLen, int64_t stateOutOffset)
{
    DataCopyExtParams copyParams;
    // DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetStateOutCopyParams(stateLen, copyParams);
    DataCopyPad<T>(convStateGm_[stateOutOffset], inLocal, copyParams);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::CopyOutY(int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<T> outLocal = outQueueY_.DeQue<T>();
    DataCopyExtParams copyParams;
    this->GetYOutCopyParams(yLen, copyParams);
    DataCopyPad<T>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace CausalConv1dUpdateOp
#endif