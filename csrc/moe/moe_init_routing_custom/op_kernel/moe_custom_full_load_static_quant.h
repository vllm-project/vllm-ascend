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
 * \file moe_custom_static_quant_full_load.h
 * \brief
 */
#ifndef MOE_CUSTOM_FULL_LOAD_STATIC_QUANT_H
#define MOE_CUSTOM_FULL_LOAD_STATIC_QUANT_H

#include "moe_custom_full_load_base.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

template <typename T>
class MoeCustomFullLoadStaticQuant : public MoeCustomFullLoadBase<T> {
public:
    __aicore__ inline MoeCustomFullLoadStaticQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOutXStaticQuant();
    __aicore__ inline void FreeLocalTensor();
    __aicore__ inline void ComputeQuant(int64_t xLocalLength);

private:
    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECOUT, 1> floatQueue_;
    TQue<QuePosition::VECOUT, 1> halfQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;

    float scale_;
    float offset_;
};

template <typename T>
__aicore__ inline void MoeCustomFullLoadStaticQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                         GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                         GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                                         const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    MoeCustomFullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    this->scale_ = scaleGm_.GetValue(0);
    this->offset_ = offsetGm_.GetValue(0);
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    int64_t curIndexStart = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t rowLength = 0;
    if (this->ep_) {
        rowLength = 1;
    } else {
        rowLength = (curIndexStart + this->coreIndicesElements_ - 1) / this->k_ - curIndexStart / this->k_ + 1;
    }
    int64_t xAlignedCount = Align(this->cols_, sizeof(int8_t));
    this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, xAlignedCount * sizeof(T) * rowLength);
    this->pipe_->InitBuffer(inputXOutQueue_, 1, xAlignedCount * sizeof(int8_t) * rowLength);
    this->pipe_->InitBuffer(floatQueue_, 1, xAlignedCount * sizeof(float) * rowLength);
    this->pipe_->InitBuffer(halfQueue_, 1, xAlignedCount * sizeof(half) * rowLength);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadStaticQuant<T>::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        this->CopyIn();
        this->Compute();

        // valid expert equal zero
        if (this->needCoreNum_ < 1) {
            if (this->blockIdx_ == 0) {
                if (this->rowIdxType_ == GATHER) {
                    this->CopyOutDefaultGatherIdx();
                }
                if (this->expertTokensNumFlag_ == 1) {
                    this->CopyOutDefaultTokenCountOrCumsum();
                }
            }
            return;
        }

        if (this->blockIdx_ == 0) {
            this->CopyOutIdx();
        }
        if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensNumFlag_ == 1) {
            this->ComputeExpertTokenCountOrCumsum();
        }
        if (this->blockIdx_ < this->needCoreNum_) {
            CopyOutXStaticQuant();
        }
        FreeLocalTensor();
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadStaticQuant<T>::ComputeQuant(int64_t xLocalLength)
{
    LocalTensor<float> floatLocal;
    LocalTensor<T> inLocal;
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<half> halfLocal = halfQueue_.AllocTensor<half>();
    uint64_t elements = Align(this->cols_, sizeof(int8_t)) * xLocalLength;
    if constexpr (IsSameType<T, float>::value) {
        floatLocal = this->xCopyInQueue_.template DeQue<float>();
    } else {
        inLocal = this->xCopyInQueue_.template DeQue<T>();
        floatLocal = floatQueue_.AllocTensor<float>();
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
    }
    Muls(floatLocal, floatLocal, this->scale_, elements);
    PipeBarrier<PIPE_V>();
    Adds(floatLocal, floatLocal, this->offset_, elements);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
    Cast(intLocal, floatLocal, RoundMode::CAST_RINT, elements);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    Cast(halfLocal, intLocal, RoundMode::CAST_ROUND, elements);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, halfLocal, RoundMode::CAST_TRUNC, elements);
    inputXOutQueue_.EnQue(outLocal);
    if constexpr (IsSameType<T, float>::value) {
        this->xCopyInQueue_.FreeTensor(floatLocal);
    } else {
        this->xCopyInQueue_.FreeTensor(inLocal);
        floatQueue_.FreeTensor(floatLocal);
    }

    halfQueue_.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadStaticQuant<T>::CopyOutXStaticQuant()
{
    int64_t curIndex = this->curIndexStart_;
    int64_t curIndexEnd = curIndex + this->coreIndicesElements_ - 1;

    if (this->ep_) {
        LocalTensor<int32_t> sortedRowIdx = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
        LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();

        DataCopyExtParams dataXCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};

        for (int64_t dstIndex = curIndex; dstIndex <= curIndexEnd; dstIndex++) {
            if (this->dropPadMode_ == DROPLESS_MODE && dstIndex >= this->activeNum_) {
                break;
            }
            int32_t srcIdx = sortedRowIdx.GetValue(dstIndex);
            int32_t expertIdx = expandedExpertIdx.GetValue(dstIndex);
            if (expertIdx < this->expertStart_ || expertIdx >= this->expertEnd_) {
                break;
            }
            LocalTensor<T> inLocal = this->xCopyInQueue_.template AllocTensor<T>();
            // copyinx
            DataCopyPad(inLocal, this->xGm_[srcIdx / this->k_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
            this->xCopyInQueue_.template EnQue<T>(inLocal);
            ComputeQuant(1);

            LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
            DataCopyPad(this->expandedXGm_[dstIndex * this->cols_], outLocal, intriParams);
            inputXOutQueue_.FreeTensor(outLocal);
        }
        this->expandDstToSrcRowQueue_.EnQue(sortedRowIdx);
        this->expandedExpertIdxCopyOutQueue_.EnQue(expandedExpertIdx);
    } else {
        LocalTensor<T> xLocal = this->xCopyInQueue_.template AllocTensor<T>();
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        int64_t inFactor = Align(this->cols_, sizeof(int8_t));
        uint32_t dstStride = (inFactor * sizeof(T) - AlignBytes(this->cols_, sizeof(T))) / BLOCK_BYTES;
        DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(this->endXRow_ - this->startXRow_ + 1),
                                          static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, dstStride, 0};
        DataCopyPad(xLocal, this->xGm_[this->startXRow_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
        this->xCopyInQueue_.EnQue(xLocal);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        ComputeQuant(this->endXRow_ - this->startXRow_ + 1);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        int64_t k = 0;
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};
        for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
            for (; k < this->coreIndicesElements_ && curIndex / this->k_ == i; curIndex++, k++) {
                int32_t outIndex = expandedRowIdx.GetValue(curIndex);
                if (outIndex < this->activeNum_) {
                    DataCopyPad(this->expandedXGm_[outIndex * this->cols_], outLocal[(i - this->startXRow_) * inFactor],
                                intriParams);
                }
            }
        }
        inputXOutQueue_.FreeTensor(outLocal);
        this->expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadStaticQuant<T>::FreeLocalTensor()
{
    if (!this->ep_) {
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        this->expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
    }
    LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();
    this->expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
    LocalTensor<int32_t> sortedRowIdx = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
    this->expandDstToSrcRowQueue_.FreeTensor(sortedRowIdx);
}

} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_FULL_LOAD_STATIC_QUANT_H