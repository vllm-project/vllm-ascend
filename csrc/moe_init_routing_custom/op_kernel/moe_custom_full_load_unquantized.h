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
 * \file moe_custom_full_load_unquantized.h
 * \brief
 */
#ifndef MOE_CUSTOM_FULL_LOAD_UNQUANTIZED_H
#define MOE_CUSTOM_FULL_LOAD_UNQUANTIZED_H

#include "moe_custom_full_load_base.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

template <typename T>
class MoeCustomFullLoadUnquantized : public MoeCustomFullLoadBase<T> {
public:
    __aicore__ inline MoeCustomFullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void FreeLocalTensor();
    __aicore__ inline void GatherOutX();
    __aicore__ inline void CopyOutScale();

protected:
    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECIN, 1> scaleCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;
};

template <typename T>
__aicore__ inline void MoeCustomFullLoadUnquantized<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                                         GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                         GM_ADDR expandedScale, GM_ADDR workspace,
                                                         const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    MoeCustomFullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    if (this->isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    int64_t buffSize = this->sortNum_ * sizeof(int32_t);
    int64_t row_length =
        (this->curIndexStart_ + this->coreIndicesElements_ - 1) / this->k_ - this->curIndexStart_ / this->k_ + 1;

    if (this->ep_) {
        this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(T)));
    } else {
        this->pipe_->InitBuffer(xCopyInQueue_, this->bufferNum_, AlignBytes(this->cols_, sizeof(T)) * row_length);
    }
    this->pipe_->InitBuffer(scaleCopyInQueue_, 1, AlignBytes(1, sizeof(float)));
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadUnquantized<T>::Process()
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
            this->GatherOutX();
            if (this->isInputScale_) {
                this->CopyOutScale();
            }
        }

        this->FreeLocalTensor();
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadUnquantized<T>::GatherOutX()
{
    if (this->ep_) {
        LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();
        LocalTensor<int32_t> expandDstToSrcRowLocal = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
        int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
        int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = expandedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = expandDstToSrcRowLocal.GetValue(i);
            int64_t srcOffset = rowIdx / this->k_ * this->cols_;
            int64_t dstOffset = i * this->cols_;
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedXGm_[dstOffset], xLocal, copyParams);
        }
        xCopyInQueue_.FreeTensor(xLocal);
        this->expandedExpertIdxCopyOutQueue_.template EnQue<int32_t>(expandedExpertIdx);
        this->expandDstToSrcRowQueue_.template EnQue<int32_t>(expandDstToSrcRowLocal);
    } else {
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(this->endXRow_ - this->startXRow_ + 1),
                                          static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[this->startXRow_ * this->cols_], dataXCopyParams, dataXCopyPadParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        int64_t inFactor = Align(this->cols_, sizeof(T));
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        int64_t curIndexStart = this->curIndexStart_;
        int64_t k = 0;
        for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
            for (; k < this->coreIndicesElements_ && curIndexStart / this->k_ == i; curIndexStart++, k++) {
                int32_t outIndex = expandedRowIdx.GetValue(curIndexStart);
                if (outIndex < this->activeNum_) {
                    DataCopyPad(expandedXGm_[outIndex * this->cols_], xLocal[(i - this->startXRow_) * inFactor],
                                copyParams);
                }
            }
        }
        xCopyInQueue_.FreeTensor(xLocal);
        this->expandedRowIdxCopyOutQueue_.template EnQue<int32_t>(expandedRowIdx);
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadUnquantized<T>::FreeLocalTensor()
{
    LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> expandDstToSrcRowLocal = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
    this->expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
    this->expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
    if (!this->ep_) {
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        this->expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadUnquantized<T>::CopyOutScale()
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (this->ep_) {
        LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();
        LocalTensor<int32_t> expandDstToSrcRowLocal = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
        int64_t startRowIdx = this->blockIdx_ * this->perCoreIndicesElements_;
        int64_t endRowIdx = startRowIdx + this->coreIndicesElements_;
        for (int64_t i = startRowIdx; i < endRowIdx && i < this->activeNum_; i++) {
            int32_t curExpertId = expandedExpertIdx.GetValue(i);
            if (curExpertId < this->expertStart_ || curExpertId >= this->expertEnd_) {
                break;
            }
            int64_t rowIdx = expandDstToSrcRowLocal.GetValue(i);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(scaleLocal, scaleGm_[rowIdx / this->k_], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedScaleGm_[i], scaleLocal, copyParams);
        }
        this->expandedExpertIdxCopyOutQueue_.template EnQue<int32_t>(expandedExpertIdx);
        this->expandDstToSrcRowQueue_.template EnQue<int32_t>(expandDstToSrcRowLocal);
    } else {
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        int64_t curIndexStart = this->curIndexStart_;
        int64_t k = 0;
        for (int64_t i = this->startXRow_; i <= this->endXRow_; i++) {
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(scaleLocal, scaleGm_[i], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            for (; k < this->coreIndicesElements_ && curIndexStart / this->k_ == i; curIndexStart++, k++) {
                int32_t outIndex = expandedRowIdx.GetValue(curIndexStart);
                if (outIndex < this->activeNum_) {
                    DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, copyParams);
                }
            }
        }
        this->expandedRowIdxCopyOutQueue_.template EnQue<int32_t>(expandedRowIdx);
    }
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_FULL_LOAD_UNQUANTIZED_H