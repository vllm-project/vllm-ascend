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
 * \file moe_custom_full_load_dynamic_quant.h
 * \brief
 */
#ifndef MOE_CUSTOM_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_CUSTOM_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_custom_full_load_base.h"
#include "moe_custom_common.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
class MoeCustomFullLoadDynamicQuant : public MoeCustomFullLoadBase<T> {
public:
    __aicore__ inline MoeCustomFullLoadDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOutXDynamicQuantFromGather();
    __aicore__ inline void CopyOutXDynamicQuantFromScatter();
    __aicore__ inline void FreeLocalTensor();
    __aicore__ inline void ComputeQuant(LocalTensor<float> &smoothLocal);

private:
    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECIN, 1> smoothInQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t colsAlign_ = 0;
};

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
    const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    MoeCustomFullLoadBase<T>::Init(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    quantSmoothGm_.SetGlobalBuffer((__gm__ float *)scale);
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    this->colsAlign_ = Align(this->cols_, sizeof(T));
    if constexpr (IsSameType<T, float>::value) {
        this->pipe_->InitBuffer(xCopyInQueue_, 1, AlignBytes(this->cols_, sizeof(float)));
    } else {
        this->pipe_->InitBuffer(xCopyInQueue_, 1, 2 * AlignBytes(this->cols_, sizeof(T)));
    }
    this->pipe_->InitBuffer(inputXOutQueue_, 1, AlignBytes(this->cols_, sizeof(int8_t)));
    this->pipe_->InitBuffer(smoothInQueue_, 1, AlignBytes(this->cols_, sizeof(float)));
    this->pipe_->InitBuffer(tmpBuff_, AlignBytes(this->cols_, sizeof(float)));
    this->pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::Process()
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
            if constexpr (!COPYOUTTYPE && SMOOTHTYPE != SCALE_EH) {
                CopyOutXDynamicQuantFromGather();
            } else {
                CopyOutXDynamicQuantFromScatter();
            }
        }

        FreeLocalTensor();
    }
}

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void
MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::ComputeQuant(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> tempLocal = tmpBuff_.Get<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> dynamicQuantLocal = scaleOutQueue_.AllocTensor<float>();
    LocalTensor<float> inLocal = xCopyInQueue_.DeQue<float>();

    if constexpr (!IsSameType<T, float>::value && !IsSameType<T, int8_t>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[colsAlign_], RoundMode::CAST_NONE, this->cols_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (SMOOTHTYPE != NO_SCALE) {
        Mul(inLocal, inLocal, smoothLocal, this->cols_);
        PipeBarrier<PIPE_V>();
    }

    Abs(tempLocal, inLocal, this->cols_);
    PipeBarrier<PIPE_V>();

    ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, this->cols_);
    PipeBarrier<PIPE_V>();

    float maxValue = dynamicQuantLocal.GetValue(0) / MAX_INT8;

    Duplicate<float>(dynamicQuantLocal, maxValue, INT32_ONE_BLOCK_NUM);
    PipeBarrier<PIPE_V>();
    Duplicate<float>(tempLocal, maxValue, this->cols_);
    PipeBarrier<PIPE_V>();

    Div(tempLocal, inLocal, tempLocal, this->cols_);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> intLocal = tempLocal.ReinterpretCast<int32_t>();
    Cast(intLocal, tempLocal, RoundMode::CAST_RINT, this->cols_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    Cast(intLocal.ReinterpretCast<half>(), intLocal, RoundMode::CAST_ROUND, this->cols_);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, intLocal.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->cols_);

    inputXOutQueue_.EnQue<int8_t>(outLocal);
    scaleOutQueue_.EnQue<float>(dynamicQuantLocal);
}

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::CopyOutXDynamicQuantFromScatter()
{
    LocalTensor<int32_t> sortedRowIdx = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();

    DataCopyExtParams dataXCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    ;

    if constexpr (SMOOTHTYPE == SCALE_1H) {
        DataCopyPad(smoothLocal, quantSmoothGm_, smoothCopyParams, {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    int64_t dstIndexStart = this->curIndexStart_;
    int64_t dstIndexEnd = dstIndexStart + this->coreIndicesElements_ - 1;
    int32_t lastExpertIdx = -1;

    for (int64_t dstIndex = dstIndexStart; dstIndex <= dstIndexEnd; dstIndex++) {
        if (this->dropPadMode_ == DROPLESS_MODE && dstIndex >= this->activeNum_) {
            break;
        }
        int32_t srcIdx = sortedRowIdx.GetValue(dstIndex);
        int32_t expertIdx = expandedExpertIdx.GetValue(dstIndex);
        if (expertIdx < this->expertStart_ || expertIdx >= this->expertEnd_) {
            break;
        }
        expertIdx = expertIdx - this->expertStart_;
        LocalTensor<T> xLocal = this->xCopyInQueue_.template AllocTensor<T>();
        // copy in single x
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(xLocal, this->xGm_[srcIdx / this->k_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(xLocal[colsAlign_], this->xGm_[srcIdx / this->k_ * this->cols_], dataXCopyParams,
                        {false, 0, 0, 0});
        }
        xCopyInQueue_.EnQue<T>(xLocal);

        // copyin dynamic scale
        if constexpr (SMOOTHTYPE == SCALE_EH) {
            if (expertIdx != lastExpertIdx) {
                DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * this->cols_], smoothCopyParams, {false, 0, 0, 0});
                smoothInQueue_.EnQue(smoothLocal);
                smoothLocal = smoothInQueue_.DeQue<float>();
                lastExpertIdx = expertIdx;
            }
        }

        ComputeQuant(smoothLocal);

        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(expandedScaleGm_[dstIndex], quantScaleLocal, quantScaleParams);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(this->expandedXGm_[dstIndex * this->cols_], outLocal, intriParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
        this->xCopyInQueue_.FreeTensor(xLocal);
    }
    smoothInQueue_.FreeTensor(smoothLocal);
    this->expandDstToSrcRowQueue_.EnQue(sortedRowIdx);
    this->expandedExpertIdxCopyOutQueue_.EnQue(expandedExpertIdx);
}

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::CopyOutXDynamicQuantFromGather()
{
    DataCopyExtParams dataXCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int64_t curIndex = this->blockIdx_ * this->perCoreIndicesElements_;
    int64_t curIndexEnd = curIndex + this->coreIndicesElements_ - 1;

    if constexpr (SMOOTHTYPE == SCALE_1H) {
        DataCopyPad(smoothLocal, quantSmoothGm_, smoothCopyParams, {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    for (int64_t row = this->startXRow_; row <= this->endXRow_; row++) {
        LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(xLocal, this->xGm_[row * this->cols_], dataXCopyParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(xLocal[colsAlign_], this->xGm_[row * this->cols_], dataXCopyParams, {false, 0, 0, 0});
        }
        xCopyInQueue_.EnQue<T>(xLocal);
        ComputeQuant(smoothLocal);

        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        while (curIndex <= curIndexEnd && curIndex / this->k_ == row) {
            int32_t outIndex = expandedRowIdx.GetValue(curIndex);
            curIndex++;
            if (outIndex == -1 || this->dropPadMode_ == DROPLESS_MODE && outIndex >= this->activeNum_) {
                continue;
            }
            DataCopyPad(expandedXGm_[outIndex * this->cols_], outLocal, intriParams);
            DataCopyPad(expandedScaleGm_[outIndex], quantScaleLocal, quantScaleParams);
        }

        xCopyInQueue_.FreeTensor(xLocal);
        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    this->expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

template <typename T, const int COPYOUTTYPE, const int SMOOTHTYPE>
__aicore__ inline void MoeCustomFullLoadDynamicQuant<T, COPYOUTTYPE, SMOOTHTYPE>::FreeLocalTensor()
{
    if constexpr (!COPYOUTTYPE) {
        LocalTensor<int32_t> expandedRowIdx = this->expandedRowIdxCopyOutQueue_.template DeQue<int32_t>();
        this->expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
    }
    LocalTensor<int32_t> sortedRowIdx = this->expandDstToSrcRowQueue_.template DeQue<int32_t>();
    LocalTensor<int32_t> expandedExpertIdx = this->expandedExpertIdxCopyOutQueue_.template DeQue<int32_t>();
    this->expandDstToSrcRowQueue_.FreeTensor(sortedRowIdx);
    this->expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
}

} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_FULL_LOAD_DYNAMIC_QUANT_H