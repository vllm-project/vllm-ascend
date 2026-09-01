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
 * \file moe_custom_gather_out.h
 * \brief
 */
#ifndef MOE_CUSTOM_GATHER_OUT_H
#define MOE_CUSTOM_GATHER_OUT_H

#include "moe_custom_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

constexpr int64_t GATHER_OUT_BUFFER_NUM = 2;

template <typename T, const int EP>
class MoeGatherOut {
public:
    __aicore__ inline MoeGatherOut(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t progress);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset);
    __aicore__ inline void GatherCopyOut(int64_t progress);
    __aicore__ inline void ScatterCopyOut(int64_t progress);

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<float> xGscaleGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t activeNum_;
    int64_t dropPadMode_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;
    int64_t curLoopElements_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;

    int64_t rowIdxType_;
    int64_t isInputScale_;
    int64_t coreNum_;
};

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx,
                                                 GM_ADDR expandedX, GM_ADDR expandedScale,
                                                 const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    coreNum_ = tilingData->coreNum;
    dropPadMode_ = tilingData->dropPadMode;
    activeNum_ = tilingData->activeNum;

    isInputScale_ = tilingData->isInputScale;
    rowIdxType_ = tilingData->rowIdxType;

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;

    if constexpr (EP) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
        expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    } else {
        expertTotalCount_ = n_ * k_;
    }

    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;
    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
    }
    indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

    xGm_.SetGlobalBuffer((__gm__ T *)x, n_ * cols_);
    xGscaleGm_.SetGlobalBuffer((__gm__ float *)scale, n_);

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_OUT_BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(scaleCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(1, sizeof(float)));

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreIndicesElements_,
                                       Align(curCoreIndicesElements_, sizeof(int32_t)));

    if constexpr (EP) {
        if (rowIdxType_ == SCATTER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        } else {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                                  blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == GATHER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        } else {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                                  blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        }
    }
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::CopyExpertIn(int64_t progress)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[progress * curCorePerLoopIndicesElements_], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
    xCopyInQueue_.EnQue(xLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::CopyScaleIn(int64_t scaleSrcOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::CopyScaleOut(int64_t scaleDstOffset)
{
    LocalTensor<float> scaleLocal = scaleCopyInQueue_.DeQue<float>();
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::GatherCopyOut(int64_t progress)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    int64_t curLoopCols = perLoopCols_;
    for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
        int64_t initialRow = blockIdx_ * perCoreIndicesElements_ + curCorePerLoopIndicesElements_ * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == colsLoops_ - 1) {
            curLoopCols = lastLoopCols_;
        }
        int64_t currentLoopStartRow = initialRow / k_;
        int64_t currentLoopLastRow = (initialRow + this->curLoopElements_ - 1) / k_;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            LocalTensor<T> inLocal = xCopyInQueue_.AllocTensor<T>();
            int64_t inputOffset = row * cols_ + colsLoop * perLoopCols_;
            DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(inLocal, xGm_[inputOffset], xCopyParams, dataCopyPadParams);
            // copy in scale
            LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            if (isInputScale_ == 1 && colsLoop == 0) {
                DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
                DataCopyPad(scaleLocal, xGscaleGm_[row], scaleCopyParams, scalePadParams);
            }
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
            while (curLoopRow < this->curLoopElements_ && initialRow / k_ == row) {
                int32_t outIndex = subRowIdxLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1 || (dropPadMode_ == DROPLESS_MODE && outIndex >= activeNum_)) {
                    continue;
                }
                int64_t outOffset = outIndex * this->cols_ + colsLoop * this->perLoopCols_;
                DataCopyPad(expandedXGm_[outOffset], inLocal, intriParams);
                // copy out scale
                if (isInputScale_ == 1 && colsLoop == 0) {
                    DataCopyPad(expandedScaleGm_[outIndex], scaleLocal, scaleCopyParams);
                }
            }
            scaleCopyInQueue_.FreeTensor(scaleLocal);
            xCopyInQueue_.FreeTensor(inLocal);
        }
    }
    expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::ScatterCopyOut(int64_t progress)
{
    int64_t curExpertLoopOffset = progress * curCorePerLoopIndicesElements_;
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
    for (int64_t indicesIndex = 0; indicesIndex < curLoopElements_; indicesIndex++) {
        int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
        int64_t rowOffset = curExpertLoopOffset + indicesIndex + blockIdx_ * perCoreIndicesElements_;
        if (activeNum_ > 0 && dropPadMode_ == DROPLESS_MODE && rowOffset >= activeNum_) {
            break;
        }
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        if (isInputScale_ == 1) {
            int64_t scaleSrcOffset = rowIdx / k_;
            CopyScaleIn(scaleSrcOffset);
            CopyScaleOut(indicesIndex + curExpertLoopOffset + blockIdx_ * perCoreIndicesElements_);
        }
        int64_t curLoopCols = perLoopCols_;
        for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
            if (colsLoop == colsLoops_ - 1) {
                curLoopCols = lastLoopCols_;
            }
            int64_t xSrcOffset = rowIdx / k_ * cols_;
            int64_t xDstOffset = (blockIdx_ * perCoreIndicesElements_ + curExpertLoopOffset + indicesIndex) * cols_;
            int64_t colsLoopOffset = colsLoop * perLoopCols_;
            CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
            CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
        }
    }
    expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOut<T, EP>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        curLoopElements_ = curCorePerLoopIndicesElements_;
        for (int64_t loop = 0; loop < indicesLoops_; loop++) {
            if (loop == indicesLoops_ - 1) {
                curLoopElements_ = curCoreLastLoopIndicesElements_;
            }
            CopyExpertIn(loop);
            if constexpr (!EP) {
                GatherCopyOut(loop);
            } else {
                ScatterCopyOut(loop);
            }
        }
    }
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_GATHER_OUT_H