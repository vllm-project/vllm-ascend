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
 * \file moe_custom_gather_out_droppad.h
 * \brief
 */
#ifndef MOE_CUSTOM_GATHER_OUT_DROPPAD_H
#define MOE_CUSTOM_GATHER_OUT_DROPPAD_H

#include "moe_custom_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

constexpr int64_t GATHER_OUT_DROPPAD_BUFFER_NUM = 2;

template <typename T>
class MoeGatherOutDroppad {
public:
    __aicore__ inline MoeGatherOutDroppad(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, GM_ADDR workspace, const MoeInitRoutingCustomTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset, LocalTensor<float> scaleLocal);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset, LocalTensor<float> scaleLocal);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, GATHER_OUT_DROPPAD_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_DROPPAD_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_DROPPAD_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<float> xGscaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;

    const MoeCustomGatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t isInputScale_;

    int64_t indicesOffset_;
    int64_t inputOffset_;
    int64_t outOffset_;
};

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyInIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandedRowIdxCopyInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyScaleIn(int64_t scaleSrcOffset, LocalTensor<float> scaleLocal)
{
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyScaleOut(int64_t scaleDstOffset, LocalTensor<float> scaleLocal)
{
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::CopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    colsTileLength_ = perLoopCols_;
    for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
        int64_t initialRow = gatherOutTilingData_->perCoreIndicesElements * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == colLoops_ - 1) {
            colsTileLength_ = lastLoopCols_;
        }
        int64_t currentLoopStartRow = initialRow / k_;
        int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            LocalTensor<float> scaleLocal = scaleCopyInQueue_.AllocTensor<float>();
            if (isInputScale_ == 1) {
                CopyScaleIn(row, scaleLocal);
                LocalTensor<float> scaleLocal = scaleCopyInQueue_.DeQue<float>();
            }
            inputOffset_ = row * cols_ + colsLoop * perLoopCols_;
            // input row position
            LocalTensor<T> inLocal = xCopyInQueue_.AllocTensor<T>();
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(inLocal, inputXGm_[inputOffset_], dataCopyParams, dataCopyPadParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
            while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1) {
                    continue;
                }
                outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
                DataCopyPad(expandedXGm_[outOffset_], inLocal, intriParams);
                if (isInputScale_ == 1) {
                    CopyScaleOut(outIndex, scaleLocal);
                }
            }
            xCopyInQueue_.FreeTensor(inLocal);
            scaleCopyInQueue_.FreeTensor(scaleLocal);
        }
    }
    expandedRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR expandedRowIdx,
                                                    GM_ADDR expandedX, GM_ADDR expandedScale, GM_ADDR workspace,
                                                    const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);

    needCoreNum_ = gatherOutTilingData_->needCoreNum;
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    isInputScale_ = tilingData->isInputScale;

    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = gatherOutTilingData_->lastCoreIndicesElements;
        perLoopRows_ = gatherOutTilingData_->lastCorePerLoopIndicesElements;
        lastLoopRows_ = gatherOutTilingData_->lastCoreLastLoopIndicesElements;
        rowLoops_ = gatherOutTilingData_->lastCoreIndicesLoops;
    } else {
        coreRows_ = gatherOutTilingData_->perCoreIndicesElements;
        perLoopRows_ = gatherOutTilingData_->perCorePerLoopIndicesElements;
        lastLoopRows_ = gatherOutTilingData_->perCoreLastLoopIndicesElements;
        rowLoops_ = gatherOutTilingData_->perCoreIndicesLoops;
    }
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX, coreRows_ * cols_);
    xGscaleGm_.SetGlobalBuffer((__gm__ float *)scale, n_);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX, n_ * k_ * cols_);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx +
                                          blockIdx_ * gatherOutTilingData_->perCoreIndicesElements,
                                      Align(coreRows_, sizeof(int32_t)));
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    pipe_->InitBuffer(xCopyInQueue_, GATHER_OUT_DROPPAD_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_OUT_DROPPAD_BUFFER_NUM,
                      AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(scaleCopyInQueue_, GATHER_OUT_DROPPAD_BUFFER_NUM, AlignBytes(1, sizeof(float)));
}

template <typename T>
__aicore__ inline void MoeGatherOutDroppad<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_; loop++) {
            if (loop == rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }
            CopyInIndices(loop);
            CopyOut(loop);
        }
    }
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_GATHER_OUT_DROPPAD_H
