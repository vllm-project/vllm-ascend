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
 * \file moe_custom_gather_quant.h
 * \brief
 */
#ifndef MOE_CUSTOM_GATHER_STATIC_QUANT_H
#define MOE_CUSTOM_GATHER_STATIC_QUANT_H

#include "moe_custom_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

constexpr int64_t GATHER_OUT_QUANT_BUFFER_NUM = 2;

template <typename T, const int EP>
class MoeGatherOutQuant {
public:
    __aicore__ inline MoeGatherOutQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx,
                                GM_ADDR expandedX, GM_ADDR workspace, const MoeInitRoutingCustomTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyExpertIn(int64_t progress);
    __aicore__ inline void Compute(int64_t curLoopCols);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);
    __aicore__ inline void ScatterCopyOut(int64_t progress);
    __aicore__ inline void GatherCopyOut(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, GATHER_OUT_QUANT_BUFFER_NUM> inputXCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_QUANT_BUFFER_NUM> expandRowIdxCopyInQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_QUANT_BUFFER_NUM> inputXCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> floatQueue_;
    TQue<QuePosition::VECOUT, 1> halfQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeCustomGatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    float scale_;
    float offset_;
    int64_t rowIdxType_;
    int64_t dropPadMode_;
    int64_t activeNum_;
    int64_t indicesOffset_;
    int64_t coreNum_;
    int64_t inputOffset_;
    int64_t outOffset_;
    int64_t expertTotalCount_;
};

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset,
                                                      GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                                                      const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    dropPadMode_ = tilingData->dropPadMode;
    activeNum_ = tilingData->activeNum;
    coreNum_ = tilingData->coreNum;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;

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

    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements_;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    if constexpr (EP) {
        if (rowIdxType_ == SCATTER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                              Align(coreRows_, sizeof(int32_t)));
        } else {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                                  blockIdx_ * perCoreRow_,
                                              Align(coreRows_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == GATHER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                              Align(coreRows_, sizeof(int32_t)));
        } else {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                                  blockIdx_ * perCoreRow_,
                                              Align(coreRows_, sizeof(int32_t)));
        }
    }


    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    scale_ = scaleGm_.GetValue(0);
    offset_ = offsetGm_.GetValue(0);

    pipe_->InitBuffer(inputXCopyInQueue_, GATHER_OUT_QUANT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(inputXCopyOutQueue_, GATHER_OUT_QUANT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(expandRowIdxCopyInQueue_, GATHER_OUT_QUANT_BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(floatQueue_, 1, AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(halfQueue_, 1, AlignBytes(perLoopCols_, sizeof(half)));
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::CopyExpertIn(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxCopyInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<T> inLocal = inputXCopyInQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    DataCopyPad(inLocal, inputXGm_[xSrcOffset], copyParams0, padParams0);
    inputXCopyInQueue_.EnQue(inLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(int8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], outLocal, copyParams2);
    inputXCopyOutQueue_.FreeTensor(outLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::Compute(int64_t curLoopCols)
{
    LocalTensor<float> floatLocal;
    LocalTensor<T> inLocal;
    LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.AllocTensor<int8_t>();
    LocalTensor<half> halfLocal = halfQueue_.AllocTensor<half>();
    uint32_t elements = Align(curLoopCols, sizeof(T));
    if constexpr (IsSameType<T, float>::value) {
        floatLocal = inputXCopyInQueue_.DeQue<float>();
    } else {
        inLocal = inputXCopyInQueue_.DeQue<T>();
        floatLocal = floatQueue_.AllocTensor<float>();
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
    }
    Muls(floatLocal, floatLocal, scale_, elements);
    PipeBarrier<PIPE_V>();
    Adds(floatLocal, floatLocal, offset_, elements);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
    Cast(intLocal, floatLocal, RoundMode::CAST_RINT, elements);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(halfLocal, intLocal, RoundMode::CAST_ROUND, elements);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, halfLocal, RoundMode::CAST_TRUNC, elements);
    inputXCopyOutQueue_.EnQue(outLocal);
    if constexpr (IsSameType<T, float>::value) {
        inputXCopyInQueue_.FreeTensor(floatLocal);
    } else {
        inputXCopyInQueue_.FreeTensor(inLocal);
        floatQueue_.FreeTensor(floatLocal);
    }
    halfQueue_.FreeTensor(halfLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::ScatterCopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t indicesIndex = 0; indicesIndex < currentLoopRows_; indicesIndex++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t rowIdx = indicesLocal.GetValue(indicesIndex);
        int64_t xSrcOffset = rowIdx / k_ * cols_;
        int64_t xDstOffset = (rowOffset + indicesIndex) * cols_;
        int64_t curLoopCols = perLoopCols_;
        if (activeNum_ > 0 && dropPadMode_ == DROPLESS_MODE && (rowOffset + indicesIndex) >= activeNum_) {
            break;
        }
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
            if (colsLoop == colLoops_ - 1) {
                curLoopCols = lastLoopCols_;
            }
            int64_t colsLoopOffset = colsLoop * perLoopCols_;
            CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
            Compute(curLoopCols);
            CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
        }
    }
    expandRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::GatherCopyOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    colsTileLength_ = perLoopCols_;
    for (int64_t colsLoop = 0; colsLoop < colLoops_; colsLoop++) {
        int64_t initialRow = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == colLoops_ - 1) {
            colsTileLength_ = lastLoopCols_;
        }
        int64_t currentLoopStartRow = initialRow / k_;
        int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            inputOffset_ = row * cols_ + colsLoop * perLoopCols_;
            // input row position
            CopyXIn(inputOffset_, colsTileLength_);
            Compute(colsTileLength_);
            LocalTensor<int8_t> outLocal = inputXCopyOutQueue_.DeQue<int8_t>();
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1 || (dropPadMode_ == DROPLESS_MODE && outIndex >= activeNum_)) {
                    continue;
                }
                outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
                DataCopyPad(expandedXGm_[outOffset_], outLocal, intriParams);
            }
            inputXCopyOutQueue_.FreeTensor(outLocal);
        }
    }
    expandRowIdxCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T, const int EP>
__aicore__ inline void MoeGatherOutQuant<T, EP>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_; loop++) {
            if (loop == rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }
            CopyExpertIn(loop);
            if constexpr (EP) {
                ScatterCopyOut(loop);
            } else {
                GatherCopyOut(loop);
            }
        }
    }
}
} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_GATHER_STATIC_QUANT_H