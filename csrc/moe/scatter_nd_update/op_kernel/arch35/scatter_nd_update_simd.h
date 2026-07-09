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
 * \file scatter_nd_simd.h
 * \brief simd kernel of scatter_nd_update
 */

#ifndef SCATTER_ND_UPDATE_SIMD_H
#define SCATTER_ND_UPDATE_SIMD_H

#include "scatter_nd_update_common.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ScatterNdUpdate {
using namespace AscendC;

template <typename T, typename U, typename OFFSET_T = U>
class ScatterNdUpdateSimd : public ScatterNdUpdateBase<T, U, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateSimd(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIndiceInSplitIndices(int64_t rowIdx, int64_t rowLen);
    __aicore__ inline void CopyUpdatesInSplitIndices(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyOutSplitIndices(int64_t rowLen, int64_t colLen, int64_t colIdx);
    __aicore__ inline void CopyOutSplitIndicesWithSort(int64_t rowLen, int64_t colLen, int64_t colIdx);
    __aicore__ inline void ProcessSplitAfter();
    __aicore__ inline void ProcessSplitIndices();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tilingData_;

    int64_t curCoreIndexCount_{0};
    uint64_t strideList[MAX_SHAPE_RANK];
};

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                                 GM_ADDR y, GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T*)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    this->indicesFactor_ = tilingData_.indicesFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->indexRankSize_ = tilingData_.indexRankSize;
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;
    this->varInAxis_ = tilingData_.outputStorageShapeSize;
    this->simdWithSort_ = tilingData_.isSimdWithSort;
    this->InitBaseBuffer(pipe_, tilingData_.indicesFactor, indices, updates, y);

    curCoreIndexCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                                 tilingData_.tailCoreIndexCount);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::ProcessSplitAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
                                                                                    tilingData_.updateTailNum;
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = tilingData_.indiceTailNum;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        this->CopyIndiceInSplitAfter(rowIdx, rowDataLen);
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
            this->SortAndComputeUniqueIdx(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
                this->CopyOutSplitAfterWithSort(this->uniqueIdNum_, colDataLen, colIdx);
            }
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            this->uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
                this->CopyOutSplitAfter(rowDataLen, colDataLen, colIdx);
            }
        }
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::CopyIndiceInSplitIndices(int64_t rowIdx, int64_t rowLen)
{
    LocalTensor<U> indicesLocal = this->indicesBuf_.template Get<U>();
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
    LocalTensor<float> dstLocal = this->maxScoreBuf_.template Get<float>();

    int64_t rankSize = tilingData_.indexRankSize;
    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.indicesFactor;
    this->template CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    this->ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
    if (tilingData_.isSimdWithSort == 0) {
        return;
    }
    if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
        IndexStatisticInt32(outOfstLocal, dstLocal, this->maxScore_, rowLen, tilingData_.afterAxis);
    } else {
        IndexStatisticInt64(outOfstLocal, dstLocal, this->maxScore_, rowLen, tilingData_.afterAxis);
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::CopyUpdatesInSplitIndices(int64_t rowIdx, int64_t colIdx,
                                                                                      int64_t rowLen, int64_t colLen)
{
    LocalTensor<T> updatesLocal = this->dataQueue_.template AllocTensor<T>();
    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.indicesFactor;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
    int64_t rowOfset = indicesOfset * tilingData_.afterAxis;
    int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
    DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
    this->dataQueue_.template EnQue(updatesLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::CopyOutSplitIndices(int64_t rowLen, int64_t colLen,
                                                                                int64_t colIdx)
{
    LocalTensor<T> dataLocal = this->dataQueue_.template DeQue<T>();
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    int64_t varInAxis = tilingData_.outputStorageShapeSize;

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t i = 0; i < rowLen; i++) {
        int64_t indicesValue = outOfstLocal(i);
        int64_t outOfset = indicesValue + colIdx * tilingData_.afterAxisFactor;
        if (indicesValue >= 0 && indicesValue < varInAxis) {
            this->template CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
        }
    }
    this->dataQueue_.FreeTensor(dataLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::CopyOutSplitIndicesWithSort(int64_t rowLen, int64_t colLen,
                                                                                        int64_t colIdx)
{
    LocalTensor<T> dataLocal = this->dataQueue_.template DeQue<T>();
    LocalTensor<OFFSET_T> sortIndicesLocal = this->sortIndicesQue_.template Get<OFFSET_T>();
    LocalTensor<OFFSET_T> shiftSortLocal = sortIndicesLocal[this->shiftOffset_];
    LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    int64_t varInAxis = tilingData_.outputStorageShapeSize;

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t i = 0; i < rowLen; i++) {
        int32_t uniqueIdx = uniqueIdCountLocal(i);
        int64_t inOfset = updatesOriginIdexLocal(uniqueIdx) * colLenAlignSize;
        int64_t outOfset = shiftSortLocal(uniqueIdx);
        outOfset += colIdx * tilingData_.afterAxisFactor;
        int64_t indicesValue = shiftSortLocal(uniqueIdx);
        if (indicesValue >= 0 && indicesValue < varInAxis) {
            this->template CopyOut<T>(yGm_[outOfset], dataLocal[inOfset], colLen);
        }
    }

    this->uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
    this->updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
    this->dataQueue_.FreeTensor(dataLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::ProcessSplitIndices()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        CopyIndiceInSplitIndices(rowIdx, rowDataLen);
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
            this->SortAndComputeUniqueIdx(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                CopyOutSplitIndicesWithSort(this->uniqueIdNum_, colDataLen, colIdx);
            }
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                CopyOutSplitIndices(rowDataLen, colDataLen, colIdx);
            }
        }
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimd<T, U, OFFSET_T>::Process()
{
    LocalTensor<OFFSET_T> strideLocal = this->strideBuf_.template Get<OFFSET_T>();
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        strideLocal(i) = tilingData_.strideList[i];
    }

    if (tilingData_.isSplitAfterAxis == 1) {
        ProcessSplitAfter();
    } else {
        ProcessSplitIndices();
    }
}
} // namespace ScatterNdUpdate
#endif // SCATTER_ND_ADD_SIMD_H