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
 * \file scatter_nd_update_simd_mask.h
 * \brief simd mask kernel of scatter_nd_update
 */

#ifndef SCATTER_ND_UPDATE_SIMD_MASK_H
#define SCATTER_ND_UPDATE_SIMD_MASK_H

#include "scatter_nd_update_common.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ScatterNdUpdate {
using namespace AscendC;

template <typename T, typename U, typename OFFSET_T = U>
class ScatterNdUpdateSimdMask : public ScatterNdUpdateBase<T, U, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateSimdMask(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIndiceIn(int64_t rowIdx, int64_t rowLen);
    __aicore__ inline void CopyOutMultiLine(int64_t rowLen);
    __aicore__ inline void ProcessMaskMultiLine();
    __aicore__ inline void CopyUpdatesIn(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyOutOneLine(int64_t colLen, int64_t colIdx);
    __aicore__ inline void ProcessMaskOneLine();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    int64_t curCoreIndexCount_{0};

    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tilingData_;
};

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                                     GM_ADDR y, GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T*)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    curCoreIndexCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                                 tilingData_.tailCoreIndexCount);

    pipe_.InitBuffer(maskBuf_, Ops::Base::CeilAlign(tilingData_.varInAxis * sizeof(int8_t),
                                                    static_cast<uint64_t>(Ops::Base::GetUbBlockSize())));
    pipe_.InitBuffer(this->outOfstBuf_, tilingData_.indicesFactor * sizeof(OFFSET_T));
    pipe_.InitBuffer(this->indicesBuf_, tilingData_.indicesFactor * tilingData_.indexRankSize * sizeof(U));
    pipe_.InitBuffer(this->strideBuf_, MAX_SHAPE_RANK * sizeof(U));
    pipe_.InitBuffer(this->dataQueue_, DOUBLE_BUFFER,
                     tilingData_.indicesFactor * tilingData_.afterAxisFactor * sizeof(T));

    LocalTensor<int8_t> maskLocal = maskBuf_.Get<int8_t>();
    Duplicate(maskLocal, static_cast<int8_t>(0), tilingData_.varStorageInAxis);

    LocalTensor<U> strideLocal = this->strideBuf_.template Get<U>();
    for (int32_t i = 0; i < tilingData_.indexRankSize; i++) {
        strideLocal(i) = tilingData_.strideList[i] / tilingData_.strideList[tilingData_.indexRankSize - 1];
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::CopyIndiceIn(int64_t rowIdx, int64_t rowLen)
{
    LocalTensor<U> indicesLocal = this->indicesBuf_.template Get<U>();
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();

    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.indicesFactor;
    this->template CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * tilingData_.indexRankSize],
                             rowLen * tilingData_.indexRankSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    this->ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, tilingData_.indexRankSize);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::CopyUpdatesIn(int64_t rowIdx, int64_t colIdx,
                                                                              int64_t rowLen, int64_t colLen)
{
    LocalTensor<T> updatesLocal = this->dataQueue_.template AllocTensor<T>();
    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.indicesFactor;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
    int64_t rowOfset = indicesOfset * tilingData_.afterAxis;
    int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
    DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
    this->dataQueue_.template EnQue(updatesLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::CopyOutOneLine(int64_t colLen, int64_t colIdx)
{
    LocalTensor<T> dataLocal = this->dataQueue_.template DeQue<T>();
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t rowOfset = outOfstLocal(0);
    int64_t outOfset = rowOfset * tilingData_.afterAxis + colIdx * tilingData_.afterAxisFactor;
    this->template CopyOut<T>(yGm_[outOfset], dataLocal[0], colLen);
    this->dataQueue_.FreeTensor(dataLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::ProcessMaskOneLine()
{
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
    LocalTensor<int8_t> maskLocal = maskBuf_.template Get<int8_t>();

    for (int64_t rowIdx = 0; rowIdx < curCoreIndexCount_; rowIdx++) {
        CopyIndiceIn(rowIdx, tilingData_.indicesFactor);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        int64_t idx = outOfstLocal(0);
        if (idx < 0 || idx >= tilingData_.varStorageInAxis) {
            continue;
        }
        if (maskLocal(idx) != 0) {
            continue;
        }

        for (int64_t colIdx = 0; colIdx < tilingData_.updateLoopSize; colIdx++) {
            int64_t colDataLen = (colIdx == tilingData_.updateLoopSize - 1) ? tilingData_.updateTailNum :
                                                                              tilingData_.afterAxisFactor;
            CopyUpdatesIn(rowIdx, colIdx, tilingData_.indicesFactor, colDataLen);
            CopyOutOneLine(colDataLen, colIdx);
        }
        maskLocal(idx) = 1;
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::CopyOutMultiLine(int64_t rowLen)
{
    LocalTensor<T> dataLocal = this->dataQueue_.template DeQue<T>();
    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
    LocalTensor<int8_t> maskLocal = maskBuf_.template Get<int8_t>();
    int64_t colLenAlignSize = Ops::Base::CeilAlign(tilingData_.afterAxis * sizeof(T),
                                                   static_cast<uint64_t>(Ops::Base::GetUbBlockSize())) /
                              sizeof(T);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int32_t i = 0; i < rowLen; i++) {
        int64_t srcOffset = i * colLenAlignSize;
        int64_t idx = outOfstLocal(i);
        int64_t outOfstVal = idx * tilingData_.strideList[tilingData_.indexRankSize - 1];
        if (outOfstVal < 0 || outOfstVal >= tilingData_.outputStorageShapeSize) {
            continue;
        }
        if (maskLocal(idx) == 0) {
            this->template CopyOut<T>(yGm_[outOfstVal], dataLocal[srcOffset], tilingData_.afterAxis);
            maskLocal(idx) = 1;
        }
    }
    this->dataQueue_.FreeTensor(dataLocal);
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::ProcessMaskMultiLine()
{
    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);
    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        CopyIndiceIn(rowIdx, rowDataLen);
        CopyUpdatesIn(rowIdx, 0, rowDataLen, tilingData_.afterAxis);
        CopyOutMultiLine(rowDataLen);
    }
}

template <typename T, typename U, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimdMask<T, U, OFFSET_T>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    if (tilingData_.isSplitOneLine == 1) {
        ProcessMaskOneLine();
    } else {
        ProcessMaskMultiLine();
    }
}
} // namespace ScatterNdUpdate
#endif // SCATTER_ND_UPDATE_SIMD_MASK_H