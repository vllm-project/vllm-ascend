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
 * \file moe_v3_full_load.h
 * \brief
 */
#ifndef MOE_V3_FULL_LOAD_H
#define MOE_V3_FULL_LOAD_H

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeV3FullLoad {
public:
    __aicore__ inline MoeV3FullLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void ExpertCountCompute();
    __aicore__ inline void CopyOutDynamicQuant();

private:
    int64_t sortNum;

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TQue<QuePosition::VECOUT, 1> expertTokensCountOrCumsumOutQueue;
    TQue<QuePosition::VECIN, 1> smoothInQueue;
    TQue<QuePosition::VECIN, 1> inputXInQueue;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue;
    TQue<QuePosition::VECOUT, 1> rowIdxOutQueue;

    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;
    TBuf<TPosition::VECCALC> quantTempBuffer;

    GlobalTensor<bfloat16_t> inputXGm;
    GlobalTensor<float> smoothGm;
    GlobalTensor<int8_t> expandedXGm;
    GlobalTensor<float> expandedScaleGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<int32_t> expendedRowIdxGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;
    GlobalTensor<int32_t> sortedexpertIdxGm;
    GlobalTensor<int32_t> expertCountTempGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int64_t> expertTokensCountOrCumsumGm;

    int64_t blockIdx = 0;
    int64_t tileLength;
    int64_t bufferNum = 1;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols_;
    int64_t expertNum_ = 256;
    int64_t rowIdxType_;
    int64_t kvFactor = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeV3FullLoad::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
    LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
    ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV3FullLoad::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdx = inLocal[0];
    LocalTensor<float> expertIdxFp32 = expertIdx.ReinterpretCast<float>();
    Cast(expertIdxFp32, expertIdx, RoundMode::CAST_ROUND, this->tileLength);
    Muls(expertIdxFp32, expertIdxFp32, (float)-1, this->tileLength);
    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertIdxFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal;
    expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);

    LocalTensor<int32_t> expertForSourceRowLocalInt32;
    expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV3FullLoad::ExpertCountCompute()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    LocalTensor<int32_t> sortedExpertId = outLocal;
    LocalTensor<int64_t> expertTokensLocalTensor = expertTokensCountOrCumsumOutQueue.AllocTensor<int64_t>();

    int64_t i = 0;
    int32_t lastExpertId = sortedExpertId.GetValue(0);
    int32_t lastIndex = 0;
    int64_t index = 0;
    for (i = 1; i < this->totalLength; i++) {
        int32_t curExpertId = sortedExpertId.GetValue(i);
        if (curExpertId != lastExpertId) {
            expertTokensLocalTensor.SetValue(index * kvFactor, lastExpertId);
            expertTokensLocalTensor.SetValue(index * kvFactor + 1, i - lastIndex);
            index++;
            lastIndex = i;
            lastExpertId = curExpertId;
        }
    }
    if (i == this->totalLength) {
        expertTokensLocalTensor.SetValue(index * kvFactor, lastExpertId);
        expertTokensLocalTensor.SetValue(index * kvFactor + 1, i - lastIndex);
        index++;
    }
    // totalLength < 256
    expertTokensLocalTensor.SetValue(index * kvFactor, 0);
    expertTokensLocalTensor.SetValue(index * kvFactor + 1, 0);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);

    expertTokensCountOrCumsumOutQueue.EnQue<int64_t>(expertTokensLocalTensor);
    sortDataCopyOutQueue.EnQue<int32_t>(outLocal);
}

__aicore__ inline void MoeV3FullLoad::CopyOutDynamicQuant()
{
    LocalTensor<int64_t> expertTokensLocalTensor = expertTokensCountOrCumsumOutQueue.DeQue<int64_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = expertNum_ * sizeof(int64_t);
    DataCopyPad(expertTokensCountOrCumsumGm, expertTokensLocalTensor, intriParams);
    expertTokensCountOrCumsumOutQueue.FreeTensor(expertTokensLocalTensor);
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();

    // 量化
    int64_t expertIdx = outLocal.GetValue(blockIdx);
    LocalTensor<bfloat16_t> xInLocal = inputXInQueue.AllocTensor<bfloat16_t>();
    LocalTensor<int8_t> xOutLocal = inputXOutQueue.AllocTensor<int8_t>();
    LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();
    LocalTensor<float> scaleLocal = scaleOutQueue.AllocTensor<float>();
    LocalTensor<float> tempLocal = quantTempBuffer.Get<float>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cols_ * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyPad(xInLocal, inputXGm, copyInParams, {false, 0, 0, 0});
    DataCopyPad(smoothLocal, smoothGm[expertIdx * cols_], smoothParams, {false, 0, 0, 0});
    smoothInQueue.EnQue<float>(smoothLocal);
    smoothLocal = smoothInQueue.DeQue<float>();
    Cast(tempLocal, xInLocal, RoundMode::CAST_NONE, cols_);
    Mul(smoothLocal, tempLocal, smoothLocal, cols_);
    // compute scale
    Abs(tempLocal, smoothLocal, cols_);
    ReduceMax(scaleLocal, tempLocal, tempLocal, cols_);
    float scaleValue = scaleLocal.GetValue(0) / 127.0f;
    Duplicate<float>(scaleLocal, scaleValue, DST_REP_STRIDE);
    Duplicate<float>(tempLocal, scaleValue, cols_);
    // compute quant
    Div(tempLocal, smoothLocal, tempLocal, cols_);
    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_ODD, cols_);  // fp32->fp16
    Cast(xOutLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_RINT, cols_); // fp16->int8
    inputXOutQueue.EnQue<int8_t>(xOutLocal);
    xOutLocal = inputXOutQueue.DeQue<int8_t>();
    scaleOutQueue.EnQue<float>(scaleLocal);
    scaleLocal = scaleOutQueue.DeQue<float>();
    DataCopyPad(expandedXGm[blockIdx * cols_], xOutLocal, copyOutParams);
    DataCopyPad(expandedScaleGm[blockIdx], scaleLocal, {1, 4, 0, 0, 0});
    smoothInQueue.FreeTensor(smoothLocal);
    inputXInQueue.FreeTensor(xInLocal);
    inputXOutQueue.FreeTensor(xOutLocal);
    scaleOutQueue.FreeTensor(scaleLocal);

    if (blockIdx == 0) {
        intriParams.blockLen = this->totalLength * sizeof(int32_t);
        if (rowIdxType_ == 1) {
            DataCopyPad(expandedRowIdxGm, outLocal[this->sortNum], intriParams);
        } else if (rowIdxType_ == 0) {
            LocalTensor rowIdxLocalTensor = rowIdxOutQueue.AllocTensor<int32_t>();
            for (int i = 0; i < this->totalLength; i++) {
                int32_t dstIdx = outLocal[this->sortNum].GetValue(i);
                rowIdxLocalTensor.SetValue(dstIdx, i);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(expandedRowIdxGm, rowIdxLocalTensor, intriParams);
            rowIdxOutQueue.FreeTensor(rowIdxLocalTensor);
        }
    }
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV3FullLoad::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                           GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                           GM_ADDR expandedScale, const MoeInitRoutingV3TilingData *tilingData,
                                           TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    cols_ = tilingData->cols;
    rowIdxType_ = tilingData->rowIdxType;

    expertIdxGm.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);

    expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength);
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum, this->tileLength);

    inputXGm.SetGlobalBuffer((__gm__ bfloat16_t *)x, this->n * cols_);
    smoothGm.SetGlobalBuffer((__gm__ float *)scale, expertNum_ * cols_);
    expandedXGm.SetGlobalBuffer((__gm__ int8_t *)expandedX, this->n * cols_ * this->k);
    expandedScaleGm.SetGlobalBuffer((__gm__ float *)expandedScale, this->n * this->k);

    // key and value
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
    pipe->InitBuffer(expertTokensCountOrCumsumOutQueue, bufferNum, Align(expertNum_ * kvFactor, sizeof(int32_t)));

    pipe->InitBuffer(smoothInQueue, bufferNum, AlignBytes(cols_, sizeof(float)));
    pipe->InitBuffer(inputXInQueue, bufferNum, AlignBytes(cols_, sizeof(bfloat16_t)));
    pipe->InitBuffer(inputXOutQueue, bufferNum, AlignBytes(cols_, sizeof(int8_t)));
    pipe->InitBuffer(quantTempBuffer, AlignBytes(cols_, sizeof(float)));
    pipe->InitBuffer(scaleOutQueue, bufferNum, AlignBytes(1, sizeof(float)));
    pipe->InitBuffer(rowIdxOutQueue, bufferNum, AlignBytes(this->totalLength, sizeof(int32_t)));
}

__aicore__ inline void MoeV3FullLoad::Process()
{
    if (this->blockIdx < GetBlockNum()) {
        CopyIn();
        SortCompute();
        ExpertCountCompute();
        CopyOutDynamicQuant();
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_H