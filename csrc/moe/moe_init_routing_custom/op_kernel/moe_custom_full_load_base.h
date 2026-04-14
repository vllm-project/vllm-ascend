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
 * \file moe_custom_base_full_load.h
 * \brief
 */
#ifndef MOE_CUSTOM_FULL_LOAD_BASE_H
#define MOE_CUSTOM_FULL_LOAD_BASE_H

#include "moe_custom_common.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

template <typename T>
class MoeCustomFullLoadBase {
public:
    __aicore__ inline MoeCustomFullLoadBase(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                GM_ADDR workspace, const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void TilingInKernel();
    __aicore__ inline void SortComputeWithRange();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOutIdx();
    __aicore__ inline void CopyOutDefaultGatherIdx();
    __aicore__ inline void CopyOutDefaultTokenCountOrCumsum();
    __aicore__ inline void ComputeExpertTokenCountOrCumsum();

protected:
    int64_t sortNum_;
    const MoeCustomGatherOutComputeTilingData *gatherOutTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t coreIndicesElements_;
    int64_t perCoreIndicesElements_;
    int64_t k_;
    int64_t n_;
    int64_t cols_;
    int64_t dropPadMode_;
    int64_t activeNum_;
    int64_t expertNum_;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t bufferNum_ = 1;
    int64_t kvFactor_ = 2;
    int64_t totalLength_;
    int64_t tileLength_;
    int64_t expertTokensNumType_ = 0;
    int64_t expertTokensNumFlag_ = 0;
    uint64_t actual_idx_num_ = 0;
    int64_t ep_ = 0;
    int64_t gatherFirstFullload_ = 0;
    int64_t isInputScale_ = 0;
    int64_t rowIdxType_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t expertCountElements_ = 0;
    int64_t curIndexStart_;
    int64_t startXRow_;
    int64_t endXRow_;
    int64_t quantMode_ = -1;

    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;
    TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> sortDataCopyInQueue_;

    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> sortedBuffer_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensCountOrCumsumGm_;

    TPipe *pipe_;
};

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                                                      GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                                      const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    this->blockIdx_ = GetBlockIdx();
    this->n_ = tilingData->n;
    this->k_ = tilingData->k;
    this->cols_ = tilingData->cols;
    this->expertStart_ = tilingData->expertStart;
    this->expertEnd_ = tilingData->expertEnd;
    this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;

    this->perCoreIndicesElements_ = this->gatherOutTilingData_->perCoreIndicesElements;
    this->dropPadMode_ = tilingData->dropPadMode;
    this->activeNum_ = tilingData->activeNum;
    this->quantMode_ = tilingData->quantMode;
    if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
        this->coreIndicesElements_ = this->gatherOutTilingData_->lastCoreIndicesElements;
    } else {
        this->coreIndicesElements_ = this->gatherOutTilingData_->perCoreIndicesElements;
    }
    this->expertTokensNumType_ = tilingData->expertTokensNumType;
    this->expertTokensNumFlag_ = tilingData->expertTokensNumFlag;
    this->expertNum_ = tilingData->expertNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->ep_ = tilingData->ep;
    this->gatherFirstFullload_ = tilingData->gatherFirstFullload;
    this->isInputScale_ = tilingData->isInputScale;
    this->tileLength_ = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->actual_idx_num_ = this->totalLength_;
    this->rowIdxType_ = tilingData->rowIdxType;
    this->actualExpertNum_ = tilingData->actualExpertNum;
    this->pipe_ = tPipe;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength_);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength_);
    if (this->expertTokensNumFlag_ > 0) {
        expertTokensCountOrCumsumGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertCountElements_ = expertNum_ * EXERPT_TOKENS_KEY_VALUE;
    } else {
        expertCountElements_ = actualExpertNum_;
    }
    int64_t buffSize = this->sortNum_ * sizeof(int32_t);

    curIndexStart_ = this->blockIdx_ * this->perCoreIndicesElements_;
    startXRow_ = curIndexStart_ / this->k_;
    endXRow_ = (curIndexStart_ + this->coreIndicesElements_ - 1) / this->k_;

    pipe_->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(expertTokensCopyOutQueue_, bufferNum_, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(expandDstToSrcRowQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum_, buffSize);
    pipe_->InitBuffer(sortDataCopyInQueue_, bufferNum_, buffSize * kvFactor_);
    pipe_->InitBuffer(tempBuffer_, buffSize * kvFactor_);
    pipe_->InitBuffer(sortedBuffer_, buffSize * kvFactor_);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0,
                                     0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, totalLength_);
    sortDataCopyInQueue_.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::Compute()
{
    if (ep_) {
        SortComputeWithRange();
    } else {
        SortCompute();
    }
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::SortComputeWithRange()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdxLocal = inLocal[0];
    LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, totalLength_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxLocalFp32, expertIdxLocalFp32, (float)-1, totalLength_);
    PipeBarrier<PIPE_V>();
    if (gatherFirstFullload_) {
        int64_t maskOffset = AlignBytes(Ceil(totalLength_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE, sizeof(int8_t));
        LocalTensor<uint8_t> compareScalarMaskLocalTensor0 = tempBuffer_.Get<uint8_t>()[maskOffset];
        LocalTensor<uint8_t> compareScalarMaskLocalTensor1 = tempBuffer_.Get<uint8_t>()[maskOffset * kvFactor_];
        LocalTensor<uint8_t> gatherMaskLocalTensor = tempBuffer_.Get<uint8_t>();

        // Find elements >= expertStart_, which means -elements <= -expertStart_
        AscendC::CompareScalar(
            compareScalarMaskLocalTensor0, expertIdxLocalFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::LE,
            (totalLength_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
        PipeBarrier<PIPE_V>();

        // Find elements < expertEnd_, which means -elements > -expertEnd_
        AscendC::CompareScalar(
            compareScalarMaskLocalTensor1, expertIdxLocalFp32, static_cast<float>(-expertEnd_), AscendC::CMPMODE::GT,
            (totalLength_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
        PipeBarrier<PIPE_V>();

        And(gatherMaskLocalTensor.ReinterpretCast<uint16_t>(),
            compareScalarMaskLocalTensor0.ReinterpretCast<uint16_t>(),
            compareScalarMaskLocalTensor1.ReinterpretCast<uint16_t>(),
            Ceil(totalLength_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / kvFactor_);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = 1;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
        gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
        GatherMask(expertIdxLocalFp32, expertIdxLocalFp32, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(totalLength_), gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        actual_idx_num_ = rsvdCnt;
        sortNum_ = Ceil(actual_idx_num_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

        GatherMask(rowIdxLocal, rowIdxLocal, gatherMaskLocalTensor.ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(totalLength_), gatherMaskParams, actual_idx_num_);
        PipeBarrier<PIPE_V>();
        TilingInKernel();
    } else {
        LocalTensor<uint8_t> maskLocalTensor = tempBuffer_.Get<uint8_t>();
        AscendC::CompareScalar(
            maskLocalTensor, expertIdxLocalFp32, static_cast<float>(-expertStart_), AscendC::CMPMODE::GT,
            (totalLength_ + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);
        LocalTensor<float> floatMinLocalTensor = sortedBuffer_.Get<float>();
        Duplicate(floatMinLocalTensor, MIN_FP32, totalLength_);
        PipeBarrier<PIPE_V>();
        Select(expertIdxLocalFp32, maskLocalTensor, floatMinLocalTensor, expertIdxLocalFp32,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, totalLength_);
        PipeBarrier<PIPE_V>();
    }
    // handle actual_idx_num_ == 0
    if (actual_idx_num_ < 1) {
        sortDataCopyInQueue_.FreeTensor(inLocal);
        return;
    }
    int64_t duplicateNum = actual_idx_num_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = actual_idx_num_ - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> concatLocal = expertIdxLocalFp32;
    LocalTensor<float> tempTensor = tempBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
    Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    Muls(expandedExpertIdxLocal, expandedExpertIdxLocal, (float)-1, actual_idx_num_);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> expandedExpertIdxLocalInt32;
    expandedExpertIdxLocalInt32 = expandedExpertIdxLocal.ReinterpretCast<int32_t>();
    Cast(expandedExpertIdxLocalInt32, expandedExpertIdxLocal, RoundMode::CAST_ROUND, actual_idx_num_);
    PipeBarrier<PIPE_V>();
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocalInt32);
    expandDstToSrcRowQueue_.EnQue<uint32_t>(expandDstToSrcRowLocal);
    sortDataCopyInQueue_.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdxLocal = inLocal[0];
    LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, totalLength_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxLocalFp32, expertIdxLocalFp32, (float)-1, totalLength_);
    PipeBarrier<PIPE_V>();
    int64_t duplicateNum = totalLength_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = totalLength_ - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<float> concatLocal = expertIdxLocalFp32;
    LocalTensor<float> tempTensor = tempBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
    LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();
    Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<uint32_t>();
    Muls(expandedExpertIdxLocal, expandedExpertIdxLocal, (float)-1, totalLength_);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> expandedExpertIdxLocalInt32;
    expandedExpertIdxLocalInt32 = expandedExpertIdxLocal.ReinterpretCast<int32_t>();
    Cast(expandedExpertIdxLocalInt32, expandedExpertIdxLocal, RoundMode::CAST_ROUND, totalLength_);
    PipeBarrier<PIPE_V>();

    Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
         totalLength_);
    PipeBarrier<PIPE_V>();
    Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, totalLength_);
    PipeBarrier<PIPE_V>();
    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, totalLength_);
    PipeBarrier<PIPE_V>();
    if (duplicateNum > 0) {
        int duplicateIndex = totalLength_ - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expandDstToSrcRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    Concat(concatLocal, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    Extract(tempTensor, expandedRowIdx, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();

    if (rowIdxType_ == SCATTER or quantMode_ == 1) {
        Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, totalLength_);
        PipeBarrier<PIPE_V>();
        Cast(expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), expandDstToSrcRowLocalFp32, RoundMode::CAST_RINT,
             totalLength_);
    }
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocalInt32);
    expandedRowIdxCopyOutQueue_.EnQue<uint32_t>(expandedRowIdx);
    expandDstToSrcRowQueue_.EnQue<uint32_t>(expandDstToSrcRowLocal);
    sortDataCopyInQueue_.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::CopyOutDefaultGatherIdx()
{
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<int32_t>();
    Duplicate(expandedRowIdx, static_cast<int32_t>(-1), static_cast<int32_t>(totalLength_));
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPad(expandedRowIdxGm_, expandedRowIdx, copyParams);
    expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::CopyOutDefaultTokenCountOrCumsum()
{
    LocalTensor<int64_t> expertTokensOut = expertTokensCopyOutQueue_.AllocTensor<int64_t>();
    Duplicate(expertTokensOut.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
              static_cast<int32_t>(expertCountElements_ * EXERPT_TOKENS_KEY_VALUE));
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountOrCumsumGm_, expertTokensOut, copyParams);
    expertTokensCopyOutQueue_.FreeTensor(expertTokensOut);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::CopyOutIdx()
{
    LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.DeQue<int32_t>();
    if (rowIdxType_ == SCATTER) {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(actual_idx_num_ * sizeof(int32_t)),
                                     0, 0, 0};
        DataCopyPad(expandedRowIdxGm_, expandDstToSrcRowLocal, copyParams);
    } else if (ep_) {
        LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<int32_t>();
        Duplicate(expandedRowIdx, static_cast<int32_t>(-1), static_cast<int32_t>(totalLength_));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (int64_t i = 0; i < actual_idx_num_; i++) {
            int32_t curExpertId = expandedExpertIdx.GetValue(i);
            if (curExpertId < expertStart_ || curExpertId >= expertEnd_) {
                break;
            }
            int64_t outIndices = expandDstToSrcRowLocal.GetValue(i);
            expandedRowIdx.SetValue(outIndices, i);
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0,
                                     0, 0};
        DataCopyPad(expandedRowIdxGm_, expandedRowIdx, copyParams);
        expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
    } else {
        LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalLength_ * sizeof(int32_t)), 0,
                                     0, 0};
        DataCopyPad(expandedRowIdxGm_, expandedRowIdx, copyParams);
        expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
    }
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdx);
    expandDstToSrcRowQueue_.EnQue<int32_t>(expandDstToSrcRowLocal);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::ComputeExpertTokenCountOrCumsum()
{
    // compute
    LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertTokensOut = expertTokensCopyOutQueue_.AllocTensor<int64_t>();
    Duplicate(expertTokensOut.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
              static_cast<int32_t>(expertCountElements_ * EXERPT_TOKENS_KEY_VALUE));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    int64_t i = 0;
    int32_t lastExpertId = expandedExpertIdx.GetValue(0);
    int32_t lastLastId = lastExpertId;
    int64_t tokenCount = 0;
    int64_t lastIndex = 0;
    int64_t Offset = 0;
    for (i = 1; i < actual_idx_num_; i++) {
        if ((lastExpertId >= expertEnd_) || (lastExpertId < expertStart_)) {
            break;
        }
        int32_t curExpertId = expandedExpertIdx.GetValue(i);
        if (curExpertId != lastExpertId || curExpertId >= expertEnd_) {
            int64_t expertOffset = lastExpertId - expertStart_;
            if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
                expertTokensOut.SetValue(Offset * EXERPT_TOKENS_KEY_VALUE, lastExpertId);
                expertTokensOut.SetValue(Offset * EXERPT_TOKENS_KEY_VALUE + 1, i - lastIndex);
                Offset += 1;
            } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
                expertTokensOut.SetValue(expertOffset, i - lastIndex);
            } else {
                for (int64_t j = lastLastId; j < lastExpertId; j++) {
                    expertTokensOut.SetValue(j - expertStart_, tokenCount);
                }
                tokenCount += i - lastIndex;
                expertTokensOut.SetValue(expertOffset, tokenCount);
            }
            lastIndex = i;
            lastLastId = lastExpertId;
            lastExpertId = curExpertId;
        }
    }
    if ((i == actual_idx_num_) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
        int64_t expertOffset = lastExpertId - expertStart_;
        if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
            expertTokensOut.SetValue(Offset * EXERPT_TOKENS_KEY_VALUE, lastExpertId);
            expertTokensOut.SetValue(Offset * EXERPT_TOKENS_KEY_VALUE + 1, i - lastIndex);
        } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
            expertTokensOut.SetValue(expertOffset, i - lastIndex);
        } else {
            for (int64_t j = lastLastId; j < lastExpertId; j++) {
                expertTokensOut.SetValue(j - expertStart_, tokenCount);
            }
            tokenCount += i - lastIndex;
            expertTokensOut.SetValue(expertOffset, tokenCount);
            for (int64_t j = lastExpertId; j < expertEnd_; j++) {
                expertTokensOut.SetValue(j - expertStart_, tokenCount);
            }
        }
    } else {
        if (expertTokensNumType_ == EXERPT_TOKENS_CUMSUM) {
            for (int64_t j = lastLastId; j < expertEnd_; j++) {
                expertTokensOut.SetValue(j - expertStart_, tokenCount);
            }
        }
    }
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdx);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountOrCumsumGm_, expertTokensOut, copyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    expertTokensCopyOutQueue_.FreeTensor(expertTokensOut);
}

template <typename T>
__aicore__ inline void MoeCustomFullLoadBase<T>::TilingInKernel()
{
    int64_t coreNum = needCoreNum_;
    perCoreIndicesElements_ = Ceil(actual_idx_num_, coreNum);
    needCoreNum_ = Ceil(actual_idx_num_, perCoreIndicesElements_);
    int64_t lastCoreIndicesElements = actual_idx_num_ - (needCoreNum_ - 1) * perCoreIndicesElements_;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreIndicesElements_ = lastCoreIndicesElements;
    } else {
        coreIndicesElements_ = perCoreIndicesElements_;
    }
    curIndexStart_ = this->blockIdx_ * this->perCoreIndicesElements_;
    startXRow_ = curIndexStart_ / this->k_;
    endXRow_ = (curIndexStart_ + this->coreIndicesElements_ - 1) / this->k_;
}

} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_FULL_LOAD_BASE_H