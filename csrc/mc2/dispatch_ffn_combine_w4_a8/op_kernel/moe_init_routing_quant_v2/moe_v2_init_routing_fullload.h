/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_init_routing_fullload.h
 * \brief
 */
#ifndef INNER_MOE_V2_FULL_LOAD_H
#define INNER_MOE_V2_FULL_LOAD_H

#include "moe_v2_mrgsort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoad : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2FullLoad(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                              const InnerMoeInitRoutingV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void CopyOutX();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();

 private:
  int64_t sortNum_;
  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData_;
  int64_t blockIdx_;
  int64_t needCoreNum_;
  int64_t coreRows_;
  int64_t perCoreRows_;
  int64_t k_;
  int64_t n_;
  int64_t cols_;
  int64_t activateRows_;
  int64_t expertNum;
  int64_t expertCapacity;

  TQue<QuePosition::VECIN, 1> xCopyInQueue_;
  TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;
  TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;

  GlobalTensor<T> xGm_;
  GlobalTensor<int32_t> expertIdxGm_;

  GlobalTensor<T> expandedXGm_;
  GlobalTensor<int32_t> expandedRowIdxGm_;
  GlobalTensor<int32_t> expandedExpertIdxGm_;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t dropPadMode = 0;
};

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)),
                                   0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
  ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
  sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::SortCompute() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertIdxLocal = inLocal[0];
  LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
  Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
  pipe_barrier(PIPE_V);
  Muls(expertIdxLocalFp32, expertIdxLocalFp32, (float)-1, this->totalLength);
  pipe_barrier(PIPE_V);
  int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
  if (duplicateNum > 0) {
    int duplicateIndex = this->totalLength - duplicateNum;
    uint64_t mask0 = UINT64_MAX;
    mask0 = mask0 << duplicateNum;
    mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
    uint64_t mask[2] = {mask0, 0};
    Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    pipe_barrier(PIPE_V);
  }
  LocalTensor<float> concatLocal;
  LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
  Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
  LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
  Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<float>();
  LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
  LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();
  Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
       this->totalLength);
  pipe_barrier(PIPE_V);
  Muls(expandedExpertIdxLocal, expandedExpertIdxLocal, (float)-1, this->totalLength);
  pipe_barrier(PIPE_V);
  LocalTensor<int32_t> expandedExpertIdxLocalInt32;
  expandedExpertIdxLocalInt32 = expandedExpertIdxLocal.ReinterpretCast<int32_t>();
  Cast(expandedExpertIdxLocalInt32, expandedExpertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
  pipe_barrier(PIPE_V);
  expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocalInt32);

  LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<uint32_t>();
  LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
  Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, this->totalLength);
  pipe_barrier(PIPE_V);
  ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
  pipe_barrier(PIPE_V);
  if (duplicateNum > 0) {
    int duplicateIndex = this->totalLength - duplicateNum;
    uint64_t mask0 = UINT64_MAX;
    mask0 = mask0 << duplicateNum;
    mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
    uint64_t mask[2] = {mask0, 0};
    Duplicate(expandDstToSrcRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    pipe_barrier(PIPE_V);
  }
  Concat(concatLocal, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Extract(tempTensor, expandedRowIdxU32, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  expandedRowIdxCopyOutQueue_.EnQue<uint32_t>(expandedRowIdx);
  sortDataCopyInQueue.FreeTensor(inLocal);

  expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutIdx() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = this->totalLength * sizeof(int32_t);
  DataCopyPad(expandedRowIdxGm_, expandedRowIdx, intriParams);
  expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::ComputeExpertTokenCountOrCumsum() {
  LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
  LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue_.AllocTensor<int32_t>();

  int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
  Duplicate(expertTokensCount, 0, expertNumAlign);
  SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

  int32_t lastExpertId = expandedExpertIdx.GetValue(0);
  int64_t tokenCount = 0;
  int64_t lastExpertCount = 0;
  for (int64_t i = 0; i < this->totalLength; i++) {
    int32_t curExpertId = expandedExpertIdx.GetValue(i);
    tokenCount++;
    while (lastExpertId < curExpertId) {
      expertTokensCount.SetValue(lastExpertId, tokenCount - 1);
      if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
        tokenCount = 1;
      }
      lastExpertId++;
    }
  }
  expertTokensCount.SetValue(lastExpertId, tokenCount);
  if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM) {
    lastExpertId++;
    while (lastExpertId < this->expertNum) {
      expertTokensCount.SetValue(lastExpertId, tokenCount);
      lastExpertId++;
    }
  }
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->expertNum * sizeof(int32_t)), 0, 0,
                               0};
  if (this->expertTokensCountOrCumsumFlag > 0) {
    DataCopyPad(expertTokensCountOrCumsumGm, expertTokensCount, copyParams);
  }
  expertTokensCopyOutQueue_.FreeTensor(expertTokensCount);
  expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutX() {
  LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = this->cols_ * sizeof(T);
  int64_t inFactor = Align(this->cols_, sizeof(T));
  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;

  DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(endXRow - startXRow + 1),
                                    static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, 0};
  DataCopyPad(xLocal, xGm_[startXRow * this->cols_], dataXCopyParams, dataXCopyPadParams);
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

  int64_t k = 0;
  for (int64_t i = startXRow; i <= endXRow; i++) {
    for (; k < this->perCoreRows_ && curRowsStart / this->k_ == i; curRowsStart++, k++) {
      int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
      if (outIndex < this->activateRows_) {
        DataCopyPad(expandedXGm_[outIndex * this->cols_], xLocal[(i - startXRow) * inFactor], intriParams);
      }
    }
  }
  expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
  xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutEmpty() {
  LocalTensor<int32_t> outLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
  expandedExpertIdxCopyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                              const InnerMoeInitRoutingV2TilingData* tilingData, TPipe* tPipe) {
  this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
  this->blockIdx_ = get_block_idx() + get_subblockid() * get_block_num();
  this->k_ = tilingData->k;
  this->n_ = tilingData->n;
  this->cols_ = tilingData->cols;
  this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;
  this->perCoreRows_ = this->gatherOutTilingData_->perCoreRows;
  this->activateRows_ = this->gatherOutTilingData_->activateRows;
  if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
    this->coreRows_ = this->gatherOutTilingData_->lastCoreRows;
  } else {
    this->coreRows_ = this->gatherOutTilingData_->perCoreRows;
  }
  this->expertNum = tilingData->expertNum;
  this->dropPadMode = tilingData->dropPadMode;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum_ = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->pipe = tPipe;

  xGm_.SetGlobalBuffer((__gm__ T*)x);
  expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);

  expandedXGm_.SetGlobalBuffer((__gm__ T*)expandedX);
  expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, this->tileLength);
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                Align(this->expertNum, sizeof(int32_t)));
  }

  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum_ * sizeof(int32_t);

  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;
  pipe->InitBuffer(xCopyInQueue_, bufferNum, AlignBytes(this->cols_, sizeof(T)) * (endXRow - startXRow + 1));
  pipe->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expertTokensCopyOutQueue_, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
  pipe->InitBuffer(expandDstToSrcRowQueue_, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
  pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
  pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Process() {
  if (this->blockIdx_ < this->needCoreNum_) {
    CopyIn();
    SortCompute();
    if (this->blockIdx_ == 0) {
      CopyOutIdx();
    }
    if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
      ComputeExpertTokenCountOrCumsum();
    } else {
      CopyOutEmpty();
    }
    CopyOutX();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_FULL_LOAD_H