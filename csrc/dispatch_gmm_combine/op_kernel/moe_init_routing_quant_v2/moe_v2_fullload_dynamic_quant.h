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
 * \file moe_v2_fullload_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_base.h"
namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadDynamicQuant : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2FullLoadDynamicQuant(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                              GM_ADDR workspace, const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void CopyOutXQuant1H();
  __aicore__ inline void CopyOutXQuantEH();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();
  __aicore__ inline void Compute(LocalTensor<float>& smoothLocal);

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
  int64_t smoothType;
  int64_t colsAlign;

  TQue<QuePosition::VECIN, 1> xCopyInQueue_;
  TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;
  TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;
  TQue<QuePosition::VECIN, 1> smoothInQueue;
  TQue<QuePosition::VECOUT, 1> calcQueue;
  TQue<QuePosition::VECOUT, 1> inputXOutQueue;
  TQue<QuePosition::VECOUT, 1> scaleOutQueue;

  GlobalTensor<T> xGm_;
  GlobalTensor<int32_t> expertIdxGm_;
  GlobalTensor<float> quantSmoothGm;
  GlobalTensor<float> dynamicQuantScaleGm;

  GlobalTensor<int8_t> expandedXGm_;
  GlobalTensor<int32_t> expandedRowIdxGm_;
  GlobalTensor<int32_t> expandedExpertIdxGm_;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t dropPadMode = 0;

  LocalTensor<uint32_t> expandDstToSrcRowLocal;
  LocalTensor<int32_t> expandedExpertIdxLocal;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)),
                                   0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
  ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
  sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::SortCompute() {
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
  expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
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
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutIdx() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = this->totalLength * sizeof(int32_t);
  DataCopyPad(expandedRowIdxGm_, expandedRowIdx, intriParams);
  expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::ComputeExpertTokenCountOrCumsum() {
  expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
  LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue_.AllocTensor<int32_t>();

  int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
  Duplicate(expertTokensCount, 0, expertNumAlign);
  SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

  int32_t lastExpertId = expandedExpertIdxLocal.GetValue(0);
  int64_t tokenCount = 0;
  int64_t lastExpertCount = 0;
  for (int64_t i = 0; i < this->totalLength; i++) {
    int32_t curExpertId = expandedExpertIdxLocal.GetValue(i);
    tokenCount++;
    while (lastExpertId < curExpertId) {
      expertTokensCount.SetValue(lastExpertId, tokenCount - 1);
      if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
        tokenCount = 1;
      }
      lastExpertId++;
    }
  }
#ifndef __CCE_KT_TEST__
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
#endif
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutEmpty() {
  expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Compute(LocalTensor<float>& smoothLocal) {
  LocalTensor<float> inLocal = xCopyInQueue_.DeQue<float>();

  LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
  LocalTensor<int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();
  LocalTensor<float> dynamicQuantLocal = scaleOutQueue.AllocTensor<float>();

  if constexpr (!IsSameType<T, float>::value) {
    Cast(inLocal, inLocal.ReinterpretCast<T>()[colsAlign], RoundMode::CAST_NONE, this->cols_);
    pipe_barrier(PIPE_V);
  }

  if (smoothType != 0) {
    Mul(inLocal, inLocal, smoothLocal, this->cols_);
    pipe_barrier(PIPE_V);
  }

  Abs(tempLocal, inLocal, this->cols_);
  pipe_barrier(PIPE_V);

  ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, this->cols_);
  pipe_barrier(PIPE_V);

  float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;

  Duplicate<float>(dynamicQuantLocal, maxValue, 8);
  Duplicate<float>(tempLocal, maxValue, this->cols_);
  pipe_barrier(PIPE_V);

  Div(tempLocal, inLocal, tempLocal, this->cols_);
  pipe_barrier(PIPE_V);

  Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, this->cols_);
  pipe_barrier(PIPE_V);

  Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, this->cols_);

  calcQueue.FreeTensor(tempLocal);
  inputXOutQueue.EnQue(outLocal);
  scaleOutQueue.EnQue(dynamicQuantLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutXQuant1H() {
  expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
  expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdxLocal);

  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t curRowsEnd = curRowsStart + this->coreRows_ - 1;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = curRowsEnd / this->k_;

  DataCopyExtParams dataXCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
  DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
  DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};

  LocalTensor<float> smoothLocal;
  if (smoothType == 1) {
    smoothLocal = smoothInQueue.AllocTensor<float>();
    DataCopyPad(smoothLocal, quantSmoothGm, smoothCopyParams, {false, 0, 0, 0});
    smoothInQueue.EnQue(smoothLocal);
    smoothLocal = smoothInQueue.DeQue<float>();
  }
  for (int64_t row = startXRow; row <= endXRow; row++) {
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(xLocal, xGm_[row * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    } else {
      DataCopyPad(xLocal[colsAlign], xGm_[row * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    }

    xCopyInQueue_.EnQue<T>(xLocal);
    Compute(smoothLocal);

    LocalTensor<float> quantScaleLocal = scaleOutQueue.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue.DeQue<int8_t>();
    while (curRowsStart <= curRowsEnd && curRowsStart / this->k_ == row) {
      int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
      curRowsStart++;
      if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows_)) {
        continue;
      }
      DataCopyPad(expandedXGm_[outIndex * cols_], outLocal, intriParams);
      DataCopyPad(dynamicQuantScaleGm[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
    }

    xCopyInQueue_.FreeTensor(xLocal);
    inputXOutQueue.FreeTensor(outLocal);
    scaleOutQueue.FreeTensor(quantScaleLocal);
  }

  if (smoothType == 1) {
    smoothInQueue.FreeTensor(smoothLocal);
  }
  expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutXQuantEH() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);

  Muls(expandDstToSrcRowLocal.ReinterpretCast<float>(), expandDstToSrcRowLocal.ReinterpretCast<float>(), (float)-1,
       this->totalLength);
  pipe_barrier(PIPE_V);
  LocalTensor<int32_t> sortedRowIdx = expandDstToSrcRowLocal.ReinterpretCast<int32_t>();
  Cast(sortedRowIdx, expandDstToSrcRowLocal.ReinterpretCast<float>(), RoundMode::CAST_ROUND, this->totalLength);

  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t curRowsEnd = curRowsStart + this->coreRows_ - 1;

  DataCopyExtParams dataXCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
  DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(this->cols_ * sizeof(float)), 0, 0, 0};
  DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0, 0};

  for (int64_t row = curRowsStart; row <= curRowsEnd; row++) {
    if (this->dropPadMode == DROPLESS_MODE && row >= this->activateRows_) {
      break;
    }
    int32_t srcIdx = sortedRowIdx.GetValue(row);
    int32_t expertIdx = expandedExpertIdxLocal.GetValue(row);

    LocalTensor<T> inLocal = xCopyInQueue_.AllocTensor<T>();
    LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(inLocal, xGm_[srcIdx / this->k_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    } else {
      DataCopyPad(inLocal[colsAlign], xGm_[srcIdx / this->k_ * this->cols_], dataXCopyParams, {false, 0, 0, 0});
    }
    DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols_], smoothCopyParams, {false, 0, 0, 0});
    xCopyInQueue_.EnQue<T>(inLocal);
    smoothInQueue.EnQue(smoothLocal);
    smoothLocal = smoothInQueue.DeQue<float>();

    Compute(smoothLocal);

    LocalTensor<float> quantScaleLocal = scaleOutQueue.DeQue<float>();
    DataCopyPad(dynamicQuantScaleGm[row], quantScaleLocal, {1, 4, 0, 0, 0});

    LocalTensor<int8_t> outLocal = inputXOutQueue.DeQue<int8_t>();
    DataCopyPad(expandedXGm_[row * this->cols_], outLocal, intriParams);

    xCopyInQueue_.FreeTensor(inLocal);
    smoothInQueue.FreeTensor(smoothLocal);
    inputXOutQueue.FreeTensor(outLocal);
    scaleOutQueue.FreeTensor(quantScaleLocal);
  }

  expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
  expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdxLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                                                          GM_ADDR workspace,
                                                          const MoeInitRoutingQuantV2TilingData* tilingData,
                                                          TPipe* tPipe) {
  this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
  //this->blockIdx_ = GetBlockIdx();
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
  this->smoothType = tilingData->smoothType;
  this->colsAlign = Align(this->cols_, sizeof(T));
  this->pipe = tPipe;

  xGm_.SetGlobalBuffer((__gm__ T*)x);
  expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);

  expandedXGm_.SetGlobalBuffer((__gm__ int8_t*)expandedX);
  expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, this->tileLength);
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                Align(this->expertNum, sizeof(int32_t)));
  }
  quantSmoothGm.SetGlobalBuffer((__gm__ float*)quantSmooth);
  dynamicQuantScaleGm.SetGlobalBuffer((__gm__ float*)dynamicQuantScale);

  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum_ * sizeof(int32_t);

  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;

  pipe->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expertTokensCopyOutQueue_, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
  pipe->InitBuffer(expandDstToSrcRowQueue_, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
  pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
  pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);

  if constexpr (IsSameType<T, float>::value) {
    pipe->InitBuffer(xCopyInQueue_, 1, AlignBytes(this->cols_, sizeof(float)));
  } else {
    pipe->InitBuffer(xCopyInQueue_, 1, 2 * AlignBytes(this->cols_, sizeof(T)));
  }
  pipe->InitBuffer(smoothInQueue, 1, AlignBytes(this->cols_, sizeof(float)));
  pipe->InitBuffer(calcQueue, 1, AlignBytes(this->cols_, sizeof(float)));
  pipe->InitBuffer(inputXOutQueue, 1, AlignBytes(this->cols_, sizeof(int8_t)));
  pipe->InitBuffer(scaleOutQueue, 1, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Process() {
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
    if (smoothType == 2) {
      CopyOutXQuantEH();
    } else {
      CopyOutXQuant1H();
    }
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_DYNAMIC_QUANT_FULL_LOAD_H