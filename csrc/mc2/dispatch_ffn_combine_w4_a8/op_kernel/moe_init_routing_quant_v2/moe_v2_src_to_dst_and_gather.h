/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v2_src_to_dst_and_gather.h
 * \brief
 */
#ifndef MOE_V2_SRC_TO_DST_AND_GATHER_H
#define MOE_V2_SRC_TO_DST_AND_GATHER_H

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T, typename TilingData>
class MoeV2SrcToDstAndGather {
 public:
  __aicore__ inline MoeV2SrcToDstAndGather(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                              GM_ADDR dynamicQuantScale, GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void CopyOut(int64_t progress);
  __aicore__ inline void CopyOutLoops(int64_t progress);
  __aicore__ inline void Compute(int32_t srcIdx, int32_t dstIdx, int32_t expertIdx);
  __aicore__ inline float ComputeMax(LocalTensor<float>& inLocal, LocalTensor<float>& tempLocal,
                                     LocalTensor<float>& dynamicQuantLocal, int32_t srcIdx, int32_t expertIdx,
                                     int64_t j);
  __aicore__ inline void ComputeScale(LocalTensor<float>& inLocal, LocalTensor<float>& tempLocal, float scaleTemp,
                                      int64_t dstIndex, int64_t j);
  __aicore__ inline void ComputeLoops(int32_t srcIdx, int32_t dstIdx, int32_t expertIdx);

  __aicore__ inline void CopyOutRemain();
  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> copyInQueue;
  TQue<QuePosition::VECOUT, 1> copyOutQueue;
  TQue<QuePosition::VECOUT, 1> copyOutZeroQueue;

  TQue<QuePosition::VECIN, 1> inputXInQueue;
  TQue<QuePosition::VECIN, 1> smoothInQueue;
  TQue<QuePosition::VECOUT, 1> calcQueue;
  TQue<QuePosition::VECOUT, 1> inputXOutQueue;
  TQue<QuePosition::VECOUT, 1> scaleOutQueue;
  TQue<QuePosition::VECOUT, 1> scaleOutZeroQueue;

  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expandedRowIdxGm;
  GlobalTensor<int32_t> expertIdxValueGm;
  GlobalTensor<int32_t> expandedExpertIdxGm;
  GlobalTensor<int8_t> expandedXGm;

  GlobalTensor<T> inputXGm;
  GlobalTensor<float> quantSmoothGm;
  GlobalTensor<float> dynamicQuantScaleGm;
  GlobalTensor<float> quantSrcGm;

  LocalTensor<int8_t> outTmpLocal;
  LocalTensor<float> scaleOutTmpLocal;
  LocalTensor<float> smoothLocal;

  const InnerMoeV2GatherOutComputeTilingData* srcToDstTilingData;

  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t rowLoops;
  int64_t expertCapacity;
  int64_t expertNum;
  int64_t cols;
  int64_t perLoopCols;
  int64_t lastLoopCols;
  int64_t colLoops;
  int64_t perLoopColsAlign;
  int64_t k;
  int64_t colsTileLength;
  int64_t smoothType;

  int64_t tokenCount = 0;
  int32_t lastExpertId = -1;
  int32_t lastCoreExpertId = 0;
  int32_t lastCoreExpertIdNum = 0;
};

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::AssistInit() {
  LocalTensor<int16_t> outLocal = copyOutZeroQueue.AllocTensor<int16_t>();
  Duplicate<int16_t>(outLocal, static_cast<int16_t>(0), this->perLoopCols);
  copyOutZeroQueue.EnQue<int16_t>(outLocal);
  LocalTensor<float> scaleOutLocal = scaleOutZeroQueue.AllocTensor<float>();
  Duplicate<float>(scaleOutLocal, 0.0f, 8);
  scaleOutZeroQueue.EnQue<float>(scaleOutLocal);

  if (this->blockIdx != 0) {
    this->lastCoreExpertId = expertIdxValueGm.GetValue((this->blockIdx - 1) * 2);
    this->lastCoreExpertIdNum = expertIdxValueGm.GetValue((this->blockIdx - 1) * 2 + 1);
    for (int64_t i = this->blockIdx - 2; i >= 0; i--) {
      int32_t lastExpertIdx = expertIdxValueGm.GetValue(i * 2);
      if (lastExpertIdx < this->lastCoreExpertId) {
        break;
      }
      int32_t lastExpertNum = expertIdxValueGm.GetValue(i * 2 + 1);
      this->lastCoreExpertIdNum += lastExpertNum;
    }
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyIn(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
  int64_t length = Align(currentLoopRows, sizeof(int32_t));
  DataCopy(inLocal, expandDstToSrcRowGm[progress * perLoopRows], length);
  DataCopy(inLocal[length], expandedExpertIdxGm[progress * perLoopRows], length);

  copyInQueue.EnQue<int32_t>(inLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Compute(int32_t srcIdx, int32_t dstIdx,
                                                                      int32_t expertIdx) {
  DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
  DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols * sizeof(float)), 0, 0, 0};
  DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};

  LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();

  if constexpr (IsSameType<T, float>::value) {
    DataCopyPad(inLocal, inputXGm[srcIdx / this->k * this->cols], copyInParams, {false, 0, 0, 0});
  } else {
    DataCopyPad(inLocal.template ReinterpretCast<T>()[perLoopColsAlign], inputXGm[srcIdx / this->k * this->cols],
                copyInParams, {false, 0, 0, 0});
  }

  if (smoothType == 2) {
    DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols], smoothParams, {false, 0, 0, 0});
  }

  inputXInQueue.EnQue<float>(inLocal);
  smoothInQueue.EnQue(smoothLocal);
  smoothLocal = smoothInQueue.DeQue<float>();

  inLocal = inputXInQueue.DeQue<float>();

  LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
  LocalTensor<int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();
  LocalTensor<float> dynamicQuantLocal = scaleOutQueue.AllocTensor<float>();

  if constexpr (!IsSameType<T, float>::value) {
    Cast(inLocal, inLocal.template ReinterpretCast<T>()[perLoopColsAlign], RoundMode::CAST_NONE, this->cols);
    pipe_barrier(PIPE_V);
  }

  if (smoothType != 0) {
    Mul(inLocal, inLocal, smoothLocal, this->cols);
    pipe_barrier(PIPE_V);
  }

  Abs(tempLocal, inLocal, this->cols);
  pipe_barrier(PIPE_V);

  ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, this->cols);
  pipe_barrier(PIPE_V);

  float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;

  Duplicate<float>(dynamicQuantLocal, maxValue, 8);
  Duplicate<float>(tempLocal, maxValue, this->cols);
  pipe_barrier(PIPE_V);

  Div(tempLocal, inLocal, tempLocal, this->cols);
  pipe_barrier(PIPE_V);

  Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, this->cols);
  pipe_barrier(PIPE_V);

  Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, this->cols);

  calcQueue.FreeTensor(tempLocal);
  inputXOutQueue.EnQue(outLocal);
  scaleOutQueue.EnQue(dynamicQuantLocal);

  LocalTensor<float> quantScaleLocal = scaleOutQueue.DeQue<float>();
  DataCopyPad(dynamicQuantScaleGm[dstIdx], quantScaleLocal, {1, 4, 0, 0, 0});

  outLocal = inputXOutQueue.DeQue<int8_t>();
#ifndef __CCE_KT_TEST__
  DataCopyPad(expandedXGm[dstIdx * this->cols], outLocal, copyOutParams);
#endif
  inputXInQueue.FreeTensor(inLocal);
  inputXOutQueue.FreeTensor(outLocal);
  scaleOutQueue.FreeTensor(quantScaleLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOut(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> outLocal = copyOutQueue.AllocTensor<int32_t>();
  int64_t length = Align(currentLoopRows, sizeof(int32_t));
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
  DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};

  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = this->lastCoreExpertId;
    this->tokenCount = this->lastCoreExpertIdNum;
  }
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    int32_t expertIdx = inLocal[length].GetValue(idx);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    int32_t index = 0;
    while (this->lastExpertId < expertIdx) {
      while (this->tokenCount < this->expertCapacity) {
        index = this->lastExpertId * this->expertCapacity + this->tokenCount;
        DataCopyPad(expandedXGm[index * this->cols], this->outTmpLocal, copyParams1);
        DataCopyPad(dynamicQuantScaleGm[index], this->scaleOutTmpLocal, {1, 4, 0, 0, 0});
        this->tokenCount++;
      }
      this->tokenCount = 0;
      this->lastExpertId++;
    }

    if (this->tokenCount < this->expertCapacity) {
      int32_t outOffset = inLocal.GetValue(idx);
      index = expertIdx * this->expertCapacity + this->tokenCount;
      outLocal.SetValue(0, index);
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      DataCopyPad(expandedRowIdxGm[outOffset], outLocal, copyParams);
      Compute(outOffset, index, expertIdx);
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
  copyInQueue.FreeTensor(inLocal);
  copyOutQueue.FreeTensor(outLocal);
}

template <typename T, typename TilingData>
__aicore__ inline float MoeV2SrcToDstAndGather<T, TilingData>::ComputeMax(LocalTensor<float>& inLocal,
                                                                          LocalTensor<float>& tempLocal,
                                                                          LocalTensor<float>& dynamicQuantLocal,
                                                                          int32_t srcIdx, int32_t expertIdx,
                                                                          int64_t j) {
  LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();

  DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength * sizeof(T)), 0, 0, 0};
  DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};

  if constexpr (!IsSameType<T, float>::value) {
    DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign], inputXGm[srcIdx * this->cols + j * this->perLoopCols],
                intriParamsT, {false, 0, 0, 0});
  } else {
    DataCopyPad(inLocal, inputXGm[srcIdx * this->cols + j * this->perLoopCols], intriParamsT, {false, 0, 0, 0});
  }

  inputXInQueue.EnQue<float>(inLocal);
  inLocal = inputXInQueue.DeQue<float>();

  if constexpr (!IsSameType<T, float>::value) {
    Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign], RoundMode::CAST_NONE, colsTileLength);
    pipe_barrier(PIPE_V);
  }

  if (smoothType != 0) {
    DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols + j * this->perLoopCols], intriParamsFp32,
                {false, 0, 0, 0});
    smoothInQueue.EnQue(smoothLocal);
    smoothLocal = smoothInQueue.DeQue<float>();

    Mul(inLocal, inLocal, smoothLocal, colsTileLength);
    pipe_barrier(PIPE_V);
  }

  Abs(tempLocal, inLocal, colsTileLength);
  pipe_barrier(PIPE_V);

  ReduceMax(dynamicQuantLocal[8], tempLocal, tempLocal, colsTileLength);

  DataCopyPad(quantSrcGm[j * this->perLoopCols], inLocal, intriParamsFp32);
  smoothInQueue.FreeTensor(smoothLocal);
  SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

  return dynamicQuantLocal.GetValue(8);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::ComputeScale(LocalTensor<float>& inLocal,
                                                                           LocalTensor<float>& tempLocal,
                                                                           float scaleTemp, int64_t dstIndex,
                                                                           int64_t j) {
  DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};
  DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength * sizeof(int8_t)), 0, 0, 0};

  LocalTensor<int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();

  DataCopyPad(inLocal, quantSrcGm[j * this->perLoopCols], copyInParams, {false, 0, 0, 0});
  inputXInQueue.EnQue<float>(inLocal);
  inLocal = inputXInQueue.DeQue<float>();

  Duplicate<float>(tempLocal, scaleTemp, colsTileLength);
  pipe_barrier(PIPE_V);

  Div(tempLocal, inLocal, tempLocal, colsTileLength);
  pipe_barrier(PIPE_V);

  Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, colsTileLength);
  pipe_barrier(PIPE_V);

  Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, colsTileLength);

  inputXOutQueue.EnQue(outLocal);
  outLocal = inputXOutQueue.DeQue<int8_t>();
  DataCopyPad(expandedXGm[dstIndex * this->cols + j * this->perLoopCols], outLocal, copyOutParams);

  inputXOutQueue.FreeTensor(outLocal);
  SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::ComputeLoops(int32_t srcIdx, int32_t dstIdx,
                                                                           int32_t expertIdx) {
  LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();
  LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
  LocalTensor<float> quantScaleLocal = scaleOutQueue.AllocTensor<float>();

  uint32_t tmp = 0xFF7FFFFF;
  float reduceMax = *((float*)&tmp);
  for (int64_t j = 0; j < this->colLoops; j++) {
    colsTileLength = this->perLoopCols;
    if (j == this->colLoops - 1) {
      colsTileLength = this->lastLoopCols;
    }
    float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, srcIdx / this->k, expertIdx, j);
    reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
  }

  float scaleTemp = reduceMax / 127.0f;
  Duplicate<float>(quantScaleLocal, scaleTemp, 8);
  scaleOutQueue.EnQue(quantScaleLocal);
  quantScaleLocal = scaleOutQueue.DeQue<float>();

  DataCopyPad(dynamicQuantScaleGm[dstIdx], quantScaleLocal, {1, 4, 0, 0, 0});

  for (int64_t j = 0; j < this->colLoops; j++) {
    colsTileLength = this->perLoopCols;
    if (j == this->colLoops - 1) {
      colsTileLength = this->lastLoopCols;
    }
    ComputeScale(inLocal, tempLocal, scaleTemp, dstIdx, j);
  }

  inputXInQueue.FreeTensor(inLocal);
  calcQueue.FreeTensor(tempLocal);
  scaleOutQueue.FreeTensor(quantScaleLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOutLoops(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> outLocal = copyOutQueue.AllocTensor<int32_t>();
  int64_t length = Align(currentLoopRows, sizeof(int32_t));
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = this->lastCoreExpertId;
    this->tokenCount = this->lastCoreExpertIdNum;
  }
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    int32_t expertIdx = inLocal[length].GetValue(idx);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    int32_t index = 0;
    while (this->lastExpertId < expertIdx) {
      while (this->tokenCount < this->expertCapacity) {
        index = this->lastExpertId * this->expertCapacity + this->tokenCount;
        int64_t col = this->perLoopCols;
        DataCopyPad(dynamicQuantScaleGm[index], this->scaleOutTmpLocal, {1, 4, 0, 0, 0});
        for (int64_t i = 0; i < this->colLoops; i++) {
          if (i == this->colLoops - 1) {
            col = this->lastLoopCols;
          }
          DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(int8_t)), 0, 0, 0};
          DataCopyPad(expandedXGm[index * this->cols + i * this->perLoopCols], this->outTmpLocal, copyParams1);
          SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        this->tokenCount++;
      }
      this->tokenCount = 0;
      this->lastExpertId++;
    }

    if (this->tokenCount < this->expertCapacity) {
      int32_t outOffset = inLocal.GetValue(idx);
      index = expertIdx * this->expertCapacity + this->tokenCount;
      outLocal.SetValue(0, index);
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      DataCopyPad(expandedRowIdxGm[outOffset], outLocal, copyParams);
      if (smoothType == 2) {
        ComputeLoops(outOffset, index, expertIdx);
      } else {
        ComputeLoops(outOffset, index, 0);
      }
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
  copyInQueue.FreeTensor(inLocal);
  copyOutQueue.FreeTensor(outLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::CopyOutRemain() {
  if (this->blockIdx != this->srcToDstTilingData->needCoreNum - 1) {
    copyOutZeroQueue.FreeTensor(this->outTmpLocal);
    scaleOutZeroQueue.FreeTensor(this->scaleOutTmpLocal);
    return;
  }
  while (this->lastExpertId < this->expertNum) {
    while (this->tokenCount < this->expertCapacity) {
      int32_t index = this->lastExpertId * this->expertCapacity + this->tokenCount;
      int64_t col = this->perLoopCols;
      DataCopyPad(dynamicQuantScaleGm[index], this->scaleOutTmpLocal, {1, 4, 0, 0, 0});
      for (int64_t i = 0; i < this->colLoops; i++) {
        if (i == this->colLoops - 1) {
          col = this->lastLoopCols;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(expandedXGm[index * this->cols + i * this->perLoopCols], this->outTmpLocal, copyParams);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      }
      this->tokenCount++;
    }
    this->tokenCount = 0;
    this->lastExpertId++;
  }
  copyOutZeroQueue.FreeTensor(this->outTmpLocal);
  scaleOutZeroQueue.FreeTensor(this->scaleOutTmpLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR expandedRowIdx,
                                                                   GM_ADDR expandedX, GM_ADDR dynamicQuantScale,
                                                                   GM_ADDR workspace, const TilingData* tilingData,
                                                                   TPipe* tPipe) {
  int64_t blockNum = GetBlockNum();
  this->pipe = tPipe;
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();

  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstCapacityComputeParamsOp);
  this->expertNum = tilingData->expertNum;
  this->expertCapacity = tilingData->expertCapacity;
  this->cols = tilingData->cols;
  this->k = tilingData->k;
  this->smoothType = tilingData->smoothType;

  if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
    this->coreRows = this->srcToDstTilingData->lastCoreRows;
    this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
    this->rowLoops = this->srcToDstTilingData->lastCoreLoops;
  } else {
    this->coreRows = this->srcToDstTilingData->perCoreRows;
    this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
    this->rowLoops = this->srcToDstTilingData->perCoreLoops;
  }
  this->perLoopCols = this->srcToDstTilingData->perLoopCols;
  this->lastLoopCols = this->srcToDstTilingData->lastLoopCols;
  this->colLoops = this->srcToDstTilingData->colLoops;
  this->perLoopColsAlign = Align(this->perLoopCols, sizeof(T));

  inputXGm.SetGlobalBuffer((__gm__ T*)x);
  quantSmoothGm.SetGlobalBuffer((__gm__ float*)scale);
  dynamicQuantScaleGm.SetGlobalBuffer((__gm__ float*)dynamicQuantScale);

  int64_t length = Align(this->totalLength, sizeof(int32_t));
  expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, length);
  expandedXGm.SetGlobalBuffer((__gm__ int8_t*)expandedX, this->expertNum * this->expertCapacity * this->cols);

  expandedExpertIdxGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));
  expandDstToSrcRowGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + length + this->blockIdx * this->srcToDstTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));
  expertIdxValueGm.SetGlobalBuffer((__gm__ int32_t*)workspace + length * 2, this->coreNum * 2);
  if (this->colLoops > 1) {
    quantSrcGm.SetGlobalBuffer((__gm__ float*)workspace + length * 2 + this->coreNum * 2 + this->blockIdx * this->cols,
                               this->cols * sizeof(float));
  }

  pipe->InitBuffer(copyInQueue, 1, AlignBytes(this->perLoopRows, sizeof(int32_t)) * 2);
  pipe->InitBuffer(copyOutQueue, 1, AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t)));
  pipe->InitBuffer(copyOutZeroQueue, 1, AlignBytes(this->perLoopCols, sizeof(int16_t)));

  int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols, sizeof(T));
  perLoopColsAlignBytes =
      Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));

  pipe->InitBuffer(inputXInQueue, 1, perLoopColsAlignBytes);
  pipe->InitBuffer(smoothInQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
  pipe->InitBuffer(calcQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
  pipe->InitBuffer(inputXOutQueue, 1, AlignBytes(this->perLoopCols, sizeof(int8_t)));
  pipe->InitBuffer(scaleOutQueue, 1, BLOCK_BYTES + BLOCK_BYTES);
  pipe->InitBuffer(scaleOutZeroQueue, 1, BLOCK_BYTES);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstAndGather<T, TilingData>::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    AssistInit();
    this->outTmpLocal = copyOutZeroQueue.DeQue<int8_t>();
    this->scaleOutTmpLocal = scaleOutZeroQueue.DeQue<float>();
    currentLoopRows = perLoopRows;
    if (colLoops > 1) {
      for (int64_t loop = 0; loop < this->rowLoops; loop++) {
        if (loop == this->rowLoops - 1) {
          currentLoopRows = lastLoopRows;
        }
        CopyIn(loop);
        CopyOutLoops(loop);
      }
    } else {
      smoothLocal = smoothInQueue.AllocTensor<float>();
      if (smoothType == 1) {
        DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols * sizeof(float)), 0, 0, 0};
        DataCopyPad(smoothLocal, quantSmoothGm, smoothParams, {false, 0, 0, 0});
      }
      for (int64_t loop = 0; loop < this->rowLoops; loop++) {
        if (loop == this->rowLoops - 1) {
          currentLoopRows = lastLoopRows;
        }
        CopyIn(loop);
        CopyOut(loop);
      }
      smoothInQueue.FreeTensor(smoothLocal);
    }
    CopyOutRemain();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_SRC_TO_DST_AND_GATHER_H