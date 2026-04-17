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
 * \file moe_v2_fullload_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_H
#define MOE_V2_FULL_LOAD_QUANT_H

#include "moe_v2_fullload_quant_base.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadQuant : public MoeV2FullLoadQuantBase {
 public:
  __aicore__ inline MoeV2FullLoadQuant(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                              GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                              const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void Compute(int64_t xLocalLength);
  __aicore__ inline void CopyOutX();

 private:
  TQue<QuePosition::VECOUT, 1> floatQueue;
  TQue<QuePosition::VECOUT, 1> halfQueue;
  TQue<QuePosition::VECOUT, 1> inputXCopyOutQueue;

  GlobalTensor<T> xGm;
  GlobalTensor<float> scaleGm;
  GlobalTensor<float> offsetGm;

  float scale;
  float offset;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Compute(int64_t xLocalLength) {
  LocalTensor<T> inLocal = xCopyInQueue.DeQue<T>();
  LocalTensor<int8_t> outLocal = inputXCopyOutQueue.AllocTensor<int8_t>();
  LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
  LocalTensor<half> halfLocal = halfQueue.AllocTensor<half>();

  uint32_t elements = Align(this->cols, sizeof(int8_t)) * xLocalLength;
  if constexpr (IsSameType<T, bfloat16_t>::value) {
    Cast(floatLocal, inLocal, RoundMode::CAST_NONE, elements);
    pipe_barrier(PIPE_V);
    Cast(halfLocal, floatLocal, RoundMode::CAST_NONE, elements);
    pipe_barrier(PIPE_V);
    Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
    pipe_barrier(PIPE_V);
    Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
    pipe_barrier(PIPE_V);
    LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
    Cast(intLocal, halfLocal, RoundMode::CAST_RINT, elements);
    pipe_barrier(PIPE_V);
    SetDeqScale((half)1.000000e+00f);
    pipe_barrier(PIPE_V);
    Cast(halfLocal, intLocal, RoundMode::CAST_RINT, elements);
    pipe_barrier(PIPE_V);
    Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
  } else if constexpr (IsSameType<T, float>::value) {
    Cast(halfLocal, inLocal, RoundMode::CAST_NONE, elements);
    pipe_barrier(PIPE_V);
    Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
    pipe_barrier(PIPE_V);
    Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
    pipe_barrier(PIPE_V);
    Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
  } else {
    Muls(inLocal, inLocal, static_cast<T>(this->scale), elements);
    pipe_barrier(PIPE_V);
    Adds(inLocal, inLocal, static_cast<T>(this->offset), elements);
    pipe_barrier(PIPE_V);
    Cast(outLocal, inLocal, RoundMode::CAST_RINT, elements);
  }
  inputXCopyOutQueue.EnQue(outLocal);
  xCopyInQueue.FreeTensor(inLocal);
  floatQueue.FreeTensor(floatLocal);
  halfQueue.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::CopyOutX() {
  LocalTensor<T> xLocal = xCopyInQueue.AllocTensor<T>();
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.DeQue<int32_t>();
  int64_t inFactor = Align(this->cols, sizeof(int8_t));
  int64_t curRowsStart = this->blockIdx * this->perCoreRows;
  int64_t startXRow = curRowsStart / this->k;
  int64_t endXRow = (curRowsStart + this->coreRows - 1) / this->k;

  uint32_t dstStride = (inFactor * sizeof(T) - AlignBytes(this->cols, sizeof(T))) / BLOCK_BYTES;
  DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(endXRow - startXRow + 1),
                                    static_cast<uint32_t>(this->cols * sizeof(T)), 0, dstStride, 0};
  DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, 0};
  DataCopyPad(xLocal, xGm[startXRow * this->cols], dataXCopyParams, dataXCopyPadParams);
  xCopyInQueue.EnQue(xLocal);
  Compute(endXRow - startXRow + 1);
  LocalTensor<int8_t> outLocal = inputXCopyOutQueue.DeQue<int8_t>();
  int64_t k = 0;
  DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};
  for (int64_t i = startXRow; i <= endXRow; i++) {
    for (; k < this->perCoreRows && curRowsStart / this->k == i; curRowsStart++, k++) {
      int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
      if (outIndex < this->activateRows) {
        DataCopyPad(expandedXGm[outIndex * this->cols], outLocal[(i - startXRow) * inFactor], intriParams);
      }
    }
  }
  expandedRowIdxCopyOutQueue.FreeTensor(expandedRowIdx);
  inputXCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                   GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                   GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                                   const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe) {
  this->InitBase(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);
  xGm.SetGlobalBuffer((__gm__ T*)x);
  scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);
  offsetGm.SetGlobalBuffer((__gm__ float*)offset, 1);
  this->scale = scaleGm.GetValue(0);
  this->offset = offsetGm.GetValue(0);

  int64_t curRowsStart = this->blockIdx * this->perCoreRows;
  int64_t rowLength = (curRowsStart + this->coreRows - 1) / this->k - curRowsStart / this->k + 1;
  int64_t xAlignedCount = Align(this->cols, sizeof(int8_t));
  pipe->InitBuffer(xCopyInQueue, bufferNum, xAlignedCount * sizeof(T) * rowLength);
  pipe->InitBuffer(inputXCopyOutQueue, 1, xAlignedCount * sizeof(int8_t) * rowLength);
  pipe->InitBuffer(floatQueue, 1, xAlignedCount * sizeof(float) * rowLength);
  pipe->InitBuffer(halfQueue, 1, xAlignedCount * sizeof(half) * rowLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Process() {
  if (this->blockIdx < this->needCoreNum) {
    this->ProcessBase();
    CopyOutX();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif