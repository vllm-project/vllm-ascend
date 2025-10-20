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
 * \file moe_v2_src_to_dst_with_capacity.h
 * \brief
 */
#ifndef INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H
#define INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T, typename TilingData>
class MoeV2SrcToDstWithCapacity {
 public:
  __aicore__ inline MoeV2SrcToDstWithCapacity(){};
  __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                              const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void CopyOut(int64_t progress);
  __aicore__ inline void CopyOutRemain();
  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> copyInQueue;
  TQue<QuePosition::VECOUT, 1> copyOutQueue;
  TQue<QuePosition::VECOUT, 1> copyOutZeroQueue;

  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expandedRowIdxGm;
  GlobalTensor<int32_t> expertIdxValueGm;
  GlobalTensor<int32_t> expandedExpertIdxGm;
  GlobalTensor<T> expandedXGm;

  LocalTensor<T> outTmpLocal;

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

  int64_t tokenCount = 0;
  int32_t lastExpertId = -1;
  int32_t lastCoreExpertId = 0;
  int32_t lastCoreExpertIdNum = 0;
};

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::AssistInit() {
  if constexpr (IsSameType<T, int8_t>::value) {
    LocalTensor<int16_t> outLocal = copyOutZeroQueue.AllocTensor<int16_t>();
    Duplicate<int16_t>(outLocal, static_cast<int16_t>(0), this->perLoopCols);
    copyOutZeroQueue.EnQue<int16_t>(outLocal);
  } else {
    LocalTensor<T> outLocal = copyOutZeroQueue.AllocTensor<T>();
    Duplicate<T>(outLocal, static_cast<T>(0), this->perLoopCols);
    copyOutZeroQueue.EnQue<T>(outLocal);
  }

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
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyIn(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
  int64_t length = Align(currentLoopRows, sizeof(int32_t));
  DataCopy(inLocal, expandDstToSrcRowGm[progress * perLoopRows], length);
  DataCopy(inLocal[length], expandedExpertIdxGm[progress * perLoopRows], length);
  copyInQueue.EnQue<int32_t>(inLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyOut(int64_t progress) {
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
        for (int64_t i = 0; i < this->colLoops; i++) {
          if (i == this->colLoops - 1) {
            col = this->lastLoopCols;
          }
#ifdef __CCE_KT_TEST__
          // CPU孪生调试无法使用多核同步，可能导致index为未初始化的脏数据，因此需要特殊处理
          if (index * this->cols + i * this->perLoopCols + col * sizeof(T) > expandedXGm.GetSize()) {
              continue;
          }
#endif
          DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(T)), 0, 0, 0};
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
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      this->tokenCount++;
    }
  }
  copyInQueue.FreeTensor(inLocal);
  copyOutQueue.FreeTensor(outLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::CopyOutRemain() {
  if (this->blockIdx != this->srcToDstTilingData->needCoreNum - 1) {
    copyOutZeroQueue.FreeTensor(this->outTmpLocal);
    return;
  }
  while (this->lastExpertId < this->expertNum) {
    while (this->tokenCount < this->expertCapacity) {
      int32_t index = this->lastExpertId * this->expertCapacity + this->tokenCount;
      int64_t col = this->perLoopCols;
      for (int64_t i = 0; i < this->colLoops; i++) {
        if (i == this->colLoops - 1) {
          col = this->lastLoopCols;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(T)), 0, 0, 0};
        DataCopyPad(expandedXGm[index * this->cols + i * this->perLoopCols], this->outTmpLocal, copyParams);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      }
      this->tokenCount++;
    }
    this->tokenCount = 0;
    this->lastExpertId++;
  }
  copyOutZeroQueue.FreeTensor(this->outTmpLocal);
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
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

  int64_t length = Align(this->totalLength, sizeof(int32_t));
  expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, length);
  expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX, this->expertNum * this->expertCapacity * this->cols);

  expandedExpertIdxGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));
  expandDstToSrcRowGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + length + this->blockIdx * this->srcToDstTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));
  expertIdxValueGm.SetGlobalBuffer((__gm__ int32_t*)workspace + length * 2, this->coreNum * 2);

  pipe->InitBuffer(copyInQueue, 1, AlignBytes(this->perLoopRows, sizeof(int32_t)) * 2);
  pipe->InitBuffer(copyOutQueue, 1, AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t)));
  if constexpr (IsSameType<T, int8_t>::value) {
    pipe->InitBuffer(copyOutZeroQueue, 1, AlignBytes(this->perLoopCols, sizeof(int16_t)));
  } else {
    pipe->InitBuffer(copyOutZeroQueue, 1, AlignBytes(this->perLoopCols, sizeof(T)));
  }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacity<T, TilingData>::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    AssistInit();
    this->outTmpLocal = copyOutZeroQueue.DeQue<T>();
    currentLoopRows = perLoopRows;
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyIn(loop);
      CopyOut(loop);
    }
    CopyOutRemain();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_SRC_TO_DST_WITH_CAPACITY_H