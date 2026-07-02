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
 * \file moe_v2_fullload_quant_base.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_BASE_H
#define MOE_V2_FULL_LOAD_QUANT_BASE_H

#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2FullLoadQuantBase {
 public:
  __aicore__ inline MoeV2FullLoadQuantBase(){};

 protected:
  __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                  GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                  const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void ProcessBase();
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();

 protected:
  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData;

  TPipe* pipe;
  int64_t tileLength;
  int64_t bufferNum = 1;
  int64_t totalLength;
  int64_t coreNum;
  int64_t sortNum;
  int64_t blockIdx;
  int64_t needCoreNum;
  int64_t coreRows;
  int64_t perCoreRows;
  int64_t k;
  int64_t n;
  int64_t cols;
  int64_t activateRows;
  int64_t expertNum;
  int64_t expertCapacity;

  TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
  TBuf<TPosition::VECCALC> tempBuffer;
  TBuf<TPosition::VECCALC> sortedBuffer;
  TQue<QuePosition::VECIN, 1> xCopyInQueue;
  TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue;
  TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue;
  TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue;
  TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue;

  GlobalTensor<int32_t> expertIdxGm;
  GlobalTensor<int8_t> expandedXGm;
  GlobalTensor<int32_t> expandedRowIdxGm;
  GlobalTensor<int32_t> expandedExpertIdxGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t dropPadMode = 0;
  static constexpr int64_t DST_BLK_STRIDE = 1;
  static constexpr int64_t DST_REP_STRIDE = 8;
  static constexpr int64_t FOUR_BLOCK_BYTES = 128;
};

__aicore__ inline void MoeV2FullLoadQuantBase::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)),
                                   0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
  ArithProgression<int32_t>(inLocal[this->sortNum], 0, 1, this->totalLength);
  sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::SortCompute() {
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
  LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum].template ReinterpretCast<uint32_t>();
  LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue.AllocTensor<float>();
  LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue.AllocTensor<uint32_t>();
  LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();
  Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
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
  expandedExpertIdxCopyOutQueue.EnQue<int32_t>(expandedExpertIdxLocalInt32);

  LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.AllocTensor<uint32_t>();
  LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
  Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, this->totalLength);
  pipe_barrier(PIPE_V);
  ArithProgression<int32_t>(inLocal[this->sortNum], 0, 1, this->totalLength);
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
  Concat(concatLocal, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Extract(tempTensor, expandedRowIdxU32, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  expandedRowIdxCopyOutQueue.EnQue<uint32_t>(expandedRowIdx);
  sortDataCopyInQueue.FreeTensor(inLocal);

  expandDstToSrcRowQueue.FreeTensor(expandDstToSrcRowLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutIdx() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.DeQue<int32_t>();
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = this->totalLength * sizeof(int32_t);
  DataCopyPad(expandedRowIdxGm, expandedRowIdx, intriParams);
  expandedRowIdxCopyOutQueue.EnQue(expandedRowIdx);
}

__aicore__ inline void MoeV2FullLoadQuantBase::ComputeExpertTokenCountOrCumsum() {
  LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue.AllocTensor<int32_t>();

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
  expertTokensCopyOutQueue.FreeTensor(expertTokensCount);
  expandedExpertIdxCopyOutQueue.FreeTensor(expandedExpertIdx);
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutEmpty() {
  LocalTensor<int32_t> outLocal = expandedExpertIdxCopyOutQueue.DeQue<int32_t>();
  expandedExpertIdxCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                        GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                        GM_ADDR workspace,
                                                        const MoeInitRoutingQuantV2TilingData* tilingData,
                                                        TPipe* tPipe) {
  this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
  this->k = tilingData->k;
  this->n = tilingData->n;
  this->cols = tilingData->cols;
  this->needCoreNum = this->gatherOutTilingData->needCoreNum;
  this->perCoreRows = this->gatherOutTilingData->perCoreRows;
  this->activateRows = this->gatherOutTilingData->activateRows;
  if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
    this->coreRows = this->gatherOutTilingData->lastCoreRows;
  } else {
    this->coreRows = this->gatherOutTilingData->perCoreRows;
  }
  this->expertNum = tilingData->expertNum;
  this->dropPadMode = tilingData->dropPadMode;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->pipe = tPipe;

  expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);

  expandedXGm.SetGlobalBuffer((__gm__ int8_t*)expandedX);
  expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, this->tileLength);
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                Align(this->expertNum, sizeof(int32_t)));
  }

  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum * sizeof(int32_t);

  pipe->InitBuffer(expandedRowIdxCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(expandedExpertIdxCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(expertTokensCopyOutQueue, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
  pipe->InitBuffer(expandDstToSrcRowQueue, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
  pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
  pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);
}

__aicore__ inline void MoeV2FullLoadQuantBase::ProcessBase() {
  if (this->blockIdx < this->needCoreNum) {
    CopyIn();
    SortCompute();
    if (this->blockIdx == 0) {
      CopyOutIdx();
    }
    if (this->blockIdx == this->needCoreNum - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
      ComputeExpertTokenCountOrCumsum();
    } else {
      CopyOutEmpty();
    }
  }
}

}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_FULL_LOAD_QUANT_BASE_H