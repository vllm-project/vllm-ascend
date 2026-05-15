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
 * \file moe_v2_src_to_dst_op.h
 * \brief
 */
#ifndef INNER_MOE_V2_SRC_TO_DST_H
#define INNER_MOE_V2_SRC_TO_DST_H

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2SrcToDstOp {
 public:
  __aicore__ inline MoeV2SrcToDstOp(){};
  template <typename TilingData>
  __aicore__ inline void Init(GM_ADDR expandSrcToDstRow, GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void Compute(int64_t progress);
  __aicore__ inline void CopyOut();
  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> copyInQueue;
  TQue<QuePosition::VECOUT, 1> copyOutQueue;
  TBuf<TPosition::VECCALC> assistBuffer;

  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expandSrcToDstRowGm;
  GlobalTensor<int32_t> assistGm;

  const InnerMoeV2GatherOutComputeTilingData* srcToDstTilingData;

  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
};

__aicore__ inline void MoeV2SrcToDstOp::AssistInit() {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
  OOMCheckAddrRange(assistGm.GetPhyAddr(), ASSIST_NUM * sizeof(int32_t));
#endif
  LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);
  DataCopy(assistTensor, assistGm, ASSIST_NUM);
  SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
  Adds(assistTensor, assistTensor, (int32_t)(this->blockIdx * this->srcToDstTilingData->perCoreRows), ASSIST_NUM);
}

__aicore__ inline void MoeV2SrcToDstOp::CopyIn(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
  DataCopy(inLocal, expandDstToSrcRowGm[progress * perLoopRows], Align(currentLoopRows, sizeof(int32_t)));
  copyInQueue.EnQue<int32_t>(inLocal);
}

__aicore__ inline void MoeV2SrcToDstOp::Compute(int64_t progress) {
  LocalTensor<int32_t> outLocal = copyOutQueue.AllocTensor<int32_t>();
  LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);

  pipe_barrier(PIPE_V);
  int64_t loops = Ceil(currentLoopRows, ASSIST_INDEX_NUM);
  for (int64_t i = 0; i < loops; i++) {
    Adds(outLocal[i * ASSIST_NUM], assistTensor,
         static_cast<int32_t>(this->perLoopRows * progress + i * ASSIST_INDEX_NUM), ASSIST_NUM);
  }
  pipe_barrier(PIPE_V);
  copyOutQueue.EnQue<int32_t>(outLocal);
}

__aicore__ inline void MoeV2SrcToDstOp::CopyOut() {
  LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> outLocal = copyOutQueue.DeQue<int32_t>();
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = sizeof(int32_t);
  uint32_t outOffset;
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    outOffset = inLocal.GetValue(idx);
    DataCopyPad(expandSrcToDstRowGm[outOffset], outLocal[idx * INT32_ONE_BLOCK_NUM], intriParams);
  }

  copyInQueue.FreeTensor(inLocal);
  copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV2SrcToDstOp::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2SrcToDstOp::Init(GM_ADDR expandSrcToDstRow, GM_ADDR workspace, const TilingData* tilingData,
                                             TPipe* tPipe) {
  int64_t blockNum = GetBlockNum();
  this->pipe = tPipe;
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();

  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);

  if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
    this->coreRows = this->srcToDstTilingData->lastCoreRows;
    this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
  } else {
    this->coreRows = this->srcToDstTilingData->perCoreRows;
    this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
  }

  expandSrcToDstRowGm.SetGlobalBuffer((__gm__ int32_t*)expandSrcToDstRow, Align(this->totalLength, sizeof(int32_t)));
  expandDstToSrcRowGm.SetGlobalBuffer((__gm__ int32_t*)workspace + Align(this->totalLength, sizeof(int32_t)) +
                                          this->blockIdx * this->srcToDstTilingData->perCoreRows,
                                      Align(this->coreRows, sizeof(int32_t)));
  assistGm.SetGlobalBuffer((__gm__ int32_t*)assist, ASSIST_NUM);

  pipe->InitBuffer(copyInQueue, 1, this->perLoopRows * BLOCK_BYTES);
  pipe->InitBuffer(copyOutQueue, 1, Ceil(this->perLoopRows, ASSIST_NUM) * ASSIST_NUM * BLOCK_BYTES);
  pipe->InitBuffer(assistBuffer, ASSIST_NUM * sizeof(int32_t));
}

__aicore__ inline void MoeV2SrcToDstOp::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
    currentLoopRows = perLoopRows;
    AssistInit();
    for (int64_t loop = 0; loop < loops - 1; loop++) {
      CopyIn(loop);
      Compute(loop);
      CopyOut();
    }
    currentLoopRows = lastLoopRows;
    CopyIn(loops - 1);
    Compute(loops - 1);
    CopyOut();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_SRC_TO_DST_H