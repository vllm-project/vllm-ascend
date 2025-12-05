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
 * \file moe_v2_expert_token_out.h
 * \brief
 */
#ifndef INNER_MOE_V2_EXPERT_TOKEN_OUT_H
#define INNER_MOE_V2_EXPERT_TOKEN_OUT_H

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
constexpr int64_t EXPERT_ID_VALUE_NUM = 2;

class MoeV2ExpertTokenOut {
 public:
  __aicore__ inline MoeV2ExpertTokenOut(){};
  template <typename TilingData>
  __aicore__ inline void Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                              GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void Compute(int64_t progress);
  __aicore__ inline void SyncAll();
  __aicore__ inline void InitLocal();
  __aicore__ inline void GetExpertTokenCount(int32_t curExpertId);
  __aicore__ inline void CopyOutTokenGm();
  __aicore__ inline void CopyOutExpertTokensCumsum(bool isTail);
  __aicore__ inline void CopyOutExpertTokensCount(bool isTail);

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> copyInQueue;
  TQue<QuePosition::VECIN, 1> expertTokenIdxCopyInQueue;
  TQue<QuePosition::VECOUT, 1> expertTokenIdxCopyOutQueue;

  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;
  GlobalTensor<int32_t> expandedExpertIdxGm;
  GlobalTensor<int32_t> expertIdxValueGm;
  GlobalTensor<int32_t> expandedRowIdxGm;

  LocalTensor<int32_t> expertTokenIdxOutLocal;

  const InnerMoeV2GatherOutComputeTilingData* srcToDstTilingData;

  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t expertNum;
  int64_t expertNumUbAlign;
  int64_t dropPadMode = 0;
  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;

  int64_t tokenCount = 0;
  int64_t expertIdx = 0;
  int32_t lastExpertId = -1;
  int32_t firstExpertId = -1;

  int32_t expertTokenValue = 0;
};

__aicore__ inline void MoeV2ExpertTokenOut::InitLocal() {
  LocalTensor<int32_t> tokenIdxLocal = expertTokenIdxCopyOutQueue.AllocTensor<int32_t>();
  Duplicate<int32_t>(tokenIdxLocal, 0, this->expertNumUbAlign);
  expertTokenIdxCopyOutQueue.EnQue<int32_t>(tokenIdxLocal);

  // expandedRowIdx initialized to -1, which is used in the src_to_dst_with_capacity step.
  // use this step SyncAll to synchronize every core data
  if (this->dropPadMode == 0) {
    return;
  }
  LocalTensor<int32_t> outLocal = copyInQueue.AllocTensor<int32_t>();
  int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
  Duplicate<int32_t>(outLocal, -1, perLoopRows);
  SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
  for (int64_t loop = 0; loop < loops; loop++) {
    int64_t copyLength = perLoopRows;
    if (loop == loops - 1) {
      copyLength = lastLoopRows;
    }
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(copyLength * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPad(expandedRowIdxGm[this->blockIdx * this->srcToDstTilingData->perCoreRows + loop * perLoopRows], outLocal,
                copyParams);
  }
  SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
  copyInQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV2ExpertTokenOut::CopyIn(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
  DataCopy(inLocal, expandedExpertIdxGm[progress * perLoopRows], Align(currentLoopRows, sizeof(int32_t)));
  copyInQueue.EnQue<int32_t>(inLocal);
}

__aicore__ inline void MoeV2ExpertTokenOut::GetExpertTokenCount(int32_t curExpertId) {
  this->tokenCount++;
  if (this->lastExpertId < curExpertId) {
    this->expertTokenIdxOutLocal.SetValue(this->expertIdx, this->tokenCount - 1);
    this->tokenCount = 1;
    this->expertIdx += (curExpertId - this->lastExpertId);
    while (curExpertId - this->firstExpertId + 1 > this->expertNumUbAlign) {
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      CopyOutExpertTokensCumsum(false);
      CopyOutExpertTokensCount(false);
      SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
      Duplicate<int32_t>(this->expertTokenIdxOutLocal, 0, this->expertNumUbAlign);
      SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
      this->firstExpertId += this->expertNumUbAlign;
      this->expertIdx = curExpertId - this->firstExpertId;
    }
    this->lastExpertId = curExpertId;
  }
}

__aicore__ inline void MoeV2ExpertTokenOut::Compute(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  if (this->lastExpertId == -1) {
    this->lastExpertId = inLocal.GetValue(0);
    this->firstExpertId = this->lastExpertId;
  }
  for (int64_t i = 0; i < currentLoopRows; i++) {
    int32_t expertId = inLocal.GetValue(i);
    GetExpertTokenCount(expertId);
  }
  this->expertTokenIdxOutLocal.SetValue(this->expertIdx, this->tokenCount);
  copyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2ExpertTokenOut::CopyOutExpertTokensCumsum(bool isTail) {
  if (this->dropPadMode != DROPLESS_MODE || expertTokensCountOrCumsumFlag != EXERPT_TOKENS_CUMSUM) {
    return;
  }
#ifdef __CCE_KT_TEST__
    // CPU twin debugging cannot use multi-core sync, so index may contain uninitialized dirty data; handle specially
    if (this->firstExpertId > expertTokensCountOrCumsumGm.GetSize()) {
        return;
    }
#endif
  int64_t copyLength = isTail ? this->lastExpertId - this->firstExpertId + 1 : this->expertNumUbAlign;
  int64_t end = this->expertNum - this->firstExpertId;
  for (int64_t i = 0; i < copyLength; i++) {
    this->expertTokenValue += this->expertTokenIdxOutLocal.GetValue(i);
    this->expertTokenIdxOutLocal.SetValue(i, this->expertTokenValue);
  }
  // if the remianing UB is sufficient, use the UB space to copy
  // otherwise, copy the calculated data first, and then copy the last tokenValue to remaining expert position
  if (isTail && end <= this->expertNumUbAlign) {
    int64_t startAlign = Min(Align(copyLength, sizeof(int32_t)), end);
    for (int64_t i = copyLength; i < startAlign; i++) {
      this->expertTokenIdxOutLocal.SetValue(i, this->expertTokenValue);
    }
    if (startAlign < end) {
      Duplicate<int32_t>(this->expertTokenIdxOutLocal[startAlign], this->expertTokenValue, end - startAlign);
    }
    copyLength = end;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
  }
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(copyLength * sizeof(int32_t)), 0, 0, 0};
  SetAtomicAdd<int32_t>();
#ifndef __CCE_KT_TEST__
  DataCopyPad(expertTokensCountOrCumsumGm[this->firstExpertId], this->expertTokenIdxOutLocal, copyParams);
#endif
  SetAtomicNone();
  if (isTail && end > this->expertNumUbAlign) {
    int64_t remainderLength = end - copyLength;
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    Duplicate<int32_t>(this->expertTokenIdxOutLocal, this->expertTokenValue, this->expertNumUbAlign);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t loopTimes = remainderLength / this->expertNumUbAlign + 1;
    for (int64_t i = 0; i < loopTimes; i++) {
      copyLength = i == loopTimes - 1 ? remainderLength - this->expertNumUbAlign * i : this->expertNumUbAlign;
      DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(copyLength * sizeof(int32_t)), 0, 0, 0};
      SetAtomicAdd<int32_t>();
      DataCopyPad(expertTokensCountOrCumsumGm[this->lastExpertId + 1 + this->expertNumUbAlign * i],
                  this->expertTokenIdxOutLocal, params);
      SetAtomicNone();
    }
  }
}

__aicore__ inline void MoeV2ExpertTokenOut::CopyOutExpertTokensCount(bool isTail) {
  int64_t copyLength = isTail ? this->lastExpertId - this->firstExpertId + 1 : this->expertNumUbAlign;
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(copyLength * sizeof(int32_t)), 0, 0, 0};
#ifdef __CCE_KT_TEST__
    // CPU twin debugging skips output copies
    return;
#endif
  SetAtomicAdd<int32_t>();
  if (this->dropPadMode == DROP_PAD_MODE && expertTokensBeforeCapacityFlag > EXERPT_TOKENS_NONE) {
    DataCopyPad(expertTokensBeforeCapacityGm[this->firstExpertId], this->expertTokenIdxOutLocal, copyParams);
  }
  if (this->dropPadMode == DROPLESS_MODE && expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
    DataCopyPad(expertTokensCountOrCumsumGm[this->firstExpertId], this->expertTokenIdxOutLocal, copyParams);
  }
  SetAtomicNone();
}

__aicore__ inline void MoeV2ExpertTokenOut::CopyOutTokenGm() {
  if (this->dropPadMode == DROPLESS_MODE) {
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    CopyOutExpertTokensCumsum(true);
    CopyOutExpertTokensCount(true);
    return;
  }
  this->expertTokenIdxOutLocal.SetValue(this->expertNumUbAlign, this->lastExpertId);
  this->expertTokenIdxOutLocal.SetValue(this->expertNumUbAlign + 1, this->tokenCount);
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(EXPERT_ID_VALUE_NUM * sizeof(int32_t)),
                               0, 0, 0};
  SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
  DataCopyPad(expertIdxValueGm[this->blockIdx * EXPERT_ID_VALUE_NUM],
              this->expertTokenIdxOutLocal[this->expertNumUbAlign], copyParams);
  CopyOutExpertTokensCount(true);
}

__aicore__ inline void MoeV2ExpertTokenOut::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2ExpertTokenOut::Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                                 GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                                 const TilingData* tilingData, TPipe* tPipe) {
  int64_t blockNum = GetBlockNum();
  this->pipe = tPipe;
  //this->blockIdx = GetBlockIdx();
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);
  this->expertNum = tilingData->expertNum;
  this->dropPadMode = tilingData->dropPadMode;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
  this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

  if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
    this->coreRows = this->srcToDstTilingData->lastCoreRows;
    this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
  } else {
    this->coreRows = this->srcToDstTilingData->perCoreRows;
    this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
  }

  expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, Align(this->totalLength, sizeof(int32_t)));
  if (this->dropPadMode == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum, this->expertNum);
  }
  if (this->dropPadMode == DROP_PAD_MODE && this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
    expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensBeforeCapacity, this->expertNum);
  }

  expandedExpertIdxGm.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));
  expertIdxValueGm.SetGlobalBuffer((__gm__ int32_t*)workspace + Align(this->totalLength, sizeof(int32_t)) * 2,
                                   this->coreNum * 2);

  this->expertNumUbAlign = Min(Align(this->expertNum, sizeof(int32_t)), MAX_EXPERT_NUM);
  pipe->InitBuffer(copyInQueue, 1, this->perLoopRows * BLOCK_BYTES);
  pipe->InitBuffer(expertTokenIdxCopyInQueue, 1, this->expertNumUbAlign * sizeof(int32_t));
  pipe->InitBuffer(expertTokenIdxCopyOutQueue, 1, (this->expertNumUbAlign + EXPERT_ID_VALUE_NUM) * sizeof(int32_t));
}

__aicore__ inline void MoeV2ExpertTokenOut::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
    currentLoopRows = perLoopRows;
    InitLocal();
    this->expertTokenIdxOutLocal = expertTokenIdxCopyOutQueue.DeQue<int32_t>();
    for (int64_t loop = 0; loop < loops - 1; loop++) {
      CopyIn(loop);
      Compute(loop);
    }
    currentLoopRows = lastLoopRows;
    CopyIn(loops - 1);
    Compute(loops - 1);
    CopyOutTokenGm();
    expertTokenIdxCopyOutQueue.FreeTensor(this->expertTokenIdxOutLocal);
  }
  this->SyncAll();
}

}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_EXPERT_TOKEN_OUT_H
