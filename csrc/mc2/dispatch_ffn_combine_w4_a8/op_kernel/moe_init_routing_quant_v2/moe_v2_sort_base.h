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
 * \file moe_v2_sort_base.h
 * \brief
 */
#ifndef INNER_MOE_V2_SORT_BASE_H
#define INNER_MOE_V2_SORT_BASE_H

#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2SortBase {
 public:
  __aicore__ inline MoeV2SortBase(){};

 protected:
  __aicore__ inline void SyncAll();

 protected:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
  TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
  TBuf<TPosition::VECCALC> tempBuffer;
  TBuf<TPosition::VECCALC> sortedBuffer;

  GlobalTensor<int32_t> expertIdxGm;
  GlobalTensor<int32_t> sortedexpertIdxGm;
  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t tileLength;
  int64_t bufferNum = 1;
  int64_t totalLength;
  int64_t coreNum;
  int64_t n;
  int64_t k;
  int64_t existRowIdx;
  int64_t expertNum;
  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;

  static constexpr int64_t SYNC_GM_NUM = 2;
  static constexpr int64_t WORK_GM_NUM = 2;
  static constexpr int64_t DST_BLK_STRIDE = 1;
  static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeV2SortBase::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_SORT_BASE_H