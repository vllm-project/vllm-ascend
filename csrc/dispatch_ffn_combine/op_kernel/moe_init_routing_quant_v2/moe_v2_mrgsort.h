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
 * \file moe_v2_mrgsort.h
 * \brief
 */
#ifndef INNER_MOE_V2_MRGSORT_H
#define INNER_MOE_V2_MRGSORT_H

#include "moe_v2_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
struct MoeV2MrgsortParam {
  int64_t perListElements;
  int64_t lastListElements;
  int64_t oneLoopMaxElements;
};

class MoeV2Mrgsort {
 public:
  __aicore__ inline MoeV2Mrgsort(){};
  __aicore__ inline void Init(MoeV2MrgsortParam* param);
  __aicore__ inline void Process();
  __aicore__ inline void SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput);
  __aicore__ inline void SetOutput(GlobalTensor<float>& gmOutput, LocalTensor<float>& ubOutput);

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void UpdateMrgParam();
  __aicore__ inline void MrgsortCompute();
  __aicore__ inline void UpdateSortInfo();
  __aicore__ inline void CopyOut();
  __aicore__ inline void ClearCache();

 private:
  MoeV2MrgsortParam* param = nullptr;

  GlobalTensor<float> gmInputs[4];
  GlobalTensor<float> gmOutput;

  LocalTensor<float> ubInputs[4];
  LocalTensor<float> ubOutput;

  int64_t listNum{0};
  int64_t remainListNum{0};
  int64_t outOffset{0};
  int64_t offsets[4];
  int64_t listRemainElements[4];
  int64_t lengths[4];
  int64_t allRemainElements{0};
  int64_t curLoopSortedNum{0};

  // for MrgSort
  uint16_t validBitTail{0};
  uint16_t elementCountListTail[4];
  uint32_t listSortedNums[4];
  LocalTensor<float> tmpUbInputs[4];
};

__aicore__ inline void MoeV2Mrgsort::ClearCache() {
  this->listNum = 0;
  this->allRemainElements = 0;
  this->outOffset = 0;
}

__aicore__ inline void MoeV2Mrgsort::SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput) {
  this->gmInputs[listNum] = gmInput;
  this->ubInputs[listNum] = ubInput;
  this->listNum += 1;
}

__aicore__ inline void MoeV2Mrgsort::SetOutput(GlobalTensor<float>& gmOutput, LocalTensor<float>& ubOutput) {
  this->gmOutput = gmOutput;
  this->ubOutput = ubOutput;
}

__aicore__ inline void MoeV2Mrgsort::UpdateMrgParam() {
  if (this->remainListNum == MERGE_LIST_TWO) {
    elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
    elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
    validBitTail = 0b0011;
  } else if (this->remainListNum == MERGE_LIST_THREE) {
    elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
    validBitTail = 0b0111;
  } else if (this->remainListNum == MERGE_LIST_FOUR) {
    validBitTail = 0b1111;
  } else {
    validBitTail = 0b0001;
  }
}

__aicore__ inline void MoeV2Mrgsort::CopyIn() {
  this->remainListNum = 0;
  SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
  for (int64_t i = 0, j = 0; i < listNum; i++) {
    lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
    if (lengths[i] > 0) {
      DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]], Align(GetSortLen<float>(lengths[i]), sizeof(float)));
      tmpUbInputs[j] = this->ubInputs[i];
      elementCountListTail[j] = lengths[i];
      this->remainListNum += 1;
      j++;
    }
  }
}

__aicore__ inline void MoeV2Mrgsort::MrgsortCompute() {
  SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
  if (this->remainListNum == MERGE_LIST_TWO) {
    MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
    MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
  } else if (this->remainListNum == MERGE_LIST_THREE) {
    MrgSortSrcList sortListTail =
        MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
    MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
  } else if (this->remainListNum == MERGE_LIST_FOUR) {
    MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO],
                                                 tmpUbInputs[MERGE_LIST_IDX_THREE]);
    MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
  } else {
    DataCopy(this->ubOutput, this->tmpUbInputs[0], Align(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
    listSortedNums[0] = elementCountListTail[0];
  }
}

__aicore__ inline void MoeV2Mrgsort::UpdateSortInfo() {
  curLoopSortedNum = 0;
  for (int64_t i = 0, j = 0; i < listNum; i++) {
    if (lengths[i] > 0) {
      // update remain size
      listRemainElements[i] -= listSortedNums[j];
      allRemainElements -= listSortedNums[j];
      // update offset
      offsets[i] += GetSortOffset<float>(listSortedNums[j]);
      // update current loop sorted nums
      curLoopSortedNum += listSortedNums[j];
      j += 1;
    }
  }
}

__aicore__ inline void MoeV2Mrgsort::CopyOut() {
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = GetSortLen<float>(curLoopSortedNum) * sizeof(float);
  SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
  DataCopyPad(this->gmOutput[outOffset], this->ubOutput, intriParams);
  outOffset += GetSortLen<float>(curLoopSortedNum);
}

__aicore__ inline void MoeV2Mrgsort::Init(MoeV2MrgsortParam* param) {
  this->param = param;
  this->remainListNum = listNum;

  for (int64_t i = 0; i < listNum; i++) {
    offsets[i] = GetSortOffset<float>(param->perListElements * i);
    if (i == listNum - 1) {
      listRemainElements[i] = param->lastListElements;
    } else {
      listRemainElements[i] = param->perListElements;
    }
    allRemainElements += listRemainElements[i];
  }
}

__aicore__ inline void MoeV2Mrgsort::Process() {
  for (; allRemainElements > 0;) {
    CopyIn();
    UpdateMrgParam();
    MrgsortCompute();
    UpdateSortInfo();
    CopyOut();
  }

  ClearCache();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_MRGSORT_H