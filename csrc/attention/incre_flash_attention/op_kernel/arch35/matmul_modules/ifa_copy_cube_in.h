/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ifa_copy_cube_in.h
 * \brief
 */

#ifndef IFA_COPY_CUBE_IN_H
#define IFA_COPY_CUBE_IN_H

#include "ifa_flag_data.h"
namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class IFACopyCubeIn {
  MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
  using TRANS_T = typename INPUT_TYPE::TRANS_T;
  using SRC_T = typename INPUT_TYPE::T;

 public:
  __aicore__ inline IFACopyCubeIn() = default;
  __aicore__ inline ~IFACopyCubeIn() = default;

  __aicore__ inline void Init(int32_t totalRow, int32_t totalCol, int32_t orgHeight, int32_t orgWidth,
                              int32_t baseHeight, int32_t baseWidth, int32_t subBlockIdx, int32_t batchNum = -1) {
    this->orgHeight = orgHeight;
    this->orgWidth = orgWidth;
    this->baseHeight = baseHeight;
    this->baseWidth = baseWidth;
    MATMUL_MODULE(CubeInBuffer)->Init();
  }

  __aicore__ inline void SetOriShape(int32_t orgHeight, int32_t orgWidth) {
    this->orgHeight = orgHeight;
    this->orgWidth = orgWidth;
  }
  __aicore__ inline void SetBaseShape(int32_t baseHeight, int32_t baseWidth) {
    this->baseHeight = baseHeight;
    this->baseWidth = baseWidth;
  }
  __aicore__ inline void SetSplitCount(int32_t totalRow, int32_t totalCol) {
  }

  __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx) {
  }

  __aicore__ inline void TransposeShape(bool isTranspose = false) {
  }

  __aicore__ inline void SetInput(const TBuffAddr& address) {
  }

  __aicore__ inline void SetInput(__gm__ SRC_T* srcGlobalAddr) {
    this->srcGlobalAddr = srcGlobalAddr;
  }

  __aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB) {
  }

  // left matrix only reuse or not, no prefetch
  __aicore__ inline LocalTensor<TRANS_T> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
                                                  int batchNum = -1) {
    IFAFlagData flag = MATMUL_PARAM_VAR.dataPtr_;
    LocalTensor<TRANS_T> l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.tscmIdx);
    if (!flag.tscmReuse) {
      GlobalTensor<SRC_T> aGlobal;
      aGlobal.SetGlobalBuffer(srcGlobalAddr);
      Nd2NzParams nd2nzParams;
      nd2nzParams.ndNum = 1;
      nd2nzParams.nValue = tileHeight;
      nd2nzParams.dValue = tileWidth;
      nd2nzParams.srcNdMatrixStride = 0;
      nd2nzParams.srcDValue = orgWidth;
      nd2nzParams.dstNzC0Stride = ToMatmulConfig(MM_CFG).basicM;
      nd2nzParams.dstNzNStride = 1;
      nd2nzParams.dstNzMatrixStride = 0;
      DataCopy(l1, aGlobal, nd2nzParams);
      MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
      MATMUL_MODULE(CubeInBuffer)->DeQue();
    }

    return l1;
  }

  __aicore__ inline void ClearLoadData(int curRow, int curCol, const LocalTensor<TRANS_T>& aMatrix) {
    MATMUL_MODULE(CubeInBuffer)->FreeTensor();
  }

  __aicore__ inline void Destroy() {
    MATMUL_MODULE(CubeInBuffer)->Destroy();
  }

 private:
  __gm__ SRC_T* srcGlobalAddr;
  int32_t baseHeight;
  int32_t baseWidth;
  int32_t orgHeight;
  int32_t orgWidth;
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // IFA_COPY_CUBE_IN_H
