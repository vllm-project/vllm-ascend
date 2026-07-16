/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_VECTOR_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRuleComputeWy {

using namespace AscendC;

constexpr uint32_t WY_ELEMS_PER_REP_FP32 = 64;

__aicore__ inline int32_t WyFindPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// Reduce srcLocal[0:count) and write the scalar sum into dstLocal[0] via VECTOR only.
// Avoids V_S + GetValue; caller must sync V_S once before any scalar read of dst.
__aicore__ inline void ReduceSumFp32ToUb(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                         uint32_t count)
{
    if (count > WY_ELEMS_PER_REP_FP32) {
        int32_t bodyCount = WyFindPowerTwo(static_cast<int32_t>(count));
        const int32_t tailCount = static_cast<int32_t>(count) - bodyCount;
        if (tailCount > 0) {
            Add(srcLocal, srcLocal, srcLocal[bodyCount], static_cast<uint32_t>(tailCount));
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > static_cast<int32_t>(WY_ELEMS_PER_REP_FP32)) {
            bodyCount /= 2;
            Add(srcLocal, srcLocal, srcLocal[bodyCount], static_cast<uint32_t>(bodyCount));
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(WY_ELEMS_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dstLocal, srcLocal, WY_ELEMS_PER_REP_FP32, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dstLocal, srcLocal, WY_ELEMS_PER_REP_FP32, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
}

class WyVectorGemm {
 public:
  __aicore__ inline void GemmATransB(const LocalTensor<float> &attn, const LocalTensor<float> &a,
                                     const LocalTensor<float> &b, const LocalTensor<float> &mulTmp, uint32_t m,
                                     uint32_t k, uint32_t lda, uint32_t ldb) const {
    if (k < WY_ELEMS_PER_REP_FP32) {
      // Small dimensions (k < 64): with guaranteed tail cleanup
      for (uint32_t i = 0; i < m; ++i) {
        const uint32_t aRowOff = i * lda;
        for (uint32_t j = 0; j < m; ++j) {
          const uint32_t bRowOff = j * ldb;
          Duplicate(mulTmp, static_cast<float>(0.0f), WY_ELEMS_PER_REP_FP32);
          PipeBarrier<PIPE_V>();
          Mul(mulTmp, a[aRowOff], b[bRowOff], k);
          PipeBarrier<PIPE_V>();
          ReduceSumFp32ToUb(attn[i * m + j], mulTmp, k);
          PipeBarrier<PIPE_V>();
        }
      }
    } else {
      // Standard dimensions (k >= 64): without unnecessary Duplicate instructions
      for (uint32_t i = 0; i < m; ++i) {
        const uint32_t aRowOff = i * lda;
        for (uint32_t j = 0; j < m; ++j) {
          const uint32_t bRowOff = j * ldb;
          Mul(mulTmp, a[aRowOff], b[bRowOff], k);
          PipeBarrier<PIPE_V>();
          ReduceSumFp32ToUb(attn[i * m + j], mulTmp, k);
          PipeBarrier<PIPE_V>();
        }
      }
    }
    // Single V_S for subsequent scalar reads of attn (e.g. decay loop).
    event_t eventVs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVs);
    WaitFlag<HardEvent::V_S>(eventVs);
  }

  __aicore__ inline void GemmAttnB(const LocalTensor<float> &c, const LocalTensor<float> &attn,
                                   const LocalTensor<float> &b, const LocalTensor<float> &mulRow, uint32_t m,
                                   uint32_t n, uint32_t ldb, uint32_t ldc) const {
    Duplicate(c, static_cast<float>(0.0f), m * ldc);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < m; ++i) {
      const uint32_t cRowOff = i * ldc;
      for (uint32_t j = 0; j <= i; ++j) {
        const float coeff = attn.GetValue(i * m + j);
        if (coeff == 0.0f) {
          continue;
        }
        const uint32_t bRowOff = j * ldb;
        Muls(mulRow, b[bRowOff], coeff, n);
        Add(c[cRowOff], c[cRowOff], mulRow, n);
      }
      PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
  }
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_VECTOR_H
