/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace ChunkGatedDeltaRuleComputeWy {

using namespace AscendC;

// Atlas 310P / inference AI Core: half x half -> float via Cube.
using WyMmAType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using WyMmBType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using WyMmBTransType = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
using WyMmCType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using WyMmBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

using WyMatmulNoTrans = matmul::MatmulImpl<WyMmAType, WyMmBType, WyMmCType, WyMmBiasType>;
using WyMatmulBTrans = matmul::MatmulImpl<WyMmAType, WyMmBTransType, WyMmCType, WyMmBiasType>;

// GM staging: A_half(64*128) + B_half(64*128) + C_float(64*128).
// Doubling applies U/W separately so N <= 128 (no stacked 256-wide staging).
constexpr uint32_t WY_CUBE_MAX_HEAD = 128;
constexpr uint32_t WY_CUBE_CHUNK = 64;
constexpr uint32_t WY_CUBE_STAGING_A_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_B_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_C_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(float);
constexpr uint32_t WY_CUBE_STAGING_A_OFF = 0;
constexpr uint32_t WY_CUBE_STAGING_B_OFF = WY_CUBE_STAGING_A_BYTES;
constexpr uint32_t WY_CUBE_STAGING_C_OFF = WY_CUBE_STAGING_A_BYTES + WY_CUBE_STAGING_B_BYTES;

class WyCubeGemm {
 public:
  __aicore__ inline void Init(const TCubeTiling *attnTiling, const TCubeTiling *squareTiling,
                              const TCubeTiling *applyUTiling, const TCubeTiling *applyWTiling, TPipe *pipe,
                              LocalTensor<uint8_t> localWs, uint32_t localWsBytes, GM_ADDR workspace,
                              uint32_t perCoreBytes, uint32_t usedCoreNum)
  {
    pipe_ = pipe;
    perCoreBytes_ = perCoreBytes;
    usedCoreNum_ = usedCoreNum == 0 ? 1 : usedCoreNum;

    mmAttn_.SetSubBlockIdx(0);
    mmAttn_.Init(attnTiling, pipe_);
    mmSquare_.SetSubBlockIdx(0);
    mmSquare_.Init(squareTiling, pipe_);
    mmApplyU_.SetSubBlockIdx(0);
    mmApplyU_.Init(applyUTiling, pipe_);
    mmApplyW_.SetSubBlockIdx(0);
    mmApplyW_.Init(applyWTiling, pipe_);

    localWs_ = localWs;
    (void)localWsBytes;

    constexpr uint64_t kSysWorkspace = 16ULL * 1024ULL * 1024ULL;
    const uint32_t blockIdx = GetBlockIdx() % usedCoreNum_;
    const uint64_t coreBase =
        kSysWorkspace + static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(perCoreBytes_);
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_A_OFF));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_B_OFF));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + coreBase + WY_CUBE_STAGING_C_OFF));
  }

  // C[64,64] = A[64,K] @ B[64,K]^T
  __aicore__ inline void GemmATransB(LocalTensor<float> cUb, const LocalTensor<half> aUb, const LocalTensor<half> bUb,
                                     uint32_t kDim, uint32_t aLda, uint32_t bLda)
  {
    CopyHalfRowsToGm(aGm_, aUb, WY_CUBE_CHUNK, kDim, aLda);
    CopyHalfRowsToGm(bGm_, bUb, WY_CUBE_CHUNK, kDim, bLda);
    PipeBarrier<PIPE_ALL>();

    mmAttn_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kDim));
    mmAttn_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kDim));
    mmAttn_.SetLocalWorkspace(localWs_);
    mmAttn_.SetTensorA(aGm_, false);
    mmAttn_.SetTensorB(bGm_, true);
    mmAttn_.IterateAll(cGm_);
    WaitFixpipeToMte2();
    PipeBarrier<PIPE_ALL>();

    CopyFloatRowsFromGm(cUb, cGm_, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    WaitMte2ToV();
  }

  // P = P @ P (64x64). halfScratch >= 64*64 halves.
  __aicore__ inline void GemmSquare(LocalTensor<float> pUb, LocalTensor<half> halfScratch)
  {
    Cast(halfScratch, pUb, RoundMode::CAST_NONE, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
    CopyHalfRowsToGm(aGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    CopyHalfRowsToGm(bGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    PipeBarrier<PIPE_ALL>();

    mmSquare_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmSquare_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmSquare_.SetLocalWorkspace(localWs_);
    mmSquare_.SetTensorA(aGm_, false);
    mmSquare_.SetTensorB(bGm_, false);
    mmSquare_.IterateAll(cGm_);
    WaitFixpipeToMte2();
    PipeBarrier<PIPE_ALL>();

    CopyFloatRowsFromGm(pUb, cGm_, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    WaitMte2ToV();
  }

  // Cast P[64,64] → halfScratch and upload to aGm_. halfScratch must stay live until
  // MTE3 finishes and must NOT alias the R half buffer used by GemmApplyAdd.
  __aicore__ inline void UploadP(const LocalTensor<float> pUb, LocalTensor<half> halfScratch)
  {
    Cast(halfScratch, pUb, RoundMode::CAST_NONE, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
    CopyHalfRowsToGm(aGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(evt);
    WaitFlag<HardEvent::MTE3_V>(evt);
  }

  // R = R + P @ R for one RHS block (U or W). Assumes P already on aGm_ (UploadP).
  // halfScratch holds contiguous R half only (>= 64*nDim) — do not reuse UploadP buffer.
  // floatScratch (>= 64*nDim floats, e.g. tmpBuf_ [64,max(K,V)]) holds C for Add.
  __aicore__ inline void GemmApplyAdd(LocalTensor<float> rUb, LocalTensor<half> halfScratch,
                                      LocalTensor<float> floatScratch, uint32_t nDim, uint32_t rLda, bool useU)
  {
    CastFloatRowsToHalfContiguous(halfScratch, rUb, WY_CUBE_CHUNK, nDim, rLda);
    CopyHalfRowsToGm(bGm_, halfScratch, WY_CUBE_CHUNK, nDim, nDim);
    PipeBarrier<PIPE_ALL>();

    if (useU) {
      mmApplyU_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyU_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyU_.SetLocalWorkspace(localWs_);
      mmApplyU_.SetTensorA(aGm_, false);
      mmApplyU_.SetTensorB(bGm_, false);
      mmApplyU_.IterateAll(cGm_);
    } else {
      mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyW_.SetLocalWorkspace(localWs_);
      mmApplyW_.SetTensorA(aGm_, false);
      mmApplyW_.SetTensorB(bGm_, false);
      mmApplyW_.IterateAll(cGm_);
    }
    WaitFixpipeToMte2();
    PipeBarrier<PIPE_ALL>();

    // C[64,nDim] from GM -> floatScratch, then R += C
    CopyFloatRowsFromGm(floatScratch, cGm_, WY_CUBE_CHUNK, nDim, nDim);
    WaitMte2ToV();
    if (rLda == nDim) {
      Add(rUb, rUb, floatScratch, WY_CUBE_CHUNK * nDim);
    } else {
      for (uint32_t row = 0; row < WY_CUBE_CHUNK; ++row) {
        Add(rUb[row * rLda], rUb[row * rLda], floatScratch[row * nDim], nDim);
      }
    }
    PipeBarrier<PIPE_V>();
  }

 private:
  __aicore__ inline void WaitFixpipeToMte2() const
  {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE2));
    SetFlag<HardEvent::FIX_MTE2>(evt);
    WaitFlag<HardEvent::FIX_MTE2>(evt);
  }

  __aicore__ inline void WaitMte2ToV() const
  {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evt);
    WaitFlag<HardEvent::MTE2_V>(evt);
  }

  __aicore__ inline void CastFloatRowsToHalfContiguous(LocalTensor<half> dst, const LocalTensor<float> src,
                                                       uint32_t rows, uint32_t cols, uint32_t srcLda) const
  {
    if (srcLda == cols) {
      Cast(dst, src, RoundMode::CAST_NONE, rows * cols);
    } else {
      for (uint32_t row = 0; row < rows; ++row) {
        Cast(dst[row * cols], src[row * srcLda], RoundMode::CAST_NONE, cols);
      }
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyHalfRowsToGm(GlobalTensor<half> dst, const LocalTensor<half> src, uint32_t rows,
                                          uint32_t cols, uint32_t lda) const
  {
    const uint32_t rowBytes = cols * sizeof(half);
    if (lda == cols) {
      DataCopyParams params{1, static_cast<uint16_t>((rows * rowBytes + 31) / 32), 0, 0};
      DataCopy(dst, src, params);
      return;
    }
    DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>((rowBytes + 31) / 32),
                          static_cast<uint16_t>(((lda - cols) * sizeof(half)) / 32), 0};
    DataCopy(dst, src, params);
  }

  __aicore__ inline void CopyFloatRowsFromGm(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t rows,
                                             uint32_t cols, uint32_t lda) const
  {
    const uint32_t rowBytes = cols * sizeof(float);
    if (lda == cols) {
      DataCopyParams params{1, static_cast<uint16_t>((rows * rowBytes + 31) / 32), 0, 0};
      DataCopy(dst, src, params);
      return;
    }
    DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>((rowBytes + 31) / 32), 0,
                          static_cast<uint16_t>(((lda - cols) * sizeof(float)) / 32)};
    DataCopy(dst, src, params);
  }

  TPipe *pipe_{nullptr};
  uint32_t perCoreBytes_{0};
  uint32_t usedCoreNum_{1};
  LocalTensor<uint8_t> localWs_;
  WyMatmulBTrans mmAttn_;
  WyMatmulNoTrans mmSquare_;
  WyMatmulNoTrans mmApplyU_;
  WyMatmulNoTrans mmApplyW_;
  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
  GlobalTensor<float> cGm_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
