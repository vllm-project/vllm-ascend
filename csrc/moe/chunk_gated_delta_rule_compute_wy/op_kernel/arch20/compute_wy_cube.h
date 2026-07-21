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

// GM staging layout inside each core's workspace slice (bytes).
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
  __aicore__ inline void Init(const TCubeTiling *attnTiling, const TCubeTiling *uTiling, const TCubeTiling *wTiling,
                              TPipe *pipe, LocalTensor<uint8_t> localWs, uint32_t localWsBytes, GM_ADDR workspace,
                              uint32_t perCoreBytes, uint32_t usedCoreNum)
  {
    pipe_ = pipe;
    perCoreBytes_ = perCoreBytes;
    usedCoreNum_ = usedCoreNum == 0 ? 1 : usedCoreNum;

    mmAttn_.SetSubBlockIdx(0);
    mmAttn_.Init(attnTiling, pipe_);
    mmU_.SetSubBlockIdx(0);
    mmU_.Init(uTiling, pipe_);
    mmW_.SetSubBlockIdx(0);
    mmW_.Init(wTiling, pipe_);

    localWs_ = localWs;
    (void)localWsBytes;

    // User staging sits after the reserved system workspace region.
    constexpr uint64_t kSysWorkspace = 16ULL * 1024ULL * 1024ULL;
    const uint32_t blockIdx = GetBlockIdx() % usedCoreNum_;
    const uint64_t coreBase =
        kSysWorkspace + static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(perCoreBytes_);
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_A_OFF));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_B_OFF));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + coreBase + WY_CUBE_STAGING_C_OFF));
  }

  // C[64,64] = A[64,K] @ B[64,K]^T  (B stored as [64,K], transpose on Cube)
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
    PipeBarrier<PIPE_ALL>();

    CopyFloatRowsFromGm(cUb, cGm_, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
  }

  // C[64,N] = A[64,64] @ B[64,N]
  // dstLda is the row stride of cUb (may be alignN > nDim).
  __aicore__ inline void GemmAB(LocalTensor<float> cUb, const LocalTensor<half> aUb, const LocalTensor<half> bUb,
                                uint32_t nDim, uint32_t aLda, uint32_t bLda, uint32_t dstLda, bool useUMatmul)
  {
    CopyHalfRowsToGm(aGm_, aUb, WY_CUBE_CHUNK, WY_CUBE_CHUNK, aLda);
    CopyHalfRowsToGm(bGm_, bUb, WY_CUBE_CHUNK, nDim, bLda);
    PipeBarrier<PIPE_ALL>();

    if (useUMatmul) {
      mmU_.SetLocalWorkspace(localWs_);
      mmU_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmU_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmU_.SetTensorA(aGm_, false);
      mmU_.SetTensorB(bGm_, false);
      mmU_.IterateAll(cGm_);
    } else {
      mmW_.SetLocalWorkspace(localWs_);
      mmW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmW_.SetTensorA(aGm_, false);
      mmW_.SetTensorB(bGm_, false);
      mmW_.IterateAll(cGm_);
    }
    PipeBarrier<PIPE_ALL>();

    CopyFloatRowsFromGm(cUb, cGm_, WY_CUBE_CHUNK, nDim, dstLda);
  }

 private:
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
  WyMatmulNoTrans mmU_;
  WyMatmulNoTrans mmW_;
  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
  GlobalTensor<float> cGm_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
