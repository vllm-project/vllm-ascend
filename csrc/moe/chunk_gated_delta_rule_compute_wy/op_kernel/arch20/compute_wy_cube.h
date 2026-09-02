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
using WyMmCType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;
using WyMmBiasType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;

using WyMatmulNoTrans = matmul::MatmulImpl<WyMmAType, WyMmBType, WyMmCType, WyMmBiasType>;
using WyMatmulBTrans = matmul::MatmulImpl<WyMmAType, WyMmBTransType, WyMmCType, WyMmBiasType>;
// Apply matmuls read A/B straight from UB — no UB->GM->L1 staging round-trip.
using WyMmAUbType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using WyMmBUbType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using WyMatmulApplyUb = matmul::MatmulImpl<WyMmAUbType, WyMmBUbType, WyMmCType, WyMmBiasType>;
// (half-C UB output wedges the 310P cube — aicore timeout; keep C fp32)

// GM staging: A_half(64*MAX_HEAD) + B_half(64*MAX_HEAD). Cube writes C directly to UB.
// W/U are applied separately so N <= MAX_HEAD (no stacked 2N-wide staging).
// 128 is UB-safe again since the kernel solves W and U in two passes (one RHS
// resident at a time) instead of keeping K,V,Kβ fp32 chunks live together.
constexpr uint32_t WY_CUBE_MAX_HEAD = 128;
constexpr uint32_t WY_CUBE_CHUNK = 64;
constexpr uint32_t WY_CUBE_STAGING_A_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_B_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_A_OFF = 0;
constexpr uint32_t WY_CUBE_STAGING_B_OFF = WY_CUBE_STAGING_A_BYTES;
class WyCubeGemm {
 public:
  __aicore__ inline void Init(const TCubeTiling *attnTiling, const TCubeTiling *squareTiling,
                              const TCubeTiling *applyUTiling, const TCubeTiling *applyWTiling, TPipe *pipe,
                              LocalTensor<uint8_t> localWs, uint32_t localWsBytes, GM_ADDR workspace,
                              uint64_t workspaceOffset, uint32_t perCoreBytes, uint32_t usedCoreNum,
                              uint32_t kHeadDim, uint32_t vHeadDim)
  {
    pipe_ = pipe;
    perCoreBytes_ = perCoreBytes;
    usedCoreNum_ = usedCoreNum == 0 ? 1 : usedCoreNum;

    mmAttn_.SetSubBlockIdx(0);
    mmAttn_.Init(attnTiling, pipe_);
    (void)squareTiling;  // Kept in the host tiling ABI; blocked-T no longer squares A.
    mmApplyU_.SetSubBlockIdx(0);
    mmApplyU_.Init(applyUTiling, pipe_);
    mmApplyW_.SetSubBlockIdx(0);
    mmApplyW_.Init(applyWTiling, pipe_);


    localWs_ = localWs;
    (void)localWsBytes;

    // Every cube call runs at <=64-wide tiles (larger single-shapes wedge the
    // 310P matmul); per-call SetOrgShape/SetSingleShape in the helpers below
    // re-set partial slices, so Init only ever configures 64^3.
    (void)kHeadDim;
    (void)vHeadDim;
    mmAttn_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmAttn_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmAttn_.SetLocalWorkspace(localWs_);
    mmApplyU_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmApplyU_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmApplyU_.SetLocalWorkspace(localWs_);
    mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmApplyW_.SetLocalWorkspace(localWs_);


    const uint32_t blockIdx = GetBlockIdx() % usedCoreNum_;
    const uint64_t coreBase =
        workspaceOffset + static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(perCoreBytes_);
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_A_OFF));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + coreBase + WY_CUBE_STAGING_B_OFF));
  }

  // C[64,64] = A[64,K] @ B[64,K]^T
  // K is split into <=64 slices and accumulated in UB: on 310P a gram matmul whose inner
  // K exceeds 64 (i.e. needs more than one K iteration) hangs the AI core (aicore timeout).
  __aicore__ inline void GemmATransB(LocalTensor<float> cUb, const LocalTensor<half> aUb, const LocalTensor<half> bUb,
                                     LocalTensor<float> accScratch, uint32_t kDim, uint32_t aLda, uint32_t bLda)
  {
    // Each K slice is restaged contiguously so no cube call — including its
    // OrgShape strides — ever sees a dimension above 64 (128 hangs the aicore).
    for (uint32_t k0 = 0; k0 < kDim; k0 += WY_CUBE_CHUNK) {
      const uint32_t kCur = (kDim - k0) < WY_CUBE_CHUNK ? (kDim - k0) : WY_CUBE_CHUNK;
      WaitVToMte3();
      CopyHalfRowsToGm(aGm_, aUb[k0], WY_CUBE_CHUNK, kCur, aLda);
      CopyHalfRowsToGm(bGm_, bUb[k0], WY_CUBE_CHUNK, kCur, bLda);
      WaitMte3ToMte2();

      // Re-configuring shapes between IterateAll calls wedges the 310P matmul;
      // full 64-slices keep Init's 64^3 config untouched.
      if (kCur != WY_CUBE_CHUNK) {
        mmAttn_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kCur));
        mmAttn_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kCur));
      }
      mmAttn_.SetTensorA(aGm_, false);
      mmAttn_.SetTensorB(bGm_, true);
      if (k0 == 0) {
        mmAttn_.IterateAll(cUb);
      } else {
        mmAttn_.IterateAll(accScratch);
      }
      PipeBarrier<PIPE_ALL>();
      if (k0 != 0) {
        Add(cUb, cUb, accScratch, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // C[64,64] = A[64,64] @ B[64,64] for one blocked-T prefix update.
  // The caller zero-pads unused rows/columns, keeping every invocation on the
  // static 64^3 shape that is reliable on 310P.
  __aicore__ inline void GemmBlockedTUpdate(LocalTensor<float> cUb, LocalTensor<half> aUb,
                                            LocalTensor<half> bUb) {
    WaitVToMte3();
    mmApplyU_.SetTensorA(aUb, false);
    mmApplyU_.SetTensorB(bUb, false);
    mmApplyU_.IterateAll(cUb);
    PipeBarrier<PIPE_ALL>();
  }

  // C[64,nCur] = T[64,64] @ B[64,nCur]. Both inputs are compact fp16
  // operands prepared by the caller and C remains fp32. Keeping this primitive
  // free of casts lets the kernel accumulate compensated hi/lo products.
  __aicore__ inline void GemmApplyRaw(LocalTensor<float> cUb, LocalTensor<half> tUbHalf,
                                      LocalTensor<half> bUb, uint32_t nCur) {
    WaitVToMte3();
    if (nCur != applyShapeN_) {
      mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
      mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
      applyShapeN_ = nCur;
    }
    mmApplyW_.SetTensorA(tUbHalf, false);
    mmApplyW_.SetTensorB(bUb, false);
    mmApplyW_.IterateAll(cUb);
    PipeBarrier<PIPE_ALL>();
  }

 private:
  uint32_t applyShapeN_{WY_CUBE_CHUNK};

  __aicore__ inline void WaitVToMte3() const
  {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evt);
    WaitFlag<HardEvent::V_MTE3>(evt);
  }

  __aicore__ inline void WaitMte3ToMte2() const
  {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(evt);
    WaitFlag<HardEvent::MTE3_MTE2>(evt);
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

  TPipe *pipe_{nullptr};
  uint32_t perCoreBytes_{0};
  uint32_t usedCoreNum_{1};
  LocalTensor<uint8_t> localWs_;
  WyMatmulBTrans mmAttn_;
  WyMatmulApplyUb mmApplyU_;
  WyMatmulApplyUb mmApplyW_;
  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
