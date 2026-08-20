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

// GM staging: A_half(64*MAX_HEAD) + B_half(64*MAX_HEAD). Cube writes C directly to UB.
// Doubling applies U/W separately so N <= MAX_HEAD (no stacked 2N-wide staging).
// 128 is UB-safe again since the kernel solves W and U in two passes (one RHS
// resident at a time) instead of keeping K,V,Kβ fp32 chunks live together.
constexpr uint32_t WY_CUBE_MAX_HEAD = 128;
constexpr uint32_t WY_CUBE_CHUNK = 64;
constexpr uint32_t WY_CUBE_STAGING_A_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_B_BYTES = WY_CUBE_CHUNK * WY_CUBE_MAX_HEAD * sizeof(half);
constexpr uint32_t WY_CUBE_STAGING_A_OFF = 0;
constexpr uint32_t WY_CUBE_STAGING_B_OFF = WY_CUBE_STAGING_A_BYTES;
// Per-core fp32 A-snapshot slot: the two-pass solve parks A here between passes
// (UB is too tight at head dim 128 to keep a second 64x64 fp32 copy resident).
constexpr uint32_t WY_CUBE_SNAP_BYTES = WY_CUBE_CHUNK * WY_CUBE_CHUNK * sizeof(float);
constexpr uint32_t WY_CUBE_SNAP_OFF = WY_CUBE_STAGING_A_BYTES + WY_CUBE_STAGING_B_BYTES;

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
    mmSquare_.SetSubBlockIdx(0);
    mmSquare_.Init(squareTiling, pipe_);
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
    mmSquare_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmSquare_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmSquare_.SetLocalWorkspace(localWs_);
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
    snapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + coreBase + WY_CUBE_SNAP_OFF));
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
      PipeBarrier<PIPE_ALL>();

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

  // P = P @ P (64x64). halfScratch >= 64*64 halves.
  __aicore__ inline void GemmSquare(LocalTensor<float> pUb, LocalTensor<half> halfScratch)
  {
    Cast(halfScratch, pUb, RoundMode::CAST_NONE, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
    WaitVToMte3();
    CopyHalfRowsToGm(aGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    CopyHalfRowsToGm(bGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    WaitMte3ToMte2();
    PipeBarrier<PIPE_ALL>();

    mmSquare_.SetTensorA(aGm_, false);
    mmSquare_.SetTensorB(bGm_, false);
    mmSquare_.IterateAll(pUb);
    PipeBarrier<PIPE_ALL>();
  }

  // Cast P[64,64] → halfScratch and upload to aGm_. halfScratch must stay live until
  // MTE3 finishes and must NOT alias the R half buffer used by GemmApplyAdd.
  __aicore__ inline void UploadP(const LocalTensor<float> pUb, LocalTensor<half> halfScratch)
  {
    Cast(halfScratch, pUb, RoundMode::CAST_NONE, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
    WaitVToMte3();
    CopyHalfRowsToGm(aGm_, halfScratch, WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    // Apply matmul reads A from GM via MTE2, so we must gate MTE3 write completion
    // with MTE3->MTE2 instead of MTE3->V.
    WaitMte3ToMte2();
  }

  // Park/restore the fp32 A block in the per-core GM snapshot slot.
  __aicore__ inline void SaveSnap(const LocalTensor<float> pUb)
  {
    WaitVToMte3();
    DataCopyParams params{1, static_cast<uint16_t>((WY_CUBE_CHUNK * WY_CUBE_CHUNK * sizeof(float)) / 32), 0, 0};
    DataCopy(snapGm_, pUb, params);
    WaitMte3ToMte2();
  }
  __aicore__ inline void LoadSnap(LocalTensor<float> pUb)
  {
    // pUb (A) was last read/written by V or Cube; order that before the MTE2 write.
    event_t evtV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(evtV);
    WaitFlag<HardEvent::V_MTE2>(evtV);
    DataCopyParams params{1, static_cast<uint16_t>((WY_CUBE_CHUNK * WY_CUBE_CHUNK * sizeof(float)) / 32), 0, 0};
    DataCopy(pUb, snapGm_, params);
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evt);
    WaitFlag<HardEvent::MTE2_V>(evt);
  }

  // R = R + P @ R for one RHS block (U or W). Assumes P already on aGm_ (UploadP).
  // halfScratch holds contiguous R half only (>= 64*min(nDim,64)) — do not reuse the
  // UploadP buffer. floatScratch (>= 64*min(nDim,64) floats) holds C for Add.
  // nDim > 64 is processed in <=64-wide column slices: a single matmul call with
  // N=128 hangs the 310P aicore even with FixSplit-64 tiling, so no cube call may
  // ever see a dimension above 64.
  __aicore__ inline void GemmApplyAdd(LocalTensor<float> rUb, LocalTensor<half> halfScratch,
                                      LocalTensor<float> floatScratch, uint32_t nDim, uint32_t rLda, bool useU)
  {
    for (uint32_t n0 = 0; n0 < nDim; n0 += WY_CUBE_CHUNK) {
      const uint32_t nCur = (nDim - n0) < WY_CUBE_CHUNK ? (nDim - n0) : WY_CUBE_CHUNK;
      CastFloatRowsToHalfContiguous(halfScratch, rUb[n0], WY_CUBE_CHUNK, nCur, rLda);
      WaitVToMte3();
      CopyHalfRowsToGm(bGm_, halfScratch, WY_CUBE_CHUNK, nCur, nCur);
      WaitMte3ToMte2();
      PipeBarrier<PIPE_ALL>();

      if (useU) {
        if (nCur != WY_CUBE_CHUNK) {
          mmApplyU_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
          mmApplyU_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
        }
        mmApplyU_.SetTensorA(aGm_, false);
        mmApplyU_.SetTensorB(bGm_, false);
        mmApplyU_.IterateAll(floatScratch);
      } else {
        if (nCur != WY_CUBE_CHUNK) {
          mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
          mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nCur), WY_CUBE_CHUNK);
        }
        mmApplyW_.SetTensorA(aGm_, false);
        mmApplyW_.SetTensorB(bGm_, false);
        mmApplyW_.IterateAll(floatScratch);
      }
      PipeBarrier<PIPE_ALL>();

      // Cube wrote C[64,nCur] directly to floatScratch; now R[:, n0:n0+nCur] += C.
      if (rLda == nCur && n0 == 0) {
        Add(rUb, rUb, floatScratch, WY_CUBE_CHUNK * nCur);
      } else {
        for (uint32_t row = 0; row < WY_CUBE_CHUNK; ++row) {
          Add(rUb[row * rLda + n0], rUb[row * rLda + n0], floatScratch[row * nCur], nCur);
        }
      }
      PipeBarrier<PIPE_V>();
    }
  }

 private:
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
  GlobalTensor<float> snapGm_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
