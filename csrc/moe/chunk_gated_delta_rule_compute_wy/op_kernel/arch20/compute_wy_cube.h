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
using WyMmBUbTransType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, true>;
using WyMatmulBTransUb = matmul::MatmulImpl<WyMmAUbType, WyMmBUbTransType, WyMmCType, WyMmBiasType>;
// (half-C UB output wedges the 310P cube — aicore timeout; keep C fp32)

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
                                     LocalTensor<float> accScratch, LocalTensor<half> halfScratch, uint32_t kDim,
                                     uint32_t aLda, uint32_t bLda)
  {
    // Each K slice is restaged contiguously so no cube call — including its
    // OrgShape strides — ever sees a dimension above 64 (128 hangs the aicore).
    for (uint32_t k0 = 0; k0 < kDim; k0 += WY_CUBE_CHUNK) {
      const uint32_t kCur = (kDim - k0) < WY_CUBE_CHUNK ? (kDim - k0) : WY_CUBE_CHUNK;
      // UB-fed: K slices are compact-copied inside UB (halfScratch) instead of
      // round-tripping through the GM staging buffers.
      LocalTensor<half> aFeed = aUb;
      LocalTensor<half> bFeed = bUb;
      if (!(k0 == 0 && aLda == kCur)) {
        Muls(halfScratch, aUb[k0], static_cast<half>(1), static_cast<uint64_t>(kCur), WY_CUBE_CHUNK,
             {1, 1, static_cast<uint8_t>(kCur * sizeof(half) / 32), static_cast<uint8_t>(aLda * sizeof(half) / 32)});
        aFeed = halfScratch;
      }
      if (!(k0 == 0 && bLda == kCur)) {
        Muls(halfScratch[WY_CUBE_CHUNK * WY_CUBE_CHUNK], bUb[k0], static_cast<half>(1), static_cast<uint64_t>(kCur),
             WY_CUBE_CHUNK,
             {1, 1, static_cast<uint8_t>(kCur * sizeof(half) / 32), static_cast<uint8_t>(bLda * sizeof(half) / 32)});
        bFeed = halfScratch[WY_CUBE_CHUNK * WY_CUBE_CHUNK];
      }
      PipeBarrier<PIPE_V>();
      WaitVToMte3();
      if (kCur != WY_CUBE_CHUNK) {
        mmAttn_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kCur));
        mmAttn_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, static_cast<int>(kCur));
      }
      mmAttn_.SetTensorA(aFeed, false);
      mmAttn_.SetTensorB(bFeed, true);
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

  // Generic UB-fed matmul C[m,n] = A[m,k] @ B[k,n] on mmApplyU_ (shapes re-set
  // per call and restored to 64^3). Caller provides half operands in UB and a
  // fp32 C target in UB; caller owns pre-call V->MTE3 ordering.
  __aicore__ inline void ApplyGeneric(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                                      uint32_t m, uint32_t n, uint32_t k)
  {
    mmApplyU_.SetOrgShape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    mmApplyU_.SetSingleShape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    mmApplyU_.SetTensorA(aUb, false);
    mmApplyU_.SetTensorB(bUb, false);
    mmApplyU_.IterateAll(cUb);
    PipeBarrier<PIPE_ALL>();
    mmApplyU_.SetOrgShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
    mmApplyU_.SetSingleShape(WY_CUBE_CHUNK, WY_CUBE_CHUNK, WY_CUBE_CHUNK);
  }

  // R[:, n0:n0+nCur] = T @ R_slice per <=64-wide column slice; tUbHalf is the
  // resident half(T) in UB, bScratch receives the slice half-cast. REPLACES the
  // slice (T includes the identity) — one matmul per slice finishes the solve.
  __aicore__ inline void GemmApplyReplace(LocalTensor<float> rUb, LocalTensor<half> tUbHalf,
                                          LocalTensor<half> bScratch, LocalTensor<float> floatScratch,
                                          uint32_t nDim, uint32_t rLda, bool useU)
  {
    // DESLICE: one full-width call per RHS — C is written straight into rUb
    // (B was cast out to bScratch first, so no aliasing), removing both the
    // second matmul call and the writeback pass. floatScratch unused here.
    // Full-width single call, C written straight into rUb. (The 2026-08-21
    // "direct C NaNs" was misattributed — the NaN was the 8KB local-workspace
    // overrun; with 32KB this is the good-era-verified layout.)
    (void)useU;
    (void)floatScratch;
    CastFloatRowsToHalfContiguous(bScratch, rUb, WY_CUBE_CHUNK, nDim, rLda);
    WaitVToMte3();
    if (nDim != applyShapeN_) {
      mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      applyShapeN_ = nDim;
    }
    mmApplyW_.SetTensorA(tUbHalf, false);
    mmApplyW_.SetTensorB(bScratch, false);
    mmApplyW_.IterateAll(rUb);
    PipeBarrier<PIPE_ALL>();
  }


  // C[64,nDim] (fp32) = T @ B; B is the ready-half beta*V the caller built in
  // UB — skips the cast-out that GemmApplyReplace performs.
  __aicore__ inline void GemmApplyPreCastB(LocalTensor<float> rUb, LocalTensor<half> tUbHalf, LocalTensor<half> bUb,
                                           LocalTensor<half> bCompact, LocalTensor<float> floatScratch, uint32_t nDim,
                                           uint32_t bLda)
  {
    // Full-width single call; bUb is the ready-half beta*V at lda==nDim
    // (contiguous), C lands straight in rUb (fp32, dead buffer in this flow).
    (void)bCompact;
    (void)floatScratch;
    (void)bLda;
    WaitVToMte3();
    if (nDim != applyShapeN_) {
      mmApplyW_.SetOrgShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      mmApplyW_.SetSingleShape(WY_CUBE_CHUNK, static_cast<int>(nDim), WY_CUBE_CHUNK);
      applyShapeN_ = nDim;
    }
    mmApplyW_.SetTensorA(tUbHalf, false);
    mmApplyW_.SetTensorB(bUb, false);
    mmApplyW_.IterateAll(rUb);
    PipeBarrier<PIPE_ALL>();
  }


  // T = T + P @ T (64x64), operands fed straight from UB: aUb holds half(P),
  // bScratch receives half(T) cast fresh each call. floatScratch holds C.
  __aicore__ inline void GemmApplyAdd(LocalTensor<float> tUb, LocalTensor<half> aUb, LocalTensor<half> bScratch,
                                      LocalTensor<float> floatScratch)
  {
    Cast(bScratch, tUb, RoundMode::CAST_NONE, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
    WaitVToMte3();
    mmApplyU_.SetTensorA(aUb, false);
    mmApplyU_.SetTensorB(bScratch, false);
    mmApplyU_.IterateAll(floatScratch);
    PipeBarrier<PIPE_ALL>();
    Add(tUb, tUb, floatScratch, WY_CUBE_CHUNK * WY_CUBE_CHUNK);
    PipeBarrier<PIPE_V>();
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

  __aicore__ inline void CastFloatRowsToHalfContiguous(LocalTensor<half> dst, const LocalTensor<float> src,
                                                       uint32_t rows, uint32_t cols, uint32_t srcLda) const
  {
    if (srcLda == cols) {
      Cast(dst, src, RoundMode::CAST_NONE, rows * cols);
    } else {
      // One strided Cast per call instead of a per-row scalar-issued loop.
      Cast(dst, src, RoundMode::CAST_NONE, static_cast<uint64_t>(cols), static_cast<uint8_t>(rows),
           {1, 1, static_cast<uint8_t>(cols * sizeof(half) / 32),
            static_cast<uint8_t>(srcLda * sizeof(float) / 32)});
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
  WyMatmulBTransUb mmAttn_;
  WyMatmulNoTrans mmSquare_;
  WyMatmulApplyUb mmApplyU_;
  WyMatmulApplyUb mmApplyW_;
  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
  GlobalTensor<float> snapGm_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_CUBE_H
