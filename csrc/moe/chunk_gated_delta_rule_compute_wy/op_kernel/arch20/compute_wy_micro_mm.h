/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRuleComputeWy {

using namespace AscendC;

// Hand-rolled m200 cube path for the two fixed shapes this kernel needs:
// C[64,n] fp32 = A[64,64] half x B[64,n] half, n in {64,128}, operands ND in
// UB. Bypasses the matmul library, whose per-call overhead (~1.8us measured)
// dwarfs the ~65ns of arithmetic at these sizes.
class WyMicroMm {
 public:
  __aicore__ inline void Init(TPipe *pipe) {
    // A-side is 64x128 so MmNz2 can hold a hi+lo operand pair ([0:4096) hi,
    // [4096:8192) lo); the single-matrix calls keep using the first half.
    pipe->InitBuffer(l1ABuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(l1BBuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(l0ABuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(l0BBuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(coBuf_, 64 * 128 * sizeof(float));
  }

  // aUb: [64,64] half, row stride 64. bUb: [64,n] half, row stride n (both
  // contiguous ND). cUb: [64,n] fp32, row stride ldc. cNz: fp32 scratch >= 64*n.
  __aicore__ inline void Mm(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                            LocalTensor<float> cNz, uint32_t n) {
    Mm(cUb, aUb, bUb, cNz, n, n);
  }

  __aicore__ inline void Mm(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                            LocalTensor<float> cNz, uint32_t n, uint32_t ldc) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    const uint16_t n1 = static_cast<uint16_t>(n / 16);
    // A/B were produced on V; the L1 staging below reads them. UB->L1 nd2nz:
    // one DataCopy per 16-column fractal, all 64 rows land contiguously.
    // MTE1_MTE3: the previous call's LoadData must be done reading L1 before
    // this call's staging overwrites it (back-to-back calls race otherwise).
    Evt<HardEvent::V_MTE3>();
    Evt<HardEvent::MTE1_MTE3>();
    for (uint32_t j = 0; j < 4; ++j) {
      DataCopy(l1A[j * 64 * 16], aUb[j * 16], {64, 1, 3, 0});
    }
    for (uint32_t j = 0; j < n1; ++j) {
      DataCopy(l1B[j * 64 * 16], bUb[j * 16], {64, 1, static_cast<uint16_t>(n1 - 1), 0});
    }
    Evt<HardEvent::MTE3_MTE1>();
    // M_MTE1: the previous call's Mmad must be done reading L0A/L0B before the
    // loads below overwrite them.
    Evt<HardEvent::M_MTE1>();
    // L1(NZ) -> L0A (Zz): each m-fractal-row gathers its 4 k-fractals.
    LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
    }
    // L1(NZ) -> L0B (Zn): each k-fractal gathers its n1 n-fractals; the
    // transpose flag is how load2d expresses non-transposed B (see
    // load_to_l0b_load2d.h in the toolkit).
    LoadData2dParams pb(0, static_cast<uint8_t>(n1), 4, 0, 0, true, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0B[i * n1 * 256], l1B[i * 256], pb);
    }
    Evt<HardEvent::MTE1_M>();
    MmadParams mp(64, static_cast<uint16_t>(n), 64, /*unitFlag=*/0, /*cmatrixSource=*/false,
                  /*cmatrixInitVal=*/true);
    Mmad(co, l0A, l0B, mp);
    Evt<HardEvent::M_V>();
    // CO1 (fp32 NZ) -> UB NZ via matrix block mode, then NZ->ND with one Muls
    // per n-fractal column (16 fp32 lanes x 64 rows each).
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, co, {n1, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < n1; ++j) {
      Muls(cUb[j * 16], cNz[j * 64 * 16], 1.0f, static_cast<uint64_t>(16), 64,
           {1, 1, static_cast<uint8_t>(ldc * sizeof(float) / 32), 2});
    }
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

  // 64^3 call with every tensor already in NZ layout (half A/B in UB, fp32 C
  // straight from CO1 into cNzOut — no ND conversion anywhere): staging is one
  // contiguous 8KB copy per operand, readback one matrix-block copy.
  __aicore__ inline void MmNz(LocalTensor<float> cNzOut, LocalTensor<half> aNzHalf, LocalTensor<half> bNzHalf) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    Evt<HardEvent::V_MTE3>();
    Evt<HardEvent::MTE1_MTE3>();
    DataCopy(l1A, aNzHalf, 64 * 64);
    DataCopy(l1B, bNzHalf, 64 * 64);
    Evt<HardEvent::MTE3_MTE1>();
    Evt<HardEvent::M_MTE1>();
    LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
    }
    LoadData2dParams pb(0, 4, 4, 0, 0, true, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0B[i * 4 * 256], l1B[i * 256], pb);
    }
    Evt<HardEvent::MTE1_M>();
    MmadParams mp(64, 64, 64, 0, false, true);
    Mmad(co, l0A, l0B, mp);
    Evt<HardEvent::M_V>();
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNzOut, co, {4, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

  // Compensated 64^3 NZ matmul: C = A@B with A/B fp32 NZ in UB. Each operand
  // splits into hi = half(x), lo = half(x - float(hi)); three Mmads accumulate
  // hi*hi + hi*lo + lo*hi in L0C (lo*lo ~2^-22 is dropped), so C carries
  // near-fp32 operand precision at ~2x the staging cost of MmNz. Used by the
  // gate-tripped doubling path, where plain fp16 casts of the growing P/T
  // compound into visible error. In-place C (cNzOut aliasing aF32/bF32) is
  // safe: both operands are fully staged to L1 before the readback writes C.
  // workHalf: 64x64 half scratch, workF32: 64x64 fp32 scratch (both clobbered).
  __aicore__ inline void MmNz2(LocalTensor<float> cNzOut, LocalTensor<float> aF32, LocalTensor<float> bF32,
                               LocalTensor<half> workHalf, LocalTensor<float> workF32) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    Evt<HardEvent::MTE1_MTE3>();
    StageSplit(l1A, aF32, workHalf, workF32);
    StageSplit(l1B, bF32, workHalf, workF32);
    Evt<HardEvent::MTE3_MTE1>();
    Evt<HardEvent::M_MTE1>();
    LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
    LoadData2dParams pb(0, 4, 4, 0, 0, true, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
      LoadData(l0A[4096 + i * 4 * 256], l1A[4096 + i * 256], pa);
      LoadData(l0B[i * 4 * 256], l1B[i * 256], pb);
      LoadData(l0B[4096 + i * 4 * 256], l1B[4096 + i * 256], pb);
    }
    Evt<HardEvent::MTE1_M>();
    MmadParams mpInit(64, 64, 64, 0, false, true);
    MmadParams mpAcc(64, 64, 64, 0, false, false);
    Mmad(co, l0A, l0B, mpInit);
    Mmad(co, l0A, l0B[4096], mpAcc);
    Mmad(co, l0A[4096], l0B, mpAcc);
    Evt<HardEvent::M_V>();
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNzOut, co, {4, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

  // Solve variant with the A operand already NZ (half): one contiguous staging
  // copy for A; B stays ND.
  __aicore__ inline void MmANz(LocalTensor<float> cUb, LocalTensor<half> aNzHalf, LocalTensor<half> bUb,
                               LocalTensor<float> cNz, uint32_t n, uint32_t ldc) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    const uint16_t n1 = static_cast<uint16_t>(n / 16);
    Evt<HardEvent::V_MTE3>();
    Evt<HardEvent::MTE1_MTE3>();
    DataCopy(l1A, aNzHalf, 64 * 64);
    for (uint32_t j = 0; j < n1; ++j) {
      DataCopy(l1B[j * 64 * 16], bUb[j * 16], {64, 1, static_cast<uint16_t>(n1 - 1), 0});
    }
    Evt<HardEvent::MTE3_MTE1>();
    Evt<HardEvent::M_MTE1>();
    LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
    }
    LoadData2dParams pb(0, static_cast<uint8_t>(n1), 4, 0, 0, true, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0B[i * n1 * 256], l1B[i * 256], pb);
    }
    Evt<HardEvent::MTE1_M>();
    MmadParams mp(64, static_cast<uint16_t>(n), 64, 0, false, true);
    Mmad(co, l0A, l0B, mp);
    Evt<HardEvent::M_V>();
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, co, {n1, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < n1; ++j) {
      Muls(cUb[j * 16], cNz[j * 64 * 16], 1.0f, static_cast<uint64_t>(16), 64,
           {1, 1, static_cast<uint8_t>(ldc * sizeof(float) / 32), 2});
    }
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

  // C[64,64] fp32 = A[64,kDim] @ B[64,kDim]^T, accumulated over 64-wide K
  // slices IN L0C (one readback total). A/B are ND in UB with row strides
  // aLda/bLda. For A@B^T the per-fractal LoadData transpose cancels the matrix
  // transpose, so B fractals load straight (no flag, stride 1).
  __aicore__ inline void GramAcc(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                                 LocalTensor<float> cNz, uint32_t kDim, uint32_t aLda, uint32_t bLda) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    Evt<HardEvent::V_MTE3>();
    for (uint32_t k0 = 0; k0 < kDim; k0 += 64) {
      Evt<HardEvent::MTE1_MTE3>();
      for (uint32_t j = 0; j < 4; ++j) {
        DataCopy(l1A[j * 64 * 16], aUb[k0 + j * 16], {64, 1, static_cast<uint16_t>(aLda / 16 - 1), 0});
        DataCopy(l1B[j * 64 * 16], bUb[k0 + j * 16], {64, 1, static_cast<uint16_t>(bLda / 16 - 1), 0});
      }
      Evt<HardEvent::MTE3_MTE1>();
      Evt<HardEvent::M_MTE1>();
      LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
      for (uint32_t i = 0; i < 4; ++i) {
        LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
      }
      LoadData2dParams pb(0, 4, 1, 0, 0, false, 0);
      for (uint32_t i = 0; i < 4; ++i) {
        LoadData(l0B[i * 4 * 256], l1B[i * 1024], pb);
      }
      Evt<HardEvent::MTE1_M>();
      MmadParams mp(64, 64, 64, /*unitFlag=*/0, /*cmatrixSource=*/false,
                    /*cmatrixInitVal=*/k0 == 0);
      Mmad(co, l0A, l0B, mp);
    }
    Evt<HardEvent::M_V>();
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, co, {4, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < 4; ++j) {
      Muls(cUb[j * 16], cNz[j * 64 * 16], 1.0f, static_cast<uint64_t>(16), 64, {1, 1, 8, 2});
    }
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

 private:
  // Split-cast one fp32 NZ matrix into L1 as a hi+lo half pair. workHalf is
  // reused for both halves, so each L1 copy must drain before the next cast
  // overwrites it (the MTE3_V waits); the trailing one also frees workHalf
  // for the caller's next StageSplit.
  __aicore__ inline void StageSplit(LocalTensor<half> l1Dst, LocalTensor<float> xF32, LocalTensor<half> workHalf,
                                    LocalTensor<float> workF32) {
    Cast(workHalf, xF32, RoundMode::CAST_NONE, 64 * 64);
    PipeBarrier<PIPE_V>();
    Cast(workF32, workHalf, RoundMode::CAST_NONE, 64 * 64);
    PipeBarrier<PIPE_V>();
    Sub(workF32, xF32, workF32, 64 * 64);
    Evt<HardEvent::V_MTE3>();
    DataCopy(l1Dst, workHalf, 64 * 64);
    Evt<HardEvent::MTE3_V>();
    Cast(workHalf, workF32, RoundMode::CAST_NONE, 64 * 64);
    Evt<HardEvent::V_MTE3>();
    DataCopy(l1Dst[4096], workHalf, 64 * 64);
    Evt<HardEvent::MTE3_V>();
  }

  template <HardEvent E>
  __aicore__ inline void Evt() {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(evt);
    WaitFlag<E>(evt);
  }

  TBuf<TPosition::A1> l1ABuf_;
  TBuf<TPosition::B1> l1BBuf_;
  TBuf<TPosition::A2> l0ABuf_;
  TBuf<TPosition::B2> l0BBuf_;
  TBuf<TPosition::CO1> coBuf_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H
