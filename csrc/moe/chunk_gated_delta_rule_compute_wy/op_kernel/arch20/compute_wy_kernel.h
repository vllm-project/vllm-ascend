/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_KERNEL_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_KERNEL_H

#include "kernel_operator.h"
#include "../chunk_gated_delta_rule_compute_wy_tiling_data.h"

#include "compute_wy_cube.h"
#include "compute_wy_micro_mm.h"
#include "compute_wy_identity_nz.h"
#include "compute_wy_lambda_table.h"

namespace ChunkGatedDeltaRuleComputeWy {

using namespace AscendC;

constexpr uint32_t FIXED_CHUNK_SIZE = 64;
constexpr uint32_t ATTEN_ELEMS = FIXED_CHUNK_SIZE * FIXED_CHUNK_SIZE;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t HALF_PER_BLOCK = BLOCK_BYTES / sizeof(half);
constexpr uint32_t FLOAT_PER_BLOCK = BLOCK_BYTES / sizeof(float);
constexpr uint32_t FLOAT_VEC_LEN = 64;
// Must match host tiling. 310P UB is 192KB; K/V=128 fits because the solve runs
// in two passes (W then U) with a single RHS resident, K staged through tmpBuf_,
// and V loaded only in pass 2 (~178KB peak instead of ~242KB).
constexpr uint32_t MAX_SAFE_HEAD_DIM = 128;
constexpr uint32_t DOUBLING_ROUNDS = 6;  // log2(64)
// Gate for the scalar fp32 fallback. The old 0.75 bound was chosen "for ample
// headroom" but sends realistic chunks (row sums of |βK·Kᵀ⊙Λ| routinely exceed
// 1) down a 2016-serial-Axpy path that costs ~60us/RHS — measured as ~85% of
// the whole op. fp16 headroom through the doubling products tolerates far more;
// 4.0 keeps the truly pathological chunks (and NaN) on the exact fp32 path and
// is validated by the 9-shape adversarial cosine battery.
constexpr float FP32_FS_ROW_SUM_THRESHOLD = 2.5f;
// Above this row-sum the compensated doubling's fp16 hi/lo split can overflow
// to NaN (T outgrows half range). Swept on gate-trip data: row sums 13.6 are
// exact (cos 1.0000000), 19.2 overflow — 14.0 keeps every validated case on
// the fast path and routes the rest to the exact scalar solve.
constexpr float COMP_DOUBLING_MAX_ROW_SUM = 14.0f;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) { return (value + align - 1) / align * align; }
__aicore__ inline uint16_t BytesToBlocks(uint32_t bytes) { return static_cast<uint16_t>(AlignUp(bytes, BLOCK_BYTES) / BLOCK_BYTES); }

template <HardEvent event>
__aicore__ inline void SyncEvent(HardEvent evt) {
  event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
  SetFlag<event>(eventId);
  WaitFlag<event>(eventId);
}

class KernelComputeWy {
 public:
  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR qKernel,
                              GM_ADDR kKernel, GM_ADDR wKernel, GM_ADDR uKernel, GM_ADDR gKernel, GM_ADDR workspace,
                              const ChunkGatedDeltaRuleComputeWyTilingData* tiling, TPipe* pipe) {
    valid_ = false;
    pipe_ = pipe;
    batch_ = static_cast<uint32_t>(tiling->batch);
    seqlen_ = static_cast<uint32_t>(tiling->seqlen);
    kNumHead_ = static_cast<uint32_t>(tiling->kNumHead);
    vNumHead_ = static_cast<uint32_t>(tiling->vNumHead);
    kHeadDim_ = static_cast<uint32_t>(tiling->kHeadDim);
    vHeadDim_ = static_cast<uint32_t>(tiling->vHeadDim);
    chunkSize_ = static_cast<uint32_t>(tiling->chunkSize);
    numChunks_ = static_cast<uint32_t>(tiling->numChunks);
    localWorkspaceSize_ = tiling->localWorkspaceSize;
    perCoreWorkspaceBytes_ = tiling->perCoreWorkspaceBytes;
    usedCoreNum_ = tiling->usedCoreNum == 0 ? 1 : tiling->usedCoreNum;
    if (kNumHead_ == 0 || vNumHead_ == 0) {
      return;
    }
    headGroups_ = vNumHead_ / kNumHead_;
    alignK_ = AlignUp(kHeadDim_, HALF_PER_BLOCK);
    alignV_ = AlignUp(vHeadDim_, HALF_PER_BLOCK);
    chunkKElems_ = FIXED_CHUNK_SIZE * alignK_;
    chunkVElems_ = FIXED_CHUNK_SIZE * alignV_;
    if (chunkSize_ != FIXED_CHUNK_SIZE || headGroups_ == 0) {
      return;
    }
    if (headGroups_ * kNumHead_ != vNumHead_) {
      return;
    }
    if (kHeadDim_ > MAX_SAFE_HEAD_DIM || vHeadDim_ > MAX_SAFE_HEAD_DIM) {
      return;
    }

    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q));
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k));
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v));
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
    betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta));
    qKernelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qKernel));
    kKernelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kKernel));
    wKernelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(wKernel));
    uKernelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(uKernel));
    gKernelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gKernel));
    iNzGm_.SetGlobalBuffer(const_cast<__gm__ float*>(WY_IDENTITY_NZ));

    const uint32_t localWsBytes = localWorkspaceSize_ == 0 ? (32 * 1024) : localWorkspaceSize_;
    pipe_->InitBuffer(mmLocalWsBuf_, localWsBytes);
    microMm_.Init(pipe_);
    cubeGemm_.Init(&tiling->mmAttn, &tiling->mmSquare, &tiling->mmApplyU, &tiling->mmApplyW, pipe_,
                   mmLocalWsBuf_.Get<uint8_t>(), localWsBytes, workspace, tiling->workspaceOffset,
                   perCoreWorkspaceBytes_, usedCoreNum_, kHeadDim_, vHeadDim_);

    uint32_t maxAlign = (alignK_ > alignV_ ? alignK_ : alignV_);
    if (maxAlign < FIXED_CHUNK_SIZE) {
      maxAlign = FIXED_CHUNK_SIZE;
    }
    maxAlign_ = maxAlign;
    // Explicit-T UB layout (fits K=V=128 in 192KB):
    //  - qHalfBuf_: βK gram operand + P/T upload staging (two 8KB half regions:
    //    [0:4096) P-halves, [4096:8192) T's B-operand halves) + Q/K passthrough.
    //    Sized >= 8192 halves even at head dim 64 for the second region.
    //  - halfBuf_:  K gram operand, R-half staging, and (reinterpreted as float)
    //    the 64x64 C scratch for the T-build applies.
    //  - tmpBuf_:   K fp32 slices, gram acc, Λ mat, then T fp32 during the solve.
    //  - rhsBuf_:   the single resident fp32 RHS (γβK for W, then βV for U).
    const uint32_t qHalfElems = FIXED_CHUNK_SIZE * (maxAlign < 2 * FIXED_CHUNK_SIZE ? 2 * FIXED_CHUNK_SIZE : maxAlign);
    pipe_->InitBuffer(qHalfBuf_, qHalfElems * sizeof(half));
    // halfBuf_ doubles (reinterpreted as fp32) as the 64x64 C scratch of the
    // T-build applies — that role needs ATTEN_ELEMS*4 bytes regardless of head
    // dim. At head dim 64 the natural size is only 8KB and the matmul C-write
    // clobbered rhsBuf_ (the cos=0.16/0.21 failures on K=V=64 shapes whenever
    // those tasks took the doubling path).
    {
      uint32_t halfBytes = FIXED_CHUNK_SIZE * maxAlign * static_cast<uint32_t>(sizeof(half));
      const uint32_t cScratchBytes = ATTEN_ELEMS * static_cast<uint32_t>(sizeof(float));
      if (halfBytes < cScratchBytes) halfBytes = cScratchBytes;
      pipe_->InitBuffer(halfBuf_, halfBytes);
    }
    pipe_->InitBuffer(rhsBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(float));
    // Dedicated output staging: W/U casts land here so their MTE3 stores drain
    // underneath the next phase instead of serializing on halfBuf_.
    pipe_->InitBuffer(storeBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(half));
    pipe_->InitBuffer(attnBuf_, ATTEN_ELEMS * sizeof(float));
    // +8 floats: slot FIXED_CHUNK_SIZE holds the Lambda mask sentinel (see BuildLambdaTable).
    pipe_->InitBuffer(gBuf_, (FIXED_CHUNK_SIZE + 8) * sizeof(float));
    pipe_->InitBuffer(expGBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    // 64x64 fp32 scratch: per-slice K staging, gram acc, Lambda mat, then T.
    // (Keeping it 64-wide is what lets K=V=128 fit in UB next to the matmul
    // library's own allocations.)
    pipe_->InitBuffer(tmpBuf_, ATTEN_ELEMS * sizeof(float));
    pipe_->InitBuffer(rowBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    pipe_->InitBuffer(negABuf_, FIXED_CHUNK_SIZE * sizeof(float));
    pipe_->InitBuffer(lamOffBuf_, ATTEN_ELEMS * sizeof(uint32_t));
    pipe_->InitBuffer(gOffBuf_, FIXED_CHUNK_SIZE * sizeof(uint32_t));
    pipe_->InitBuffer(betaOffBuf_, FIXED_CHUNK_SIZE * sizeof(uint32_t));
    pipe_->InitBuffer(lane0OffBuf_, FIXED_CHUNK_SIZE * sizeof(uint32_t));
    pipe_->InitBuffer(betaHalfBuf_, FIXED_CHUNK_SIZE * sizeof(half));
    BuildLambdaTable();
    BuildColumnGatherTables();
    valid_ = true;
  }

  // Byte-offset table for the Lambda row-broadcast Gather, built once per kernel launch.
  // off[i][j] = &a[i] for j < i (strict lower), else &a[FIXED_CHUNK_SIZE] which holds a
  // large negative sentinel, so Exp() drives those lanes to 0. This folds the triangular
  // mask into the gather and costs no extra vector op per task.
  __aicore__ inline void BuildLambdaTable() {
    // Table is compile-time GM constant data; one 16KB MTE2 burst replaces the
    // 4096-iteration scalar SetValue loop every core paid on every launch.
    GlobalTensor<uint32_t> tabGm;
    tabGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ uint32_t*>(WY_LAMBDA_OFF_TABLE)));
    DataCopy(lamOffBuf_.Get<uint32_t>(), tabGm, ATTEN_ELEMS);
    SyncEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    gBuf_.Get<float>().SetValue(FIXED_CHUNK_SIZE, -1.0e20f);
  }

  // g and beta arrive as [64, vNumHead] blocks; we need one strided column. Byte-offset
  // tables let a single Gather extract it, so the column never round-trips through the
  // scalar unit. numHeads is fixed for the launch, so these are built once.
  __aicore__ inline void BuildColumnGatherTables() {
    LocalTensor<uint32_t> gOff = gOffBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> betaOff = betaOffBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> lane0 = lane0OffBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      gOff.SetValue(i, i * vNumHead_ * static_cast<uint32_t>(sizeof(float)));
      betaOff.SetValue(i, i * vNumHead_ * static_cast<uint32_t>(sizeof(half)));
      lane0.SetValue(i, i * FIXED_CHUNK_SIZE * static_cast<uint32_t>(sizeof(float)));
    }
  }

  __aicore__ inline void Process() {
    if (!valid_) {
      return;
    }
    if (chunkSize_ != FIXED_CHUNK_SIZE || headGroups_ == 0 || kNumHead_ == 0 ||
        headGroups_ * kNumHead_ != vNumHead_) {
      return;
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = usedCoreNum_;
    if (blockIdx >= blockNum) {
      return;
    }
    const uint64_t totalTasks = static_cast<uint64_t>(batch_) * vNumHead_ * numChunks_;
    for (uint64_t task = blockIdx; task < totalTasks; task += blockNum) {
      const uint32_t chunkIdx = static_cast<uint32_t>(task % numChunks_);
      const uint32_t tmp = static_cast<uint32_t>(task / numChunks_);
      const uint32_t vHeadIdx = tmp % vNumHead_;
      const uint32_t batchIdx = tmp / vNumHead_;
      const uint32_t kHeadIdx = vHeadIdx / headGroups_;
      ProcessOneTask(batchIdx, kHeadIdx, vHeadIdx, chunkIdx);
    }
  }

 private:
  __aicore__ inline uint64_t BthdOffset(uint32_t b, uint32_t t, uint32_t h, uint32_t d, uint32_t numHeads,
                                        uint32_t headDim) const {
    return ((static_cast<uint64_t>(b) * seqlen_ + t) * numHeads + h) * headDim + d;
  }
  __aicore__ inline uint64_t BhtdOffset(uint32_t b, uint32_t h, uint32_t t, uint32_t d, uint32_t numHeads,
                                        uint32_t headDim) const {
    return ((static_cast<uint64_t>(b) * numHeads + h) * seqlen_ + t) * headDim + d;
  }
  __aicore__ inline uint64_t BhtOffset(uint32_t b, uint32_t h, uint32_t t, uint32_t numHeads) const {
    return (static_cast<uint64_t>(b) * numHeads + h) * seqlen_ + t;
  }
  __aicore__ inline void LoadBthdChunk(GlobalTensor<half> srcGm, LocalTensor<half> dstLocal, uint32_t b,
                                       uint32_t tokenStart, uint32_t headIdx, uint32_t numHeads, uint32_t headDim,
                                       uint32_t alignDim) const {
    const uint32_t rowBytes = headDim * sizeof(half);
    const uint32_t srcRowStrideBytes = numHeads * headDim * sizeof(half);
    const uint32_t dstRowStrideBytes = alignDim * sizeof(half);
    const uint64_t base = BthdOffset(b, tokenStart, headIdx, 0, numHeads, headDim);
    const uint16_t dstStrideBlocks =
        (alignDim > headDim) ? BytesToBlocks(dstRowStrideBytes - rowBytes) : static_cast<uint16_t>(0);
    DataCopyParams copyParams{FIXED_CHUNK_SIZE, BytesToBlocks(rowBytes), BytesToBlocks(srcRowStrideBytes - rowBytes),
                              dstStrideBlocks};
    DataCopy(dstLocal, srcGm[base], copyParams);
    if (alignDim > headDim) {
      SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      for (uint32_t row = 0; row < FIXED_CHUNK_SIZE; ++row) {
        Duplicate(dstLocal[row * alignDim + headDim], static_cast<half>(0), alignDim - headDim);
      }
    }
  }
  // Loads columns [d0, d0+dCur) of a [64, headDim] chunk into the same column
  // window of dstLocal (row stride alignDim). d0/dCur must be 16-multiples.
  __aicore__ inline void LoadBthdChunkSlice(GlobalTensor<half> srcGm, LocalTensor<half> dstLocal, uint32_t b,
                                            uint32_t tokenStart, uint32_t headIdx, uint32_t numHeads,
                                            uint32_t headDim, uint32_t alignDim, uint32_t d0, uint32_t dCur) const {
    const uint32_t rowBytes = dCur * sizeof(half);
    const uint32_t srcRowStrideBytes = numHeads * headDim * sizeof(half);
    const uint32_t dstRowStrideBytes = alignDim * sizeof(half);
    const uint64_t base = BthdOffset(b, tokenStart, headIdx, d0, numHeads, headDim);
    DataCopyParams copyParams{FIXED_CHUNK_SIZE, BytesToBlocks(rowBytes),
                              BytesToBlocks(srcRowStrideBytes - rowBytes),
                              BytesToBlocks(dstRowStrideBytes - rowBytes)};
    DataCopy(dstLocal[d0], srcGm[base], copyParams);
  }

  __aicore__ inline void StoreBhtdChunk(GlobalTensor<half> dstGm, LocalTensor<half> srcLocal, uint32_t b,
                                        uint32_t headIdx, uint32_t tokenStart, uint32_t numHeads, uint32_t headDim,
                                        uint32_t alignDim) const {
    const uint32_t rowBytes = headDim * sizeof(half);
    const uint32_t srcRowStrideBytes = alignDim * sizeof(half);
    const uint64_t base = BhtdOffset(b, headIdx, tokenStart, 0, numHeads, headDim);
    const uint16_t srcStrideBlocks =
        (alignDim > headDim) ? BytesToBlocks(srcRowStrideBytes - rowBytes) : static_cast<uint16_t>(0);
    DataCopyParams copyParams{FIXED_CHUNK_SIZE, BytesToBlocks(rowBytes), srcStrideBlocks, 0};
    DataCopy(dstGm[base], srcLocal, copyParams);
  }
  __aicore__ inline void LoadHeadScalarChunk(GlobalTensor<float> srcGm, LocalTensor<float> dstLocal,
                                             LocalTensor<float> scratch, uint32_t scratchElems, uint32_t b,
                                             uint32_t tokenStart, uint32_t headIdx, uint32_t numHeads) {
    const uint32_t span = FIXED_CHUNK_SIZE * numHeads;
    const uint64_t chunkBase = (static_cast<uint64_t>(b) * seqlen_ + tokenStart) * numHeads;
    // Gather table is built for vNumHead_; fall back to the scalar path otherwise.
    if (span <= scratchElems && numHeads == vNumHead_) {
      DataCopyParams copyParams{1, BytesToBlocks(span * sizeof(float)), 0, 0};
      DataCopy(scratch, srcGm[chunkBase], copyParams);
      SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      Gather(dstLocal, scratch, gOffBuf_.Get<uint32_t>(),
             headIdx * static_cast<uint32_t>(sizeof(float)), FIXED_CHUNK_SIZE);
      // Vector wrote dstLocal; the cumsum below reads it on the scalar unit.
      SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    } else {
      const uint64_t base = chunkBase + headIdx;
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        dstLocal.SetValue(i, srcGm.GetValue(base + static_cast<uint64_t>(i) * numHeads));
      }
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void LoadBetaChunk(LocalTensor<float> dstLocal, LocalTensor<half> scratch, uint32_t scratchElems,
                                       uint32_t b, uint32_t tokenStart, uint32_t vHeadIdx) {
    const uint32_t span = FIXED_CHUNK_SIZE * vNumHead_;
    const uint64_t chunkBase = (static_cast<uint64_t>(b) * seqlen_ + tokenStart) * vNumHead_;
    if (span <= scratchElems) {
      DataCopyParams copyParams{1, BytesToBlocks(span * sizeof(half)), 0, 0};
      DataCopy(scratch, betaGm_[chunkBase], copyParams);
      SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      LocalTensor<half> betaHalf = betaHalfBuf_.Get<half>();
      Gather(betaHalf, scratch, betaOffBuf_.Get<uint32_t>(),
             vHeadIdx * static_cast<uint32_t>(sizeof(half)), FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
      Cast(dstLocal, betaHalf, RoundMode::CAST_NONE, FIXED_CHUNK_SIZE);
    } else {
      const uint64_t base = chunkBase + vHeadIdx;
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        const half val = betaGm_.GetValue(base + static_cast<uint64_t>(i) * vNumHead_);
        dstLocal.SetValue(i, static_cast<float>(val));
      }
      // Scalar SetValue → vector Brcb/Mul in BroadcastMulRowsFloat.
      SyncEvent<HardEvent::S_V>(HardEvent::S_V);
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void LoadKChunk(uint32_t b, uint32_t kHeadIdx, uint32_t tokenStart, LocalTensor<half> kHalf) const {
    LoadBthdChunk(kGm_, kHalf, b, tokenStart, kHeadIdx, kNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
  }
  __aicore__ inline void StoreQKKernel(uint32_t b, uint32_t kHeadIdx, uint32_t tokenStart,
                                       LocalTensor<half> qHalf) const {
    LoadBthdChunk(qGm_, qHalf, b, tokenStart, kHeadIdx, kNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    StoreBhtdChunk(qKernelGm_, qHalf, b, kHeadIdx, tokenStart, kNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    LoadBthdChunk(kGm_, qHalf, b, tokenStart, kHeadIdx, kNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    StoreBhtdChunk(kKernelGm_, qHalf, b, kHeadIdx, tokenStart, kNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
  }
  __aicore__ inline void BuildCumulativeG(uint32_t b, uint32_t vHeadIdx, uint32_t tokenStart, LocalTensor<float> gLocal,
                                          LocalTensor<float> expGLocal, LocalTensor<float> gRaw,
                                          LocalTensor<float> loadScratch, uint32_t loadScratchElems) {
    LoadHeadScalarChunk(gGm_, gRaw, loadScratch, loadScratchElems, b, tokenStart, vHeadIdx, vNumHead_);
    // Inclusive prefix sum as a 6-round vector scan (was a 64-step scalar RMW
    // loop — one of the serial scalar hotspots behind the 0.73 issue ratio).
    Adds(gLocal, gRaw, 0.0f, FIXED_CHUNK_SIZE);
    PipeBarrier<PIPE_V>();
    for (uint32_t sh = 1; sh < FIXED_CHUNK_SIZE; sh <<= 1) {
      Add(gLocal[sh], gLocal[sh], gLocal, FIXED_CHUNK_SIZE - sh);
      PipeBarrier<PIPE_V>();
    }
    Exp(expGLocal, gLocal, FIXED_CHUNK_SIZE);
    PipeBarrier<PIPE_V>();
    const uint64_t dstOffset = BhtOffset(b, vHeadIdx, tokenStart, vNumHead_);
    DataCopyParams dstCopyParams{1, BytesToBlocks(FIXED_CHUNK_SIZE * sizeof(float)), 0, 0};
    DataCopy(gKernelGm_[dstOffset], gLocal, dstCopyParams);
  }

  // In-place: gramLocal = A = −strictlower(G ⊙ Λ) with Λ_ij = exp(a_i − a_j) for j < i.
  // Never forms γ_i/γ_j (overflow-safe: a_i − a_j ≤ 0 on the strict lower triangle).
  // Whole-block: Gather row-broadcasts a_i (masking folded into the offsets), one Sub with
  // src1RepStride=0 subtracts a_j without materialising a second 64x64, then Exp and one Mul.
  // gramLocal must already be negated by the caller. No value passes through the scalar unit.
  __aicore__ inline void ApplyLambdaNegStrictLower(LocalTensor<float> gramLocal, const LocalTensor<float> aLocal,
                                                  LocalTensor<float> mat) {
    Gather(mat, aLocal, lamOffBuf_.Get<uint32_t>(), 0U, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    // mat[i][j] -= a_j : src1 repeat stride 0 replays the 64-float a vector on every row.
    const BinaryRepeatParams rp{1, 1, 1, 8, 8, 0};
    Sub(mat, mat, aLocal, FIXED_CHUNK_SIZE, FIXED_CHUNK_SIZE, rp);
    PipeBarrier<PIPE_V>();
    Exp(mat, mat, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    Mul(gramLocal, gramLocal, mat, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
  }

  // Row-scale: dst[i, :] *= scalePerRow[i]. Pure vector path — Brcb expands each
  // scale into one 32B block, then Mul replays that block across the row
  // (src1BlkStride=0) and advances one block per row (src1RepStride=1).
  // brcbScratch must hold rowsCount * FLOAT_PER_BLOCK floats.
  __aicore__ inline void BroadcastMulRowsFloat(LocalTensor<float> dst, const LocalTensor<float> rows,
                                               const LocalTensor<float> scalePerRow, LocalTensor<float> brcbScratch,
                                               uint32_t rowsCount, uint32_t cols, uint32_t dstLda,
                                               uint32_t srcLda) const {
    const uint32_t blockCount = (rowsCount + FLOAT_PER_BLOCK - 1) / FLOAT_PER_BLOCK;
    Brcb(brcbScratch, scalePerRow, blockCount, {1, static_cast<uint16_t>(FLOAT_PER_BLOCK)});
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams rp{1, 1, 0, static_cast<uint8_t>(dstLda / FLOAT_PER_BLOCK),
                                static_cast<uint8_t>(srcLda / FLOAT_PER_BLOCK), 1};
    uint32_t col = 0;
    for (; col + FLOAT_VEC_LEN <= cols; col += FLOAT_VEC_LEN) {
      Mul(dst[col], rows[col], brcbScratch, FLOAT_VEC_LEN, rowsCount, rp);
    }
    if (col < cols) {
      Mul(dst[col], rows[col], brcbScratch, cols - col, rowsCount, rp);
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void BroadcastMulRowsHalf(LocalTensor<half> dst, const LocalTensor<half> rows,
                                              const LocalTensor<half> scalePerRow, LocalTensor<half> brcbScratch,
                                              uint32_t rowsCount, uint32_t cols, uint32_t dstLda,
                                              uint32_t srcLda) const {
    constexpr uint32_t HALF_PER_BLOCK = 16;
    constexpr uint32_t HALF_VEC_LEN = 128;
    const uint32_t blockCount = (rowsCount + FLOAT_PER_BLOCK - 1) / FLOAT_PER_BLOCK;
    Brcb(brcbScratch, scalePerRow, blockCount, {1, static_cast<uint16_t>(FLOAT_PER_BLOCK)});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams rp{1, 1, 0, static_cast<uint8_t>(dstLda / HALF_PER_BLOCK),
                                static_cast<uint8_t>(srcLda / HALF_PER_BLOCK), 1};
    uint32_t col = 0;
    for (; col + HALF_VEC_LEN <= cols; col += HALF_VEC_LEN) {
      Mul(dst[col], rows[col], brcbScratch, HALF_VEC_LEN, rowsCount, rp);
    }
    if (col < cols) {
      Mul(dst[col], rows[col], brcbScratch, cols - col, rowsCount, rp);
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void CastFloatRowsToHalf(LocalTensor<half> dst, const LocalTensor<float> src, uint32_t rows,
                                             uint32_t cols, uint32_t srcLda) {
    if (srcLda == cols) {
      Cast(dst, src, RoundMode::CAST_NONE, rows * cols);
      PipeBarrier<PIPE_V>();
      return;
    }
    for (uint32_t row = 0; row < rows; ++row) {
      Cast(dst[row * cols], src[row * srcLda], RoundMode::CAST_NONE, cols);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline bool NeedsFp32ForwardSubstitution(const LocalTensor<float> a,
                                                      LocalTensor<float> absScratch,
                                                      LocalTensor<float> rowSums,
                                                      LocalTensor<float> reduceScratch) {
    // A is strict-lower, so full-row sums equal the |lower-triangle| sums.
    // Fold columns in 6 strided vector Adds (was 64 serialized ReduceSums),
    // then two whole-tensor reductions; exactly TWO scalars cross to S.
    Abs(absScratch, a, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    const uint32_t rowBlk = FIXED_CHUNK_SIZE * sizeof(float) / BLOCK_BYTES;
    for (uint32_t w = FIXED_CHUNK_SIZE / 2; w >= FLOAT_PER_BLOCK; w >>= 1) {
      const BinaryRepeatParams rp{1, 1, 1, static_cast<uint8_t>(rowBlk), static_cast<uint8_t>(rowBlk),
                                  static_cast<uint8_t>(rowBlk)};
      Add(absScratch, absScratch, absScratch[w], static_cast<uint64_t>(w), FIXED_CHUNK_SIZE, rp);
      PipeBarrier<PIPE_V>();
    }
    // Rows now hold their sum spread over the first 8 lanes; ReduceSum of that
    // 8-lane block per row == row sum. Gather the per-row 8-lane partials into
    // rowSums via one strided Add tree is done; finish with global max + sum.
    // Max of (partial-lane values) >= threshold/8 is NOT exact — instead reduce
    // each row's 8 lanes with one more strided fold to lane 0:
    for (uint32_t w = FLOAT_PER_BLOCK / 2; w >= 1; w >>= 1) {
      const BinaryRepeatParams rp{1, 1, 1, static_cast<uint8_t>(rowBlk), static_cast<uint8_t>(rowBlk),
                                  static_cast<uint8_t>(rowBlk)};
      Add(absScratch, absScratch, absScratch[w], static_cast<uint64_t>(w), FIXED_CHUNK_SIZE, rp);
      PipeBarrier<PIPE_V>();
    }
    // Lane 0 of each row now holds the full |row| sum; pack the 64 strided
    // lane-0 values with one Gather (offset table built once per launch).
    Gather(rowSums, absScratch, lane0OffBuf_.Get<uint32_t>(), 0U, FIXED_CHUNK_SIZE);
    PipeBarrier<PIPE_V>();
    ReduceMax(reduceScratch, rowSums, reduceScratch[8], FIXED_CHUNK_SIZE, /*calIndex=*/false);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceScratch[1], rowSums, reduceScratch[32], FIXED_CHUNK_SIZE);
    SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    const float maxSum = reduceScratch.GetValue(0);
    const float totSum = reduceScratch.GetValue(1);
    // NaN propagates into gateMaxRowSum_ and fails every < compare, so NaN
    // input data routes to the scalar path like before.
    gateMaxRowSum_ = (totSum != totSum) ? totSum : maxSum;
    return (totSum != totSum) || (maxSum >= FP32_FS_ROW_SUM_THRESHOLD);
  }

  // Exact in-place solve R = (I-A)^-1 R for ANY ||A|| (fp32 coefficients).
  // Blocked forward substitution: rows in 4 blocks of 16. Off-diagonal updates
  // R_b += A[b,<b] @ R_<b run on the cube against the already-solved prefix
  // (fp16 operands, fp32 accumulate — same precision class as the RHS itself);
  // only the 16x16 diagonal blocks stay on the serial scalar path, cutting the
  // per-RHS scalar issues from 2016 to 480.
  __aicore__ inline void Fp32ForwardSubstitution(const LocalTensor<float> a, LocalTensor<float> r, uint32_t dim,
                                                 uint32_t lda, LocalTensor<half> aHalf, LocalTensor<half> bHalf,
                                                 LocalTensor<float> cScratch) {
    constexpr uint32_t BLK = 16;
    SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    for (uint32_t b0 = 0; b0 < FIXED_CHUNK_SIZE; b0 += BLK) {
      if (b0 > 0) {
        // Cast the A block [16, b0] (lda 64) and the solved prefix rows [b0, dim]
        // (lda) to half, then R_b += A_blk @ R_prefix on the cube.
        Cast(aHalf, a[b0 * FIXED_CHUNK_SIZE], RoundMode::CAST_NONE, static_cast<uint64_t>(b0), BLK,
             {1, 1, static_cast<uint8_t>(b0 * sizeof(half) / 32 == 0 ? 1 : b0 * sizeof(half) / 32),
              static_cast<uint8_t>(FIXED_CHUNK_SIZE * sizeof(float) / 32)});
        PipeBarrier<PIPE_V>();
        CastFloatRowsToHalfSized(bHalf, r, b0, dim, lda);
        SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        cubeGemm_.ApplyGeneric(cScratch, aHalf, bHalf, BLK, dim, b0);
        // R_b += C
        const BinaryRepeatParams arp{1, 1, 1, static_cast<uint8_t>(lda * sizeof(float) / 32),
                                     static_cast<uint8_t>(lda * sizeof(float) / 32),
                                     static_cast<uint8_t>(dim * sizeof(float) / 32)};
        Add(r[b0 * lda], r[b0 * lda], cScratch, static_cast<uint64_t>(dim > 64 ? 64 : dim), BLK, arp);
        if (dim > 64) {
          Add(r[b0 * lda + 64], r[b0 * lda + 64], cScratch[64], static_cast<uint64_t>(dim - 64), BLK, arp);
        }
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_S>(HardEvent::V_S);
      }
      for (uint32_t row = b0 + 1; row < b0 + BLK; ++row) {
        const uint32_t aRowOffset = row * FIXED_CHUNK_SIZE;
        const uint32_t rRowOffset = row * lda;
        for (uint32_t col = b0; col < row; ++col) {
          const float coefficient = a.GetValue(aRowOffset + col);
          // No per-Axpy barrier: the V queue executes in order, and every
          // dependency in this chain is V-to-V.
          Axpy(r[rRowOffset], r[col * lda], coefficient, static_cast<int32_t>(dim));
        }
      }
      SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    }
  }

  // Rows-limited compact cast: dst[rows, cols] (lda cols) <- src[rows, srcLda].
  __aicore__ inline void CastFloatRowsToHalfSized(LocalTensor<half> dst, const LocalTensor<float> src, uint32_t rows,
                                                  uint32_t cols, uint32_t srcLda) {
    if (srcLda == cols) {
      Cast(dst, src, RoundMode::CAST_NONE, rows * cols);
    } else {
      Cast(dst, src, RoundMode::CAST_NONE, static_cast<uint64_t>(cols > 64 ? 64 : cols), static_cast<uint8_t>(rows),
           {1, 1, static_cast<uint8_t>(cols * sizeof(half) / 32),
            static_cast<uint8_t>(srcLda * sizeof(float) / 32)});
      if (cols > 64) {
        Cast(dst[64], src[64], RoundMode::CAST_NONE, static_cast<uint64_t>(cols - 64), static_cast<uint8_t>(rows),
             {1, 1, static_cast<uint8_t>(cols * sizeof(half) / 32),
              static_cast<uint8_t>(srcLda * sizeof(float) / 32)});
      }
    }
    PipeBarrier<PIPE_V>();
  }


  // Stable in-place solve R = (I-A)^-1 R for a single RHS block. Processing rows
  // top to bottom means each source row has already been solved, exactly matching
  // the torch WY forward substitution. A and R stay fp32 until the output cast.
  // Called once per pass (W in pass 1, U in pass 2); A is read-only here.
  __aicore__ inline void Fp32ForwardSubstitution(const LocalTensor<float> a, LocalTensor<float> r, uint32_t dim,
                                                 uint32_t lda) const {
    SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    for (uint32_t row = 1; row < FIXED_CHUNK_SIZE; ++row) {
      const uint32_t aRowOffset = row * FIXED_CHUNK_SIZE;
      const uint32_t rRowOffset = row * lda;
      for (uint32_t col = 0; col < row; ++col) {
        const float coefficient = a.GetValue(aRowOffset + col);
        Axpy(r[rRowOffset], r[col * lda], coefficient, static_cast<int32_t>(dim));
        PipeBarrier<PIPE_V>();
      }
    }
  }

  // Materialises T = (I−A)⁻¹ = Π(I + A^{2^k}) once via doubling ON T:
  //   T ← I; per round: T ← T + P·T, P ← P². All 64×64, all fp32-accumulated.
  // Afterwards each RHS solve is ONE matmul per 64-wide slice (GemmApplyReplace),
  // instead of re-running the whole chain per RHS as the two-pass solve did.
  // Layout (no new buffers): T fp32 lives in tOut (tmpBuf); the C scratch for the
  // T-build applies is halfLocal reinterpreted as float (K half is dead by now);
  // qHalf[0:4096] stages P uploads, qHalf[4096:8192] stages T's B-operand halves.
  __aicore__ inline void BuildTMat(LocalTensor<float> attnLocal, LocalTensor<float> tOut,
                                LocalTensor<float> cScratch, LocalTensor<half> pHalf, LocalTensor<half> tHalf,
                                LocalTensor<float> cNz) {
    (void)cNz;
    // NZ-resident doubling: A is converted to NZ once; P, T, casts and cube
    // C's all stay NZ (elementwise ops are layout-agnostic), so no per-round
    // conversions and single contiguous L1 stagings. tHalf leaves this
    // function in NZ — the solves stage it with MmANz.
    // The GM identity load overlaps the layout convert; cScratch is free here.
    SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopy(cScratch, iNzGm_, ATTEN_ELEMS);
    for (uint32_t j = 0; j < 4; ++j) {
      Muls(tOut[j * 64 * 16], attnLocal[j * 16], 1.0f, static_cast<uint64_t>(16), 64, {1, 1, 2, 8});
    }
    PipeBarrier<PIPE_V>();
    Adds(attnLocal, tOut, 0.0f, ATTEN_ELEMS);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    // T = I + A in NZ, no scalar writes (A's diagonal is strictly zero).
    Add(tOut, tOut, cScratch, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    // Round 0's T-update is T = I + A, already materialized above; square P.
    Cast(pHalf, attnLocal, RoundMode::CAST_NONE, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    microMm_.MmNz(attnLocal, pHalf, pHalf);
    for (uint32_t round = 1; round < DOUBLING_ROUNDS; ++round) {
      Cast(pHalf, attnLocal, RoundMode::CAST_NONE, ATTEN_ELEMS);
      Cast(tHalf, tOut, RoundMode::CAST_NONE, ATTEN_ELEMS);
      PipeBarrier<PIPE_V>();
      microMm_.MmNz(cScratch, pHalf, tHalf);
      Add(tOut, tOut, cScratch, ATTEN_ELEMS);
      PipeBarrier<PIPE_V>();
      if (round + 1 < DOUBLING_ROUNDS) {
        microMm_.MmNz(attnLocal, pHalf, pHalf);
      }
    }
    // Keep half(T) resident in UB (tHalf, NZ); both RHS applies read it.
    Cast(tHalf, tOut, RoundMode::CAST_NONE, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
  }

  // Compensated doubling for gate-tripped tasks (row sums >= 2.5): same
  // NZ-resident structure as BuildTMat, but every product runs MmNz2 (hi+lo
  // split operands, fp32-accumulated), so the growing P/T survive the six
  // squarings at near-fp32 precision. ~2x BuildTMat's cost — still ~50x cheaper
  // than the scalar forward substitution it replaces. The final T is cast to
  // half ONCE for the solves; that single rounding does not compound.
  // workHalf = qHalf[0:ATTEN), tHalf = qHalf[ATTEN:2*ATTEN),
  // workF32 = halfLocal reinterpreted (V is NOT yet loaded on this path).
  __aicore__ inline void BuildTMatComp(LocalTensor<float> attnLocal, LocalTensor<float> tOut,
                                    LocalTensor<float> cScratch, LocalTensor<half> workHalf,
                                    LocalTensor<half> tHalf, LocalTensor<float> workF32) {
    SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopy(cScratch, iNzGm_, ATTEN_ELEMS);
    for (uint32_t j = 0; j < 4; ++j) {
      Muls(tOut[j * 64 * 16], attnLocal[j * 16], 1.0f, static_cast<uint64_t>(16), 64, {1, 1, 2, 8});
    }
    PipeBarrier<PIPE_V>();
    Adds(attnLocal, tOut, 0.0f, ATTEN_ELEMS);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Add(tOut, tOut, cScratch, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    // Round 0: T = I + A is materialized; square P in place.
    microMm_.MmNz2(attnLocal, attnLocal, attnLocal, workHalf, workF32);
    for (uint32_t round = 1; round < DOUBLING_ROUNDS; ++round) {
      microMm_.MmNz2(cScratch, attnLocal, tOut, workHalf, workF32);
      Add(tOut, tOut, cScratch, ATTEN_ELEMS);
      PipeBarrier<PIPE_V>();
      if (round + 1 < DOUBLING_ROUNDS) {
        microMm_.MmNz2(attnLocal, attnLocal, attnLocal, workHalf, workF32);
      }
    }
    Cast(tHalf, tOut, RoundMode::CAST_NONE, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
  }


  __aicore__ inline void ProcessOneTask(uint32_t b, uint32_t kHeadIdx, uint32_t vHeadIdx, uint32_t chunkIdx) {
    // The compensated doubling can overflow half range on rare tasks whose T
    // outgrows the row-sum gate's estimate (real model data overflows below
    // the synthetic-sweep bound). The attempt returns false on a non-finite T
    // BEFORE any GM store; the retry re-derives everything from GM on the
    // exact scalar path, so a mis-predicted gate costs one task redo, not NaN.
    if (!ProcessOneTaskAttempt(b, kHeadIdx, vHeadIdx, chunkIdx, /*allowCompDoubling=*/true)) {
      ProcessOneTaskAttempt(b, kHeadIdx, vHeadIdx, chunkIdx, /*allowCompDoubling=*/false);
    }
  }

  __aicore__ inline bool ProcessOneTaskAttempt(uint32_t b, uint32_t kHeadIdx, uint32_t vHeadIdx, uint32_t chunkIdx,
                                               bool allowCompDoubling) {
    const uint32_t tokenStart = chunkIdx * FIXED_CHUNK_SIZE;
    if (tokenStart + FIXED_CHUNK_SIZE > seqlen_ || kHeadIdx >= kNumHead_ || vHeadIdx >= vNumHead_) {
      return true;
    }
    LocalTensor<half> halfLocal = halfBuf_.Get<half>();
    LocalTensor<half> qHalf = qHalfBuf_.Get<half>();
    LocalTensor<float> rhs = rhsBuf_.Get<float>();
    LocalTensor<float> gLocal = gBuf_.Get<float>();
    LocalTensor<float> expGLocal = expGBuf_.Get<float>();
    LocalTensor<float> betaLocal = rowBuf_.Get<float>();
    LocalTensor<float> reduceScratch = negABuf_.Get<float>();
    LocalTensor<float> attnLocal = attnBuf_.Get<float>();
    LocalTensor<float> scratch = tmpBuf_.Get<float>();

    // ---- Stage 1: K → halfLocal (half, full width) + Kβ → rhs, one 64-wide
    // slice at a time so the fp32 staging (scratch) stays 64x64. ----
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    LoadBetaChunk(betaLocal, qHalf, FIXED_CHUNK_SIZE * maxAlign_, b, tokenStart, vHeadIdx);
    for (uint32_t k0 = 0; k0 < kHeadDim_; k0 += FIXED_CHUNK_SIZE) {
      const uint32_t kCur = (kHeadDim_ - k0) < FIXED_CHUNK_SIZE ? (kHeadDim_ - k0) : FIXED_CHUNK_SIZE;
      // MTE2 write below races prior V work on halfLocal/scratch without this.
      SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
      LoadBthdChunkSlice(kGm_, halfLocal, b, tokenStart, kHeadIdx, kNumHead_, kHeadDim_, alignK_, k0, kCur);
      SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      // One strided Cast (dst rows lda=64 fp32, src rows lda=alignK half):
      // replaces 64 scalar-issued per-row Casts — the biggest scalar-issue sink.
      Cast(scratch, halfLocal[k0], RoundMode::CAST_NONE, static_cast<uint64_t>(kCur), FIXED_CHUNK_SIZE,
           {1, 1, static_cast<uint8_t>(FIXED_CHUNK_SIZE * sizeof(float) / BLOCK_BYTES),
            static_cast<uint8_t>(alignK_ * sizeof(half) / BLOCK_BYTES)});
      PipeBarrier<PIPE_V>();
      // attnLocal doubles as the Brcb workspace (needs 64*8 floats) while free.
      BroadcastMulRowsFloat(rhs[k0], scratch, betaLocal, attnLocal, FIXED_CHUNK_SIZE, kCur, alignK_,
                            FIXED_CHUNK_SIZE);
    }
    // reduceScratch holds gRaw; attnLocal is the g-load scratch
    // (Hv<=64 ⇒ FIXED_CHUNK*Hv <= ATTEN_ELEMS). attnLocal was just V-written
    // (Brcb workspace) and is DataCopy'd (MTE2) inside — sync or it corrupts
    // both the g gather source and the in-flight Kβ product.
    SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    BuildCumulativeG(b, vHeadIdx, tokenStart, gLocal, expGLocal, reduceScratch, attnLocal, ATTEN_ELEMS);

    // Cube: G = Kβ @ K^T → attnLocal; then A = −strictlower(G ⊙ Λ).
    CastFloatRowsToHalf(qHalf, rhs, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    // halfLocal already holds the full-width K half block from the slice loads.
    if (kHeadDim_ >= 128) {
      microMm_.GramAcc(attnLocal, qHalf, halfLocal, storeBuf_.Get<half>().ReinterpretCast<float>(), kHeadDim_,
                       kHeadDim_, alignK_);
    } else {
      // kHeadDim<128: the micro gram's rounding shifts the fallback-gate
      // boundary on weak-decay data (battery K=64 shape drops to cos 0.978);
      // the library gram keeps it at 0.9995. Cost: 2 lib calls on small dims.
      cubeGemm_.GemmATransB(attnLocal, qHalf, halfLocal, scratch, storeBuf_.Get<half>(), kHeadDim_, kHeadDim_,
                            alignK_);
    }
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Muls(attnLocal, attnLocal, -1.0f, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    ApplyLambdaNegStrictLower(attnLocal, gLocal, scratch);

    // Pass-1 RHS: rhs already holds βK; apply γ = exp(a) in place → W RHS.
    BroadcastMulRowsFloat(rhs, rhs, expGLocal, scratch, FIXED_CHUNK_SIZE, kHeadDim_, alignK_, alignK_);
    // expGLocal is dead after γ was applied and can hold the 64 row sums;
    // betaLocal must survive for the pass-2 βV product.
    const bool useFp32ForwardSubstitution =
        NeedsFp32ForwardSubstitution(attnLocal, scratch, expGLocal, reduceScratch);
    // Gate-tripped tasks on the all-micro path take the compensated doubling
    // (hi+lo split matmuls) instead of the ~50x slower scalar substitution;
    // small head dims keep the scalar fallback (their gram runs on the lib and
    // the K=64 shapes sit right on the precision boundary). Beyond
    // COMP_DOUBLING_MAX_ROW_SUM the intermediate T can outgrow fp16 range
    // (||T|| ~ e^rowsum worst case; half caps at 65504 ~ e^11) and the split
    // cast turns inf — those rows stay on the exact scalar path.
    const bool useCompDoubling = allowCompDoubling && useFp32ForwardSubstitution && kHeadDim_ >= 128 &&
                                 gateMaxRowSum_ < COMP_DOUBLING_MAX_ROW_SUM;
    const bool useScalarFallback = useFp32ForwardSubstitution && !useCompDoubling;
    // ---- Build T = (I−A)⁻¹ once (skipped on the fp32 fallback path, which
    // consumes A directly). tmpBuf holds T fp32; halfLocal (K half, dead after
    // the gram) is the reinterpreted C scratch; qHalf hosts both half stagings.
    // On the 128-dim all-micro path the C scratch lives in storeBuf so
    // halfLocal is free right after the gram and the V load overlaps the
    // doubling loop; the small-dim library-gram flow keeps the old layout.
    LocalTensor<float> cScratch = kHeadDim_ >= 128 ? storeBuf_.Get<half>().ReinterpretCast<float>()
                                                   : halfLocal.template ReinterpretCast<float>();
    // Kick the V load now: halfLocal (K) is dead after the gram, and the MTE2
    // transfer hides under the whole doubling loop. Only on the all-micro
    // 128-dim path — the small-dim library-gram flow regresses with it.
    if (kHeadDim_ >= 128 && !useCompDoubling) {
      SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
      LoadBthdChunk(vGm_, halfLocal, b, tokenStart, vHeadIdx, vNumHead_, vHeadDim_, alignV_);
    }
    if (!useFp32ForwardSubstitution) {
      BuildTMat(attnLocal, scratch, cScratch, qHalf, qHalf[ATTEN_ELEMS], cScratch);
    } else if (useCompDoubling) {
      // halfLocal is the fp32 work scratch here, so the V load waits for the
      // build and then overlaps the W solve (which only touches qHalf/rhs).
      BuildTMatComp(attnLocal, scratch, cScratch, qHalf, qHalf[ATTEN_ELEMS],
                 halfLocal.template ReinterpretCast<float>());
      // Overflow rollback: T (fp32, in scratch) must survive the half cast the
      // solves consume (tHalf). |T| beyond half range means the doubling
      // overflowed — bail before any GM store and let the caller retry scalar.
      Abs(cScratch, scratch, ATTEN_ELEMS);
      PipeBarrier<PIPE_V>();
      // attnLocal (the doubling's P) is dead once T is final — reuse it as the
      // 4096-element reduce's work area (reduceScratch is only 64 floats).
      ReduceMax(reduceScratch, cScratch, attnLocal, ATTEN_ELEMS, /*calIndex=*/false);
      SyncEvent<HardEvent::V_S>(HardEvent::V_S);
      const float tMax = reduceScratch.GetValue(0);
      if (!(tMax < 60000.0f)) {  // catches inf, NaN, and finite half-overflow
        PipeBarrier<PIPE_ALL>();  // leave clean pipe state for the retry
        return false;
      }
      SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
      LoadBthdChunk(vGm_, halfLocal, b, tokenStart, vHeadIdx, vNumHead_, vHeadDim_, alignV_);
    }
    // STAGECUT_3_gate_buildT

    // ---- W = T @ (γβK), resident in rhs. halfLocal stages R halves. ----
    if (useScalarFallback) {
      Fp32ForwardSubstitution(attnLocal, rhs, kHeadDim_, alignK_);
    } else {
      // Micro-path solve: per 64-wide slice, compact-cast B then C into the
      // rhs slice (ldc = alignK). cNz scratch: storeBuf as fp32.
      LocalTensor<float> wNz = storeBuf_.Get<half>().ReinterpretCast<float>();
      for (uint32_t n0 = 0; n0 < kHeadDim_; n0 += FIXED_CHUNK_SIZE) {
        const uint32_t nCur = (kHeadDim_ - n0) < FIXED_CHUNK_SIZE ? (kHeadDim_ - n0) : FIXED_CHUNK_SIZE;
        LocalTensor<half> wB = kHeadDim_ >= 128 ? qHalf : halfLocal;
        CastFloatRowsToHalfSized(wB, rhs[n0], FIXED_CHUNK_SIZE, nCur, alignK_);
        microMm_.MmANz(rhs[n0], qHalf[ATTEN_ELEMS], wB, wNz, nCur, alignK_);
      }
    }
    LocalTensor<half> storeLocal = storeBuf_.Get<half>();
    if (kHeadDim_ < 128) {
      SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
      LoadBthdChunk(vGm_, halfLocal, b, tokenStart, vHeadIdx, vNumHead_, vHeadDim_, alignV_);
    }
    Cast(storeLocal, rhs, RoundMode::CAST_NONE, chunkKElems_);
    SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    StoreBhtdChunk(wKernelGm_, storeLocal, b, vHeadIdx, tokenStart, vNumHead_, kHeadDim_, alignK_);

    // ---- U = T @ (βV). ----
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    if (useScalarFallback) {
      Cast(rhs, halfLocal, RoundMode::CAST_NONE, chunkVElems_);
      PipeBarrier<PIPE_V>();
      BroadcastMulRowsFloat(rhs, rhs, betaLocal, scratch, FIXED_CHUNK_SIZE, vHeadDim_, alignV_, alignV_);
      Fp32ForwardSubstitution(attnLocal, rhs, vHeadDim_, alignV_);
      // storeBuf may still be feeding the W store (MTE3 reads) — order the U cast.
      SyncEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
      Cast(storeLocal, rhs, RoundMode::CAST_NONE, chunkVElems_);
      SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
      StoreBhtdChunk(uKernelGm_, storeLocal, b, vHeadIdx, tokenStart, vNumHead_, vHeadDim_, alignV_);
    } else {
      // Half-domain U path: βV is computed directly in half, in place on the
      // loaded V (the product was always rounded to half before the cube, so
      // the rounding point is unchanged); the h→f V cast and the apply's
      // cast-out both disappear. C stays fp32 (half C wedges the 310P cube).
      LocalTensor<half> betaHalfVec = betaHalfBuf_.Get<half>();
      LocalTensor<half> brcbHalf = scratch.template ReinterpretCast<half>();
      Cast(betaHalfVec, betaLocal, RoundMode::CAST_NONE, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
      BroadcastMulRowsHalf(halfLocal, halfLocal, betaHalfVec, brcbHalf, FIXED_CHUNK_SIZE, vHeadDim_, alignV_,
                           alignV_);
      LocalTensor<float> uNz = storeBuf_.Get<half>().ReinterpretCast<float>();
      // storeBuf may still be feeding the W store (MTE3 reads) — the micro
      // path writes its NZ scratch there on V, so order it explicitly.
      SyncEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
      for (uint32_t n0 = 0; n0 < vHeadDim_; n0 += FIXED_CHUNK_SIZE) {
        const uint32_t nCur = (vHeadDim_ - n0) < FIXED_CHUNK_SIZE ? (vHeadDim_ - n0) : FIXED_CHUNK_SIZE;
        LocalTensor<half> bFeedU = halfLocal;
        if (!(n0 == 0 && alignV_ == nCur)) {
          Muls(qHalf, halfLocal[n0], static_cast<half>(1), static_cast<uint64_t>(nCur), FIXED_CHUNK_SIZE,
               {1, 1, static_cast<uint8_t>(nCur * sizeof(half) / 32),
                static_cast<uint8_t>(alignV_ * sizeof(half) / 32)});
          PipeBarrier<PIPE_V>();
          bFeedU = qHalf;
        }
        microMm_.MmANz(rhs[n0], qHalf[ATTEN_ELEMS], bFeedU, uNz, nCur, alignV_);
      }
      SyncEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
      Cast(storeLocal, rhs, RoundMode::CAST_NONE, chunkVElems_);
      SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
      StoreBhtdChunk(uKernelGm_, storeLocal, b, vHeadIdx, tokenStart, vNumHead_, vHeadDim_, alignV_);
    }
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    if ((vHeadIdx % headGroups_) == 0) {
      StoreQKKernel(b, kHeadIdx, tokenStart, qHalf);
    }
    PipeBarrier<PIPE_ALL>();
    return true;
  }

  TPipe* pipe_{nullptr};
  bool valid_{false};
  // Max |row sum| of A from the last gate check; NaN when the data was NaN.
  float gateMaxRowSum_{0.0f};
  uint32_t batch_{0}, seqlen_{0}, kNumHead_{0}, vNumHead_{0}, kHeadDim_{0}, vHeadDim_{0};
  uint32_t chunkSize_{0}, numChunks_{0}, headGroups_{0}, alignK_{0}, alignV_{0}, chunkKElems_{0}, chunkVElems_{0};
  uint32_t maxAlign_{0};
  uint32_t localWorkspaceSize_{0}, perCoreWorkspaceBytes_{0}, usedCoreNum_{1};
  GlobalTensor<half> qGm_, kGm_, vGm_, betaGm_, qKernelGm_, kKernelGm_, wKernelGm_, uKernelGm_;
  GlobalTensor<float> gGm_, gKernelGm_, iNzGm_;
  WyCubeGemm cubeGemm_;
  WyMicroMm microMm_;
  TBuf<TPosition::VECCALC> halfBuf_, qHalfBuf_, rhsBuf_, attnBuf_, gBuf_,
      expGBuf_, tmpBuf_, rowBuf_, negABuf_, lamOffBuf_, gOffBuf_, betaOffBuf_, lane0OffBuf_, betaHalfBuf_, storeBuf_, mmLocalWsBuf_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif
