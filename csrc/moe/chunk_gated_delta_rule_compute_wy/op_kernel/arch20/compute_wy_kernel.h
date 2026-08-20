/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_KERNEL_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_KERNEL_H

#include "kernel_operator.h"
#include "../chunk_gated_delta_rule_compute_wy_tiling_data.h"

#include "compute_wy_cube.h"

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
// ||A||_inf below this value bounds the forward solve by 1/(1-threshold),
// leaving ample fp16 headroom for the fast Cube doubling path.
constexpr float FP32_FS_ROW_SUM_THRESHOLD = 0.75f;

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
    debugStage_ = tiling->debugStage;
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

    const uint32_t localWsBytes = localWorkspaceSize_ == 0 ? (32 * 1024) : localWorkspaceSize_;
    pipe_->InitBuffer(mmLocalWsBuf_, localWsBytes);
    cubeGemm_.Init(&tiling->mmAttn, &tiling->mmSquare, &tiling->mmApplyU, &tiling->mmApplyW, pipe_,
                   mmLocalWsBuf_.Get<uint8_t>(), localWsBytes, workspace, tiling->workspaceOffset,
                   perCoreWorkspaceBytes_, usedCoreNum_, kHeadDim_, vHeadDim_);

    uint32_t maxAlign = (alignK_ > alignV_ ? alignK_ : alignV_);
    if (maxAlign < FIXED_CHUNK_SIZE) {
      maxAlign = FIXED_CHUNK_SIZE;
    }
    maxAlign_ = maxAlign;
    // Two-pass UB layout (fits K=V=128 in 192KB):
    //  - qHalfBuf_: βK gram operand + P-upload scratch + Q/K passthrough staging.
    //  - halfBuf_:  K gram operand (pass 1) and the R-half staging for whichever
    //    RHS is being solved; W and U are never live simultaneously.
    //  - tmpBuf_:   K fp32 while building Kβ, then gram acc / Λ mat / apply-C scratch.
    //  - rhsBuf_:   the single resident fp32 RHS (γβK in pass 1, βV in pass 2).
    //  - A snapshot for pass 2 lives in the per-core GM slot (cubeGemm_.SaveSnap).
    pipe_->InitBuffer(qHalfBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(half));
    pipe_->InitBuffer(halfBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(half));
    pipe_->InitBuffer(rhsBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(float));
    pipe_->InitBuffer(attnBuf_, ATTEN_ELEMS * sizeof(float));
    // +8 floats: slot FIXED_CHUNK_SIZE holds the Lambda mask sentinel (see BuildLambdaTable).
    pipe_->InitBuffer(gBuf_, (FIXED_CHUNK_SIZE + 8) * sizeof(float));
    pipe_->InitBuffer(expGBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    // 64x64 fp32 scratch: per-slice K staging, gram acc, Lambda mat, apply C.
    // (Keeping it 64-wide is what lets K=V=128 fit in UB next to the matmul
    // library's own allocations; A parks in GM between the two solve passes.)
    pipe_->InitBuffer(tmpBuf_, ATTEN_ELEMS * sizeof(float));
    pipe_->InitBuffer(rowBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    pipe_->InitBuffer(negABuf_, FIXED_CHUNK_SIZE * sizeof(float));
    pipe_->InitBuffer(lamOffBuf_, ATTEN_ELEMS * sizeof(uint32_t));
    pipe_->InitBuffer(gOffBuf_, FIXED_CHUNK_SIZE * sizeof(uint32_t));
    pipe_->InitBuffer(betaOffBuf_, FIXED_CHUNK_SIZE * sizeof(uint32_t));
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
    LocalTensor<uint32_t> off = lamOffBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      const uint32_t rowByte = i * static_cast<uint32_t>(sizeof(float));
      for (uint32_t j = 0; j < FIXED_CHUNK_SIZE; ++j) {
        off.SetValue(i * FIXED_CHUNK_SIZE + j,
                     (j < i) ? rowByte : FIXED_CHUNK_SIZE * static_cast<uint32_t>(sizeof(float)));
      }
    }
    gBuf_.Get<float>().SetValue(FIXED_CHUNK_SIZE, -1.0e20f);
  }

  // g and beta arrive as [64, vNumHead] blocks; we need one strided column. Byte-offset
  // tables let a single Gather extract it, so the column never round-trips through the
  // scalar unit. numHeads is fixed for the launch, so these are built once.
  __aicore__ inline void BuildColumnGatherTables() {
    LocalTensor<uint32_t> gOff = gOffBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> betaOff = betaOffBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      gOff.SetValue(i, i * vNumHead_ * static_cast<uint32_t>(sizeof(float)));
      betaOff.SetValue(i, i * vNumHead_ * static_cast<uint32_t>(sizeof(half)));
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
    float running = 0.0f;
    for (uint32_t localT = 0; localT < FIXED_CHUNK_SIZE; ++localT) {
      running += gRaw.GetValue(localT);
      gLocal.SetValue(localT, running);
    }
    PipeBarrier<PIPE_V>();
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
                                                      LocalTensor<float> reduceScratch) const {
    // A is strict-lower, so reducing full rows also computes their absolute
    // lower-triangle sums. Keep the scan on the vector pipe; only 64 reduced
    // scalars cross to the scalar unit.
    Abs(absScratch, a, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < FIXED_CHUNK_SIZE; ++row) {
      ReduceSum(rowSums[row], absScratch[row * FIXED_CHUNK_SIZE], reduceScratch, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
    }
    SyncEvent<HardEvent::V_S>(HardEvent::V_S);
    for (uint32_t row = 0; row < FIXED_CHUNK_SIZE; ++row) {
      const float rowSum = rowSums.GetValue(row);
      if (rowSum != rowSum || rowSum >= FP32_FS_ROW_SUM_THRESHOLD) {
        return true;
      }
    }
    return false;
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

  // Applies (I−A)⁻¹ in place to one RHS block, choosing the scalar or doubling path.
  // Doubling mutates attnLocal (squarings); the caller restores it between passes.
  __aicore__ inline void SolveOneRhs(bool useFp32, LocalTensor<float> attnLocal, LocalTensor<float> rhs,
                                     LocalTensor<half> rHalf, LocalTensor<half> pHalf, LocalTensor<float> scratch,
                                     uint32_t dim, uint32_t lda, bool useU) {
    if (useFp32) {
      Fp32ForwardSubstitution(attnLocal, rhs, dim, lda);
      return;
    }
    // Fast nilpotent doubling: R ← (I−A)⁻¹ R without forming T. pHalf (qHalf)
    // stages P uploads; rHalf stages the RHS — they must not alias.
    for (uint32_t round = 0; round < DOUBLING_ROUNDS; ++round) {
      cubeGemm_.UploadP(attnLocal, pHalf);
      cubeGemm_.GemmApplyAdd(rhs, rHalf, scratch, dim, lda, useU);
      if (round + 1 < DOUBLING_ROUNDS) {
        cubeGemm_.GemmSquare(attnLocal, pHalf);
        SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      }
    }
  }

  __aicore__ inline void ProcessOneTask(uint32_t b, uint32_t kHeadIdx, uint32_t vHeadIdx, uint32_t chunkIdx) {
    const uint32_t tokenStart = chunkIdx * FIXED_CHUNK_SIZE;
    if (tokenStart + FIXED_CHUNK_SIZE > seqlen_ || kHeadIdx >= kNumHead_ || vHeadIdx >= vNumHead_) {
      return;
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
      LoadBthdChunkSlice(kGm_, halfLocal, b, tokenStart, kHeadIdx, kNumHead_, kHeadDim_, alignK_, k0, kCur);
      SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      for (uint32_t row = 0; row < FIXED_CHUNK_SIZE; ++row) {
        Cast(scratch[row * FIXED_CHUNK_SIZE], halfLocal[row * alignK_ + k0], RoundMode::CAST_NONE, kCur);
      }
      PipeBarrier<PIPE_V>();
      // attnLocal doubles as the Brcb workspace (needs 64*8 floats) while free.
      BroadcastMulRowsFloat(rhs[k0], scratch, betaLocal, attnLocal, FIXED_CHUNK_SIZE, kCur, alignK_,
                            FIXED_CHUNK_SIZE);
    }
    // reduceScratch holds gRaw; attnLocal is the g-load scratch
    // (Hv<=64 ⇒ FIXED_CHUNK*Hv <= ATTEN_ELEMS). Both are free until the gram.
    BuildCumulativeG(b, vHeadIdx, tokenStart, gLocal, expGLocal, reduceScratch, attnLocal, ATTEN_ELEMS);

    // Cube: G = Kβ @ K^T → attnLocal; then A = −strictlower(G ⊙ Λ).
    CastFloatRowsToHalf(qHalf, rhs, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    // halfLocal already holds the full-width K half block from the slice loads.
    cubeGemm_.GemmATransB(attnLocal, qHalf, halfLocal, scratch, kHeadDim_, kHeadDim_, alignK_);
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
    // Snapshot A to the per-core GM slot: the doubling squarings below consume
    // it, and pass 2 must restart from P₀. (Unconditional; also fp32 path.)
    cubeGemm_.SaveSnap(attnLocal);

    // ---- Pass 1: W = (I−A)⁻¹ (γβK), resident in rhs. halfLocal stages R halves. ----
    SolveOneRhs(useFp32ForwardSubstitution, attnLocal, rhs, halfLocal, qHalf, scratch, kHeadDim_, alignK_,
                /*useU=*/false);
    Cast(halfLocal, rhs, RoundMode::CAST_NONE, chunkKElems_);
    SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    StoreBhtdChunk(wKernelGm_, halfLocal, b, vHeadIdx, tokenStart, vNumHead_, kHeadDim_, alignK_);
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    // ---- Pass 2: load V now, U = (I−A)⁻¹ (βV). ----
    cubeGemm_.LoadSnap(attnLocal);
    LoadBthdChunk(vGm_, halfLocal, b, tokenStart, vHeadIdx, vNumHead_, vHeadDim_, alignV_);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Cast(rhs, halfLocal, RoundMode::CAST_NONE, chunkVElems_);
    PipeBarrier<PIPE_V>();
    BroadcastMulRowsFloat(rhs, rhs, betaLocal, scratch, FIXED_CHUNK_SIZE, vHeadDim_, alignV_, alignV_);
    SolveOneRhs(useFp32ForwardSubstitution, attnLocal, rhs, halfLocal, qHalf, scratch, vHeadDim_, alignV_,
                /*useU=*/true);
    Cast(halfLocal, rhs, RoundMode::CAST_NONE, chunkVElems_);
    SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    StoreBhtdChunk(uKernelGm_, halfLocal, b, vHeadIdx, tokenStart, vNumHead_, vHeadDim_, alignV_);
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    if ((vHeadIdx % headGroups_) == 0) {
      StoreQKKernel(b, kHeadIdx, tokenStart, qHalf);
    }
    PipeBarrier<PIPE_ALL>();
  }

  TPipe* pipe_{nullptr};
  bool valid_{false};
  uint32_t batch_{0}, seqlen_{0}, kNumHead_{0}, vNumHead_{0}, kHeadDim_{0}, vHeadDim_{0};
  uint32_t chunkSize_{0}, numChunks_{0}, headGroups_{0}, alignK_{0}, alignV_{0}, chunkKElems_{0}, chunkVElems_{0};
  uint32_t maxAlign_{0};
  uint32_t localWorkspaceSize_{0}, perCoreWorkspaceBytes_{0}, usedCoreNum_{1};
  uint32_t debugStage_{0};
  GlobalTensor<half> qGm_, kGm_, vGm_, betaGm_, qKernelGm_, kKernelGm_, wKernelGm_, uKernelGm_;
  GlobalTensor<float> gGm_, gKernelGm_;
  WyCubeGemm cubeGemm_;
  TBuf<TPosition::VECCALC> halfBuf_, qHalfBuf_, rhsBuf_, attnBuf_, gBuf_,
      expGBuf_, tmpBuf_, rowBuf_, negABuf_, lamOffBuf_, gOffBuf_, betaOffBuf_, betaHalfBuf_, mmLocalWsBuf_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif
