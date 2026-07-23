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
constexpr uint32_t MAX_SAFE_HEAD_DIM = 128;
constexpr uint32_t DOUBLING_ROUNDS = 6;  // log2(64)

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

    const uint32_t localWsBytes = localWorkspaceSize_ == 0 ? (32 * 1024) : localWorkspaceSize_;
    pipe_->InitBuffer(mmLocalWsBuf_, localWsBytes);
    cubeGemm_.Init(&tiling->mmAttn, &tiling->mmSquare, &tiling->mmApplyU, &tiling->mmApplyW, pipe_,
                   mmLocalWsBuf_.Get<uint8_t>(), localWsBytes, workspace, perCoreWorkspaceBytes_, usedCoreNum_);

    uint32_t maxAlign = (alignK_ > alignV_ ? alignK_ : alignV_);
    if (maxAlign < FIXED_CHUNK_SIZE) {
      maxAlign = FIXED_CHUNK_SIZE;
    }
    // halfScratch for GemmApplyAdd/GemmSquare: P is 64x64, R is 64xN — need 64*max(64,K,V).
    pipe_->InitBuffer(qHalfBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(half));
    pipe_->InitBuffer(kHalfBuf_, chunkKElems_ * sizeof(half));
    pipe_->InitBuffer(vHalfBuf_, chunkVElems_ * sizeof(half));
    pipe_->InitBuffer(kFloatBuf_, chunkKElems_ * sizeof(float));
    pipe_->InitBuffer(vFloatBuf_, chunkVElems_ * sizeof(float));
    pipe_->InitBuffer(kBetaBuf_, chunkKElems_ * sizeof(float));
    pipe_->InitBuffer(attnBuf_, ATTEN_ELEMS * sizeof(float));
    pipe_->InitBuffer(gBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    pipe_->InitBuffer(expGBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    // Contiguous C scratch for bulk GemmApplyAdd: [64, max(K,V,64)].
    pipe_->InitBuffer(tmpBuf_, FIXED_CHUNK_SIZE * maxAlign * sizeof(float));
    pipe_->InitBuffer(rowBuf_, FIXED_CHUNK_SIZE * sizeof(float));
    valid_ = true;
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
                                             uint32_t tokenStart, uint32_t headIdx, uint32_t numHeads) const {
    const uint32_t span = FIXED_CHUNK_SIZE * numHeads;
    const uint64_t chunkBase = (static_cast<uint64_t>(b) * seqlen_ + tokenStart) * numHeads;
    if (span <= scratchElems) {
      DataCopyParams copyParams{1, BytesToBlocks(span * sizeof(float)), 0, 0};
      DataCopy(scratch, srcGm[chunkBase], copyParams);
      SyncEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        dstLocal.SetValue(i, scratch.GetValue(i * numHeads + headIdx));
      }
    } else {
      const uint64_t base = chunkBase + headIdx;
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        dstLocal.SetValue(i, srcGm.GetValue(base + static_cast<uint64_t>(i) * numHeads));
      }
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void LoadBetaChunk(LocalTensor<float> dstLocal, LocalTensor<half> scratch, uint32_t scratchElems,
                                       uint32_t b, uint32_t tokenStart, uint32_t vHeadIdx) const {
    const uint32_t span = FIXED_CHUNK_SIZE * vNumHead_;
    const uint64_t chunkBase = (static_cast<uint64_t>(b) * seqlen_ + tokenStart) * vNumHead_;
    if (span <= scratchElems) {
      DataCopyParams copyParams{1, BytesToBlocks(span * sizeof(half)), 0, 0};
      DataCopy(scratch, betaGm_[chunkBase], copyParams);
      SyncEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        dstLocal.SetValue(i, static_cast<float>(scratch.GetValue(i * vNumHead_ + vHeadIdx)));
      }
    } else {
      const uint64_t base = chunkBase + vHeadIdx;
      for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
        const half val = betaGm_.GetValue(base + static_cast<uint64_t>(i) * vNumHead_);
        dstLocal.SetValue(i, static_cast<float>(val));
      }
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
  // Never forms γ_i/γ_j (overflow-safe). One outer-sub + Exp + Mul + Neg per row.
  __aicore__ inline void ApplyLambdaNegStrictLower(LocalTensor<float> gramLocal, const LocalTensor<float> aLocal,
                                                   LocalTensor<float> rowTmp) {
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      const float ai = aLocal.GetValue(i);
      Duplicate(rowTmp, ai, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
      Sub(rowTmp, rowTmp, aLocal, FIXED_CHUNK_SIZE);  // a_i − a_j
      PipeBarrier<PIPE_V>();
      Exp(rowTmp, rowTmp, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
      // Zero Λ on diag+upper so those entries of A become 0.
      Duplicate(rowTmp[i], 0.0f, FIXED_CHUNK_SIZE - i);
      PipeBarrier<PIPE_V>();
      Mul(gramLocal[i * FIXED_CHUNK_SIZE], gramLocal[i * FIXED_CHUNK_SIZE], rowTmp, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
      Muls(gramLocal[i * FIXED_CHUNK_SIZE], gramLocal[i * FIXED_CHUNK_SIZE], -1.0f, FIXED_CHUNK_SIZE);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void BroadcastMulRowsFloat(LocalTensor<float> dst, const LocalTensor<float> rows,
                                               const LocalTensor<float> scalePerRow, uint32_t rowsCount,
                                               uint32_t cols, uint32_t lda) const {
    for (uint32_t row = 0; row < rowsCount; ++row) {
      Muls(dst[row * lda], rows[row * lda], scalePerRow.GetValue(row), cols);
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void BuildKBetaExpFloat(LocalTensor<float> dst, const LocalTensor<float> kLocal,
                                            const LocalTensor<float> betaLocal, const LocalTensor<float> expGLocal,
                                            const LocalTensor<float> rowTmp, uint32_t headDim, uint32_t lda) const {
    for (uint32_t row = 0; row < FIXED_CHUNK_SIZE; ++row) {
      const float beta = betaLocal.GetValue(row);
      Muls(rowTmp, kLocal[row * lda], beta, headDim);
      Muls(dst[row * lda], rowTmp, expGLocal.GetValue(row), headDim);
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

  __aicore__ inline void ProcessOneTask(uint32_t b, uint32_t kHeadIdx, uint32_t vHeadIdx, uint32_t chunkIdx) {
    const uint32_t tokenStart = chunkIdx * FIXED_CHUNK_SIZE;
    if (tokenStart + FIXED_CHUNK_SIZE > seqlen_ || kHeadIdx >= kNumHead_ || vHeadIdx >= vNumHead_) {
      return;
    }
    LocalTensor<half> kHalf = kHalfBuf_.Get<half>();
    LocalTensor<half> qHalf = qHalfBuf_.Get<half>();
    LocalTensor<half> vHalf = vHalfBuf_.Get<half>();
    LocalTensor<float> kFloat = kFloatBuf_.Get<float>();
    LocalTensor<float> vFloat = vFloatBuf_.Get<float>();
    LocalTensor<float> kBeta = kBetaBuf_.Get<float>();
    LocalTensor<float> gLocal = gBuf_.Get<float>();
    LocalTensor<float> expGLocal = expGBuf_.Get<float>();
    LocalTensor<float> betaLocal = rowBuf_.Get<float>();
    LocalTensor<float> attnLocal = attnBuf_.Get<float>();
    LocalTensor<float> scratch = tmpBuf_.Get<float>();

    LoadKChunk(b, kHeadIdx, tokenStart, kHalf);
    LoadBthdChunk(vGm_, vHalf, b, tokenStart, vHeadIdx, vNumHead_, vHeadDim_, alignV_);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Cast(kFloat, kHalf, RoundMode::CAST_NONE, chunkKElems_);
    Cast(vFloat, vHalf, RoundMode::CAST_NONE, chunkVElems_);
    SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SyncEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    LoadBetaChunk(betaLocal, kHalf, chunkKElems_, b, tokenStart, vHeadIdx);
    BroadcastMulRowsFloat(kBeta, kFloat, betaLocal, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    // Hv<=64 ⇒ FIXED_CHUNK*Hv <= ATTEN_ELEMS, so attnLocal is always a safe g-load scratch.
    BuildCumulativeG(b, vHeadIdx, tokenStart, gLocal, expGLocal, scratch, attnLocal, ATTEN_ELEMS);

    // Cube: G = Kβ @ K^T → attnLocal; then A = −strictlower(G ⊙ Λ).
    CastFloatRowsToHalf(qHalf, kBeta, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    CastFloatRowsToHalf(kHalf, kFloat, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    cubeGemm_.GemmATransB(attnLocal, qHalf, kHalf, kHeadDim_, kHeadDim_, kHeadDim_);
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    ApplyLambdaNegStrictLower(attnLocal, gLocal, scratch);

    // RHS halves: βV in vFloat, γ·Kβ in kBeta (in-place doubling targets).
    BroadcastMulRowsFloat(vFloat, vFloat, betaLocal, FIXED_CHUNK_SIZE, vHeadDim_, alignV_);
    BuildKBetaExpFloat(kBeta, kFloat, betaLocal, expGLocal, scratch, kHeadDim_, alignK_);

    // Nilpotent doubling: R ← (I−A)⁻¹ R without forming T.
    // UploadP uses qHalf (>=64*64). U R-half uses vHalf (>=64*V); W R-half uses kHalf (>=64*K).
    // scratch (tmpBuf_) is [64, max(K,V)] — contiguous C staging for bulk ApplyAdd.
    for (uint32_t round = 0; round < DOUBLING_ROUNDS; ++round) {
      cubeGemm_.UploadP(attnLocal, qHalf);
      cubeGemm_.GemmApplyAdd(vFloat, vHalf, scratch, vHeadDim_, alignV_, /*useU=*/true);
      cubeGemm_.GemmApplyAdd(kBeta, kHalf, scratch, kHeadDim_, alignK_, /*useU=*/false);
      if (round + 1 < DOUBLING_ROUNDS) {
        cubeGemm_.GemmSquare(attnLocal, qHalf);
        SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      }
    }

    Cast(vHalf, vFloat, RoundMode::CAST_NONE, chunkVElems_);
    CastFloatRowsToHalf(kHalf, kBeta, FIXED_CHUNK_SIZE, kHeadDim_, alignK_);
    SyncEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    StoreBhtdChunk(uKernelGm_, vHalf, b, vHeadIdx, tokenStart, vNumHead_, vHeadDim_, alignV_);
    StoreBhtdChunk(wKernelGm_, kHalf, b, vHeadIdx, tokenStart, vNumHead_, kHeadDim_, alignK_);
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
  uint32_t localWorkspaceSize_{0}, perCoreWorkspaceBytes_{0}, usedCoreNum_{1};
  GlobalTensor<half> qGm_, kGm_, vGm_, betaGm_, qKernelGm_, kKernelGm_, wKernelGm_, uKernelGm_;
  GlobalTensor<float> gGm_, gKernelGm_;
  WyCubeGemm cubeGemm_;
  TBuf<TPosition::VECCALC> kHalfBuf_, qHalfBuf_, vHalfBuf_, kFloatBuf_, vFloatBuf_, kBetaBuf_, attnBuf_, gBuf_,
      expGBuf_, tmpBuf_, rowBuf_, mmLocalWsBuf_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif
