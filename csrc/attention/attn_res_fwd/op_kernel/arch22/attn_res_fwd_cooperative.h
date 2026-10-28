// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_COOPERATIVE_H
#define ATTN_RES_FWD_COOPERATIVE_H

#include "kernel_operator.h"
#include "../attn_res_fwd_tiling_data.h"
#include "../attn_res_fwd_ub_layout.h"
#include "attn_res_fwd_reload.h"

namespace AttnResFwd {

// The arch22 reload header supplies InitParams and the common
// vector helpers. Scores use whole rows or cooperative H slices as selected by
// the host; output cores are independently reassigned to token/H slices.
template <typename D_IN>
class AttnResFwdCooperative {
 public:
  __aicore__ inline AttnResFwdCooperative(TPipe* pipe, const AttnResFwdTilingData* tiling)
      : pipe_(pipe), tiling_(tiling) {}

  __aicore__ inline void Init(const AttnResFwdInitParams& params, GM_ADDR workspace) {
    core_ = GetBlockIdx();
    hidden_ = tiling_->hiddenSize;
    rows_ = tiling_->blockCount;
    splits_ = tiling_->outputSplits;
    scoreSplits_ = tiling_->scoreSplits;
    tileElems_ = tiling_->cooperativeTileElems;
    const uint64_t partialTasks = static_cast<uint64_t>(tiling_->numTokens) * rows_ * scoreSplits_;
    outputStatsBase_ = partialTasks * (scoreSplits_ > 1 ? 2 : 1);
    metaAlign_ = Align(rows_, FLOAT_BLOCK_ELEMS);
    metaElems_ = metaAlign_ > VECTOR_FLOAT_ELEMS ? metaAlign_ : VECTOR_FLOAT_ELEMS;
    uint32_t slotCount = metaAlign_ > splits_ ? metaAlign_ : splits_;
    slotCount = rows_ * scoreSplits_ > slotCount ? rows_ * scoreSplits_ : slotCount;

    prefix_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.prefixSum));
    bank_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.blockResidual));
    proj_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.projWeight));
    norm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.normWeight));
    prefixOut_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.prefixOut));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.hiddenStates));
    materialized_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.materialized));
    if (tiling_->fusedAdd != 0) {
      addend_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.addend));
    }
    if (tiling_->outputNormEps > 0.0f) {
      outputNorm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN*>(params.outputNorm));
    }
    workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

    const bool wholeRow = scoreSplits_ == 1 && splits_ == 1;
    pipe_->InitBuffer(inQue_, wholeRow ? 2 : 1, tileElems_ * sizeof(D_IN));
    pipe_->InitBuffer(outQue_, 1, tileElems_ * sizeof(D_IN));
    pipe_->InitBuffer(statQue_, 1, FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(metaQue_, 1, slotCount * FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(rowBf16Buf_, tileElems_ * sizeof(D_IN));
    // Only the whole-row path changes its UB layout; the H-split fallback
    // keeps its original allocation and access pattern.
    const bool alignedPeriod =
        (static_cast<uint64_t>(tileElems_) * sizeof(float)) % COOPERATIVE_UB_ADDRESS_PERIOD_BYTES == 0;
    const uint32_t rowPadding = wholeRow && alignedPeriod ? COOPERATIVE_ROW_PADDING_BYTES : 0;
    pipe_->InitBuffer(rowBuf_, tileElems_ * sizeof(float) + rowPadding);
    pipe_->InitBuffer(weightBuf_, tileElems_ * sizeof(float));
    pipe_->InitBuffer(accBuf_, tileElems_ * sizeof(float));
    pipe_->InitBuffer(scoresBuf_, metaElems_ * sizeof(float));
    pipe_->InitBuffer(scratchBuf_, metaElems_ * sizeof(float));
    pipe_->InitBuffer(broadcastBuf_, metaAlign_ * FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(scalarBuf_, 2 * FLOAT_BLOCK_ELEMS * sizeof(float));
    rowBf16_ = rowBf16Buf_.template Get<D_IN>();
    row_ = rowBuf_.template Get<float>();
    weight_ = weightBuf_.template Get<float>();
    acc_ = accBuf_.template Get<float>();
    scores_ = scoresBuf_.template Get<float>();
    scratch_ = scratchBuf_.template Get<float>();
    broadcast_ = broadcastBuf_.template Get<float>();
    scalar_ = scalarBuf_.template Get<float>();
  }

  __aicore__ inline void Process() {
    if (scoreSplits_ == 1 && splits_ == 1) {
      ProcessRowParallel();
      return;
    }
    const uint64_t rowTasks = static_cast<uint64_t>(tiling_->numTokens) * rows_;
    // Only UB-limited rows use this H-split fallback.
    ProcessSplitScores(rowTasks);

    // Every launched AIV participates, including cores without score work.
    // true means AIV-only synchronization; no Cube core joins this kernel.
    SyncAll<true>();
    // 等所有核计算完会得到 B, N+1的score；H切分时读取各段分数后再求和。
    const uint32_t outputTasks = tiling_->numTokens * splits_;
    // 每一层的H乘以score然后再加起来
    if (core_ < outputTasks) {
      InitOutputSlice();
      ReadScores();
      Softmax();
      WeightedOutput();
      CastToBf16(rowBf16_, acc_, outputCount_);
      if (tiling_->saveMaterialized != 0) {
        WriteRow(materialized_, outputOffset_, outputCount_);
      }
      if (tiling_->outputNormEps > 0.0f) {
        CastToFloat(row_, rowBf16_, outputCount_);
        SquareSum(outputCount_);
        WriteScalar(outputStatsBase_ + core_);
      } else {
        WriteRow(output_, outputOffset_, outputCount_);
      }
    }
    if (tiling_->outputNormEps > 0.0f) {
      // The condition is common tiling, not per-core output ownership.
      FinishOutputNorm(outputStatsBase_, outputTasks);
    }
  }

 private:
  __aicore__ inline void ProcessRowParallel() {
    const uint64_t rowTasks = static_cast<uint64_t>(tiling_->numTokens) * rows_;
    ProcessWholeScores(rowTasks);
    SyncAll<true>();
    // The owner of each token's first task (its prefix) is also its output
    // leader. A core can lead several tokens when row tasks wrap around C.
    for (uint32_t token = 0; token < tiling_->numTokens; ++token) {
      if ((static_cast<uint64_t>(token) * rows_) % tiling_->usedCoreNum != core_) {
        continue;
      }
      outputToken_ = token;
      outputStart_ = 0;
      outputCount_ = hidden_;
      outputOffset_ = static_cast<uint64_t>(token) * hidden_;
      ReadScores();
      Softmax();
      // With one task per core, the leader's original BF16 prefix is still
      // in rowBf16_. Wrapped task schedules use prefixOut GM instead.
      WeightedOutput(rowTasks <= tiling_->usedCoreNum);
      CastToBf16(rowBf16_, acc_, outputCount_);
      if (tiling_->saveMaterialized != 0) {
        WriteRow(materialized_, outputOffset_, outputCount_);
      }
      if (tiling_->outputNormEps > 0.0f) {
        CastToFloat(row_, rowBf16_, outputCount_);
        SquareSum(outputCount_);
        InverseRms(tiling_->outputNormEps);
        ApplyOutputNorm();
      }
      WriteRow(output_, outputOffset_, outputCount_);
    }
  }

  __aicore__ inline void ProcessWholeScores(uint64_t rowTasks) {
    if (core_ < rowTasks) {
      LoadScoreWeight(0, hidden_);
    }
    for (uint64_t task = core_; task < rowTasks; task += tiling_->usedCoreNum) {
      const uint32_t token = static_cast<uint32_t>(task / rows_);
      const uint32_t position = static_cast<uint32_t>(task % rows_);
      const uint32_t residual = position == 0 ? tiling_->numBlocks : position - 1;
      ScoreRow(token, residual);
      WriteScalar(static_cast<uint64_t>(token) * rows_ + residual);
    }
  }

  __aicore__ inline void ProcessSplitScores(uint64_t rowTasks) {
    const uint64_t partialTasks = rowTasks * scoreSplits_;
    // Large H can require more slices than physical cores. Each worker
    // processes a strided list of row/H tasks before the common barrier.
    for (uint64_t task = core_; task < partialTasks; task += tiling_->usedCoreNum) {
      LoadScoreSlice(task);
      SquareSum(scoreCount_);
      WriteScalar(partialTasks + task);
    }
    // All launched AIVs participate, including output-only cores.
    SyncAll<true>();
    for (uint64_t task = core_; task < partialTasks; task += tiling_->usedCoreNum) {
      // With one task per core the candidate and weight remain resident.
      // Otherwise reload the matching slice, rather than using the last task.
      if (partialTasks > tiling_->usedCoreNum) {
        LoadScoreSlice(task);
      }
      const uint64_t groupStart = (task / scoreSplits_) * scoreSplits_;
      auto in = ReadSlots(partialTasks + groupStart, scoreSplits_);
      ReduceSumHalfInterval(scalar_, in, static_cast<int32_t>(scoreSplits_ * FLOAT_BLOCK_ELEMS));
      metaQue_.FreeTensor(in);
      // Use the full H denominator, never scoreCount_. Keep the original
      // normalize-then-dot operation order within each slice.
      InverseRms(tiling_->normEps);
      NormalizeAndScore(scoreCount_);
      // Separate score slots prevent overwriting another core's RMS input.
      WriteScalar(task);
    }
  }

  __aicore__ inline void LoadScoreSlice(uint64_t task) {
    InitSlice(scoreSplits_, static_cast<uint32_t>(task % scoreSplits_), scoreStart_, scoreCount_);
    const uint64_t rowTask = task / scoreSplits_;
    LoadScoreWeight(scoreStart_, scoreCount_);
    LoadCandidateRow(static_cast<uint32_t>(rowTask / rows_), static_cast<uint32_t>(rowTask % rows_), scoreStart_,
                     scoreCount_);
  }

  __aicore__ inline void FinishOutputNorm(uint64_t outputStatsBase, uint32_t outputTasks) {
    // All launched cores enter, including score-only workers.
    SyncAll<true>();
    if (core_ < outputTasks) {
      NormalizeOutput(outputStatsBase);
      WriteRow(output_, outputOffset_, outputCount_);
    }
  }

  static constexpr uint32_t FLOAT_BLOCK_ELEMS = 8;
  static constexpr uint32_t VECTOR_FLOAT_ELEMS = 64;
  static constexpr uint32_t VECTOR_COPY_BYTES = 256;
  static constexpr uint32_t VECTOR_MAX_REPEATS = 255;

  __aicore__ inline uint32_t Align(uint32_t value, uint32_t alignment) const {
    return (value + alignment - 1) / alignment * alignment;
  }

  __aicore__ inline void EnqueueRow(const GlobalTensor<D_IN>& gm, uint64_t offset, uint32_t count) {
    auto in = inQue_.template AllocTensor<D_IN>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(D_IN)), 0, 0, 0};
    DataCopyPadExtParams<D_IN> pad{false, 0, 0, 0};
    DataCopyPad(in, gm[offset], copy, pad);
    inQue_.EnQue(in);
  }

  __aicore__ inline LocalTensor<D_IN> ReadRow(const GlobalTensor<D_IN>& gm, uint64_t offset, uint32_t count) {
    EnqueueRow(gm, offset, count);
    return inQue_.template DeQue<D_IN>();
  }

  __aicore__ inline void CopyBf16(const LocalTensor<D_IN>& dst, const LocalTensor<D_IN>& src, uint32_t count) {
    constexpr uint32_t step = VECTOR_COPY_BYTES / sizeof(D_IN);
    uint32_t offset = 0;
    uint32_t remaining = count;
    while (remaining >= step) {
      const uint32_t repeats = remaining / step > VECTOR_MAX_REPEATS ? VECTOR_MAX_REPEATS : remaining / step;
      Copy(dst[offset], src[offset], static_cast<uint64_t>(step), static_cast<uint8_t>(repeats), {1, 1, 8, 8});
      offset += repeats * step;
      remaining -= repeats * step;
    }
    if (remaining != 0) {
      Copy(dst[offset], src[offset], static_cast<uint64_t>(remaining), 1, {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CastToFloat(const LocalTensor<float>& dst, const LocalTensor<D_IN>& src, uint32_t count) {
    Cast(dst, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CastToBf16(const LocalTensor<D_IN>& dst, const LocalTensor<float>& src, uint32_t count) {
    Cast(dst, src, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void SquareSum(uint32_t count) {
    ClearReductionTail(count);
    Mul(acc_, row_, row_, count);
    PipeBarrier<PIPE_V>();
    ReduceSumHalfInterval(scalar_, acc_, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void InverseRms(float epsilon) {
    InvRmsInPlace(scalar_, tiling_->invHiddenSize, epsilon, scratch_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void LoadScoreWeight(uint32_t start, uint32_t count) {
    auto in = ReadRow(proj_, start, count);
    CastToFloat(row_, in, count);
    inQue_.FreeTensor(in);
    in = ReadRow(norm_, start, count);
    CastToFloat(acc_, in, count);
    inQue_.FreeTensor(in);
    Mul(weight_, row_, acc_, count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void PreparePrefix(uint32_t token, uint32_t start, uint32_t count) {
    const uint64_t offset = static_cast<uint64_t>(token) * hidden_ + start;
    auto in = ReadRow(prefix_, offset, count);
    CopyBf16(rowBf16_, in, count);
    inQue_.FreeTensor(in);
    if (tiling_->fusedAdd != 0) {
      CastToFloat(row_, rowBf16_, count);
      in = ReadRow(addend_, offset, count);
      CastToFloat(acc_, in, count);
      inQue_.FreeTensor(in);
      Add(row_, row_, acc_, count);
      PipeBarrier<PIPE_V>();
      CastToBf16(rowBf16_, row_, count);
      WriteRow(prefixOut_, offset, count);
    }
    if (tiling_->blockWriteIdx >= 0) {
      const uint64_t bankOffset = static_cast<uint64_t>(token) * tiling_->blockTokenStride +
                                  static_cast<uint64_t>(tiling_->blockWriteIdx) * hidden_ + start;
      WriteRow(bank_, bankOffset, count);
    }
  }

  __aicore__ inline void LoadCandidateRow(uint32_t token, uint32_t residual, uint32_t start, uint32_t count) {
    // 取出当前的候选x
    if (residual == tiling_->numBlocks) {
      PreparePrefix(token, start, count);
      CastToFloat(row_, rowBf16_, count);
    } else {
      const uint64_t offset =
          static_cast<uint64_t>(token) * tiling_->blockTokenStride + static_cast<uint64_t>(residual) * hidden_ + start;
      auto in = ReadRow(bank_, offset, count);
      CastToFloat(row_, in, count);
      inQue_.FreeTensor(in);
    }
  }

  __aicore__ inline void NormalizeAndScore(uint32_t count) {
    BroadcastScalarMulTensor(row_, row_, scalar_, scratch_, broadcast_, count, Align(count, FLOAT_BLOCK_ELEMS));
    ClearReductionTail(count);
    Mul(acc_, row_, weight_, count);
    PipeBarrier<PIPE_V>();
    ReduceSumHalfInterval(scalar_, acc_, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ClearReductionTail(uint32_t count) {
    // Mul overwrites every valid element. Only the aligned reduction tail
    // needs initialization; keep the source address aligned to 32 bytes.
    if (count % FLOAT_BLOCK_ELEMS != 0) {
      const uint32_t start = count / FLOAT_BLOCK_ELEMS * FLOAT_BLOCK_ELEMS;
      Duplicate(acc_[start], 0.0f, FLOAT_BLOCK_ELEMS);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void ScoreRow(uint32_t token, uint32_t residual) {
    LoadCandidateRow(token, residual, 0, hidden_);
    SquareSum(hidden_);
    InverseRms(tiling_->normEps);
    NormalizeAndScore(hidden_);
  }

  __aicore__ inline void WaitForOutput() {
    const auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
  }

  __aicore__ inline void WriteScalar(uint64_t slot) {
    auto out = statQue_.template AllocTensor<float>();
    Duplicate(out, 0.0f, FLOAT_BLOCK_ELEMS);
    PipeBarrier<PIPE_V>();
    CopyMetaScalarToLocal(out, scalar_);
    PipeBarrier<PIPE_V>();
    statQue_.EnQue(out);
    out = statQue_.template DeQue<float>();
    DataCopyExtParams copy{1, FLOAT_BLOCK_ELEMS * sizeof(float), 0, 0, 0};
    DataCopyPad(workspace_[slot * FLOAT_BLOCK_ELEMS], out, copy);
    WaitForOutput();
    statQue_.FreeTensor(out);
  }

  __aicore__ inline LocalTensor<float> ReadSlots(uint64_t slot, uint32_t count) {
    auto in = metaQue_.template AllocTensor<float>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * FLOAT_BLOCK_ELEMS * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(in, workspace_[slot * FLOAT_BLOCK_ELEMS], copy, pad);
    metaQue_.EnQue(in);
    return metaQue_.template DeQue<float>();
  }

  __aicore__ inline void InitSlice(uint32_t splits, uint32_t slice, uint32_t& start, uint32_t& count) {
    const uint32_t units = Align(hidden_, VECTOR_FLOAT_ELEMS) / VECTOR_FLOAT_ELEMS;
    const uint32_t base = units / splits;
    const uint32_t tail = units % splits;
    start = (slice * base + (slice < tail ? slice : tail)) * VECTOR_FLOAT_ELEMS;
    count = (base + (slice < tail ? 1 : 0)) * VECTOR_FLOAT_ELEMS;
    if (start + count > hidden_) {
      count = hidden_ - start;
    }
  }

  __aicore__ inline void InitOutputSlice() {
    outputToken_ = core_ / splits_;
    InitSlice(splits_, core_ % splits_, outputStart_, outputCount_);
    outputOffset_ = static_cast<uint64_t>(outputToken_) * hidden_ + outputStart_;
  }

  __aicore__ inline void ReadScores() {
    auto in = ReadSlots(static_cast<uint64_t>(outputToken_) * rows_ * scoreSplits_, rows_ * scoreSplits_);
    Duplicate(scores_, SOFTMAX_PAD, metaElems_);
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < rows_; ++n) {
      // WholeReduceSum permits a compact, four-byte-aligned destination.
      // Each slot contains one score partial and seven initialized zeros.
      // This merges all H slices of one row before the softmax.
      ReduceSumHalfInterval(scores_[n], in[n * scoreSplits_ * FLOAT_BLOCK_ELEMS],
                            static_cast<int32_t>(scoreSplits_ * FLOAT_BLOCK_ELEMS));
    }
    metaQue_.FreeTensor(in);
  }

  __aicore__ inline void Softmax() {
    Duplicate(scratch_, SOFTMAX_PAD, metaElems_);
    PipeBarrier<PIPE_V>();
    CopyCompactFloatsUb(scratch_, scores_, rows_);
    ReduceMaxHalfInterval(scalar_, scratch_, static_cast<int32_t>(rows_));
    BrcbScalarRow1(broadcast_, scalar_);
    SubLastDimRow1NoBrc(scores_, scores_, broadcast_, static_cast<int32_t>(rows_));
    Exp(scores_, scores_, metaAlign_);
    PipeBarrier<PIPE_V>();
    CopyCompactFloatsUb(scratch_, scores_, metaAlign_);
    ReduceSumHalfInterval(scalar_, scratch_, static_cast<int32_t>(rows_));
    Duplicate(scratch_, 1.0f, FLOAT_BLOCK_ELEMS);
    PipeBarrier<PIPE_V>();
    Div(scalar_, scratch_, scalar_, 1);
    PipeBarrier<PIPE_V>();
    BrcbScalarRow1(broadcast_, scalar_);
    MulLastDimRow1NoBrc(scores_, scores_, broadcast_, static_cast<int32_t>(rows_));
    Brcb(broadcast_, scores_, static_cast<uint8_t>(metaAlign_ / FLOAT_BLOCK_ELEMS), {1, MOV_8});
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void EnqueueOutputRow(uint32_t n) {
    if (n < tiling_->numBlocks) {
      const uint64_t offset = static_cast<uint64_t>(outputToken_) * tiling_->blockTokenStride +
                              static_cast<uint64_t>(n) * hidden_ + outputStart_;
      EnqueueRow(bank_, offset, outputCount_);
    } else {
      EnqueueRow(tiling_->fusedAdd != 0 ? prefixOut_ : prefix_, outputOffset_, outputCount_);
    }
  }

  __aicore__ inline void WeightedOutputDoubleBuffer(bool cachedPrefix) {
    Duplicate(acc_, 0.0f, Align(outputCount_, FLOAT_BLOCK_ELEMS));
    PipeBarrier<PIPE_V>();
    EnqueueOutputRow(0);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(outputCount_);
    for (uint32_t n = 0; n < rows_; ++n) {
      const bool residentPrefix = cachedPrefix && n == tiling_->numBlocks;
      LocalTensor<D_IN> in = residentPrefix ? rowBf16_ : inQue_.template DeQue<D_IN>();
      // At most one row is queued, with two physical buffers. The current
      // buffer remains owned until Cast finishes; the next copy uses the other.
      if (n + 1 < rows_ && !(cachedPrefix && n + 1 == tiling_->numBlocks)) {
        EnqueueOutputRow(n + 1);
      }
      CastRowToFp32CounterNoSetMask(row_, in);
      PipeBarrier<PIPE_V>();
      MulAddRowByBrcBlock(acc_, row_, broadcast_[n * FLOAT_BLOCK_ELEMS], outputCount_,
                          Align(outputCount_, FLOAT_BLOCK_ELEMS), false);
      if (!residentPrefix) {
        inQue_.FreeTensor(in);
      }
    }
    SetMaskNorm();
    ResetMask();
  }

  __aicore__ inline void WeightedOutput(bool cachedPrefix = false) {
    if (scoreSplits_ == 1 && splits_ == 1) {
      WeightedOutputDoubleBuffer(cachedPrefix);
      return;
    }
    Duplicate(acc_, 0.0f, Align(outputCount_, FLOAT_BLOCK_ELEMS));
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < rows_; ++n) {
      LocalTensor<D_IN> in;
      if (n < tiling_->numBlocks) {
        const uint64_t offset = static_cast<uint64_t>(outputToken_) * tiling_->blockTokenStride +
                                static_cast<uint64_t>(n) * hidden_ + outputStart_;
        in = ReadRow(bank_, offset, outputCount_);
      } else if (cachedPrefix) {
        in = rowBf16_;
      } else {
        in = ReadRow(tiling_->fusedAdd != 0 ? prefixOut_ : prefix_, outputOffset_, outputCount_);
      }
      CastToFloat(row_, in, outputCount_);
      MulAddRowByBrcBlock(acc_, row_, broadcast_[n * FLOAT_BLOCK_ELEMS], outputCount_,
                          Align(outputCount_, FLOAT_BLOCK_ELEMS));
      PipeBarrier<PIPE_V>();
      if (!cachedPrefix || n < tiling_->numBlocks) {
        inQue_.FreeTensor(in);
      }
    }
  }

  __aicore__ inline void NormalizeOutput(uint64_t outputStatsBase) {
    auto in = ReadSlots(outputStatsBase + static_cast<uint64_t>(outputToken_) * splits_, splits_);
    ReduceSumHalfInterval(scalar_, in, static_cast<int32_t>(splits_ * FLOAT_BLOCK_ELEMS));
    metaQue_.FreeTensor(in);
    InverseRms(tiling_->outputNormEps);
    ApplyOutputNorm();
  }

  __aicore__ inline void ApplyOutputNorm() {
    // row_ still contains this slice of the BF16-rounded mixture.
    BroadcastScalarMulTensor(row_, row_, scalar_, scratch_, broadcast_, outputCount_,
                             Align(outputCount_, FLOAT_BLOCK_ELEMS));
    auto gamma = ReadRow(outputNorm_, outputStart_, outputCount_);
    CastToFloat(weight_, gamma, outputCount_);
    inQue_.FreeTensor(gamma);
    Mul(row_, row_, weight_, outputCount_);
    PipeBarrier<PIPE_V>();
    CastToBf16(rowBf16_, row_, outputCount_);
  }

  __aicore__ inline void WriteRow(const GlobalTensor<D_IN>& gm, uint64_t offset, uint32_t count) {
    auto out = outQue_.template AllocTensor<D_IN>();
    CopyBf16(out, rowBf16_, count);
    outQue_.EnQue(out);
    out = outQue_.template DeQue<D_IN>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(D_IN)), 0, 0, 0};
    DataCopyPad(gm[offset], out, copy);
    WaitForOutput();
    outQue_.FreeTensor(out);
  }

  TPipe* pipe_;
  const AttnResFwdTilingData* tiling_;
  uint32_t core_;
  uint32_t hidden_;
  uint32_t rows_;
  uint32_t splits_;
  uint32_t scoreSplits_;
  uint32_t tileElems_;
  uint32_t metaAlign_;
  uint32_t metaElems_;
  uint32_t scoreStart_{0};
  uint32_t scoreCount_{0};
  uint32_t outputToken_{0};
  uint32_t outputStart_{0};
  uint32_t outputCount_{0};
  uint64_t outputOffset_{0};
  uint64_t outputStatsBase_{0};
  GlobalTensor<D_IN> prefix_;
  GlobalTensor<D_IN> bank_;
  GlobalTensor<D_IN> proj_;
  GlobalTensor<D_IN> norm_;
  GlobalTensor<D_IN> addend_;
  GlobalTensor<D_IN> prefixOut_;
  GlobalTensor<D_IN> output_;
  GlobalTensor<D_IN> materialized_;
  GlobalTensor<D_IN> outputNorm_;
  GlobalTensor<float> workspace_;
  TQue<QuePosition::VECIN, 1> inQue_;
  TQue<QuePosition::VECIN, 1> metaQue_;
  TQue<QuePosition::VECOUT, 1> outQue_;
  TQue<QuePosition::VECOUT, 1> statQue_;
  TBuf<TPosition::VECCALC> rowBf16Buf_;
  TBuf<TPosition::VECCALC> rowBuf_;
  TBuf<TPosition::VECCALC> weightBuf_;
  TBuf<TPosition::VECCALC> accBuf_;
  TBuf<TPosition::VECCALC> scoresBuf_;
  TBuf<TPosition::VECCALC> scratchBuf_;
  TBuf<TPosition::VECCALC> broadcastBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  LocalTensor<D_IN> rowBf16_;
  LocalTensor<float> row_;
  LocalTensor<float> weight_;
  LocalTensor<float> acc_;
  LocalTensor<float> scores_;
  LocalTensor<float> scratch_;
  LocalTensor<float> broadcast_;
  LocalTensor<float> scalar_;
};

}  // namespace AttnResFwd
#endif
