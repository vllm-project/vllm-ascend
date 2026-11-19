// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_TOKEN_GROUP_H
#define ATTN_RES_FWD_TOKEN_GROUP_H

#include "kernel_operator.h"
#include "../attn_res_fwd_tiling_data.h"
#include "../attn_res_fwd_ub_layout.h"
#include "attn_res_fwd_reload.h"

namespace AttnResFwd {

// Whole-H token groups retain the output owner's original BF16 rows.
// Host dispatch admits only groups with multiple score workers.
template <typename D_IN>
class AttnResFwdTokenGroup {
 public:
  __aicore__ inline AttnResFwdTokenGroup(TPipe* pipe, const AttnResFwdTilingData* tiling)
      : pipe_(pipe), tiling_(tiling) {}

  __aicore__ inline void Init(const AttnResFwdInitParams& params, GM_ADDR workspace) {
    core_ = GetBlockIdx();
    hidden_ = tiling_->hiddenSize;
    rows_ = tiling_->blockCount;
    tileElems_ = tiling_->cooperativeTileElems;
    groupSize_ = tiling_->tokenGroupSize;
    cachedRows_ = tiling_->cachedRows;
    metaAlign_ = Align(rows_, FLOAT_BLOCK_ELEMS);
    metaElems_ = metaAlign_ > VECTOR_FLOAT_ELEMS ? metaAlign_ : VECTOR_FLOAT_ELEMS;
    const uint32_t slotCount = metaAlign_;

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

    pipe_->InitBuffer(inQue_, 2, tileElems_ * sizeof(D_IN));
    // Preserve the validated whole-row UB layout, including this reserved queue.
    pipe_->InitBuffer(outQue_, 1, tileElems_ * sizeof(D_IN));
    pipe_->InitBuffer(statQue_, 1, FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(metaQue_, 1, slotCount * FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(rowBf16Buf_, tileElems_ * sizeof(D_IN));
    // Preserve the established A2/A3 whole-row bank padding.
    const bool alignedPeriod =
        (static_cast<uint64_t>(tileElems_) * sizeof(float)) % COOPERATIVE_UB_ADDRESS_PERIOD_BYTES == 0;
    const uint32_t rowPadding = alignedPeriod ? COOPERATIVE_ROW_PADDING_BYTES : 0;
    pipe_->InitBuffer(rowBuf_, tileElems_ * sizeof(float) + rowPadding);
    pipe_->InitBuffer(weightBuf_, tileElems_ * sizeof(float));
    pipe_->InitBuffer(accBuf_, tileElems_ * sizeof(float));
    pipe_->InitBuffer(scoresBuf_, metaElems_ * sizeof(float));
    pipe_->InitBuffer(scratchBuf_, metaElems_ * sizeof(float));
    pipe_->InitBuffer(broadcastBuf_, metaAlign_ * FLOAT_BLOCK_ELEMS * sizeof(float));
    pipe_->InitBuffer(scalarBuf_, 2 * FLOAT_BLOCK_ELEMS * sizeof(float));
    if (groupSize_ != 0 && core_ % groupSize_ == 0 && cachedRows_ != 0) {
      pipe_->InitBuffer(cachedRowsBuf_, cachedRows_ * tileElems_ * sizeof(D_IN));
      cachedRowsData_ = cachedRowsBuf_.template Get<D_IN>();
    }
    rowBf16_ = rowBf16Buf_.template Get<D_IN>();
    row_ = rowBuf_.template Get<float>();
    weight_ = weightBuf_.template Get<float>();
    acc_ = accBuf_.template Get<float>();
    scores_ = scoresBuf_.template Get<float>();
    scratch_ = scratchBuf_.template Get<float>();
    broadcast_ = broadcastBuf_.template Get<float>();
    scalar_ = scalarBuf_.template Get<float>();
  }

  __aicore__ inline void Process() { ProcessTokenGroup(); }

 private:
  __aicore__ inline void ProcessTokenGroup() {
    const uint32_t token = core_ / groupSize_;
    const uint32_t lane = core_ % groupSize_;
    LoadScoreWeight(0, hidden_);
    if (lane == 0) {
      Duplicate(scores_, SOFTMAX_PAD, metaElems_);
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t position = lane; position < rows_; position += groupSize_) {
      const uint32_t residual = position == 0 ? tiling_->numBlocks : position - 1;
      ScoreRow(token, residual, lane == 0 ? scores_[residual] : scalar_);
      if (lane != 0) {
        WriteScalar(static_cast<uint64_t>(token) * rows_ + residual);
      }
    }
    // Prefix GM emission may overlap read-only scoring of its BF16 source.
    // Retire every pending write before cross-core readiness or output reuse.
    if (prefixWritePending_) {
      WaitForOutput();
      prefixWritePending_ = false;
    }
    // Every launched score worker participates in the readiness barrier.
    SyncAll<true>();
    if (lane == 0) {
      FinishWholeRow(token, true);
    }
  }

  __aicore__ inline void FinishWholeRow(uint32_t token, bool cachedPrefix) {
    outputToken_ = token;
    outputStart_ = 0;
    outputCount_ = hidden_;
    outputOffset_ = static_cast<uint64_t>(token) * hidden_;
    ReadScores();
    Softmax();
    // The group leader retains prefix and its cached original candidate rows.
    WeightedOutput(cachedPrefix);
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

  __aicore__ inline bool RetainCandidate(uint32_t residual) {
    return cachedRows_ != 0 && groupSize_ != 0 && core_ % groupSize_ == 0 && (residual + 1) % groupSize_ == 0 &&
           (residual + 1) / groupSize_ <= cachedRows_;
  }

  __aicore__ inline LocalTensor<D_IN> ReadRetainedCandidate(uint64_t offset, uint32_t residual, uint32_t count) {
    auto row = CachedOutputRow(residual);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(D_IN)), 0, 0, 0};
    DataCopyPadExtParams<D_IN> pad{false, 0, 0, 0};
    DataCopyPad(row, bank_[offset], copy, pad);
    const auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    return row;
  }

  __aicore__ inline bool IsCachedOutput(uint32_t residual, bool cachedPrefix) {
    if (residual == tiling_->numBlocks) {
      return cachedPrefix;
    }
    return cachedRows_ != 0 && groupSize_ != 0 && (residual + 1) % groupSize_ == 0 &&
           (residual + 1) / groupSize_ <= cachedRows_;
  }

  __aicore__ inline LocalTensor<D_IN> CachedOutputRow(uint32_t residual) {
    if (residual == tiling_->numBlocks) {
      return rowBf16_;
    }
    return cachedRowsData_[((residual + 1) / groupSize_ - 1) * tileElems_];
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
    if (tiling_->fusedAdd != 0) {
      // The dequeued prefix owns one physical buffer. Prefetch addend into
      // the other while casting prefix, and avoid the redundant BF16 copy.
      EnqueueRow(addend_, offset, count);
      CastToFloat(row_, in, count);
      inQue_.FreeTensor(in);
      in = inQue_.template DeQue<D_IN>();
      CastToFloat(acc_, in, count);
      inQue_.FreeTensor(in);
      Add(row_, row_, acc_, count);
      PipeBarrier<PIPE_V>();
      CastToBf16(rowBf16_, row_, count);
      WritePrefixRow(prefixOut_, offset, count);
    } else {
      CopyBf16(rowBf16_, in, count);
      inQue_.FreeTensor(in);
    }
    if (tiling_->blockWriteIdx >= 0) {
      const uint64_t bankOffset = static_cast<uint64_t>(token) * tiling_->blockTokenStride +
                                  static_cast<uint64_t>(tiling_->blockWriteIdx) * hidden_ + start;
      WritePrefixRow(bank_, bankOffset, count);
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
      const bool retained = RetainCandidate(residual);
      auto in = retained ? ReadRetainedCandidate(offset, residual, count) : ReadRow(bank_, offset, count);
      CastToFloat(row_, in, count);
      if (!retained) {
        inQue_.FreeTensor(in);
      }
    }
  }

  __aicore__ inline void NormalizeAndScore(uint32_t count, const LocalTensor<float>& score) {
    BroadcastScalarMulTensor(row_, row_, scalar_, scratch_, broadcast_, count, Align(count, FLOAT_BLOCK_ELEMS));
    ClearReductionTail(count);
    Mul(acc_, row_, weight_, count);
    PipeBarrier<PIPE_V>();
    ReduceSumHalfInterval(score, acc_, static_cast<int32_t>(count));
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

  __aicore__ inline void ScoreRow(uint32_t token, uint32_t residual, const LocalTensor<float>& score) {
    LoadCandidateRow(token, residual, 0, hidden_);
    SquareSum(hidden_);
    InverseRms(tiling_->normEps);
    NormalizeAndScore(hidden_, score);
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
    prefixWritePending_ = false;
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

  __aicore__ inline void ReadScores() {
    // Leader-owned scores already reside in scores_. Read only remote slots;
    // those runs stop before each leader-owned candidate and before prefix.
    const uint64_t base = static_cast<uint64_t>(outputToken_) * rows_;
    for (uint32_t start = 0; start < tiling_->numBlocks;) {
      if ((start + 1) % groupSize_ == 0) {
        ++start;
        continue;
      }
      const uint32_t run = groupSize_ - 1 - start % groupSize_;
      const uint32_t count = run < tiling_->numBlocks - start ? run : tiling_->numBlocks - start;
      auto in = ReadSlots(base + start, count);
      for (uint32_t i = 0; i < count; ++i) {
        ReduceSumHalfInterval(scores_[start + i], in[i * FLOAT_BLOCK_ELEMS], FLOAT_BLOCK_ELEMS);
      }
      metaQue_.FreeTensor(in);
      start += count;
    }
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

  __aicore__ inline void WeightedOutputCachedRows(bool cachedPrefix) {
    Duplicate(acc_, 0.0f, Align(outputCount_, FLOAT_BLOCK_ELEMS));
    PipeBarrier<PIPE_V>();
    uint32_t next = 0;
    while (next < rows_ && IsCachedOutput(next, cachedPrefix)) {
      ++next;
    }
    if (next < rows_) {
      EnqueueOutputRow(next++);
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(outputCount_);
    for (uint32_t n = 0; n < rows_; ++n) {
      const bool residentRow = IsCachedOutput(n, cachedPrefix);
      LocalTensor<D_IN> in = residentRow ? CachedOutputRow(n) : inQue_.template DeQue<D_IN>();
      // At most one row is queued, with two physical buffers. The current
      // buffer remains owned until Cast finishes; the next copy uses the other.
      if (!residentRow) {
        while (next < rows_ && IsCachedOutput(next, cachedPrefix)) {
          ++next;
        }
        if (next < rows_) {
          EnqueueOutputRow(next++);
        }
      }
      CastRowToFp32CounterNoSetMask(row_, in);
      PipeBarrier<PIPE_V>();
      MulAddRowByBrcBlock(acc_, row_, broadcast_[n * FLOAT_BLOCK_ELEMS], outputCount_,
                          Align(outputCount_, FLOAT_BLOCK_ELEMS), false);
      if (!residentRow) {
        inQue_.FreeTensor(in);
      }
    }
    SetMaskNorm();
    ResetMask();
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

  __aicore__ inline void WeightedOutput(bool cachedPrefix) {
    if (cachedRows_ == 0) {
      WeightedOutputDoubleBuffer(cachedPrefix);
    } else {
      WeightedOutputCachedRows(cachedPrefix);
    }
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

  __aicore__ inline void WritePrefixRow(const GlobalTensor<D_IN>& gm, uint64_t offset, uint32_t count) {
    // Direct UB output: fence Vector production before MTE3 consumption.
    const auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(D_IN)), 0, 0, 0};
    DataCopyPad(gm[offset], rowBf16_, copy);
    prefixWritePending_ = true;
  }

  __aicore__ inline void WriteRow(const GlobalTensor<D_IN>& gm, uint64_t offset, uint32_t count) {
    // Direct UB output: fence Vector production before MTE3 consumption.
    const auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(D_IN)), 0, 0, 0};
    DataCopyPad(gm[offset], rowBf16_, copy);
    WaitForOutput();
  }

  bool prefixWritePending_{false};
  TPipe* pipe_;
  const AttnResFwdTilingData* tiling_;
  uint32_t core_;
  uint32_t hidden_;
  uint32_t rows_;
  uint32_t tileElems_;
  uint32_t groupSize_{0};
  uint32_t cachedRows_{0};
  uint32_t metaAlign_;
  uint32_t metaElems_;
  uint32_t outputToken_{0};
  uint32_t outputStart_{0};
  uint32_t outputCount_{0};
  uint64_t outputOffset_{0};
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
  TBuf<TPosition::VECCALC> cachedRowsBuf_;
  TBuf<TPosition::VECCALC> rowBuf_;
  TBuf<TPosition::VECCALC> weightBuf_;
  TBuf<TPosition::VECCALC> accBuf_;
  TBuf<TPosition::VECCALC> scoresBuf_;
  TBuf<TPosition::VECCALC> scratchBuf_;
  TBuf<TPosition::VECCALC> broadcastBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  LocalTensor<D_IN> rowBf16_;
  LocalTensor<D_IN> cachedRowsData_;
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
