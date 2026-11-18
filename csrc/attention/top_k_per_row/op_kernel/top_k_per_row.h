/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Ascend C kernel for TopKPerRow.
 *
 * Algorithm (per row, per AIV core):
 *   1. the row window [start, end) is streamed in `chunk_size` blocks, each
 *      block padded with -FLT_MAX up to the chunk size;
 *   2. every block is reduced to its own top-k with the Ascend C high level
 *      TopK API, and the block-local positions are turned into *column
 *      indices* with a single scalar `Adds`;
 *   3. blocks are merged `max_chunks` at a time against a running candidate
 *      list (again with TopK over a (1 + max_chunks) * topk_pad row).  The
 *      TopK output is a position inside the concatenated value buffer, so a
 *      single `Gather` maps positions back to the column indices kept in a
 *      layout-identical index buffer;
 *   4. the tail past `min(topk, window_len)` is filled with -1.
 */
#ifndef TOP_K_PER_ROW_H
#define TOP_K_PER_ROW_H

#include "kernel_operator.h"

namespace TopKPerRowNS {
using namespace AscendC;

constexpr uint32_t K_ALIGN = 32;
constexpr float NEG_BIG = -3.402823466e+38f;  // -FLT_MAX

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t MaxU32(uint32_t a, uint32_t b) { return a > b ? a : b; }

template <typename T>
class KernelTopKPerRow {
public:
    __aicore__ inline KernelTopKPerRow() = default;

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR rowStarts, GM_ADDR rowEnds, GM_ADDR out,
                                const TopKPerRowTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        numRows_ = tiling->num_rows;
        width_ = tiling->width;
        topk_ = tiling->topk;
        topkPad_ = tiling->topk_pad;
        rowsPerCore_ = tiling->rows_per_core;
        chunk_ = tiling->chunk_size;
        maxChunks_ = tiling->max_chunks;
        tmpSize_ = tiling->tmp_size;
        topkTilingChunk_ = tiling->topk_tiling_chunk;
        topkTilingMerge_ = tiling->topk_tiling_merge;

        const uint32_t blockIdx = GetBlockIdx();
        rowBegin_ = blockIdx * rowsPerCore_;
        rowEnd_ = rowBegin_ + rowsPerCore_;
        if (rowEnd_ > numRows_) {
            rowEnd_ = numRows_;
        }

        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(logits),
                                  static_cast<uint64_t>(numRows_) * width_);
        startsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowStarts), numRows_);
        endsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowEnds), numRows_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out),
                               static_cast<uint64_t>(numRows_) * topk_);

        if constexpr (sizeof(T) != sizeof(float)) {
            pipe_->InitBuffer(rawBuf_, chunk_ * sizeof(T));
        }
        pipe_->InitBuffer(srcBuf_, chunk_ * sizeof(float));
        pipe_->InitBuffer(mergeValBuf_, (maxChunks_ + 1) * topkPad_ * sizeof(float));
        pipe_->InitBuffer(mergeIdxBuf_, (maxChunks_ + 1) * topkPad_ * sizeof(int32_t));
        pipe_->InitBuffer(curValBuf_, topkPad_ * sizeof(float));
        pipe_->InitBuffer(curIdxBuf_, topkPad_ * sizeof(int32_t));
        pipe_->InitBuffer(posBuf_, topkPad_ * sizeof(int32_t));
        pipe_->InitBuffer(posBytesBuf_, topkPad_ * sizeof(uint32_t));
        // srcIndex operand of AscendC::TopK is not read by the TOPK_NORMAL path,
        // but it must not alias the dst index buffer, so give it its own space.
        const uint32_t dummyLen = MaxU32(chunk_, (maxChunks_ + 1) * topkPad_);
        pipe_->InitBuffer(dummyIdxBuf_, dummyLen * sizeof(int32_t));
        pipe_->InitBuffer(tmpBuf_, tmpSize_);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = rowBegin_; row < rowEnd_; ++row) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline int32_t Clamp(int32_t v, int32_t lo, int32_t hi) const
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // ---------------------------------------------------------------- loads
    // The chunk tail is initialised explicitly instead of through the copy's padding
    // count: DataCopyPadExtParams carries `paddingCount` in a uint8_t, so a tail
    // longer than 255 elements is truncated (e.g. chunk_=32768, validLen=256 keeps
    // 0 of the 32512 pads and the stale UB contents reach TopK). Pre-filling the
    // whole chunk with the same sentinel makes the initialisation independent of
    // the tail length; the copy below only overwrites the valid prefix.
    __aicore__ inline void LoadChunkToFp32(uint64_t rowBase, int32_t colStart, uint32_t validLen)
    {
        LocalTensor<float> src = srcBuf_.Get<float>();
        const T padValue = static_cast<T>(NEG_BIG);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, padValue};
        const int32_t chunkLen = static_cast<int32_t>(chunk_);

        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t written = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        if constexpr (sizeof(T) == sizeof(float)) {
            Duplicate(src, static_cast<float>(padValue), chunkLen);
            SetFlag<HardEvent::V_MTE2>(written);
            WaitFlag<HardEvent::V_MTE2>(written);
            DataCopyPad(src, logitsGm_[rowBase + static_cast<uint64_t>(colStart)], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(ready);
            WaitFlag<HardEvent::MTE2_V>(ready);
        } else {
            LocalTensor<T> raw = rawBuf_.Get<T>();
            Duplicate(raw, padValue, chunkLen);
            SetFlag<HardEvent::V_MTE2>(written);
            WaitFlag<HardEvent::V_MTE2>(written);
            DataCopyPad(raw, logitsGm_[rowBase + static_cast<uint64_t>(colStart)], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(ready);
            WaitFlag<HardEvent::MTE2_V>(ready);
            Cast(src, raw, RoundMode::CAST_NONE, chunk_);
            PipeBarrier<PIPE_V>();
        }
    }

    // ------------------------------------------------------------- top-k ops
    __aicore__ inline void BlockTopK(LocalTensor<float> dstVal, LocalTensor<int32_t> dstIdx,
                                     LocalTensor<float> srcVal, int32_t colOffset)
    {
        LocalTensor<int32_t> dummyIdx = dummyIdxBuf_.Get<int32_t>();
        LocalTensor<bool> finish;
        LocalTensor<uint8_t> tmp = tmpBuf_.Get<uint8_t>();
        TopKInfo info{1, static_cast<int32_t>(chunk_), static_cast<int32_t>(chunk_)};
        AscendC::TopK<float, false, false, false, TopKMode::TOPK_NORMAL>(
            dstVal, dstIdx, srcVal, dummyIdx, finish, tmp, topk_, topkTilingChunk_, info);
        PipeBarrier<PIPE_V>();
        // block-local position -> absolute column index
        Adds(dstIdx, dstIdx, colOffset, topkPad_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeTopK(uint32_t inner)
    {
        LocalTensor<float> mergeVal = mergeValBuf_.Get<float>();
        LocalTensor<float> curVal = curValBuf_.Get<float>();
        LocalTensor<int32_t> curIdx = curIdxBuf_.Get<int32_t>();
        LocalTensor<int32_t> pos = posBuf_.Get<int32_t>();
        LocalTensor<int32_t> dummyIdx = dummyIdxBuf_.Get<int32_t>();
        LocalTensor<bool> finish;
        LocalTensor<uint8_t> tmp = tmpBuf_.Get<uint8_t>();

        TopKInfo info{1, static_cast<int32_t>(inner), static_cast<int32_t>(inner)};
        AscendC::TopK<float, false, false, false, TopKMode::TOPK_NORMAL>(
            curVal, pos, mergeVal, dummyIdx, finish, tmp, topk_, topkTilingMerge_, info);
        PipeBarrier<PIPE_V>();

        // position -> column index (layout of idx buffer mirrors the value buffer)
        LocalTensor<int32_t> mergeIdx = mergeIdxBuf_.Get<int32_t>();
        LocalTensor<uint32_t> posBytes = posBytesBuf_.Get<uint32_t>();
        Muls(posBytes.ReinterpretCast<int32_t>(), pos, static_cast<int32_t>(sizeof(int32_t)), topkPad_);
        PipeBarrier<PIPE_V>();
        Gather(curIdx, mergeIdx, posBytes, static_cast<uint32_t>(0), topkPad_);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------ row driver
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const int32_t w = static_cast<int32_t>(width_);
        int32_t s = Clamp(startsGm_.GetValue(row), 0, w);
        int32_t e = Clamp(endsGm_.GetValue(row), 0, w);
        const uint32_t len = (e > s) ? static_cast<uint32_t>(e - s) : 0U;

        LocalTensor<int32_t> curIdx = curIdxBuf_.Get<int32_t>();
        LocalTensor<float> curVal = curValBuf_.Get<float>();
        LocalTensor<float> mergeVal = mergeValBuf_.Get<float>();
        LocalTensor<int32_t> mergeIdx = mergeIdxBuf_.Get<int32_t>();

        if (len == 0U) {
            Duplicate(curIdx, static_cast<int32_t>(-1), topkPad_);
            PipeBarrier<PIPE_V>();
            WriteOut(row, curIdx);
            return;
        }

        const uint32_t nChunks = (len + chunk_ - 1U) / chunk_;
        const uint64_t rowBase = static_cast<uint64_t>(row) * width_;
        bool haveCur = false;
        uint32_t processed = 0U;

        while (processed < nChunks) {
            const uint32_t groupSize = MinU32(maxChunks_, nChunks - processed);
            const uint32_t dstBase = haveCur ? 1U : 0U;

            for (uint32_t i = 0U; i < groupSize; ++i) {
                const uint32_t off = (processed + i) * chunk_;
                const uint32_t valid = MinU32(chunk_, len - off);
                LoadChunkToFp32(rowBase, s + static_cast<int32_t>(off), valid);
                BlockTopK(mergeVal[(dstBase + i) * topkPad_], mergeIdx[(dstBase + i) * topkPad_],
                          srcBuf_.Get<float>(), s + static_cast<int32_t>(off));
            }

            // merge: keep the running candidates at slot 0 of the merge buffers
            if (haveCur) {
                // keep the running candidates in slot 0; the index buffer keeps
                // the same layout so positions map 1:1 after the merge TopK.
                Adds(mergeVal, curVal, 0.0f, topkPad_);
                Adds(mergeIdx, curIdx, static_cast<int32_t>(0), topkPad_);
                PipeBarrier<PIPE_V>();
            }

            const uint32_t inner = (dstBase + groupSize) * topkPad_;
            MergeTopK(inner);
            haveCur = true;
            processed += groupSize;
        }

        const uint32_t validOut = MinU32(topk_, len);
        if (validOut < topkPad_) {
            Duplicate(curIdx[validOut], static_cast<int32_t>(-1), topkPad_ - validOut);
            PipeBarrier<PIPE_V>();
        }
        WriteOut(row, curIdx);
    }

    // The output row is exactly `topk_` entries wide, and every entry is
    // initialized before this call: the selection itself, then the -1 padding
    // up to topk_pad (>= topk). Rows shorter than topk must therefore write the
    // full padded row -- writing only `validOut` entries would leave the tail of
    // `indices` holding whatever the caller's buffer already contained.
    __aicore__ inline void WriteOut(uint32_t row, LocalTensor<int32_t> idx)
    {
        const uint32_t count = topk_;
        event_t syncV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(syncV);
        WaitFlag<HardEvent::V_MTE3>(syncV);

        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(outGm_[static_cast<uint64_t>(row) * topk_], idx, params);
        event_t syncOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(syncOut);
        WaitFlag<HardEvent::MTE3_V>(syncOut);
    }

    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> srcBuf_;
    TBuf<TPosition::VECCALC> mergeValBuf_;
    TBuf<TPosition::VECCALC> mergeIdxBuf_;
    TBuf<TPosition::VECCALC> curValBuf_;
    TBuf<TPosition::VECCALC> curIdxBuf_;
    TBuf<TPosition::VECCALC> posBuf_;
    TBuf<TPosition::VECCALC> posBytesBuf_;
    TBuf<TPosition::VECCALC> dummyIdxBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<T> logitsGm_;
    GlobalTensor<int32_t> startsGm_;
    GlobalTensor<int32_t> endsGm_;
    GlobalTensor<int32_t> outGm_;

    uint32_t numRows_ = 0;
    uint32_t width_ = 0;
    uint32_t topk_ = 0;
    uint32_t topkPad_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t chunk_ = 0;
    uint32_t maxChunks_ = 0;
    uint32_t tmpSize_ = 0;
    uint32_t rowBegin_ = 0;
    uint32_t rowEnd_ = 0;
    TopkTiling topkTilingChunk_;
    TopkTiling topkTilingMerge_;
};
}  // namespace TopKPerRowNS
#endif
