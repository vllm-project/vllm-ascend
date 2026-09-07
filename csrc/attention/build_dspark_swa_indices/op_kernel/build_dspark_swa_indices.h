/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef BUILD_DSPARK_SWA_INDICES_H
#define BUILD_DSPARK_SWA_INDICES_H

#include "kernel_operator.h"

namespace BuildDsparkSwaIndices {
using namespace AscendC;

constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t INVALID_BLOCK_NUM = 0xFFFFFFFFU;  // sentinel for prev_block_num

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

__aicore__ inline uint32_t Int32BytesU32(uint32_t elems)
{
    return elems * static_cast<uint32_t>(sizeof(int32_t));
}

__aicore__ inline void PipeMte2ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

__aicore__ inline void PipeMte3ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

__aicore__ inline void PipeSToMte3()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventID);
}

// Field order MUST match TILING_DATA_FIELD_DEF in build_dspark_swa_indices_tiling.h
struct BuildDsparkSwaIndicesTilingData {
    uint32_t numReqs;
    uint32_t numDecodeTokens;
    uint32_t numSpeculativeTokens;
    uint32_t windowSize;
    uint32_t blockSize;
    uint32_t indexWidth;
    uint32_t blockTableStride;
    uint32_t usedCoreNum;
};

class BuildDsparkSwaIndicesKernel {
public:
    __aicore__ inline BuildDsparkSwaIndicesKernel() {}

    __aicore__ inline void Init(BuildDsparkSwaIndicesTilingData* tilingData, TPipe* pipe)
    {
        numReqs_ = tilingData->numReqs;
        numDecodeTokens_ = tilingData->numDecodeTokens;
        numSpeculativeTokens_ = tilingData->numSpeculativeTokens;
        windowSize_ = tilingData->windowSize;
        blockSize_ = tilingData->blockSize;
        indexWidth_ = tilingData->indexWidth;
        blockTableStride_ = tilingData->blockTableStride;

        qslBytes_ = AlignUpU32(Int32BytesU32(numReqs_ + 1), ALIGN_BYTES);
        seqLensBytes_ = AlignUpU32(Int32BytesU32(numReqs_), ALIGN_BYTES);
        blockTableBytes_ = AlignUpU32(Int32BytesU32(1), ALIGN_BYTES);
        slotRowBytes_ = AlignUpU32(Int32BytesU32(indexWidth_), ALIGN_BYTES);

        pipe->InitBuffer(qslBuf_, qslBytes_);
        pipe->InitBuffer(seqLensBuf_, seqLensBytes_);
        pipe->InitBuffer(blockTableBuf_, blockTableBytes_);
        pipe->InitBuffer(slotRowBuf_, slotRowBytes_);
    }

    __aicore__ inline void Process(
        GM_ADDR kvBlockTable,
        GM_ADDR queryStartLoc,
        GM_ADDR seqLens,
        GM_ADDR perTokenSlots,
        GM_ADDR workspace)
    {
        kvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kvBlockTable));
        queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
        seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
        perTokenSlotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(perTokenSlots));

        // Bulk-load metadata once per core into UB; all per-request access is
        // then scalar GetValue from UB (S pipe), avoiding repeated GM reads.
        LocalTensor<int32_t> qslLocal = qslBuf_.Get<int32_t>();
        LocalTensor<int32_t> seqLensLocal = seqLensBuf_.Get<int32_t>();
        DataCopyExtParams qslCopyParams{1, Int32BytesU32(numReqs_ + 1), 0, 0, 0};
        DataCopyExtParams seqLensCopyParams{1, Int32BytesU32(numReqs_), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(qslLocal, queryStartLocGm_, qslCopyParams, padParams);
        DataCopyPad(seqLensLocal, seqLensGm_, seqLensCopyParams, padParams);
        PipeMte2ToS();

        uint32_t begin = 0;
        uint32_t end = 0;
        SplitRange(numReqs_, begin, end);

        for (uint32_t reqIdx = begin; reqIdx < end; ++reqIdx) {
            ProcessRequest(qslLocal, seqLensLocal, reqIdx);
        }
    }

private:
    __aicore__ inline void SplitRange(uint32_t totalReqs, uint32_t& begin, uint32_t& end)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = MaxU32(GetBlockNum(), 1);
        uint32_t reqsPerBlock = (totalReqs + blockNum - 1) / blockNum;
        begin = MinU32(blockIdx * reqsPerBlock, totalReqs);
        end = MinU32(begin + reqsPerBlock, totalReqs);
    }

    __aicore__ inline void ProcessRequest(
        LocalTensor<int32_t>& qslLocal,
        LocalTensor<int32_t>& seqLensLocal,
        uint32_t reqIdx)
    {
        int32_t seqLen = seqLensLocal.GetValue(reqIdx);
        int32_t queryLen = qslLocal.GetValue(reqIdx + 1) - qslLocal.GetValue(reqIdx);

        // visible_len = min(seq_len, query_len + window_size)
        int32_t visibleLen = MinI32(seqLen, queryLen + static_cast<int32_t>(windowSize_));
        // start_pos = seq_len - visible_len  (= max(seq_len - query_len - window_size, 0))
        int32_t startPos = seqLen - visibleLen;

        LocalTensor<int32_t> slotRowLocal = slotRowBuf_.Get<int32_t>();
        LocalTensor<int32_t> blockTableLocal = blockTableBuf_.Get<int32_t>();
        DataCopyPadExtParams<int32_t> blockPadParams{true, 0, 0, 0};

        uint32_t prevBlockNum = INVALID_BLOCK_NUM;
        for (uint32_t col = 0; col < indexWidth_; ++col) {
            if (col >= static_cast<uint32_t>(visibleLen)) {
                // Padded column — every token in the draft block shares the
                // same visible window, so out-of-range cols are masked to -1.
                slotRowLocal.SetValue(col, -1);
                continue;
            }
            int32_t pos = startPos + static_cast<int32_t>(col);
            uint32_t blockNum = static_cast<uint32_t>(pos) / blockSize_;
            if (blockNum != prevBlockNum) {
                // Clamp to valid block-table columns (same as the PyTorch
                // path's safe_nums = block_nums.clamp(0, stride-1)) so the
                // GM read never goes out of bounds.
                uint32_t safeBn = MinU32(blockNum, blockTableStride_ - 1);
                uint64_t gmOffset = static_cast<uint64_t>(reqIdx) * blockTableStride_ + safeBn;
                DataCopyExtParams blockCopyParams{1, Int32BytesU32(1), 0, 0, 0};
                DataCopyPad(blockTableLocal, kvBlockTableGm_[gmOffset], blockCopyParams, blockPadParams);
                PipeMte2ToS();
                prevBlockNum = blockNum;
            }
            int32_t blockId = blockTableLocal.GetValue(0);
            int32_t blockOffset = pos % static_cast<int32_t>(blockSize_);
            int32_t slot = blockId * static_cast<int32_t>(blockSize_) + blockOffset;
            slotRowLocal.SetValue(col, slot);
        }

        // Replicate the single computed row for every token in the draft
        // block (non-causal SWA: all draft tokens share the same window);
        // equivalent to the PyTorch path's repeat_interleave.
        PipeSToMte3();
        uint32_t copies = static_cast<uint32_t>(MaxI32(queryLen, 0));
        DataCopyExtParams outCopyParams{1, Int32BytesU32(indexWidth_), 0, 0, 0};
        for (uint32_t copyIdx = 0; copyIdx < copies; ++copyIdx) {
            uint32_t outputRow = reqIdx * numSpeculativeTokens_ + copyIdx;
            if (outputRow >= numDecodeTokens_) {
                break;
            }
            uint64_t gmOutOffset = static_cast<uint64_t>(outputRow) * indexWidth_;
            DataCopyPad(perTokenSlotsGm_[gmOutOffset], slotRowLocal, outCopyParams);
            PipeMte3ToS();
        }
    }

    __aicore__ inline int32_t MinI32(int32_t lhs, int32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline int32_t MaxI32(int32_t lhs, int32_t rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }

private:
    uint32_t numReqs_{0};
    uint32_t numDecodeTokens_{0};
    uint32_t numSpeculativeTokens_{0};
    uint32_t windowSize_{0};
    uint32_t blockSize_{0};
    uint32_t indexWidth_{0};
    uint32_t blockTableStride_{0};
    uint32_t qslBytes_{0};
    uint32_t seqLensBytes_{0};
    uint32_t blockTableBytes_{0};
    uint32_t slotRowBytes_{0};

    TBuf<TPosition::VECCALC> qslBuf_;
    TBuf<TPosition::VECCALC> seqLensBuf_;
    TBuf<TPosition::VECCALC> blockTableBuf_;
    TBuf<TPosition::VECCALC> slotRowBuf_;

    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<int32_t> perTokenSlotsGm_;
};
}  // namespace BuildDsparkSwaIndices

#endif
