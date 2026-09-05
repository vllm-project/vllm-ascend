/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vector_paged_attention.h
 * \brief Kernel implementation for VectorPagedAttention
 */
#ifndef VECTOR_PAGED_ATTENTION_H
#define VECTOR_PAGED_ATTENTION_H

#include "kernel_operator.h"

namespace VectorPagedAttentionOp {
using namespace AscendC;

// The declared domain, enforced in the torch adapter and again in tiling:
// 64-wide heads, multi-head (no GQA) and a power-of-two page.
constexpr uint32_t HEAD_DIM = 64;
constexpr uint32_t MAX_BLOCK_SIZE = 128;
constexpr uint32_t MAX_CAPACITY = 4096;
constexpr uint32_t MAX_BLOCK_TABLE = MAX_CAPACITY / 8;
constexpr uint32_t MAX_BATCH = 32;
// One repeat of a float vector instruction is 256 B.
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t BLOCKS_PER_REPEAT = 8;
// Small enough that exp() underflows to zero, large enough not to be a max.
constexpr float MASKED_SCORE = -1.0e30F;

class KernelVectorPagedAttention {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache,
                                GM_ADDR blockTable, GM_ADDR seqLens, GM_ADDR attnOut,
                                const VectorPagedAttentionTilingData &tiling)
    {
        batch_ = tiling.batch;
        numHeads_ = tiling.numHeads;
        headDim_ = tiling.headDim;
        blockSize_ = tiling.blockSize;
        maxBlocks_ = tiling.maxBlocks;
        kvStride_ = tiling.kvStride;
        kvCapacity_ = tiling.kvCapacity;
        scale_ = tiling.scale;

        // One core owns one (request, head): it reads that head's whole KV
        // prefix and writes its own 64 outputs, so no core ever waits for
        // another and there is no combine pass.
        const uint32_t task = GetBlockIdx();
        batchIdx_ = task / numHeads_;
        headIdx_ = task % numHeads_;

        const uint64_t kvElems = static_cast<uint64_t>(tiling.numBlocks) * blockSize_ * kvStride_;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query),
                                 static_cast<uint64_t>(batch_) * numHeads_ * headDim_);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(keyCache), kvElems);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(valueCache), kvElems);
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable),
                                      static_cast<uint64_t>(batch_) * maxBlocks_);
        seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLens), batch_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(attnOut),
                               static_cast<uint64_t>(batch_) * numHeads_ * headDim_);

        pipe_.InitBuffer(seqBuf_, MAX_BATCH * sizeof(int32_t));
        pipe_.InitBuffer(blockTableBuf_, MAX_BLOCK_TABLE * sizeof(int32_t));
        pipe_.InitBuffer(qHalfBuf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(qBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(tileHalfBuf_, MAX_BLOCK_SIZE * HEAD_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(tileBuf_, MAX_BLOCK_SIZE * HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(scoreBuf_, MAX_CAPACITY * sizeof(float));
        pipe_.InitBuffer(weightBuf_, MAX_BLOCK_SIZE * BLOCKS_PER_REPEAT * sizeof(float));
        pipe_.InitBuffer(accumBuf_, HEAD_DIM * sizeof(float));
        pipe_.InitBuffer(outHalfBuf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(reduceBuf_, FP32_PER_REPEAT * sizeof(float));
        pipe_.InitBuffer(workBuf_, 2 * FP32_PER_REPEAT * BLOCKS_PER_REPEAT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> seqLocal = seqBuf_.Get<int32_t>();
        LocalTensor<int32_t> blockIds = blockTableBuf_.Get<int32_t>();
        LocalTensor<bfloat16_t> qHalf = qHalfBuf_.Get<bfloat16_t>();
        const DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};
        DataCopyPad(seqLocal, seqLensGm_,
                    DataCopyExtParams{1, batch_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0}, intPad);
        DataCopyPad(blockIds, blockTableGm_[batchIdx_ * maxBlocks_],
                    DataCopyExtParams{1, maxBlocks_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0}, intPad);
        DataCopy(qHalf, queryGm_[(batchIdx_ * numHeads_ + headIdx_) * headDim_], headDim_);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int32_t length = seqLocal.GetValue(batchIdx_);
        if (length < 1) length = 1;
        if (length > static_cast<int32_t>(kvCapacity_)) length = static_cast<int32_t>(kvCapacity_);
        const uint32_t seqLen = static_cast<uint32_t>(length);
        // Only the pages the sequence actually occupies are read. The declared
        // capacity bounds the buffers; it does not bound the traffic, which is
        // the whole reason this operator can be cheaper than a fixed-capacity
        // general attention.
        const uint32_t pages = (seqLen + blockSize_ - 1) / blockSize_;
        const uint32_t covered = pages * blockSize_;

        LocalTensor<float> q = qBuf_.Get<float>();
        Cast(q, qHalf, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        // The scale rides on the query, so it costs one 64-wide op instead of
        // one pass over every score.
        Muls(q, q, scale_, headDim_);
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> tileHalf = tileHalfBuf_.Get<bfloat16_t>();
        LocalTensor<float> tile = tileBuf_.Get<float>();
        LocalTensor<float> scores = scoreBuf_.Get<float>();
        // A page holds `blockSize` rows of `kvStride` elements; this head owns
        // `headDim` of each row, so one strided burst per page reads it all.
        const DataCopyParams pageCopy{
            static_cast<uint16_t>(blockSize_),
            static_cast<uint16_t>(headDim_ * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((kvStride_ - headDim_) * sizeof(bfloat16_t) / 32),
            0};
        // Every repeat is one KV row against the whole query: src1 walks its
        // eight blocks (blkStride 1) and restarts each repeat (repStride 0).
        const BinaryRepeatParams queryBroadcast{1, 1, 1, BLOCKS_PER_REPEAT, BLOCKS_PER_REPEAT, 0};

        for (uint32_t page = 0; page < pages; ++page) {
            LoadPage(tileHalf, keyGm_, blockIds.GetValue(page), pageCopy);
            Cast(tile, tileHalf, RoundMode::CAST_NONE, blockSize_ * headDim_);
            PipeBarrier<PIPE_V>();
            Mul(tile, tile, q, FP32_PER_REPEAT, blockSize_, queryBroadcast);
            PipeBarrier<PIPE_V>();
            WholeReduceSum<float>(scores[page * blockSize_], tile, FP32_PER_REPEAT,
                                  blockSize_, 1, 1, BLOCKS_PER_REPEAT);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        // Mask the tail of the last page. A vector instruction's operand has
        // to start on a 32 B boundary, and `seqLen` is an arbitrary number of
        // floats into the buffer, so the few elements below the next boundary
        // are written as scalars and the aligned remainder in one go.
        if (covered > seqLen) {
            const uint32_t aligned = (seqLen + 7U) & ~7U;
            if (aligned > seqLen) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t index = seqLen; index < aligned && index < covered; ++index) {
                    scores.SetValue(index, MASKED_SCORE);
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            if (covered > aligned) {
                Duplicate(scores[aligned], MASKED_SCORE, static_cast<int32_t>(covered - aligned));
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        ReduceMax<float>(reduce, scores, work, static_cast<int32_t>(covered), false);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float peak = reduce.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(scores, scores, -peak, static_cast<int32_t>(covered));
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, static_cast<int32_t>(covered));
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(reduce, scores, work, static_cast<int32_t>(covered));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float denominator = reduce.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        LocalTensor<float> accum = accumBuf_.Get<float>();
        LocalTensor<float> weights = weightBuf_.Get<float>();
        Duplicate(accum, 0.0F, static_cast<int32_t>(headDim_));
        PipeBarrier<PIPE_V>();
        // Each repeat of this multiply wants one score replicated across the
        // whole 64-wide row; Brcb puts it in a block and src1BlkStride=0 makes
        // all eight blocks of the repeat read that one block.
        const BinaryRepeatParams weightBroadcast{1, 1, 0, BLOCKS_PER_REPEAT, BLOCKS_PER_REPEAT, 1};
        for (uint32_t page = 0; page < pages; ++page) {
            LoadPage(tileHalf, valueGm_, blockIds.GetValue(page), pageCopy);
            Cast(tile, tileHalf, RoundMode::CAST_NONE, blockSize_ * headDim_);
            PipeBarrier<PIPE_V>();
            Brcb(weights, scores[page * blockSize_],
                 static_cast<uint8_t>(blockSize_ / BLOCKS_PER_REPEAT),
                 BrcbRepeatParams{1, BLOCKS_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            Mul(tile, tile, weights, FP32_PER_REPEAT, blockSize_, weightBroadcast);
            PipeBarrier<PIPE_V>();
            // Fold the page's rows into row 0 by halving; a page holds a
            // power-of-two number of rows, so this needs no tail handling.
            for (uint32_t half = blockSize_ >> 1; half > 0; half >>= 1) {
                Add(tile, tile, tile[half * headDim_], FP32_PER_REPEAT, half,
                    BinaryRepeatParams{1, 1, 1, BLOCKS_PER_REPEAT, BLOCKS_PER_REPEAT, BLOCKS_PER_REPEAT});
                PipeBarrier<PIPE_V>();
            }
            Add(accum, accum, tile, static_cast<int32_t>(headDim_));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        Muls(accum, accum, 1.0F / denominator, static_cast<int32_t>(headDim_));
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> outHalf = outHalfBuf_.Get<bfloat16_t>();
        Cast(outHalf, accum, RoundMode::CAST_RINT, headDim_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm_[(batchIdx_ * numHeads_ + headIdx_) * headDim_], outHalf, headDim_);
    }

private:
    __aicore__ inline void LoadPage(const LocalTensor<bfloat16_t> &dst,
                                    const GlobalTensor<bfloat16_t> &cache,
                                    int32_t blockId, const DataCopyParams &params)
    {
        const uint64_t base =
            static_cast<uint64_t>(blockId) * blockSize_ * kvStride_ + headIdx_ * headDim_;
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopy(dst, cache[base], params);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> seqBuf_;
    TBuf<TPosition::VECCALC> blockTableBuf_;
    TBuf<TPosition::VECCALC> qHalfBuf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> tileHalfBuf_;
    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> accumBuf_;
    TBuf<TPosition::VECCALC> outHalfBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<bfloat16_t> outGm_;
    uint32_t batch_ = 1;
    uint32_t numHeads_ = 1;
    uint32_t headDim_ = HEAD_DIM;
    uint32_t blockSize_ = MAX_BLOCK_SIZE;
    uint32_t maxBlocks_ = 1;
    uint32_t kvStride_ = HEAD_DIM;
    uint32_t kvCapacity_ = MAX_BLOCK_SIZE;
    uint32_t batchIdx_ = 0;
    uint32_t headIdx_ = 0;
    float scale_ = 1.0F;
};
}  // namespace VectorPagedAttentionOp

#endif  // VECTOR_PAGED_ATTENTION_H
