/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_decode_update_service_vector.h
 * \brief
 */
#ifndef lightning_indexer_decode_update_SERVICE_VECTOR_H
#define lightning_indexer_decode_update_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_decode_update_common.h"
#include "lightning_indexer_decode_update_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t OUTPUT_CAPACITY = 16384;
constexpr uint32_t FIRST_FILL_PASS_COUNT = OUTPUT_CAPACITY / BASE_TOPK;
constexpr uint32_t OUTPUT_CLEAR_CHUNK = 8192;
constexpr uint32_t REQUEST_METADATA_TAIL_OFFSET = B32_BLOCK_ALIGN_NUM;
constexpr uint32_t REQUEST_METADATA_SCALARS = REQUEST_METADATA_TAIL_OFFSET + 2;
constexpr uint32_t DCACHE_LINE_INT32 = 64 / sizeof(int32_t);
constexpr uint32_t S2_BASE_SIZE = 512;
constexpr uint32_t MAX_G_SIZE = 64;
constexpr uint32_t GROUP_INNER = 16;
constexpr uint32_t PAYLOAD_BUF_SLOTS = 4;
constexpr uint32_t EVICT_CANDIDATE_CAP = BASE_TOPK;
constexpr uint32_t SORT_TMP_FLOATS = 512 * 8;
constexpr uint32_t CHUNK_PAIR_FLOATS = 512 * VALUE_AND_INDEX_NUM;
constexpr uint32_t TOPK_PAIR_FLOATS = BASE_TOPK * VALUE_AND_INDEX_NUM;
constexpr uint32_t EVICT_PAIR_FLOATS = EVICT_CANDIDATE_CAP * VALUE_AND_INDEX_NUM;
constexpr uint32_t SORTED_SCRATCH_FLOATS = SORT_TMP_FLOATS + CHUNK_PAIR_FLOATS;
constexpr uint32_t SORT_BUFFER_FLOATS = TOPK_PAIR_FLOATS + EVICT_PAIR_FLOATS + SORTED_SCRATCH_FLOATS;

#ifndef LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS
#define LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS 8
#endif

static_assert(LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS >= 0,
              "LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS must be non-negative");
static_assert(LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS <= 512,
              "LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS must not exceed 512");
constexpr uint32_t EVICT_EXTRA_SCAN_CHUNKS = LI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS;

__aicore__ inline uint32_t HashEvictScanSeed(uint32_t actualSeqLen, uint32_t cacheRowIdx)
{
    uint32_t value = actualSeqLen ^ ((cacheRowIdx + 1U) * 0x9e3779b9U);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

template <typename LIT>
class LIVector {
public:
    using K_T = typename LIT::keyType;

    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(uint32_t kSeqSize, uint32_t qHeadNum, uint32_t cacheRowWidth);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<int32_t> cacheSlotsGm,
                                                GlobalTensor<int32_t> topkIndexGm,
                                                GlobalTensor<int32_t> topkSlotsGm,
                                                GlobalTensor<int32_t> missCountGm,
                                                GlobalTensor<int32_t> tailInfoGm,
                                                GlobalTensor<float> scoresGm);
    __aicore__ inline void CleanInvalidOutput(int64_t outputOffset, uint32_t bIdx);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> tailInfoGm;
    GlobalTensor<float> scoresGm;

private:
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> payloadBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> evictCandidateUb_;
    LocalTensor<float> SortedBasicBlock_;

    static constexpr int32_t s2BaseSize_ = S2_BASE_SIZE;
    uint32_t scoreStride_ = 0;
    uint32_t qHeadNum_ = 0;
    uint32_t outerGroupNum_ = 0;
    uint32_t cacheRowWidth_ = 0;
    uint32_t metadataIndex_ = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    __aicore__ inline void StartPayloadCopy(LocalTensor<int32_t> &payloadLocal, uint32_t cacheRowIdx,
                                            int32_t s2BaseIdx,
                                            int32_t validLen, int32_t alignedLen);
    __aicore__ inline void FinishPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx, int32_t validLen);
    __aicore__ inline void BuildIndexPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx,
                                             int32_t validLen);
    __aicore__ inline void SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                          const LocalTensor<float> &keyLocal,
                                          const LocalTensor<int32_t> &payloadBase,
                                          LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void WriteScoreChunk(uint32_t bIdx, int32_t s2BaseIdx,
                                           const LocalTensor<float> &scoreLocal, int32_t alignedLen);
    __aicore__ inline void SortEvictCandidateChunk(uint32_t bIdx, uint32_t cacheRowIdx, uint32_t chunkIdx,
                                                   uint32_t actualSeqLen,
                                                   const LocalTensor<float> &pairOut,
                                                   LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void MergeEvictCandidateChunk(const LocalTensor<float> &chunkPairLocal,
                                                     uint32_t candidateCap,
                                                     LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void MergeExtraEvictChunks512(uint32_t bIdx, uint32_t cacheRowIdx,
                                                    uint32_t actualSeqLen, uint32_t startChunk,
                                                    uint32_t chunkNum, uint32_t firstScanOffset,
                                                    uint32_t extraChunks, LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void FindEvictCandidates(uint32_t bIdx, uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                               uint32_t scanSeed, uint32_t missCount, uint32_t candidateCap,
                                               float thresholdScore, LocalTensor<float> &tmpSortBuf);
    __aicore__ inline uint32_t CopyDecodedPayloadOut(int64_t outOffset, const LocalTensor<uint32_t> &payloadLocal,
                                                     const LocalTensor<int32_t> &indexLocal,
                                                     const LocalTensor<int32_t> &slotLocal,
                                                     const LocalTensor<int32_t> &scratchLocal,
                                                     int64_t count, bool mayHaveInvalid);
    __aicore__ inline bool IsTopKIndex(const LocalTensor<int32_t> &indexLocal, uint32_t candidateIndex,
                                       uint32_t count) const;
    __aicore__ inline bool FindFallbackEvict(uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                             const LocalTensor<int32_t> &indexLocal, uint32_t &scanCursor,
                                             uint32_t &evictIndex, int32_t &evictSlot);
    __aicore__ inline bool CacheRowsShareDCacheLine() const;
    __aicore__ inline void FlushCacheSlotsDCache();
    __aicore__ inline void WriteCacheSlot(uint64_t offset, int32_t value,
                                          const LocalTensor<int32_t> &scalarLocal);
    __aicore__ inline void UpdateCacheAndWriteTopkSlots(uint32_t bIdx, uint32_t cacheRowIdx,
                                                         int64_t outOffset, uint32_t actualSeqLen, uint32_t scanSeed,
                                                         float thresholdScore, uint32_t missCount,
                                                         const LocalTensor<int32_t> &indexLocal,
                                                         const LocalTensor<int32_t> &topkSlotsLocal,
                                                         const LocalTensor<int32_t> &scratchLocal);
    __aicore__ inline void ClearOutputTail(int64_t outOffset);
    __aicore__ inline void WriteRequestMetadata(uint32_t bIdx, int32_t missCount,
                                                 int32_t tailSlotStart, int32_t tailTokenCount,
                                                 const LocalTensor<int32_t> &scalarLocal);
    __aicore__ inline void WriteNonOffloadedOutput(const LICommon::RunInfo &info,
                                                   const LocalTensor<float> &valueLocal);
    __aicore__ inline void AccumulateColdTopKChunk(uint32_t bIdx, uint32_t chunkIdx,
                                                   uint32_t actualSeqLen,
                                                   LocalTensor<float> &tmpSortBuf);
    __aicore__ inline void ProcessFirstFill(const LICommon::RunInfo &info,
                                            const LocalTensor<float> &valueLocal);
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }
    uint32_t outNeedBufSize = TOPK_PAIR_FLOATS * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + GROUP_INNER * S2_BASE_SIZE * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    uint32_t clearOutputSize = (OUTPUT_CLEAR_CHUNK + REQUEST_METADATA_SCALARS) * sizeof(int32_t);
    outNeedBufSize = clearOutputSize > outNeedBufSize ? clearOutputSize : outNeedBufSize;

    pipe->InitBuffer(inQueue_, 2,
                     GROUP_INNER * S2_BASE_SIZE * sizeof(float) + S2_BASE_SIZE * sizeof(float));
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
    pipe->InitBuffer(sortOutBuf_, SORT_BUFFER_FLOATS * sizeof(float));
    pipe->InitBuffer(indexBuf_, S2_BASE_SIZE * sizeof(int32_t));
    pipe->InitBuffer(payloadBuf_, S2_BASE_SIZE * PAYLOAD_BUF_SLOTS * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, S2_BASE_SIZE * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, GROUP_INNER * 8 * sizeof(float));

    //
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    evictCandidateUb_ = globalTopkUb_[TOPK_PAIR_FLOATS];
    SortedBasicBlock_ = evictCandidateUb_[EVICT_PAIR_FLOATS];

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, S2_BASE_SIZE);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(uint32_t kSeqSize, uint32_t qHeadNum,
                                                 uint32_t cacheRowWidth)
{
    scoreStride_ = CeilDiv(kSeqSize, static_cast<uint32_t>(s2BaseSize_)) *
                   static_cast<uint32_t>(s2BaseSize_);
    qHeadNum_ = qHeadNum;
    outerGroupNum_ = qHeadNum / GROUP_INNER;
    cacheRowWidth_ = cacheRowWidth;
    metadataIndex_ = cacheRowWidth - 1U;
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::CacheRowsShareDCacheLine() const
{
    return cacheRowWidth_ % DCACHE_LINE_INT32 != 0U;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FlushCacheSlotsDCache()
{
    PipeBarrier<PIPE_ALL>();
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(cacheSlotsGm);
    __asm__ __volatile__("");
    PipeBarrier<PIPE_ALL>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteCacheSlot(uint64_t offset, int32_t value,
                                                     const LocalTensor<int32_t> &scalarLocal)
{
    if (!CacheRowsShareDCacheLine()) {
        cacheSlotsGm.SetValue(offset, value);
        return;
    }

    // Unaligned row strides share boundary cache lines across cores; DMA avoids stale DCache writeback.
    scalarLocal.SetValue(0, value);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(cacheSlotsGm[offset], scalarLocal, 1);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                    GlobalTensor<int32_t> cacheSlotsGm, GlobalTensor<int32_t> topkIndexGm,
                                    GlobalTensor<int32_t> topkSlotsGm, GlobalTensor<int32_t> missCountGm,
                                    GlobalTensor<int32_t> tailInfoGm,
                                    GlobalTensor<float> scoresGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkIndexGm = topkIndexGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
    this->tailInfoGm = tailInfoGm;
    this->scoresGm = scoresGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t outputOffset, uint32_t bIdx)
{
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> outputLocal = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(outputLocal, LICommon::ConstInfo::INVALID_IDX, OUTPUT_CLEAR_CHUNK);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    for (uint32_t offset = 0; offset < OUTPUT_CAPACITY; offset += OUTPUT_CLEAR_CHUNK) {
        LIServiceVec::CopyOut(topkIndexGm[outputOffset + offset], outputLocal, OUTPUT_CLEAR_CHUNK);
        LIServiceVec::CopyOut(topkSlotsGm[outputOffset + offset], outputLocal, OUTPUT_CLEAR_CHUNK);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    LocalTensor<int32_t> scalarLocal = outputLocal[OUTPUT_CLEAR_CHUNK];
    WriteRequestMetadata(bIdx, 0, LICommon::ConstInfo::INVALID_IDX, 0, scalarLocal);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::StartPayloadCopy(LocalTensor<int32_t> &payloadLocal, uint32_t cacheRowIdx,
                                                       int32_t s2BaseIdx, int32_t validLen, int32_t alignedLen)
{
    if (validLen < alignedLen) {
        Duplicate(payloadLocal, LICommon::ConstInfo::INVALID_IDX, alignedLen);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    }
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheRowWidth_;
    DataCopyPad(payloadLocal, cacheSlotsGm[rowBase + static_cast<uint32_t>(s2BaseIdx)],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FinishPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx,
                                                    int32_t validLen)
{
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(), payloadLocal.template ReinterpretCast<uint32_t>(),
              INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal, s2BaseIdx, validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::BuildIndexPayload(LocalTensor<int32_t> &payloadLocal, int32_t s2BaseIdx,
                                                        int32_t validLen)
{
    if (validLen < s2BaseSize_) {
        Duplicate(payloadLocal, LICommon::ConstInfo::INVALID_IDX, s2BaseSize_);
        PipeBarrier<PIPE_V>();
    }
    Adds(payloadLocal, globalTopkIndice_, s2BaseIdx, validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortTopKBySlot(const LocalTensor<float> &pairLocal,
                                                     const LocalTensor<float> &keyLocal,
                                                     const LocalTensor<int32_t> &payloadBase,
                                                     LocalTensor<float> &tmpSortBuf)
{
    for (uint32_t blockIdx = 0; blockIdx < PAYLOAD_BUF_SLOTS; ++blockIdx) {
        uint32_t pairOffset = blockIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM;
        LocalTensor<int32_t> payloadLocal = payloadBase[blockIdx * s2BaseSize_];
        ExtractIndex(payloadLocal.template ReinterpretCast<uint32_t>(),
                     pairLocal[pairOffset].template ReinterpretCast<uint32_t>(), s2BaseSize_);
        BuildHitMissKey(keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), s2BaseSize_);
        SortByKeyWithPayload512(pairLocal[pairOffset], keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(),
                                tmpSortBuf, s2BaseSize_ / BLOCK_BYTES);
    }

    MrgBasicBlock(tmpSortBuf, pairLocal, PAYLOAD_BUF_SLOTS, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    DataCopy(pairLocal, tmpSortBuf, BASE_TOPK * VALUE_AND_INDEX_NUM);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteScoreChunk(uint32_t bIdx, int32_t s2BaseIdx,
                                                      const LocalTensor<float> &scoreLocal, int32_t alignedLen)
{
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(scoresGm[bIdx * scoreStride_ + static_cast<uint32_t>(s2BaseIdx)], scoreLocal, alignedLen);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortEvictCandidateChunk(uint32_t bIdx, uint32_t cacheRowIdx,
                                                              uint32_t chunkIdx,
                                                              uint32_t actualSeqLen,
                                                              const LocalTensor<float> &pairOut,
                                                              LocalTensor<float> &tmpSortBuf)
{
    uint32_t s2BaseIdx = chunkIdx * static_cast<uint32_t>(s2BaseSize_);
    uint32_t validLen = (s2BaseIdx + static_cast<uint32_t>(s2BaseSize_) > actualSeqLen)
                            ? (actualSeqLen - s2BaseIdx)
                            : static_cast<uint32_t>(s2BaseSize_);

    LocalTensor<float> scoreLocal = reduceOutBuf_.Get<float>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    DataCopyPad(scoreLocal, scoresGm[bIdx * scoreStride_ + s2BaseIdx],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(s2BaseSize_ * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});

    LocalTensor<int32_t> payloadLocal = payloadBuf_.Get<int32_t>();
    if (validLen < static_cast<uint32_t>(s2BaseSize_)) {
        Duplicate(payloadLocal, LICommon::ConstInfo::INVALID_IDX, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    }
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheRowWidth_;
    uint8_t payloadRightPadding = static_cast<uint8_t>(
        (B32_BLOCK_ALIGN_NUM - validLen % B32_BLOCK_ALIGN_NUM) % B32_BLOCK_ALIGN_NUM);
    DataCopyPad(payloadLocal, cacheSlotsGm[rowBase + s2BaseIdx],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{payloadRightPadding != 0, 0, payloadRightPadding,
                                                       LICommon::ConstInfo::INVALID_IDX});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(), payloadLocal.template ReinterpretCast<uint32_t>(),
              INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal, static_cast<int32_t>(s2BaseIdx), validLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> keyLocal = reduceOutBuf_.Get<float>()[s2BaseSize_];
    BuildEvictCandidateKeyFromPayload(keyLocal, scoreLocal, payloadLocal.template ReinterpretCast<uint32_t>(),
                                      tmpSortBuf, s2BaseSize_);
    SortByKeyWithPayload512(pairOut, keyLocal, payloadLocal.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                            s2BaseSize_ / BLOCK_BYTES);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeEvictCandidateChunk(const LocalTensor<float> &chunkPairLocal,
                                                               uint32_t candidateCap,
                                                               LocalTensor<float> &tmpSortBuf)
{
    uint32_t candidateBlockNum = candidateCap / static_cast<uint32_t>(s2BaseSize_);
    if (candidateBlockNum == 2U || candidateBlockNum == 3U) {
        MrgBasicBlock(tmpSortBuf, evictCandidateUb_, static_cast<int64_t>(candidateBlockNum + 1U), s2BaseSize_);
        PipeBarrier<PIPE_V>();
        DataCopy(evictCandidateUb_, tmpSortBuf, candidateCap * VALUE_AND_INDEX_NUM);
        PipeBarrier<PIPE_V>();
        return;
    }
    if (candidateBlockNum == 4U) {
        MergeSort(evictCandidateUb_, static_cast<int32_t>(candidateCap), chunkPairLocal, s2BaseSize_, tmpSortBuf);
        return;
    }

    uint32_t tailBlockOffset = (candidateBlockNum - 1U) * CHUNK_PAIR_FLOATS;
    LocalTensor<float> tailBlock = evictCandidateUb_[tailBlockOffset];
    MergeSort(tailBlock, s2BaseSize_, chunkPairLocal, s2BaseSize_, tmpSortBuf);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeExtraEvictChunks512(uint32_t bIdx, uint32_t cacheRowIdx,
                                                               uint32_t actualSeqLen, uint32_t startChunk,
                                                               uint32_t chunkNum, uint32_t firstScanOffset,
                                                               uint32_t extraChunks,
                                                               LocalTensor<float> &tmpSortBuf)
{
    uint32_t scanned = 0;
    while (scanned < extraChunks) {
        uint32_t batchChunks = Min<uint32_t, uint32_t>(PAYLOAD_BUF_SLOTS, extraChunks - scanned);
        for (uint32_t batchIdx = 0; batchIdx < batchChunks; ++batchIdx) {
            uint32_t scanOffset = firstScanOffset + scanned + batchIdx;
            uint32_t chunkIdx = startChunk + scanOffset;
            if (chunkIdx >= chunkNum) {
                chunkIdx -= chunkNum;
            }
            LocalTensor<float> chunkPairLocal = globalTopkUb_[batchIdx * CHUNK_PAIR_FLOATS];
            SortEvictCandidateChunk(bIdx, cacheRowIdx, chunkIdx, actualSeqLen, chunkPairLocal, tmpSortBuf);
            PipeBarrier<PIPE_V>();
        }

        if (batchChunks == 1U) {
            MergeSort(evictCandidateUb_, s2BaseSize_, globalTopkUb_, s2BaseSize_, tmpSortBuf);
        } else {
            MrgBasicBlock(tmpSortBuf, globalTopkUb_, static_cast<int64_t>(batchChunks), s2BaseSize_);
            PipeBarrier<PIPE_V>();
            MergeSort(evictCandidateUb_, s2BaseSize_, tmpSortBuf, s2BaseSize_, globalTopkUb_);
        }
        scanned += batchChunks;
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FindEvictCandidates(uint32_t bIdx, uint32_t cacheRowIdx,
                                                          uint32_t actualSeqLen, uint32_t scanSeed,
                                                          uint32_t missCount, uint32_t candidateCap,
                                                          float thresholdScore,
                                                          LocalTensor<float> &tmpSortBuf)
{
    InitSortOutBuf(evictCandidateUb_, candidateCap * VALUE_AND_INDEX_NUM);
    float stopKey = -thresholdScore;
    uint32_t chunkNum = CeilDiv(actualSeqLen, static_cast<uint32_t>(s2BaseSize_));
    uint32_t startChunk = HashEvictScanSeed(scanSeed, cacheRowIdx) % chunkNum;
    uint32_t stopScanOffset = chunkNum;
    LocalTensor<float> chunkPairLocal = tmpSortBuf[SORT_TMP_FLOATS];
    if (candidateCap > static_cast<uint32_t>(s2BaseSize_) && candidateCap < EVICT_CANDIDATE_CAP) {
        chunkPairLocal = evictCandidateUb_[candidateCap * VALUE_AND_INDEX_NUM];
    } else if (candidateCap == EVICT_CANDIDATE_CAP) {
        chunkPairLocal = globalTopkUb_;
    }
    for (uint32_t scanOffset = 0; scanOffset < chunkNum; ++scanOffset) {
        uint32_t chunkIdx = startChunk + scanOffset;
        if (chunkIdx >= chunkNum) {
            chunkIdx -= chunkNum;
        }
        SortEvictCandidateChunk(bIdx, cacheRowIdx, chunkIdx, actualSeqLen, chunkPairLocal, tmpSortBuf);
        PipeBarrier<PIPE_V>();
        MergeEvictCandidateChunk(chunkPairLocal, candidateCap, tmpSortBuf);

        if (stopScanOffset != chunkNum) {
            if (scanOffset >= stopScanOffset) {
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                break;
            }
            continue;
        }

        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        if (evictCandidateUb_.GetValue((missCount - 1U) * VALUE_AND_INDEX_NUM) > stopKey) {
            uint32_t remainingChunks = chunkNum - scanOffset - 1U;
            uint32_t extraChunks = EVICT_EXTRA_SCAN_CHUNKS < remainingChunks
                                       ? EVICT_EXTRA_SCAN_CHUNKS
                                       : remainingChunks;
            if (extraChunks == 0U) {
                break;
            }
            if (candidateCap == static_cast<uint32_t>(s2BaseSize_)) {
                MergeExtraEvictChunks512(bIdx, cacheRowIdx, actualSeqLen, startChunk, chunkNum,
                                         scanOffset + 1U, extraChunks, tmpSortBuf);
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                break;
            }
            stopScanOffset = scanOffset + extraChunks;
        }
    }
}

template <typename LIT>
__aicore__ inline uint32_t LIVector<LIT>::CopyDecodedPayloadOut(int64_t outOffset,
                                                                const LocalTensor<uint32_t> &payloadLocal,
                                                                const LocalTensor<int32_t> &indexLocal,
                                                                const LocalTensor<int32_t> &slotLocal,
                                                                const LocalTensor<int32_t> &scratchLocal,
                                                                int64_t count, bool mayHaveInvalid)
{
    DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(), payloadLocal, BASE_TOPK);
    if (mayHaveInvalid) {
        FixInvalidIndex(indexLocal, payloadLocal.template ReinterpretCast<int32_t>(), BASE_TOPK);
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(topkIndexGm[outOffset], indexLocal,
                {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    DecodeSlotFromPayload(slotLocal.template ReinterpretCast<uint32_t>(), payloadLocal, scratchLocal, BASE_TOPK);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    uint32_t missCount = 0;
    while (missCount < static_cast<uint32_t>(count) &&
           slotLocal.GetValue(missCount) == LICommon::ConstInfo::INVALID_IDX) {
        ++missCount;
    }
    return missCount;
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::IsTopKIndex(const LocalTensor<int32_t> &indexLocal,
                                                  uint32_t candidateIndex, uint32_t count) const
{
    for (uint32_t topkIdx = 0; topkIdx < count; ++topkIdx) {
        if (static_cast<uint32_t>(indexLocal.GetValue(topkIdx)) == candidateIndex) {
            return true;
        }
    }
    return false;
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::FindFallbackEvict(uint32_t cacheRowIdx, uint32_t actualSeqLen,
                                                        const LocalTensor<int32_t> &indexLocal,
                                                        uint32_t &scanCursor, uint32_t &evictIndex,
                                                        int32_t &evictSlot)
{
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheRowWidth_;
    while (scanCursor < actualSeqLen) {
        int32_t slot = cacheSlotsGm.GetValue(rowBase + scanCursor);
        uint32_t candidateIndex = scanCursor;
        ++scanCursor;
        if (slot >= 0 && !IsTopKIndex(indexLocal, candidateIndex, BASE_TOPK)) {
            evictIndex = candidateIndex;
            evictSlot = slot;
            return true;
        }
    }
    evictIndex = 0;
    evictSlot = LICommon::ConstInfo::INVALID_IDX;
    return false;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::UpdateCacheAndWriteTopkSlots(
    uint32_t bIdx, uint32_t cacheRowIdx, int64_t outOffset, uint32_t actualSeqLen, uint32_t scanSeed,
    float thresholdScore, uint32_t missCount,
    const LocalTensor<int32_t> &indexLocal, const LocalTensor<int32_t> &topkSlotsLocal,
    const LocalTensor<int32_t> &scratchLocal)
{
    uint32_t candidateCap = 0;
    if (missCount > 0) {
        candidateCap = ((missCount + S2_BASE_SIZE - 1U) / S2_BASE_SIZE) * S2_BASE_SIZE;
        if (candidateCap > EVICT_CANDIDATE_CAP) {
            candidateCap = EVICT_CANDIDATE_CAP;
        }
        FindEvictCandidates(bIdx, cacheRowIdx, actualSeqLen, scanSeed, missCount, candidateCap,
                            thresholdScore, SortedBasicBlock_);
    }

    uint32_t candidateCursor = 0;
    uint32_t fallbackCursor = 0;
    uint64_t rowBase = static_cast<uint64_t>(cacheRowIdx) * cacheRowWidth_;
    float stopKey = -thresholdScore;
    LocalTensor<uint32_t> candidateBitsLocal = evictCandidateUb_.template ReinterpretCast<uint32_t>();
    for (uint32_t missIdx = 0; missIdx < missCount; ++missIdx) {
        uint32_t evictIndex = 0;
        int32_t evictSlot = LICommon::ConstInfo::INVALID_IDX;
        bool foundCandidate = false;
        while (candidateCursor < candidateCap) {
            float candidateKey = evictCandidateUb_.GetValue(candidateCursor * VALUE_AND_INDEX_NUM);
            if (candidateKey <= stopKey) {
                break;
            }
            uint32_t payload = candidateBitsLocal.GetValue(candidateCursor * VALUE_AND_INDEX_NUM + 1);
            ++candidateCursor;
            int32_t slot = static_cast<int32_t>(payload >> INDEX_BITS);
            uint32_t index = payload & INDEX_MASK;
            if (slot >= 0 && index < actualSeqLen) {
                evictIndex = index;
                evictSlot = slot;
                foundCandidate = true;
                break;
            }
        }
        if (!foundCandidate) {
            if (CacheRowsShareDCacheLine()) {
                FlushCacheSlotsDCache();
            }
            foundCandidate = FindFallbackEvict(cacheRowIdx, actualSeqLen, indexLocal, fallbackCursor,
                                               evictIndex, evictSlot);
        }
        if (foundCandidate) {
            uint32_t missIndex = static_cast<uint32_t>(indexLocal.GetValue(missIdx));
            topkSlotsLocal.SetValue(missIdx, evictSlot);
            WriteCacheSlot(rowBase + evictIndex, LICommon::ConstInfo::INVALID_IDX, scratchLocal);
            WriteCacheSlot(rowBase + missIndex, evictSlot, scratchLocal);
        }
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(topkSlotsGm[outOffset], topkSlotsLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ClearOutputTail(int64_t outOffset)
{
    LocalTensor<float> valueLocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> invalidLocal = valueLocal.template ReinterpretCast<int32_t>();
    Duplicate(invalidLocal, LICommon::ConstInfo::INVALID_IDX, OUTPUT_CLEAR_CHUNK);
    outQueue_.EnQue<float>(valueLocal);
    valueLocal = outQueue_.DeQue<float>();
    uint32_t remaining = OUTPUT_CAPACITY - BASE_TOPK;
    for (uint32_t offset = BASE_TOPK; remaining > 0; offset += OUTPUT_CLEAR_CHUNK) {
        uint32_t copyCount = Min(remaining, OUTPUT_CLEAR_CHUNK);
        LIServiceVec::CopyOut(topkIndexGm[outOffset + offset], invalidLocal, copyCount);
        LIServiceVec::CopyOut(topkSlotsGm[outOffset + offset], invalidLocal, copyCount);
        remaining -= copyCount;
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(valueLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteRequestMetadata(uint32_t bIdx, int32_t missCount,
                                                           int32_t tailSlotStart, int32_t tailTokenCount,
                                                           const LocalTensor<int32_t> &scalarLocal)
{
    scalarLocal.SetValue(0, missCount);
    scalarLocal.SetValue(REQUEST_METADATA_TAIL_OFFSET, tailSlotStart);
    scalarLocal.SetValue(REQUEST_METADATA_TAIL_OFFSET + 1, tailTokenCount);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(missCountGm[bIdx], scalarLocal, 1);
    LIServiceVec::CopyOut(tailInfoGm[static_cast<int64_t>(bIdx) * 2],
                          scalarLocal[REQUEST_METADATA_TAIL_OFFSET], 2);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::WriteNonOffloadedOutput(const LICommon::RunInfo &info,
                                                              const LocalTensor<float> &valueLocal)
{
    LocalTensor<uint32_t> payloadLocal = valueLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> indexLocal = valueLocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    LocalTensor<int32_t> scalarLocal = valueLocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
    ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
    DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(), payloadLocal, BASE_TOPK);
    if (info.actS2Size < BASE_TOPK) {
        FixInvalidIndex(indexLocal, payloadLocal.template ReinterpretCast<int32_t>(), BASE_TOPK);
    }

    int64_t outOffset = static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(topkIndexGm[outOffset], indexLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    DataCopyPad(topkSlotsGm[outOffset], indexLocal,
                {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
    WriteRequestMetadata(info.bIdx, 0, LICommon::ConstInfo::INVALID_IDX, 0, scalarLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AccumulateColdTopKChunk(uint32_t bIdx, uint32_t chunkIdx,
                                                              uint32_t actualSeqLen,
                                                              LocalTensor<float> &tmpSortBuf)
{
    uint32_t s2BaseIdx = chunkIdx * static_cast<uint32_t>(s2BaseSize_);
    uint32_t validLen = Min(static_cast<uint32_t>(s2BaseSize_), actualSeqLen - s2BaseIdx);
    LocalTensor<float> scoreLocal = reduceOutBuf_.Get<float>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    DataCopyPad(scoreLocal, scoresGm[static_cast<uint64_t>(bIdx) * scoreStride_ + s2BaseIdx],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(s2BaseSize_ * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});

    LocalTensor<int32_t> payloadLocal = payloadBuf_.Get<int32_t>();
    BuildIndexPayload(payloadLocal, static_cast<int32_t>(s2BaseIdx), static_cast<int32_t>(validLen));
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    uint32_t cachedChunkIdx = chunkIdx % PAYLOAD_BUF_SLOTS;
    Sort<float, true>(SortedBasicBlock_[cachedChunkIdx * CHUNK_PAIR_FLOATS], scoreLocal,
                      payloadLocal.template ReinterpretCast<uint32_t>(), tmpSortBuf, s2BaseSize_ / BLOCK_BYTES);
    PipeBarrier<PIPE_V>();
    bool isLastChunk = chunkIdx + 1U == CeilDiv(actualSeqLen, static_cast<uint32_t>(s2BaseSize_));
    if (cachedChunkIdx == PAYLOAD_BUF_SLOTS - 1U || isLastChunk) {
        uint32_t mergedChunkCount = cachedChunkIdx + 1U;
        if (chunkIdx < PAYLOAD_BUF_SLOTS) {
            MrgBasicBlock(globalTopkUb_, SortedBasicBlock_, mergedChunkCount, s2BaseSize_);
        } else {
            if (cachedChunkIdx > 0) {
                MrgBasicBlock(tmpSortBuf, SortedBasicBlock_, mergedChunkCount, s2BaseSize_);
                PipeBarrier<PIPE_V>();
                DataCopy(SortedBasicBlock_, tmpSortBuf, mergedChunkCount * CHUNK_PAIR_FLOATS);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       s2BaseSize_ * mergedChunkCount);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessFirstFill(const LICommon::RunInfo &info,
                                                       const LocalTensor<float> &valueLocal)
{
    LocalTensor<float> tmpSortBuf = valueLocal;
    LocalTensor<uint32_t> payloadLocal = valueLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> indexLocal = valueLocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    LocalTensor<int32_t> scalarLocal = valueLocal.template ReinterpretCast<int32_t>()[OUTPUT_CLEAR_CHUNK];
    uint32_t targetCount = static_cast<uint32_t>(-info.cacheMetadata);
    uint64_t rowBase = static_cast<uint64_t>(info.cacheRowIdx) * cacheRowWidth_;
    uint32_t chunkCount = CeilDiv(info.actS2Size, static_cast<uint32_t>(s2BaseSize_));
    int64_t outOffset = static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY;
    uint64_t scoreRowBase = static_cast<uint64_t>(info.bIdx) * scoreStride_;
    uint32_t validOutputCount = Min(info.actS2Size, OUTPUT_CAPACITY);
    uint32_t passCount = CeilDiv(validOutputCount, BASE_TOPK);
    GlobalTensor<int32_t> scoreBitsLocalGm;
    scoreBitsLocalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scoresGm.GetPhyAddr(0)));

    for (uint32_t passIdx = 0; passIdx < passCount; ++passIdx) {
        uint32_t passOffsetU32 = passIdx * BASE_TOPK;
        uint32_t validInPass = Min(BASE_TOPK, validOutputCount - passOffsetU32);
        InitSortOutBuf(globalTopkUb_, TOPK_PAIR_FLOATS);
        for (uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
            AccumulateColdTopKChunk(info.bIdx, chunkIdx, info.actS2Size, tmpSortBuf);
        }

        ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        DecodeIndexFromPayload(indexLocal.template ReinterpretCast<uint32_t>(), payloadLocal, BASE_TOPK);
        FixInvalidIndex(indexLocal, payloadLocal.template ReinterpretCast<int32_t>(), BASE_TOPK);
        for (uint32_t idx = validInPass; idx < BASE_TOPK; ++idx) {
            indexLocal.SetValue(idx, LICommon::ConstInfo::INVALID_IDX);
        }
        int64_t passOffset = static_cast<int64_t>(passOffsetU32);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(topkIndexGm[outOffset + passOffset], indexLocal,
                    {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

        for (uint32_t idx = 0; idx < validInPass; ++idx) {
            int32_t tokenIdx = indexLocal.GetValue(idx);
            if (tokenIdx < 0 || static_cast<uint32_t>(tokenIdx) >= info.actS2Size) {
                continue;
            }
            uint32_t rank = passIdx * BASE_TOPK + idx;
            if (rank < targetCount) {
                WriteCacheSlot(rowBase + static_cast<uint32_t>(tokenIdx), static_cast<int32_t>(rank), scalarLocal);
            }
            scoreBitsLocalGm.SetValue(scoreRowBase + static_cast<uint32_t>(tokenIdx), LIServiceVec::NEG_INF);
        }
        PipeBarrier<PIPE_ALL>();
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(scoreBitsLocalGm);
        __asm__ __volatile__("");
        PipeBarrier<PIPE_ALL>();
    }

    Duplicate(indexLocal, LICommon::ConstInfo::INVALID_IDX, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    for (uint32_t passIdx = passCount; passIdx < FIRST_FILL_PASS_COUNT; ++passIdx) {
        int64_t passOffset = static_cast<int64_t>(passIdx) * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(topkIndexGm[outOffset + passOffset], indexLocal,
                    {1, static_cast<uint16_t>(BASE_TOPK * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    LocalTensor<int32_t> slotFirstLocal = valueLocal.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> slotSecondLocal = globalTopkUb_.template ReinterpretCast<int32_t>();
    ArithProgression(slotFirstLocal, 0, 1, OUTPUT_CLEAR_CHUNK);
    PipeBarrier<PIPE_V>();
    ArithProgression(slotSecondLocal, static_cast<int32_t>(OUTPUT_CLEAR_CHUNK), 1, OUTPUT_CLEAR_CHUNK);
    PipeBarrier<PIPE_V>();
    uint32_t validFirstHalf = Min(validOutputCount, OUTPUT_CLEAR_CHUNK);
    uint32_t validSecondHalf = validOutputCount > OUTPUT_CLEAR_CHUNK ?
                                   validOutputCount - OUTPUT_CLEAR_CHUNK : 0U;
    if (validFirstHalf < OUTPUT_CLEAR_CHUNK) {
        Duplicate(slotFirstLocal[validFirstHalf], LICommon::ConstInfo::INVALID_IDX,
                  OUTPUT_CLEAR_CHUNK - validFirstHalf);
        PipeBarrier<PIPE_V>();
    }
    if (validSecondHalf < OUTPUT_CLEAR_CHUNK) {
        Duplicate(slotSecondLocal[validSecondHalf], LICommon::ConstInfo::INVALID_IDX,
                  OUTPUT_CLEAR_CHUNK - validSecondHalf);
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(topkSlotsGm[outOffset], slotFirstLocal,
                {1, static_cast<uint16_t>(OUTPUT_CLEAR_CHUNK * sizeof(int32_t)), 0, 0});
    DataCopyPad(topkSlotsGm[outOffset + OUTPUT_CLEAR_CHUNK], slotSecondLocal,
                {1, static_cast<uint16_t>(OUTPUT_CLEAR_CHUNK * sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    WriteCacheSlot(rowBase + metadataIndex_, static_cast<int32_t>(targetCount), scalarLocal);
    if (!CacheRowsShareDCacheLine()) {
        FlushCacheSlotsDCache();
    }
    WriteRequestMetadata(info.bIdx, static_cast<int32_t>(targetCount),
                         static_cast<int32_t>(targetCount),
                         static_cast<int32_t>(info.originalSeqLen - info.actS2Size), scalarLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }

    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;
    int32_t cuS2Len = info.actualSingleProcessSInnerSize;
    int64_t mmGmOffset = (info.loop % 2) * (MAX_G_SIZE * s2BaseSize_);
    int64_t weightGmOffset = static_cast<int64_t>(info.bIdx) * qHeadNum_;
    if (info.isFirstS2InnerLoop && !info.isFirstFill) {
        InitSortOutBuf(globalTopkUb_, TOPK_PAIR_FLOATS);
    }

    int32_t mmUbStride = (s2BaseSize_ - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
    int64_t payloadBufIdx = info.s2Idx % PAYLOAD_BUF_SLOTS;
    LocalTensor<int32_t> payloadUb = payloadBuf_.Get<int32_t>()[payloadBufIdx * s2BaseSize_];
    bool isSteadyUpdate = info.isSparse && !info.isFirstFill;
    if (isSteadyUpdate) {
        StartPayloadCopy(payloadUb, info.cacheRowIdx, cuBaseS2Idx, cuS2Len, s2BaseSize_);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    PipeBarrier<PIPE_V>();
    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
    for (uint32_t outerGidx = 0; outerGidx < outerGroupNum_; ++outerGidx) {
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[GROUP_INNER * s2BaseSize_];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>();
        weightsInTUb = weightsInTUb[GROUP_INNER];
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                             mmGmOffset + outerGidx * GROUP_INNER * info.actualSingleProcessSInnerSizeAlign,
                             weightGmOffset + outerGidx * GROUP_INNER, GROUP_INNER,
                             info.actualSingleProcessSInnerSizeAlign, mmUbStride);
        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        weightsInUb = mmInUb[GROUP_INNER * s2BaseSize_];
        LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                              brcBuf, GROUP_INNER, s2BaseSize_, outerGidx);
        inQueue_.FreeTensor(mmInUb);
    }

    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, GROUP_INNER, s2BaseSize_);
    outQueue_.FreeTensor(reduceCacheBuf);

    LocalTensor<float> sortScoreUb = reduceOutBuff;
    PipeBarrier<PIPE_V>();
    Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
    PipeBarrier<PIPE_V>();
    if (info.isSparse) {
        WriteScoreChunk(info.bIdx, cuBaseS2Idx, sortScoreUb, s2BaseSize_);
    }
    if (isSteadyUpdate) {
        FinishPayload(payloadUb, cuBaseS2Idx, cuS2Len);
    } else if (!info.isFirstFill) {
        BuildIndexPayload(payloadUb, cuBaseS2Idx, cuS2Len);
    }

    if (info.isFirstFill) {
        if (info.isLastS2InnerLoop) {
            LocalTensor<float> coldLocal = outQueue_.AllocTensor<float>();
            ProcessFirstFill(info, coldLocal);
            outQueue_.FreeTensor(coldLocal);
        }
        return;
    }

    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    uint32_t cachedChunkIdx = info.s2Idx % 4;
    Sort<float, true>(SortedBasicBlock_[cachedChunkIdx * s2BaseSize_ * VALUE_AND_INDEX_NUM], reduceOutBuff,
                      payloadUb.template ReinterpretCast<uint32_t>(), tmpSortBuf, s2BaseSize_ / 32);
    PipeBarrier<PIPE_V>();
    if (cachedChunkIdx == 3 || info.isLastS2InnerLoop) {
        if (info.s2Idx < 4) {
            MrgBasicBlock(globalTopkUb_, SortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), s2BaseSize_);
        } else {
            if (cachedChunkIdx > 0) {
                MrgBasicBlock(tmpSortBuf, SortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), s2BaseSize_);
                PipeBarrier<PIPE_V>();
                DataCopy(SortedBasicBlock_, tmpSortBuf, (cachedChunkIdx + 1) * s2BaseSize_ * VALUE_AND_INDEX_NUM);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       s2BaseSize_ * (cachedChunkIdx + 1));
        }
    }
    PipeBarrier<PIPE_V>();

    float thresholdScore = 0.0f;
    if (info.isLastS2InnerLoop && isSteadyUpdate) {
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        thresholdScore = globalTopkUb_.GetValue((BASE_TOPK - 1) * VALUE_AND_INDEX_NUM);
        SortTopKBySlot(globalTopkUb_, reduceOutBuff, payloadBuf_.Get<int32_t>(), tmpSortBuf);
    }
    outQueue_.FreeTensor(tmpSortBuf);

    if (info.isLastS2InnerLoop) {
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        if (!info.isSparse) {
            WriteNonOffloadedOutput(info, valueULocal);
        } else {
            LocalTensor<uint32_t> payloadLocal = valueULocal.template ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
            LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
            LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 3];
            ExtractIndex(payloadLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
            PipeBarrier<PIPE_V>();
            int64_t outOffset = static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY;
            uint32_t missCount = CopyDecodedPayloadOut(outOffset, payloadLocal, indexLocal, slotLocal, scratchLocal,
                                                       BASE_TOPK, info.actS2Size < BASE_TOPK);
            UpdateCacheAndWriteTopkSlots(info.bIdx, info.cacheRowIdx, outOffset, info.actS2Size,
                                         info.originalSeqLen,
                                         thresholdScore, missCount, indexLocal, slotLocal, scratchLocal);
            WriteRequestMetadata(info.bIdx, static_cast<int32_t>(missCount), info.cacheMetadata,
                                 static_cast<int32_t>(info.originalSeqLen - info.actS2Size), scratchLocal);
        }
        outQueue_.FreeTensor(valueULocal);
        ClearOutputTail(static_cast<int64_t>(info.bIdx) * OUTPUT_CAPACITY);
    }
}

} // namespace LIKernel
#endif
