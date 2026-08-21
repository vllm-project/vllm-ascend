/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "catlass/gemm_coord.hpp"
using namespace Catlass;

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP

// constexpr uint32_t PING_PONG_STAGES = 1;
constexpr uint32_t PING_PONG_STAGES = 2;
constexpr uint32_t BYTE_SIZE_16_BIT = 2;
constexpr uint32_t BYTES_PER_C0 = 32;
constexpr uint32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr uint32_t SIZE_16_NUM_PER_C0 = BYTES_PER_C0 / BYTE_SIZE_16_BIT;
constexpr uint32_t FLOAT_NUM_PER_REPEAT = BYTE_SIZE_PER_REPEAT / sizeof(float);
constexpr uint32_t NZ_BLOCK_SIZE = 16;

template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <typename T>
CATLASS_DEVICE T Min(T a, T b) {
    return (a > b) ? b : a;
}

template <typename T>
CATLASS_DEVICE T Max(T a, T b) {
    return (a > b) ? a : b;
}

namespace Catlass::Gemm::Block {

struct GDNFwdHOffsets {
    uint32_t hSrcOffset;
    uint32_t hDstOffset;
    uint32_t uvOffset;
    uint32_t wkOffset;
    uint32_t wOffset;
    uint32_t gOffset;
    uint32_t gkOffset;
    uint32_t hWorkOffset;
    uint32_t vWorkOffset;
    uint32_t kDecayWorkOffset;
    uint32_t vBlockOffset;
    uint32_t vBlockDim;
    uint32_t initialStateOffset;
    uint32_t finalStateOffset;
    bool isInitialState;
    bool isFinalState;
    uint32_t blockTokens;
    uint32_t streamId;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct GDNFwdHStream {
    uint32_t batchIdx;
    uint32_t chunkIdx{0};
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t chunkOffset;
    uint32_t tokenOffset;
    uint32_t batchChunks{0};
    uint32_t batchTokens;
    uint32_t nextTaskIdx{0};
    bool active{false};

    GDNFwdHOffsets offset;
};

struct GDNFwdHRunningQ {
    GDNFwdHStream streams[PING_PONG_STAGES];
};

struct BlockSchedulerGdnFwdH {
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t vBlockSize{128};
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    bool useInitialState;
    bool storeFinalState;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;

    uint32_t taskIdx;
    uint32_t taskStride;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t taskNum;
    uint32_t headGroups;
    uint32_t totalChunks;
    uint32_t totalTokens;

    GDNFwdHRunningQ runningQ;

    bool isRunning;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Arch::CrossCoreFlag cube1Done[PING_PONG_STAGES] = {0, 1};
    Arch::CrossCoreFlag vec1Done[PING_PONG_STAGES] = {2, 3};
    Arch::CrossCoreFlag cube2Done[PING_PONG_STAGES] = {4, 5};
    Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {6, 7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdH() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if (isVariedLen) {
            gmNumChunks.SetValue(0, 0);
            gmNumSeq.SetValue(0, 0);
            uint32_t actualBatch = 0;
            int64_t prevSeq = 0, currSeq;
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                currSeq = gmSeqlen.GetValue(b);
                int64_t batchSeqLen = currSeq - prevSeq;
                if (batchSeqLen > 0) {
                    actualBatch++;
                    gmNumSeq.SetValue(actualBatch, currSeq);
                    int64_t batchChunk = (batchSeqLen + chunkSize - 1) / chunkSize;
                    gmNumChunks.SetValue(actualBatch, gmNumChunks.GetValue(actualBatch - 1) + batchChunk);
                }
                prevSeq = currSeq;
            }
            tokenBatch = actualBatch;
            batch = actualBatch;
            totalChunks = gmNumChunks.GetValue(tokenBatch);
            totalTokens = gmNumSeq.GetValue(tokenBatch);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vBlockSize = vHeadDim;
        taskNum = batch * vNumHead;
        headGroups = vNumHead / kNumHead;
        uint32_t maxTaskCntPerLoop = taskNum > cubeCoreNum ? PING_PONG_STAGES : 1;
        taskStride = cubeCoreNum * maxTaskCntPerLoop;
        for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
            auto& stream = runningQ.streams[streamId];
            stream.nextTaskIdx = (streamId < maxTaskCntPerLoop) ? (cubeCoreIdx * maxTaskCntPerLoop + streamId) : taskNum;
            stream.chunkIdx = 0;
            stream.batchChunks = 0;
            stream.active = false;
        }
        isRunning = cubeCoreIdx * maxTaskCntPerLoop < taskNum;

    }


    CATLASS_DEVICE
    void InitNewStream(GDNFwdHStream& newStream) {
        newStream.batchIdx = taskIdx / vNumHead;
        newStream.vHeadIdx = taskIdx % vNumHead;
        newStream.kHeadIdx = newStream.vHeadIdx / headGroups;
        newStream.shapeBatchIdx = isVariedLen ? 0 : newStream.batchIdx;
        newStream.tokenBatchIdx = isVariedLen ? newStream.batchIdx : 0;
        newStream.chunkOffset = isVariedLen ? gmNumChunks.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchChunks = isVariedLen ? (gmNumChunks.GetValue(newStream.tokenBatchIdx + 1) - newStream.chunkOffset) : totalChunks;
        newStream.tokenOffset = isVariedLen ? gmNumSeq.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchTokens = isVariedLen ? (gmNumSeq.GetValue(newStream.tokenBatchIdx + 1) - newStream.tokenOffset) : totalTokens;
        newStream.chunkIdx = 0;
    }

    CATLASS_DEVICE
    void AssignNextStream(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        taskIdx = stream.nextTaskIdx;
        if (taskIdx >= taskNum) {
            stream.active = false;
            stream.batchChunks = 0;
            return;
        }

        stream.nextTaskIdx += taskStride;
        InitNewStream(stream);
        stream.active = stream.batchChunks > 0;
        if (stream.active) {
            UpdateTask(streamId);
        }
    }

    CATLASS_DEVICE
    void UpdateTask(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        auto& offset = stream.offset;

        offset.isInitialState = stream.chunkIdx == 0;
        offset.isFinalState = stream.chunkIdx == (stream.batchChunks - 1);
        uint32_t vBlockOffset = 0;
        uint32_t vBlockDim = vBlockSize;
        offset.initialStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.finalStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.hSrcOffset = (stream.shapeBatchIdx * vNumHead * totalChunks + stream.vHeadIdx * totalChunks + stream.chunkOffset + stream.chunkIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.hDstOffset = offset.hSrcOffset + kHeadDim * vHeadDim;
        if (storeFinalState && offset.isFinalState) {
            offset.hDstOffset = offset.hSrcOffset;
        }
        offset.uvOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * vHeadDim + vBlockOffset;
        offset.wkOffset = (stream.shapeBatchIdx * kNumHead * totalTokens + stream.kHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.wOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.gOffset = stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize;
        offset.gkOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.hWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * kHeadDim * vBlockSize;
        offset.vWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * chunkSize * vBlockSize;
        offset.kDecayWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * chunkSize * kHeadDim;
        offset.vBlockOffset = vBlockOffset;
        offset.vBlockDim = vBlockDim;
        offset.blockTokens = offset.isFinalState ? (stream.batchTokens - stream.chunkIdx * chunkSize) : chunkSize;
        offset.streamId = streamId;
        offset.batchIdx = stream.batchIdx;
        offset.headIdx = stream.vHeadIdx;
        offset.chunkIdx = stream.chunkIdx;
    }

    CATLASS_DEVICE
    void InitTasks() {
        isRunning = false;
        for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
            auto& stream = runningQ.streams[streamId];
            if (stream.active) {
                stream.chunkIdx += 1;
                if (stream.chunkIdx >= stream.batchChunks) {
                    stream.active = false;
                    stream.batchChunks = 0;
                }
            }
            if (!stream.active) {
                AssignNextStream(streamId);
            } else {
                UpdateTask(streamId);
            }
            if (stream.active) {
                isRunning = true;
            }
        }
    }

    CATLASS_DEVICE
    const GDNFwdHStream& GetStream(uint32_t i) const {
        return runningQ.streams[i];
    }

    CATLASS_DEVICE
    uint32_t GetStreamId(uint32_t i) const {
        return i;
    }

    CATLASS_DEVICE
    const GDNFwdHOffsets& GetCurTaskOffsets(const GDNFwdHStream& stream) const {
        return stream.offset;
    }

    CATLASS_DEVICE
    bool StreamIsDone(const GDNFwdHStream& stream) const {
        return !stream.active;
    }

    CATLASS_DEVICE
    bool NeedProcessStage2(const GDNFwdHStream& stream) {
        return storeFinalState || !stream.offset.isFinalState;
    }
};

struct BlockSchedulerGdnFwdHCube : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHVec : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdH::Init(
            cu_seqlens, chunk_indices, tiling, user,
            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),
            AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP
