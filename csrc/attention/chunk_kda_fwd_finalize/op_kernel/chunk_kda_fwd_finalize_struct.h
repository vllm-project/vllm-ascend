/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_STRUCT_H
#define CHUNK_KDA_FWD_FINALIZE_STRUCT_H

#include "kernel_operator.h"

namespace KdaFinalize {

// 字段类型和顺序与 op_host 的 TILING_DATA_FIELD_DEF 严格一致。
struct ChunkKdaFwdFinalizeTilingData {
    uint32_t batch;
    uint32_t seqNum;
    uint32_t seqLen;
    uint32_t valueHeadNum;
    uint32_t totalChunks;
    uint32_t usedCoreNum;
    uint32_t headsPerPartition;
    bool isVarLen;
    bool outputSequenceMajor;
    bool stateVFirst;
};

struct FinalizeChunk {
    uint32_t batch = 0;
    uint32_t globalChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validRows = 0;
};

struct FinalizeArgs {
    GM_ADDR qgScaled = nullptr;
    GM_ADDR aqk = nullptr;
    GM_ADDR vNew = nullptr;
    GM_ADDR h = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    GM_ADDR attnOut = nullptr;
    ChunkKdaFwdFinalizeTilingData tiling{};
};

namespace Shape {
constexpr uint32_t kChunkRows = 64;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHeadsPerGroup = 4;
constexpr uint32_t kProductBytes = kChunkRows * kHeadDim * sizeof(float); // 32 KiB
} // namespace Shape

namespace A5Sync {
// mode 0x4 下，两个 AIV 各自使用本地 flag 0/1、2/3 与 4/5；
// AIC 访问 AIV1 时由硬件 flag 空间加 16。
constexpr uint8_t kCrossCoreMode = 0x4;
constexpr uint16_t kAivQhReadyFlagId[2] = {0, 1};
constexpr uint16_t kAivAvReadyFlagId[2] = {2, 3};
constexpr uint16_t kAivL1ReusableFlagId[2] = {4, 5};
constexpr uint16_t kAicQhReadyFlagId[4] = {0, 16, 1, 17};
constexpr uint16_t kAicAvReadyFlagId[4] = {2, 18, 3, 19};
constexpr uint16_t kAicL1ReusableFlagId[4] = {4, 20, 5, 21};
} // namespace A5Sync

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t WorkgroupId()
{
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(AscendC::GetBlockIdx()) /
               static_cast<uint32_t>(AscendC::GetSubBlockNum());
    }
    return static_cast<uint32_t>(AscendC::GetBlockIdx());
}

__aicore__ inline uint32_t TotalWorkItems(const FinalizeArgs &args)
{
    const uint32_t chunks = args.tiling.isVarLen
                                ? args.tiling.totalChunks
                                : args.tiling.batch * args.tiling.totalChunks;
    return chunks * CeilDiv(args.tiling.valueHeadNum, args.tiling.headsPerPartition);
}

__aicore__ inline uint32_t WorkBegin(uint32_t total, uint32_t core, uint32_t cores)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(total) * core) / cores);
}

__aicore__ inline uint32_t WorkEnd(uint32_t total, uint32_t core, uint32_t cores)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(total) * (core + 1)) / cores);
}

__aicore__ inline bool ResolveChunk(const FinalizeArgs &args, uint32_t chunkId, FinalizeChunk &chunk)
{
    chunk.globalChunk = chunkId;
    if (!args.tiling.isVarLen) {
        const uint32_t chunksPerSequence = args.tiling.totalChunks;
        chunk.batch = chunkId / chunksPerSequence;
        chunk.tokenBegin = (chunkId % chunksPerSequence) * Shape::kChunkRows;
        chunk.validRows = args.tiling.seqLen - chunk.tokenBegin;
        if (chunk.validRows > Shape::kChunkRows) {
            chunk.validRows = Shape::kChunkRows;
        }
        return chunk.batch < args.tiling.batch && chunk.validRows != 0;
    }

    const __gm__ int64_t *cu = reinterpret_cast<const __gm__ int64_t *>(args.cuSeqlens);
    const __gm__ int64_t *indices = reinterpret_cast<const __gm__ int64_t *>(args.chunkIndices);
    uint32_t sequence = 0;
    uint32_t localChunk = 0;
    if (indices != nullptr) {
        sequence = static_cast<uint32_t>(indices[chunkId * 2]);
        localChunk = static_cast<uint32_t>(indices[chunkId * 2 + 1]);
    } else {
        uint32_t prefix = 0;
        for (uint32_t seq = 0; seq < args.tiling.seqNum; ++seq) {
            const uint32_t length = static_cast<uint32_t>(cu[seq + 1] - cu[seq]);
            const uint32_t count = CeilDiv(length, Shape::kChunkRows);
            if (chunkId < prefix + count) {
                sequence = seq;
                localChunk = chunkId - prefix;
                break;
            }
            prefix += count;
        }
    }
    const uint32_t begin = static_cast<uint32_t>(cu[sequence]) + localChunk * Shape::kChunkRows;
    const uint32_t end = static_cast<uint32_t>(cu[sequence + 1]);
    chunk.batch = 0;
    chunk.tokenBegin = begin;
    chunk.validRows = end - begin;
    if (chunk.validRows > Shape::kChunkRows) {
        chunk.validRows = Shape::kChunkRows;
    }
    return chunk.validRows != 0;
}

__aicore__ inline uint64_t InputOffset(const FinalizeArgs &args, const FinalizeChunk &chunk,
                                       uint32_t valueHead, uint32_t dim)
{
    return ((static_cast<uint64_t>(chunk.batch) * args.tiling.valueHeadNum + valueHead) *
            args.tiling.seqLen + chunk.tokenBegin) * dim;
}

__aicore__ inline uint64_t StateOffset(const FinalizeArgs &args, const FinalizeChunk &chunk,
                                       uint32_t valueHead)
{
    return ((static_cast<uint64_t>(chunk.batch) * args.tiling.valueHeadNum + valueHead) *
            args.tiling.totalChunks +
            (args.tiling.isVarLen ? chunk.globalChunk : chunk.globalChunk % args.tiling.totalChunks)) *
           Shape::kHeadDim * Shape::kHeadDim;
}

__aicore__ inline uint64_t OutputOffset(const FinalizeArgs &args, const FinalizeChunk &chunk,
                                        uint32_t valueHead, uint32_t row)
{
    if (args.tiling.outputSequenceMajor) {
        return ((static_cast<uint64_t>(chunk.batch) * args.tiling.seqLen + chunk.tokenBegin + row) *
                args.tiling.valueHeadNum + valueHead) * Shape::kHeadDim;
    }
    return InputOffset(args, chunk, valueHead, Shape::kHeadDim) + row * Shape::kHeadDim;
}

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_STRUCT_H
