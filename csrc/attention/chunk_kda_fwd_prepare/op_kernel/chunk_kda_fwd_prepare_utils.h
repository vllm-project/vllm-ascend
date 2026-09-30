/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_UTILS_H
#define CHUNK_KDA_FWD_PREPARE_UTILS_H

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_struct.h"

namespace KdaPrepare {

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + (value % divisor != 0);
}

__aicore__ inline uint32_t WorkgroupId()
{
    if ASCEND_IS_AIV {
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        return subBlockNum == 0 ? 0 : AscendC::GetBlockIdx() / subBlockNum;
    }
    return AscendC::GetBlockIdx();
}

__aicore__ inline uint32_t WorkBegin(uint32_t total, uint32_t rank,
                                     uint32_t rankCount)
{
    if (rankCount == 0) {
        return 0;
    }
    if (rank >= rankCount) {
        return total;
    }
    return static_cast<uint32_t>(static_cast<uint64_t>(total) * rank /
                                 rankCount);
}

__aicore__ inline uint32_t WorkEnd(uint32_t total, uint32_t rank,
                                   uint32_t rankCount)
{
    if (rankCount == 0) {
        return 0;
    }
    if (rank >= rankCount) {
        return total;
    }
    return static_cast<uint32_t>(static_cast<uint64_t>(total) *
                                 (rank + 1) / rankCount);
}

__aicore__ inline bool UseHeadSplit(
    const PrepareRuntimeTiling &tiling)
{
    const uint32_t chunkWork = tiling.isVarLen
                                   ? tiling.totalChunks
                                   : tiling.batch * tiling.totalChunks;
    return chunkWork < tiling.usedCoreNum;
}

__aicore__ inline uint32_t HeadPartitionCount(
    const PrepareRuntimeTiling &tiling)
{
    const uint32_t headsPerPartition =
        tiling.headsPerPartition == 0 ? tiling.valueHeadNum
                                      : tiling.headsPerPartition;
    return CeilDiv(tiling.valueHeadNum, headsPerPartition);
}

__aicore__ inline uint32_t TotalWorkItems(
    const PrepareRuntimeTiling &tiling)
{
    // dense 的 totalChunks 是每个 batch 的 chunk 数；varlen 的 totalChunks
    // 已经是所有 sequence 的压平总数，与现有 chunk_kda_fwd tiling 一致。
    const uint32_t chunkWork = tiling.isVarLen
                                   ? tiling.totalChunks
                                   : tiling.batch * tiling.totalChunks;
    return UseHeadSplit(tiling)
               ? chunkWork * HeadPartitionCount(tiling)
               : chunkWork;
}

__aicore__ inline void DecodeWorkItem(
    const PrepareRuntimeTiling &tiling, uint32_t workItem,
    uint32_t &chunk, uint32_t &headPartition)
{
    if (UseHeadSplit(tiling)) {
        const uint32_t partitionCount = HeadPartitionCount(tiling);
        chunk = workItem / partitionCount;
        headPartition = workItem % partitionCount;
        return;
    }
    chunk = workItem;
    headPartition = 0;
}

__aicore__ inline void HeadRange(
    const PrepareRuntimeTiling &tiling, uint32_t headPartition,
    uint32_t &headBegin, uint32_t &headEnd)
{
    if (!UseHeadSplit(tiling)) {
        headBegin = 0;
        headEnd = tiling.valueHeadNum;
        return;
    }
    const uint32_t headsPerPartition =
        tiling.headsPerPartition == 0 ? tiling.valueHeadNum
                                      : tiling.headsPerPartition;
    headBegin = headPartition * headsPerPartition;
    headEnd = headBegin + headsPerPartition;
    if (headEnd > tiling.valueHeadNum) {
        headEnd = tiling.valueHeadNum;
    }
}

__aicore__ inline bool ResolveChunk(const PrepareKernelArgs &args,
                                    uint32_t globalChunk, ChunkRange &chunk)
{
    chunk.globalChunk = globalChunk;
    if (!args.tiling.isVarLen) {
        const uint32_t chunksPerSequence =
            CeilDiv(args.tiling.seqLen, Shape::kChunkRows);
        if (chunksPerSequence == 0) {
            return false;
        }
        chunk.sequence = globalChunk / chunksPerSequence;
        chunk.batchIndex = chunk.sequence;
        const uint32_t chunkInSequence = globalChunk % chunksPerSequence;
        chunk.tokenBegin = chunkInSequence * Shape::kChunkRows;
        chunk.validRows = args.tiling.seqLen - chunk.tokenBegin;
        if (chunk.validRows > Shape::kChunkRows) {
            chunk.validRows = Shape::kChunkRows;
        }
        return chunk.sequence < args.tiling.batch && chunk.validRows != 0;
    }

    // 变长路径沿用仓内 chunkIndices[globalChunk] = {sequence, localChunk} 合同；
    // chunkIndices 可空，此时按 cuSeqlens 顺序扫描到 globalChunk。
    const __gm__ int64_t *indices =
        reinterpret_cast<const __gm__ int64_t *>(args.chunkIndices);
    const __gm__ int64_t *cu =
        reinterpret_cast<const __gm__ int64_t *>(args.cuSeqlens);
    if (cu == nullptr || args.tiling.seqNum == 0) {
        return false;
    }
    int64_t sequence = -1;
    int64_t localChunk = -1;
    if (indices != nullptr) {
        sequence = indices[globalChunk * 2];
        localChunk = indices[globalChunk * 2 + 1];
    } else {
        uint32_t chunkPrefix = 0;
        for (uint32_t seq = 0; seq < args.tiling.seqNum; ++seq) {
            const int64_t sequenceBegin = cu[seq];
            const int64_t sequenceEnd = cu[seq + 1];
            if (sequenceBegin < 0 || sequenceEnd < sequenceBegin) {
                return false;
            }
            const uint32_t sequenceChunks = CeilDiv(
                static_cast<uint32_t>(sequenceEnd - sequenceBegin),
                Shape::kChunkRows);
            if (globalChunk < chunkPrefix + sequenceChunks) {
                sequence = static_cast<int64_t>(seq);
                localChunk = static_cast<int64_t>(globalChunk - chunkPrefix);
                break;
            }
            chunkPrefix += sequenceChunks;
        }
    }
    if (sequence < 0 || localChunk < 0 ||
        sequence >= static_cast<int64_t>(args.tiling.seqNum)) {
        return false;
    }
    const int64_t sequenceBegin = cu[sequence];
    const int64_t sequenceEnd = cu[sequence + 1];
    const int64_t tokenBegin =
        sequenceBegin + localChunk * Shape::kChunkRows;
    if (sequenceBegin < 0 || sequenceEnd < sequenceBegin ||
        sequenceEnd <= tokenBegin || tokenBegin >= args.tiling.seqLen ||
        sequenceEnd > args.tiling.seqLen) {
        return false;
    }
    chunk.sequence = static_cast<uint32_t>(sequence);
    chunk.batchIndex = 0;
    chunk.tokenBegin = static_cast<uint32_t>(tokenBegin);
    chunk.validRows = static_cast<uint32_t>(sequenceEnd - tokenBegin);
    if (chunk.validRows > Shape::kChunkRows) {
        chunk.validRows = Shape::kChunkRows;
    }
    return true;
}

__aicore__ inline uint32_t QkHeadForValueHead(
    const PrepareRuntimeTiling &tiling, uint32_t valueHead)
{
    if (tiling.qkHeadNum == 0 || tiling.valueHeadNum < tiling.qkHeadNum ||
        tiling.valueHeadNum % tiling.qkHeadNum != 0) {
        return 0;
    }
    const uint32_t headsPerQk = tiling.valueHeadNum / tiling.qkHeadNum;
    if (headsPerQk == 0) {
        return 0;
    }
    return valueHead / headsPerQk;
}

// Q/K 归一化结果按 HK 保存。GVA 中一个 HK 对应多个连续 HV，只有该
// QK 头组的第一个 HV 写公开输出，避免多个 AIV 重叠写同一段 GM。
__aicore__ inline bool IsQkOutputOwner(
    const PrepareRuntimeTiling &tiling, uint32_t valueHead)
{
    if (tiling.qkHeadNum == 0 || tiling.valueHeadNum < tiling.qkHeadNum ||
        tiling.valueHeadNum % tiling.qkHeadNum != 0) {
        return false;
    }
    const uint32_t headsPerQk = tiling.valueHeadNum / tiling.qkHeadNum;
    return headsPerQk != 0 && valueHead % headsPerQk == 0;
}

__aicore__ inline uint64_t QkInputOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t qkHead)
{
    if (tiling.inputSequenceMajor) {
        return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.seqLen +
                 chunk.tokenBegin) *
                    tiling.qkHeadNum +
                qkHead) *
               Shape::kHeadDim;
    }
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.qkHeadNum +
             qkHead) * tiling.seqLen + chunk.tokenBegin) * Shape::kHeadDim;
}

// q_hat/k_hat/q_rstd/k_rstd 固定写成 head-major，供反向直接读取；
// 输入即使是 sequence-major，也只影响上面的 QkInputOffset。
__aicore__ inline uint64_t QkHeadTensorOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t qkHead, uint32_t dimension)
{
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.qkHeadNum +
             qkHead) * tiling.seqLen + chunk.tokenBegin) * dimension;
}

__aicore__ inline uint64_t QkHeadScalarOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t qkHead)
{
    return (static_cast<uint64_t>(chunk.batchIndex) * tiling.qkHeadNum +
            qkHead) * tiling.seqLen + chunk.tokenBegin;
}

__aicore__ inline uint64_t ValueInputOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead)
{
    if (tiling.inputSequenceMajor) {
        return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.seqLen +
                 chunk.tokenBegin) * tiling.valueHeadNum + valueHead) *
               Shape::kValueDim;
    }
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
             valueHead) * tiling.seqLen + chunk.tokenBegin) *
           Shape::kValueDim;
}

__aicore__ inline uint64_t RawGateInputOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead)
{
    if (tiling.inputSequenceMajor) {
        return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.seqLen +
                 chunk.tokenBegin) * tiling.valueHeadNum + valueHead) *
               Shape::kHeadDim;
    }
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
             valueHead) * tiling.seqLen + chunk.tokenBegin) *
           Shape::kHeadDim;
}

__aicore__ inline uint64_t BetaInputOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead)
{
    if (tiling.inputSequenceMajor) {
        return (static_cast<uint64_t>(chunk.batchIndex) * tiling.seqLen +
                chunk.tokenBegin) * tiling.valueHeadNum + valueHead;
    }
    return (static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
            valueHead) * tiling.seqLen + chunk.tokenBegin;
}

// betaEff、qg/qgScaled/kg/gk 以及矩阵输出固定使用 head-major 布局。
__aicore__ inline uint64_t HeadTensorOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead, uint32_t dimension)
{
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
             valueHead) * tiling.seqLen + chunk.tokenBegin) * dimension;
}

__aicore__ inline uint64_t HeadScalarOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead)
{
    return (static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
            valueHead) * tiling.seqLen + chunk.tokenBegin;
}

__aicore__ inline uint64_t AOutputOffset(
    const PrepareRuntimeTiling &tiling, const ChunkRange &chunk,
    uint32_t valueHead)
{
    return ((static_cast<uint64_t>(chunk.batchIndex) * tiling.valueHeadNum +
             valueHead) * tiling.seqLen + chunk.tokenBegin) *
           Shape::kChunkRows;
}

__aicore__ inline uint64_t WorkspaceSlotBase(
    uint32_t workgroup, uint32_t slot, uint32_t workgroupStride)
{
    return static_cast<uint64_t>(workgroup) * workgroupStride +
           static_cast<uint64_t>(slot) * Workspace::kSlotStride;
}

__aicore__ inline uint64_t WorkspaceSlotBase(
    uint32_t workgroup, uint32_t slot, uint32_t workgroupStride,
    uint32_t slotStride)
{
    return static_cast<uint64_t>(workgroup) * workgroupStride +
           static_cast<uint64_t>(slot) * slotStride;
}

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_UTILS_H
