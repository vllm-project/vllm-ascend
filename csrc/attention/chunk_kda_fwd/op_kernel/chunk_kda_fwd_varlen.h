#pragma once

#include "kernel_operator.h"

namespace KdaVarlen {

__aicore__ inline bool ResolveChunkRange(
    const __gm__ int64_t *cuSeqlens, const __gm__ int64_t *chunkIndices,
    uint64_t seqNum, uint64_t totalTokens, uint64_t chunkSize,
    uint64_t flatChunk, uint64_t &seq, uint64_t &start, uint64_t &end)
{
    if (cuSeqlens == nullptr || chunkSize == 0) {
        return false;
    }

    int64_t seqValue = -1;
    int64_t localChunkValue = -1;
    if (chunkIndices != nullptr) {
        const uint64_t metadataOffset = flatChunk * 2;
        seqValue = chunkIndices[metadataOffset];
        localChunkValue = chunkIndices[metadataOffset + 1];
    } else {
        uint64_t chunkPrefix = 0;
        for (uint64_t seqIdx = 0; seqIdx < seqNum; ++seqIdx) {
            const int64_t seqStart = cuSeqlens[seqIdx];
            const int64_t seqEnd = cuSeqlens[seqIdx + 1];
            if (seqStart < 0 || seqEnd < seqStart) {
                return false;
            }
            const uint64_t seqLength = static_cast<uint64_t>(seqEnd - seqStart);
            const uint64_t seqChunks = (seqLength + chunkSize - 1) / chunkSize;
            if (flatChunk < chunkPrefix + seqChunks) {
                seqValue = static_cast<int64_t>(seqIdx);
                localChunkValue = static_cast<int64_t>(flatChunk - chunkPrefix);
                break;
            }
            chunkPrefix += seqChunks;
        }
    }

    if (seqValue < 0 || localChunkValue < 0 ||
        static_cast<uint64_t>(seqValue) >= seqNum) {
        return false;
    }
    const int64_t seqStartValue = cuSeqlens[seqValue];
    const int64_t seqEndValue = cuSeqlens[seqValue + 1];
    if (seqStartValue < 0 || seqEndValue < seqStartValue) {
        return false;
    }
    const uint64_t seqStart = static_cast<uint64_t>(seqStartValue);
    const uint64_t seqEnd = static_cast<uint64_t>(seqEndValue);
    const uint64_t localChunk = static_cast<uint64_t>(localChunkValue);
    start = seqStart + localChunk * chunkSize;
    if (start >= seqEnd || start >= totalTokens) {
        return false;
    }
    end = start + chunkSize;
    if (end > seqEnd) {
        end = seqEnd;
    }
    if (end > totalTokens) {
        return false;
    }
    seq = static_cast<uint64_t>(seqValue);
    return start < end;
}

} // namespace KdaVarlen
