/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_ARCH35_LAYOUT_UTILS_HPP
#define GBSA_ARCH35_LAYOUT_UTILS_HPP

#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#endif

namespace GbsaKernelArch35 {

enum class Format {
    TND = 0,
    BNSD = 1,
    BSND = 2
};

// Sorted valid-prefix indices. FD ranges use virtual block coordinates: explicit
// blocks [0, explicitCount), optionally followed by one implicit residual block.
template <bool hasResidualBlock, class Indices>
__aicore__ inline void CalcSparseKvSize(Indices& indices, uint32_t explicitCount, uint32_t rangeBegin,
                                        uint32_t rangeEnd, uint32_t blockShapeY, uint32_t visibleEnd,
                                        uint32_t& validTokens)
{
    if constexpr (hasResidualBlock) {
        const uint32_t explicitBegin = rangeBegin < explicitCount ? rangeBegin : explicitCount;
        const uint32_t explicitEnd = rangeEnd < explicitCount ? rangeEnd : explicitCount;
        validTokens = (explicitEnd - explicitBegin) * blockShapeY;
        const uint32_t residualSize = visibleEnd % blockShapeY;
        if (residualSize != 0U && rangeBegin <= explicitCount && rangeEnd > explicitCount) {
            validTokens += residualSize;
        }
    } else {
        const uint32_t count = rangeEnd - rangeBegin;
        validTokens = count * blockShapeY;
        if (count != 0U) {
            const uint64_t lastStart = static_cast<uint64_t>(indices.GetValue(rangeEnd - 1U)) * blockShapeY;
            const uint32_t remaining = static_cast<uint32_t>(visibleEnd - lastStart);
            const uint32_t lastSize = remaining < blockShapeY ? remaining : blockShapeY;
            validTokens -= blockShapeY - lastSize;
        }
    }
}

template <bool hasResidualBlock, class Indices>
__aicore__ inline int32_t GetSparseBlockId(Indices& indices, uint32_t localBlockIdx, uint32_t localExplicitCount,
                                           uint32_t residualBlockId)
{
    if constexpr (hasResidualBlock) {
        if (localBlockIdx == localExplicitCount) {
            return static_cast<int32_t>(residualBlockId);
        }
    }
    return indices.GetValue(localBlockIdx);
}

// Four slots cover QK/PV prelaunch and one tile of address lookahead.
// Each group: [copy count, 7 reserved words], followed by plain uint64 entries.
// Meta packs (sourceStride << 32) | (length << 1) | (ndNum - 1).
// Dense: [element offset, meta], retaining the original four-word capacity.
// PA: [K element offset, V element offset, K meta, V meta].
constexpr uint32_t KV_OFFSET_STAGES = 4U;
constexpr uint32_t KV_OFFSET_FLAG_BASE = 7U;
constexpr uint32_t KV_OFFSET_FIRST_READY_FLAG_BASE = KV_OFFSET_FLAG_BASE + KV_OFFSET_STAGES;

template <bool isPaged>
__aicore__ inline uint32_t CalcKvOffsetGroupWords(uint32_t l1TileN, uint32_t blockY)
{
    uint32_t blocks = (l1TileN + 2U * blockY - 2U) / blockY;
    if constexpr (isPaged) {
        blocks = blocks * 2U < l1TileN ? blocks * 2U : l1TileN;
    }
    constexpr uint32_t recordWords = 4U;
    return (8U + blocks * recordWords + 7U) / 8U * 8U;
}

// BSA-style tensor strides. Packed GQA traverses heads along the GEMM M axis,
// so its Q/O row stride is HeadStride, not TokenStride.
template <Format format>
struct GbsaLayoutHelper {
    __aicore__ static inline uint64_t TokenStride(uint32_t heads, uint32_t dim)
    {
        if constexpr (format == Format::BNSD) {
            return dim;
        }
        return static_cast<uint64_t>(heads) * dim;
    }

    __aicore__ static inline uint64_t HeadStride(uint32_t seqCapacity, uint32_t dim)
    {
        if constexpr (format == Format::BNSD) {
            return static_cast<uint64_t>(seqCapacity) * dim;
        }
        return dim;
    }

    __aicore__ static inline uint64_t Offset(uint64_t batchBase, uint32_t token, uint32_t head, uint32_t heads,
                                             uint32_t dim, uint32_t seqCapacity = 0)
    {
        return batchBase + static_cast<uint64_t>(token) * TokenStride(heads, dim) +
               static_cast<uint64_t>(head) * HeadStride(seqCapacity, dim);
    }
};

struct BatchSeqInfo {
    // Flattened storage-token base: cu[b] for TND, b * seqCapacity for batched layouts.
    // This is not yet an element offset; each tensor layout supplies its own strides.
    uint64_t storageStart;
    uint32_t storageLen;
    uint32_t actualLen;
};

template <Format format, class CuTensor, class UsedTensor>
__aicore__ inline void ReadBatchSeqInfo(CuTensor& cu, UsedTensor& used, bool hasUsed, uint32_t batch, BatchSeqInfo& seq,
                                        uint32_t seqCapacity = 0)
{
    if constexpr (format == Format::TND) {
        const int64_t start = cu.GetValue(batch);
        const int64_t end = cu.GetValue(static_cast<int64_t>(batch) + 1);
        seq.storageStart = static_cast<uint64_t>(start);
        seq.storageLen = static_cast<uint32_t>(end - start);
        seq.actualLen = hasUsed ? static_cast<uint32_t>(used.GetValue(batch)) : seq.storageLen;
    } else {
        seq.storageStart = static_cast<uint64_t>(batch) * seqCapacity;
        seq.storageLen = seqCapacity;
        seq.actualLen = static_cast<uint32_t>(used.GetValue(batch));
    }
}

template <class UsedTensor>
__aicore__ inline void DecodeBatchedQToken(uint32_t qToken, uint32_t batchCount, uint32_t seqCapacity, UsedTensor& used,
                                           uint32_t& batchIdx, uint32_t& tokenInBatch, BatchSeqInfo& seq)
{
    uint32_t actualPrefix = 0U;
    for (uint32_t b = 0U; b < batchCount; ++b) {
        seq.storageStart = static_cast<uint64_t>(b) * seqCapacity;
        seq.storageLen = seqCapacity;
        seq.actualLen = static_cast<uint32_t>(used.GetValue(b));
        if (qToken < actualPrefix + seq.actualLen) {
            batchIdx = b;
            tokenInBatch = qToken - actualPrefix;
            return;
        }
        actualPrefix += seq.actualLen;
    }
}

// Q/O addressing uses the same batch-local token/head coordinates for every layout.
template <Format format>
__aicore__ inline uint64_t GetTensorOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head, uint32_t heads,
                                           uint32_t dim)
{
    return GbsaLayoutHelper<format>::Offset(seq.storageStart * heads * dim, token, head, heads, dim, seq.storageLen);
}

template <Format qFormat>
__aicore__ inline uint64_t GetLseOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head, uint32_t heads)
{
    // GBSA API: TND -> [T,N,1]; both BSND and BNSD -> [B,N,S,1].
    constexpr Format lseLayout = qFormat == Format::TND ? Format::TND : Format::BNSD;
    return GetTensorOffset<lseLayout>(seq, token, head, heads, 1);
}

template <Format kvFormat, bool isPaged>
__aicore__ inline uint64_t KvBlockOffset(uint32_t blockId, const BatchSeqInfo& seq, uint32_t head, uint32_t heads,
                                         uint32_t dim, uint32_t blockShapeY, uint64_t pageStride)
{
    if constexpr (isPaged) {
        return GbsaLayoutHelper<kvFormat>::Offset(static_cast<uint64_t>(blockId) * pageStride, 0, head, heads, dim,
                                                  blockShapeY);
    } else {
        return GetTensorOffset<kvFormat>(seq, 0, head, heads, dim) +
               static_cast<uint64_t>(blockId) * blockShapeY * GbsaLayoutHelper<kvFormat>::TokenStride(heads, dim);
    }
}

} // namespace GbsaKernelArch35

#endif // GBSA_ARCH35_LAYOUT_UTILS_HPP
