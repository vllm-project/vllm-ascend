/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * Description: Sparse token-band geometry for the arch22 regular kernel.
 */
#ifndef GBSA_SPARSE_KV_GEOMETRY_HPP
#define GBSA_SPARSE_KV_GEOMETRY_HPP

#include <cstdint>

#if defined(__CCE_AICORE__)
#define GBSA_GEOMETRY_INLINE __aicore__ inline
#else
#define GBSA_GEOMETRY_INLINE inline
#endif

namespace GbsaSparseGeometry {

// Visible length of one logical Y-block under causal/KV end.
GBSA_GEOMETRY_INLINE uint32_t ClipBlockVisibleLen(uint32_t logicalBlockId, uint32_t blockShapeY,
                                                 uint32_t visibleEnd)
{
    const uint32_t blockBegin = logicalBlockId * blockShapeY;
    if (blockBegin >= visibleEnd) {
        return 0U;
    }
    const uint32_t remaining = visibleEnd - blockBegin;
    return remaining < blockShapeY ? remaining : blockShapeY;
}

// Selected blocks may appear in any order; each contributes ClipBlockVisibleLen.
template <class Indices>
GBSA_GEOMETRY_INLINE void CalcSparseKvSize(Indices &indices, uint32_t count, uint32_t blockShapeY,
                                         uint32_t visibleEnd, uint32_t &validTokens, bool &hasClipped,
                                         uint32_t &fullPrefixSlots)
{
    validTokens = 0;
    hasClipped = false;
    fullPrefixSlots = 0;
    bool stillFullPrefix = true;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t id = indices.GetValue(i);
        if (id < 0) {
            break;
        }
        const uint32_t len = ClipBlockVisibleLen(static_cast<uint32_t>(id), blockShapeY, visibleEnd);
        if (len != blockShapeY) {
            hasClipped = true;
            stillFullPrefix = false;
        } else if (stillFullPrefix) {
            ++fullPrefixSlots;
        }
        validTokens += len;
    }
}

// FD coordinates include the implicit residual block after the explicit list.
// In residual mode, explicit indices contain only full visible blocks and exclude the tail.
template <bool hasResidualBlock, class Indices>
GBSA_GEOMETRY_INLINE void CalcSparseKvSize(Indices &indices, uint32_t explicitCount, uint32_t rangeBegin,
                                         uint32_t rangeEnd, uint32_t blockShapeY, uint32_t visibleEnd,
                                         uint32_t &validTokens, bool &hasClipped, uint32_t &fullPrefixSlots)
{
    if constexpr (hasResidualBlock) {
        const uint32_t explicitBegin = rangeBegin < explicitCount ? rangeBegin : explicitCount;
        const uint32_t explicitEnd = rangeEnd < explicitCount ? rangeEnd : explicitCount;
        validTokens = (explicitEnd - explicitBegin) * blockShapeY;
        fullPrefixSlots = explicitEnd - explicitBegin;
        hasClipped = false;
        const uint32_t residualSize = visibleEnd % blockShapeY;
        if (residualSize != 0U && rangeBegin <= explicitCount && rangeEnd > explicitCount) {
            validTokens += residualSize;
            hasClipped = true;
        }
    } else {
        auto gTaskSparseBlockIdx = indices[rangeBegin];
        CalcSparseKvSize(gTaskSparseBlockIdx, rangeEnd - rangeBegin, blockShapeY, visibleEnd, validTokens,
                         hasClipped, fullPrefixSlots);
    }
}

template <bool hasResidualBlock, class Indices>
GBSA_GEOMETRY_INLINE int32_t GetSparseBlockId(Indices &indices, uint32_t localBlockIdx,
                                           uint32_t localExplicitCount, uint32_t residualBlockId)
{
    if constexpr (hasResidualBlock) {
        if (localBlockIdx == localExplicitCount) {
            return static_cast<int32_t>(residualBlockId);
        }
    }
    return indices.GetValue(localBlockIdx);
}

// Map an offset in the compressed validKvTokens band to the owning select slot.
template <bool hasResidualBlock, class Indices>
GBSA_GEOMETRY_INLINE void ResolveSparseBandOffset(Indices &indices, uint32_t bandOffset, uint32_t selectNum,
                                                  uint32_t blockShapeY, uint32_t visibleEnd,
                                                  uint32_t localExplicitCount, uint32_t residualBlockId,
                                                  uint32_t &localSelectIdx, uint32_t &offsetInBlock,
                                                  uint32_t &logicalBlockId)
{
    localSelectIdx = 0U;
    offsetInBlock = 0U;
    logicalBlockId = 0U;
    uint32_t remain = bandOffset;
    while (localSelectIdx < selectNum) {
        logicalBlockId = static_cast<uint32_t>(GetSparseBlockId<hasResidualBlock>(
            indices, localSelectIdx, localExplicitCount, residualBlockId));
        const uint32_t yActual = ClipBlockVisibleLen(logicalBlockId, blockShapeY, visibleEnd);
        if (yActual == 0U) {
            break;
        }
        if (remain < yActual) {
            offsetInBlock = remain;
            return;
        }
        remain -= yActual;
        ++localSelectIdx;
    }
    offsetInBlock = 0U;
    if (localSelectIdx < selectNum) {
        logicalBlockId = static_cast<uint32_t>(GetSparseBlockId<hasResidualBlock>(
            indices, localSelectIdx, localExplicitCount, residualBlockId));
    }
}

template <bool hasResidualBlock, class Indices>
GBSA_GEOMETRY_INLINE void LocateSparseBand(Indices &indices, uint32_t bandOffset, uint32_t selectNum,
                                         uint32_t blockShapeY, uint32_t visibleEnd,
                                         uint32_t localExplicitCount, uint32_t residualBlockId, bool hasClipped,
                                         uint32_t fullPrefixSlots, uint32_t &localSelectIdx,
                                         uint32_t &offsetInBlock, uint32_t &logicalBlockId)
{
    if constexpr (hasResidualBlock) {
        localSelectIdx = bandOffset / blockShapeY;
        offsetInBlock = bandOffset % blockShapeY;
    } else if (!hasClipped) {
        localSelectIdx = bandOffset / blockShapeY;
        offsetInBlock = bandOffset % blockShapeY;
    } else if (bandOffset < fullPrefixSlots * blockShapeY) {
        localSelectIdx = bandOffset / blockShapeY;
        offsetInBlock = bandOffset % blockShapeY;
    } else {
        ResolveSparseBandOffset<hasResidualBlock>(indices, bandOffset, selectNum, blockShapeY, visibleEnd,
                                                   localExplicitCount, residualBlockId, localSelectIdx, offsetInBlock,
                                                   logicalBlockId);
        return;
    }
    logicalBlockId = static_cast<uint32_t>(
        GetSparseBlockId<hasResidualBlock>(indices, localSelectIdx, localExplicitCount, residualBlockId));
}

} // namespace GbsaSparseGeometry
#endif // GBSA_SPARSE_KV_GEOMETRY_HPP
