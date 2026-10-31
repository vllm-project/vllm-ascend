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

// Indices contain an ascending valid prefix.
template <class Indices>
GBSA_GEOMETRY_INLINE void CalcSparseKvSize(Indices &indices, uint32_t count, uint32_t blockShapeY,
                                         uint32_t visibleEnd, uint32_t &validTokens)
{
    validTokens = 0;
    if (count == 0U) {
        return;
    }
    const uint64_t lastStart = static_cast<uint64_t>(indices.GetValue(count - 1U)) * blockShapeY;
    if (lastStart < visibleEnd) {
        const uint64_t remaining = visibleEnd - lastStart;
        validTokens = (count - 1U) * blockShapeY +
                      (remaining < blockShapeY ? static_cast<uint32_t>(remaining) : blockShapeY);
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t id = indices.GetValue(i);
        const uint64_t start = static_cast<uint64_t>(id) * blockShapeY;
        if (start >= visibleEnd) {
            break;
        }
        const uint64_t remaining = visibleEnd - start;
        validTokens += remaining < blockShapeY ? static_cast<uint32_t>(remaining) : blockShapeY;
    }
}

// Residual mode: full explicit blocks followed by an implicit partial block.
template <bool hasResidualBlock, class Indices>
GBSA_GEOMETRY_INLINE void CalcSparseKvSize(Indices &indices, uint32_t explicitCount, uint32_t rangeBegin,
                                         uint32_t rangeEnd, uint32_t blockShapeY, uint32_t visibleEnd,
                                         uint32_t &validTokens)
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
        auto gTaskSparseBlockIdx = indices[rangeBegin];
        CalcSparseKvSize(gTaskSparseBlockIdx, rangeEnd - rangeBegin, blockShapeY, visibleEnd, validTokens);
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

#if defined(__CCE_AICORE__)
template <bool hasResidualBlock>
GBSA_GEOMETRY_INLINE int32_t ReadSparseBlockId(const __gm__ int32_t *indices, uint32_t localBlockIdx,
                                            uint32_t localExplicitCount, uint32_t residualBlockId)
{
    if constexpr (hasResidualBlock) {
        if (localBlockIdx == localExplicitCount) {
            return static_cast<int32_t>(residualBlockId);
        }
    }
    return indices[localBlockIdx];
}
#endif

} // namespace GbsaSparseGeometry
#endif // GBSA_SPARSE_KV_GEOMETRY_HPP
