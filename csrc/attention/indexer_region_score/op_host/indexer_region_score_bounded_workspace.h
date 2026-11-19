// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_BOUNDED_WORKSPACE_H
#define INDEXER_REGION_SCORE_BOUNDED_WORKSPACE_H
#include <algorithm>
#include <cstdint>
namespace IndexerRegionScoreHost {
// Bounds the IRS layout; outer Contiguous/ViewCopy allocations are separate.
constexpr uint64_t kTopKOnlyWorkspaceBudgetBytes = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t kBoundedExecutorReserve = 4096U;
struct BoundedWorkspacePlan {
    uint32_t batchTokens = 0U;
    uint64_t regionStride = 0U;
    uint64_t scorePoolOffset = 0U;
    uint64_t totalBytes = 0U;
};
inline bool PlanBoundedWorkspace(uint64_t libraryBytes, uint32_t groups,
    uint32_t tokens, uint32_t regions, uint32_t qMicroTile, BoundedWorkspacePlan &plan)
{
    plan = {};
    const uint64_t budget = kTopKOnlyWorkspaceBudgetBytes - kBoundedExecutorReserve;
    if (groups == 0U || tokens == 0U || regions > 16777216U || libraryBytes > budget - 31U) return false;
    const uint64_t prefix = (libraryBytes + 31U) & ~uint64_t(31U);
    // R=0 still uses a positive physical stride; selection reads no scores.
    plan.regionStride = std::max(uint64_t(8U), (uint64_t(regions) + 7U) & ~uint64_t(7U));
    const uint64_t bytesPerToken = uint64_t(groups) * plan.regionStride * sizeof(float);
    uint64_t batch = std::min(uint64_t(tokens), (budget - prefix) / bytesPerToken);
    if (batch == 0U) return false;
    if (batch < tokens && qMicroTile > 1U && batch >= qMicroTile) batch = batch / qMicroTile * qMicroTile;
    plan.batchTokens = static_cast<uint32_t>(batch);
    plan.scorePoolOffset = prefix - libraryBytes;
    plan.totalBytes = prefix + batch * bytesPerToken;
    return true;
}
} // namespace IndexerRegionScoreHost
#endif
