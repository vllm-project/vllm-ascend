// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_PAGED_TILING_DATA_H
#define INDEXER_REGION_SCORE_PAGED_TILING_DATA_H

#include <cstdint>
#include <cstddef>

namespace IndexerRegionScorePaged {
constexpr uint64_t PAGE_LOAD_ALIGNED = 0U;
constexpr uint64_t PAGE_LOAD_WITH_OFFSET = 1U;
// This POD is shared between host and device; all strides are in elements.
struct alignas(8) PagedTilingData {
    uint32_t totalTokens;
    uint32_t groups;
    uint32_t headsPerGroup;
    uint32_t proxyDim;
    uint32_t numRequests;
    uint32_t pageCols;
    uint32_t summariesPerPage;
    uint32_t seqK;                 // Logical R length; bounded by pageCols*summariesPerPage, fits in uint32.
    uint32_t qTile;                // Token granularity for Q-range sharding.
    uint32_t kTile;
    uint32_t usedCubeCores;
    uint32_t maskMode;
    // When headScale != 1.0, scale all head weights before the weighted ReLU reduction.
    float headScale;
    uint32_t keyPagesPerMacro;
    uint32_t qShardCount;          // Number of contiguous Q-range shards.
    uint32_t kShardCount;          // Number of contiguous K-macro shards per Q shard.
    // The Scalar page-segment planner requires positive block_table page IDs to be smaller than the physical page count of paged_key_summary.
    uint32_t numPhysicalPages;
    // Dtype encoding: 0=FP32, 1=BF16, 2=FP16; the AIV path widens the weights to FP32.
    uint32_t weightDtype;
    uint64_t qStrideToken;
    uint64_t qStrideGroup;
    uint64_t qStrideHead;
    uint64_t weightStrideToken;
    uint64_t weightStrideGroup;
    uint64_t cacheStridePage;
    uint64_t cacheStrideRegion;
    uint64_t blockTableStrideRequest;
    uint64_t outputStrideGroup;
    uint64_t outputStrideToken;
    uint64_t reservedStride0; // Byte offset of the scratch relative to the kernel workspace pointer.
    // Host-derived page-load mode, not a user attribute; 0=page-aligned, 1=in-page start supported.
    uint64_t pageLoadMode;
    uint32_t regionSize;
    // Bits [15:0] hold topK, bit 16 the ID-sort flag, bit 17 the scratch alignment.
    uint32_t selectionConfig;
    // Zero batchTokens selects the full-score path; coordinates stay global.
    uint32_t batchTokens;
    uint32_t batchBegin;
    uint32_t batchCount;
    uint32_t batchReserved;
    uint64_t scorePoolOffset; // bytes relative to the kernel workspace pointer
};
static_assert(offsetof(PagedTilingData, pageLoadMode) == 160, "page-load ABI offset");
static_assert(offsetof(PagedTilingData, regionSize) == 168, "region-size ABI offset");
static_assert(offsetof(PagedTilingData, selectionConfig) == 172, "selection ABI offset");
static_assert(sizeof(PagedTilingData) % 8 == 0, "paged tiling must be naturally aligned");
static_assert(offsetof(PagedTilingData, keyPagesPerMacro) == 52, "key-pages ABI offset");
static_assert(sizeof(PagedTilingData) == 200, "update every paged tiling-data packer");
} // namespace IndexerRegionScorePaged

// The CANN tiling registration macros require the standard host namespace; this alias is shared by the host and device entry points.
namespace optiling {
using PagedTilingData = IndexerRegionScorePaged::PagedTilingData;
} // namespace optiling

#endif
