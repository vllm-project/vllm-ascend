/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 */
#ifndef TRIANGLE_PAGED_SPARSE_ATTENTION_TILING_H
#define TRIANGLE_PAGED_SPARSE_ATTENTION_TILING_H

#include <cstdint>

/*
 * ABI shared by the host tiler and the AscendC kernel.
 *
 * implementationStatus values:
 *   0: build-only skeleton (must not dispatch);
 *   1: bounded AIV numerical reference kernel;
 *   2: direct-paged production Cube/Vector pipeline.
 *
 * ABI v2 appends the production task geometry and a byte-addressed workspace
 * layout.  The v1 prefix is intentionally stable so the numerical oracle can
 * remain in the same binary while the long-prefill path is brought up.
 */
struct TrianglePagedSparseAttentionTilingData {
    uint32_t magic;
    uint32_t abiVersion;
    uint32_t implementationStatus;
    uint32_t blockDim;

    uint32_t queryTokens;
    uint32_t queryStart;
    uint32_t seqLen;
    uint32_t promptLen;

    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    uint32_t pageSize;

    uint32_t physicalPageCount;
    uint32_t blockTablePageCapacity;
    uint32_t queryTile;
    uint32_t taskCount;

    uint32_t sinkTokens;
    uint32_t localWindow;
    uint32_t denseTail;
    uint32_t sparseBegin;

    uint32_t sparseEnd;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;

    float scale;

    // ABI v2: one task is (query_tile, kv_head).  Four GQA query heads
    // belonging to one KV head are folded into Cube M:
    //   M = queryTile(32) * groupSize(4) = 128.
    uint32_t kvTile;
    uint32_t groupSize;
    uint32_t queryTileCount;
    uint32_t activeAicCores;

    // Per-AIC workspace.  Every offset and stride is in bytes; no packed K/V,
    // block mask, or selected-index array exists in this ABI.
    uint32_t workspacePerCoreBytes;
    uint32_t scoreOffsetBytes;
    uint32_t probabilityOffsetBytes;
    uint32_t outputTmpOffsetBytes;

    uint32_t outputUpdateOffsetBytes;
    uint32_t lseScratchOffsetBytes;
    uint32_t workspaceBytes;
    uint32_t pipelineStages;
};

#endif  // TRIANGLE_PAGED_SPARSE_ATTENTION_TILING_H
