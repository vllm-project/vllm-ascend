#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_TILING_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_TILING_H

#include <cstdint>

struct FusedQuantLightningIndexerManageTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t totalQueryRows;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t indexHeads;
    uint32_t maxBlockNumPerBatch;
    uint32_t weightStride;
    uint32_t keyStride;
    uint32_t scaleStride;
    uint32_t managementWorkspaceOffset;
    uint32_t scoreWorkspaceStride;
    uint32_t scoreRowStride;
    uint32_t queryTileSize;
    // Reserve max(4 * B, T) route rows. Four-route launches keep their
    // original per-request layout; wider launches index rows by query ID.
    uint32_t routeStorageRows;
    uint32_t compactRouteRows;
};

#endif
