#ifndef KVTC_RESTORE_ROPE_TILING_H
#define KVTC_RESTORE_ROPE_TILING_H

#include <cstdint>

struct KvtcRestoreRopeTilingData {
    uint32_t numLayers;
    uint32_t numGpuBlocks;
    uint32_t selectedBlocks;
    uint32_t blockSize;
    uint32_t numKvHeads;
    uint32_t headSize;

    uint32_t ropeCacheRows;
    uint32_t layerSize;
    uint32_t k;

    uint32_t rowsPerTile;
    uint32_t rowTilesPerBlock;
    uint32_t layerGroups;
    uint32_t taskCount;
    uint32_t usedAiv;

    uint32_t tileElements;
    uint32_t tileBytes;
    uint32_t ropeTileElements;
    uint32_t ropeTileBytes;

    uint32_t blockLen32;
    uint32_t inputStride32;
};

#endif // KVTC_RESTORE_ROPE_TILING_H
