#ifndef KVTC_PREPARE_TILING_H
#define KVTC_PREPARE_TILING_H

#include <cstdint>

struct KvtcPrepareTilingData {
    uint32_t numLayers;
    uint32_t numGpuBlocks;
    uint32_t selectedBlocks;
    uint32_t blockSize;
    uint32_t numKvHeads;
    uint32_t headSize;

    uint32_t layerSize;
    uint32_t k;

    uint32_t rowsPerTile;
    uint32_t rowTilesPerBlock;
    uint32_t taskCount;
    uint32_t usedAiv;

    uint32_t tileElements;
    uint32_t tileBytes;

    // GM <-> UB DataCopyParams fields use 32-byte blocks on Ascend 910B.
    uint32_t blockLen32;
    uint32_t outputStride32;
};

#endif // KVTC_PREPARE_TILING_H
