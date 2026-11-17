#ifndef ATTN_METADATA_UPDATE_TILING_H
#define ATTN_METADATA_UPDATE_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct AttnMetadataUpdateInfo {
    uint32_t maxBs;
    uint32_t originalBs;
    uint32_t tableSize;
    uint32_t layerIdx;
    uint32_t maxDiscardBs;
    uint64_t totalUbSize;
};

struct AttnMetadataUpdateTilingData {
    AttnMetadataUpdateInfo info;
};

#endif  // ATTN_METADATA_UPDATE_TILING_H
