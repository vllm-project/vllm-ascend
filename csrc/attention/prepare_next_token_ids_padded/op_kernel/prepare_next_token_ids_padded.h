#ifndef PREPARE_NEXT_TOKEN_IDS_PADDED_H
#define PREPARE_NEXT_TOKEN_IDS_PADDED_H

#include "kernel_tiling/kernel_tiling.h"


struct PrepareNextTokenIdsPaddedTilingInfo{
    uint32_t batchSize;
    uint32_t sampledTokensPerRequest;
    uint32_t sampledTokensAligned;
    uint32_t usedCoreNum;
    uint32_t rowsPerCore;
    uint32_t extraRowCoreNum;
    uint32_t rowsPerTile;
    int32_t vocabSize;
};

#endif
