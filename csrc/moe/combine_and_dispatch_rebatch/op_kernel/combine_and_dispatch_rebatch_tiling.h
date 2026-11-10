#ifndef COMBINE_AND_DISPATCH_REBATCH_TILING_H
#define COMBINE_AND_DISPATCH_REBATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct CombineAndDispatchRebatchInfo {
    uint32_t isCombineRebatch;   // flag: is combine rebatch mode
    uint32_t hiddenSize;         // hidden dimension size
    uint32_t maxBs;              // max batch size capacity
    uint32_t k;                  // top-k value per token
    // uint32_t aivNum;             // number of AIV cores
    uint64_t totalUbSize;        // total UB size
};

struct CombineAndDispatchRebatchTilingData {
    CombineAndDispatchRebatchInfo combineAndDispatchRebatchInfo;
};

#endif // COMBINE_AND_DISPATCH_REBATCH_TILING_H
