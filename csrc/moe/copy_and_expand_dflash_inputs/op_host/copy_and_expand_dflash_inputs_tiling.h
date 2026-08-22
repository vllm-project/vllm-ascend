#ifndef COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H
#define COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyAndExpandDflashInputsTilingData)
    // ---- core partition ----
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);            // cores actually used (always 1)
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);                // number of requests

    // ---- shape metadata ----
    TILING_DATA_FIELD_DEF(uint32_t, numContext);             // total context tokens
    TILING_DATA_FIELD_DEF(uint32_t, blockTableStride);       // stride of block_table dim 0 (elements)

    // ---- op attributes ----
    TILING_DATA_FIELD_DEF(int32_t, parallelDraftingTokenId); // mask token id for drafting
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);              // kv-cache block size
    TILING_DATA_FIELD_DEF(uint32_t, numQueryPerReq);         // 1 + K (dflash) or K (dspark)
    TILING_DATA_FIELD_DEF(uint32_t, numSpeculativeTokens);   // K
    TILING_DATA_FIELD_DEF(uint32_t, sampleFromAnchor);       // 0 = dflash, 1 = dspark
END_TILING_DATA_DEF;

struct CopyAndExpandDflashInputsCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(CopyAndExpandDflashInputs, CopyAndExpandDflashInputsTilingData)

}  // namespace optiling

#endif  // COPY_AND_EXPAND_DFLASH_INPUTS_TILING_H
