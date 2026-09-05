#ifndef BUILD_DSPARK_SWA_INDICES_TILING_H
#define BUILD_DSPARK_SWA_INDICES_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BuildDsparkSwaIndicesTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, numDecodeTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numSpeculativeTokens);
    TILING_DATA_FIELD_DEF(uint32_t, windowSize);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, indexWidth);
    TILING_DATA_FIELD_DEF(uint32_t, blockTableStride);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BuildDsparkSwaIndices, BuildDsparkSwaIndicesTilingData)

struct BuildDsparkSwaIndicesCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
};
}  // namespace optiling

#endif
