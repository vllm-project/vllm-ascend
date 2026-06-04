#ifndef GUMBEL_SAMPLE_TILING_H
#define GUMBEL_SAMPLE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GumbelSampleTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, numReqStates);
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, nRowsLarge);
    TILING_DATA_FIELD_DEF(uint32_t, nRowsSmall);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, numTiles);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, applyTemp);
    TILING_DATA_FIELD_DEF(uint32_t, hasProcessedLogits);
    TILING_DATA_FIELD_DEF(uint32_t, hasProcessedLogitsCol);
    TILING_DATA_FIELD_DEF(uint32_t, processedLogitsStride);
    TILING_DATA_FIELD_DEF(uint32_t, numSpeculativeSteps);
END_TILING_DATA_DEF;

// CompileInfo：编译期缓存硬件核数，避免每次 Tiling 重新查询平台信息。
struct GumbelSampleCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GumbelSample, GumbelSampleTilingData)

}  // namespace optiling

#endif  // GUMBEL_SAMPLE_TILING_H
