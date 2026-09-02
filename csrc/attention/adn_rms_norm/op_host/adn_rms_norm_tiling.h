#ifndef ADN_RMS_NORM_TILING_H
#define ADN_RMS_NORM_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AdnRmsNormTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerTile);
    TILING_DATA_FIELD_DEF(uint64_t, baseRowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, extraRowCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, reducePartCount);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, invHiddenSize);
END_TILING_DATA_DEF;

struct AdnRmsNormCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

REGISTER_TILING_DATA_CLASS(AdnRmsNorm, AdnRmsNormTilingData)

}  // namespace optiling

#endif  // ADN_RMS_NORM_TILING_H
