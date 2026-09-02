#ifndef REJECTION_SAMPLE_GREEDY_310_TILING_H
#define REJECTION_SAMPLE_GREEDY_310_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RejectionSampleGreedyV310TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, maxSpecLen);
    TILING_DATA_FIELD_DEF(uint32_t, alignedOutputLen);
END_TILING_DATA_DEF;

struct RejectionSampleGreedyV310CompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(RejectionSampleGreedyV310, RejectionSampleGreedyV310TilingData)

}  // namespace optiling

#endif  // REJECTION_SAMPLE_GREEDY_310_TILING_H
