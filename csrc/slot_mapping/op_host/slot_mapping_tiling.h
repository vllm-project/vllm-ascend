/**
 * @file slot_mapping_tiling.h
 * @brief SlotMapping 算子 TilingData 定义（vllm-ascend Host 侧集成版）
 */
#ifndef SLOT_MAPPING_TILING_H
#define SLOT_MAPPING_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SlotMappingTilingData)
    TILING_DATA_FIELD_DEF(int32_t, numReqs);
    TILING_DATA_FIELD_DEF(int32_t, blockTableStride);
    TILING_DATA_FIELD_DEF(int32_t, totalCpWorldSize);
    TILING_DATA_FIELD_DEF(int32_t, totalCpRank);
    TILING_DATA_FIELD_DEF(int32_t, cpKvCacheInterleaveSize);
    TILING_DATA_FIELD_DEF(int32_t, padId);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, maxNumTokens);
    TILING_DATA_FIELD_DEF(int32_t, numTokens);
END_TILING_DATA_DEF;

struct SlotMappingCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(SlotMapping, SlotMappingTilingData)

}  // namespace optiling

#endif  // SLOT_MAPPING_TILING_H
