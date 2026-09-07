#ifndef DSA_LOCAL_METADATA_TILING_H
#define DSA_LOCAL_METADATA_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DsaLocalMetadataTilingData)
    // ---- 算子属性 ----
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);            // 请求数
    TILING_DATA_FIELD_DEF(int32_t, localStart);          // 本 rank token 区间起点
    TILING_DATA_FIELD_DEF(int32_t, localEnd);            // 本 rank token 区间终点
    TILING_DATA_FIELD_DEF(uint32_t, computeStartPos);    // 0 = 不写 start_pos_out, 1 = 写
END_TILING_DATA_DEF;

struct DsaLocalMetadataCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(DsaLocalMetadata, DsaLocalMetadataTilingData)

}  // namespace optiling

#endif  // DSA_LOCAL_METADATA_TILING_H
