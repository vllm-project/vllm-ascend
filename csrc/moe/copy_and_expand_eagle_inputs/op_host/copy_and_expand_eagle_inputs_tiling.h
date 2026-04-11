#ifndef COPY_AND_EXPAND_EAGLE_INPUTS_TILING_H
#define COPY_AND_EXPAND_EAGLE_INPUTS_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyAndExpandEagleInputsTilingData)
    // ---- 分核参数 ----
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);            // 实际使用的核数
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);                // 总请求数
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);            // 每核基础请求数
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);          // 余数（前 remainder 个核多处理 1 个请求）

    // ---- 算子属性 ----
    TILING_DATA_FIELD_DEF(int32_t, paddingTokenId);          // 填充 token ID
    TILING_DATA_FIELD_DEF(int32_t, parallelDraftingTokenId); // 并行推测解码 token ID
    TILING_DATA_FIELD_DEF(uint32_t, numPaddingSlotsPerReq);  // 每个请求的 padding 槽位数
    TILING_DATA_FIELD_DEF(uint32_t, totalInputTokens);       // 输入 token 总数（用于 clamp）
    TILING_DATA_FIELD_DEF(uint32_t, shiftInputIds);          // 0 = false, 1 = true

    // ---- 输出尺寸 ----
    TILING_DATA_FIELD_DEF(uint32_t, totalDraftTokens);       // 输出 token 总数
END_TILING_DATA_DEF;

struct CopyAndExpandEagleInputsCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(CopyAndExpandEagleInputs, CopyAndExpandEagleInputsTilingData)

}  // namespace optiling

#endif  // COPY_AND_EXPAND_EAGLE_INPUTS_TILING_H
