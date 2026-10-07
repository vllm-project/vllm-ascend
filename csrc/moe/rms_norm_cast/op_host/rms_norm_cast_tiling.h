/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_TILING_H
#define VLLM_ASCEND_RMS_NORM_CAST_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormCastTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row);
TILING_DATA_FIELD_DEF(uint32_t, num_col);
TILING_DATA_FIELD_DEF(uint32_t, num_col_aligned);
TILING_DATA_FIELD_DEF(uint32_t, rows_per_core);
TILING_DATA_FIELD_DEF(float, inv_num_col);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

struct RmsNormCastCompileInfo {
    uint32_t total_core_num = 0;
    uint64_t total_ub_size = 0;
};

REGISTER_TILING_DATA_CLASS(RmsNormCast, RmsNormCastTilingData)
}  // namespace optiling
#endif
