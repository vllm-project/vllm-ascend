/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef QSA_UNION_INPUTS_TILING_H
#define QSA_UNION_INPUTS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QsaUnionInputsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, groups);
TILING_DATA_FIELD_DEF(uint32_t, input_groups);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(uint32_t, compact_row_stride);
TILING_DATA_FIELD_DEF(uint32_t, sort_tmp_bytes);
TILING_DATA_FIELD_DEF(uint32_t, groups_per_core);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QsaUnionInputs, QsaUnionInputsTilingData)
}  // namespace optiling
#endif
