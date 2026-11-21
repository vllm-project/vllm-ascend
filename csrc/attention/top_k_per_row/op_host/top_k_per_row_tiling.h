/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef TOP_K_PER_ROW_TILING_H
#define TOP_K_PER_ROW_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TopKPerRowTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_rows);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, topk_pad);      // CeilAlign(topk, 32)
TILING_DATA_FIELD_DEF(uint32_t, rows_per_core);
TILING_DATA_FIELD_DEF(uint32_t, chunk_size);    // elements per chunk, 32-aligned
TILING_DATA_FIELD_DEF(uint32_t, max_chunks);    // chunks merged in one round
TILING_DATA_FIELD_DEF(uint32_t, num_round);
TILING_DATA_FIELD_DEF(uint32_t, tmp_size);      // bytes of TopK tmp buffer
TILING_DATA_FIELD_DEF(uint32_t, dtype_size);    // 2 or 4
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topk_tiling_chunk);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topk_tiling_merge);
END_TILING_DATA_DEF;

struct TopKPerRowCompileInfo {
    uint32_t total_core_num = 0;
    uint64_t total_ub_size = 0;
};

REGISTER_TILING_DATA_CLASS(TopKPerRow, TopKPerRowTilingData)
}  // namespace optiling
#endif
