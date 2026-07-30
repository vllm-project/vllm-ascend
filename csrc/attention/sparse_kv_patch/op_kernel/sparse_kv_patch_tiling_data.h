/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SPARSE_KV_PATCH_TILING_DATA_H
#define SPARSE_KV_PATCH_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SKP_BLOCK_SIZE = 128;
constexpr uint32_t SKP_CTKV_DIM = 512;
constexpr uint32_t SKP_KPE_DIM = 64;
constexpr uint32_t SKP_COMBINED_DIM = SKP_CTKV_DIM + SKP_KPE_DIM;
constexpr uint32_t SKP_INDEX_TYPE_INT32 = 0;
constexpr uint32_t SKP_INDEX_TYPE_INT64 = 1;

BEGIN_TILING_DATA_DEF(SparseKvPatchTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numActual)
    TILING_DATA_FIELD_DEF(uint32_t, topkN)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, slotMappingType)
    TILING_DATA_FIELD_DEF(uint32_t, numPhysicalSlots)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseKvPatch, SparseKvPatchTilingData)

}  // namespace optiling

#endif  // SPARSE_KV_PATCH_TILING_DATA_H
