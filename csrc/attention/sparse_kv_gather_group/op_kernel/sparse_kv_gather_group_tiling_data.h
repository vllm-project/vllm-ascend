/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file sparse_kv_gather_tiling_data.h
 * \brief Shared tiling-data definition for SparseKvGather.
 *
 * Included by both the host tiling layer and the AscendC device kernel.
 */

#ifndef SPARSE_KV_GATHER_GROUP_TILING_DATA_H
#define SPARSE_KV_GATHER_GROUP_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SKG_BLOCK_SIZE = 128;
constexpr uint32_t SKG_CTKV_DIM   = 512;
constexpr uint32_t SKG_KPE_DIM    = 64;
constexpr uint32_t SKG_HEAD_NUM   = 1;

enum class SKGIndexType : uint32_t {
    INT32 = 0,
    INT64 = 1,
};

BEGIN_TILING_DATA_DEF(SparseKvGatherGroupTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, numActual)
    TILING_DATA_FIELD_DEF(uint32_t, maxBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, topkN)
    TILING_DATA_FIELD_DEF(uint32_t, numCacheLayers)
    TILING_DATA_FIELD_DEF(uint64_t, totalSlots)
    TILING_DATA_FIELD_DEF(uint64_t, slotsPerCore)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableType)
    TILING_DATA_FIELD_DEF(uint32_t, topkIndicesType)
    TILING_DATA_FIELD_DEF(uint32_t, curPosType)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseKvGatherGroup, SparseKvGatherGroupTilingData)

}  // namespace optiling

#endif  // SPARSE_KV_GATHER_GROUP_TILING_DATA_H
