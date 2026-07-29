/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef KV_CACHE_FULL_BLOCK_DUMP_TILING_H
#define KV_CACHE_FULL_BLOCK_DUMP_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KvCacheFullBlockDumpTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, taskCount);
TILING_DATA_FIELD_DEF(int64_t, srcBlockNum);
TILING_DATA_FIELD_DEF(int64_t, dstBlockNum);
TILING_DATA_FIELD_DEF(int64_t, plane0ElementsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, plane1ElementsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, plane0ChunkElements);
TILING_DATA_FIELD_DEF(int64_t, plane1ChunkElements);
TILING_DATA_FIELD_DEF(int64_t, plane0TasksPerRow);
TILING_DATA_FIELD_DEF(int64_t, tasksPerRow);
TILING_DATA_FIELD_DEF(int64_t, bufferBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    KvCacheFullBlockDump, KvCacheFullBlockDumpTilingData)
}  // namespace optiling

#endif  // KV_CACHE_FULL_BLOCK_DUMP_TILING_H
