/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef SFA_REMAP_SPARSE_INDICES_TILING_H
#define SFA_REMAP_SPARSE_INDICES_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SfaRemapSparseIndicesTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, topK);
    TILING_DATA_FIELD_DEF(uint32_t, dcpSize);
    TILING_DATA_FIELD_DEF(uint32_t, dcpRank);
    TILING_DATA_FIELD_DEF(uint32_t, interleaveSize);
    TILING_DATA_FIELD_DEF(uint32_t, interleaveShift);
    TILING_DATA_FIELD_DEF(uint32_t, dcpInterleaveShift);
    TILING_DATA_FIELD_DEF(uint32_t, usePowerOfTwo);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, bufferBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SfaRemapSparseIndices, SfaRemapSparseIndicesTilingData)

struct SfaRemapSparseIndicesCompileInfo {};
}  // namespace optiling

#endif
