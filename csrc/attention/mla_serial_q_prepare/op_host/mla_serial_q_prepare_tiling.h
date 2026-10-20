// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MlaSerialQPrepareTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokens);
TILING_DATA_FIELD_DEF(uint32_t, rank);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, nope);
TILING_DATA_FIELD_DEF(uint32_t, headsPerTask);
TILING_DATA_FIELD_DEF(uint32_t, useRope);
TILING_DATA_FIELD_DEF(uint32_t, cacheBlocks);
TILING_DATA_FIELD_DEF(uint64_t, cacheStride);
TILING_DATA_FIELD_DEF(uint64_t, ropeStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MlaSerialQPrepare, MlaSerialQPrepareTilingData)
}
