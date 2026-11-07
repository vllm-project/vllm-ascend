// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EngramHashIdsTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, numLayers);
TILING_DATA_FIELD_DEF(int64_t, maxNgram);
TILING_DATA_FIELD_DEF(int64_t, numHeads);
TILING_DATA_FIELD_DEF(int64_t, numSlots);
TILING_DATA_FIELD_DEF(int64_t, numTableRows);
TILING_DATA_FIELD_DEF(int64_t, maxBlocks);
TILING_DATA_FIELD_DEF(int64_t, lookbackDepth);
TILING_DATA_FIELD_DEF(int64_t, padId);
TILING_DATA_FIELD_DEF(int64_t, deadId);
TILING_DATA_FIELD_DEF(int64_t, cacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, inputStride);
TILING_DATA_FIELD_DEF(int64_t, maskStride);
TILING_DATA_FIELD_DEF(int64_t, positionStride);
TILING_DATA_FIELD_DEF(int64_t, tableRowStride);
TILING_DATA_FIELD_DEF(int64_t, tableColStride);
TILING_DATA_FIELD_DEF(int64_t, queryStride);
TILING_DATA_FIELD_DEF(int64_t, lookbackRowStride);
TILING_DATA_FIELD_DEF(int64_t, lookbackColStride);
TILING_DATA_FIELD_DEF(int64_t, lookbackMaskRowStride);
TILING_DATA_FIELD_DEF(int64_t, lookbackMaskColStride);
TILING_DATA_FIELD_DEF(uint32_t, tokenTileSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, hasCache);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EngramHashIds, EngramHashIdsTilingData)
struct EngramHashIdsCompileInfo {};
} // namespace optiling
