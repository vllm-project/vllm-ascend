// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherMlaPrefillTilingData)
TILING_DATA_FIELD_DEF(int64_t, requests);
TILING_DATA_FIELD_DEF(int64_t, pages);
TILING_DATA_FIELD_DEF(int64_t, tableColumns);
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, tilesPerRequest);
TILING_DATA_FIELD_DEF(int64_t, latentPageStride);
TILING_DATA_FIELD_DEF(int64_t, latentRowStride);
TILING_DATA_FIELD_DEF(int64_t, ropePageStride);
TILING_DATA_FIELD_DEF(int64_t, ropeRowStride);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherMlaPrefill, GatherMlaPrefillTilingData)
struct GatherMlaPrefillCompileInfo {};
} // namespace optiling
