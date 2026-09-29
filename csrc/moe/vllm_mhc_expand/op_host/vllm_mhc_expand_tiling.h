// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(VllmMhcExpandTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, tokens);
    TILING_DATA_FIELD_DEF(uint64_t, hidden);
    TILING_DATA_FIELD_DEF(uint64_t, mhcMult);
    TILING_DATA_FIELD_DEF(uint64_t, tilesPerRow);
    TILING_DATA_FIELD_DEF(uint64_t, totalTiles);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(VllmMhcExpand, VllmMhcExpandTilingData)
}  // namespace optiling
