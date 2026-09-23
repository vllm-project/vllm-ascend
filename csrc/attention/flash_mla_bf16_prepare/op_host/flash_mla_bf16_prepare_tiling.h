// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(FlashMlaBf16PrepareTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens);
TILING_DATA_FIELD_DEF(int64_t, heads);
TILING_DATA_FIELD_DEF(int64_t, ropeHeads);
TILING_DATA_FIELD_DEF(int64_t, keyTokenStride);
TILING_DATA_FIELD_DEF(int64_t, keyHeadStride);
TILING_DATA_FIELD_DEF(int64_t, valueTokenStride);
TILING_DATA_FIELD_DEF(int64_t, valueHeadStride);
TILING_DATA_FIELD_DEF(int64_t, ropeTokenStride);
TILING_DATA_FIELD_DEF(int64_t, ropeHeadStride);
TILING_DATA_FIELD_DEF(uint32_t, tileTokens);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashMlaBf16Prepare, FlashMlaBf16PrepareTilingData)
struct FlashMlaBf16PrepareCompileInfo {};
} // namespace optiling
