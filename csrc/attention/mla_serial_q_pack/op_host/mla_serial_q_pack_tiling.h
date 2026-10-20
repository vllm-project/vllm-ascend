// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MlaSerialQPackTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokens);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, nope);
TILING_DATA_FIELD_DEF(uint32_t, tileTokens);
TILING_DATA_FIELD_DEF(uint32_t, useRope);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MlaSerialQPack, MlaSerialQPackTilingData)
}
