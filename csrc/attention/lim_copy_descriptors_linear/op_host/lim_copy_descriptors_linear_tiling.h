/**
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(LimCopyDescriptorsLinearTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LimCopyDescriptorsLinear, LimCopyDescriptorsLinearTilingData)

struct LimCopyDescriptorsLinearCompileInfo {};
} // namespace optiling
