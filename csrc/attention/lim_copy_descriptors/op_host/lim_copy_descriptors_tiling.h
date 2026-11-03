/**
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(LimCopyDescriptorsTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LimCopyDescriptors, LimCopyDescriptorsTilingData)

struct LimCopyDescriptorsCompileInfo {};
} // namespace optiling
