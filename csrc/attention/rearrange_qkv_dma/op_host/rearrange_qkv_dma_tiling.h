// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#pragma once

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RearrangeQkvDmaTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, tokens);
    TILING_DATA_FIELD_DEF(uint64_t, qDim);
    TILING_DATA_FIELD_DEF(uint64_t, kDim);
    TILING_DATA_FIELD_DEF(uint64_t, vDim);
    TILING_DATA_FIELD_DEF(uint64_t, rowDim);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RearrangeQkvDma, RearrangeQkvDmaTilingData)

struct RearrangeQkvDmaCompileInfo {};
}  // namespace optiling
