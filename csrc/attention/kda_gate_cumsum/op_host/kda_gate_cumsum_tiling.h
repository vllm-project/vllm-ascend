/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(KdaGateCumsumTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, hv);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, rank);
TILING_DATA_FIELD_DEF(int64_t, layout);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
TILING_DATA_FIELD_DEF(int64_t, hasALog);
TILING_DATA_FIELD_DEF(int64_t, hasDtBias);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, useGateInKernel);
TILING_DATA_FIELD_DEF(int64_t, safeGate);
TILING_DATA_FIELD_DEF(float, lowerBound);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KdaGateCumsum, KdaGateCumsumTilingData)

struct KdaGateCumsumCompileInfo {};
} // namespace optiling
