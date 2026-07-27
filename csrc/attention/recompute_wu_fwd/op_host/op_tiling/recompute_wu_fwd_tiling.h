/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_tiling.h
 * \brief Registered tiling data for recompute_wu_fwd.
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(RecomputeWUFwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, Hk);
TILING_DATA_FIELD_DEF(int64_t, Hv);
TILING_DATA_FIELD_DEF(int64_t, hvPerHk);
TILING_DATA_FIELD_DEF(int64_t, T);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, V);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, vbVecRow);
TILING_DATA_FIELD_DEF(int64_t, kbgExpVecRow);
TILING_DATA_FIELD_DEF(int64_t, isVariable);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RecomputeWUFwd, RecomputeWUFwdTilingData)

struct RecomputeWUFwdCompileInfo {};

} // namespace optiling
