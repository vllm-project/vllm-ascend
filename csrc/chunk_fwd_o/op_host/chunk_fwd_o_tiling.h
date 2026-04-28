/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkFwdOTilingData)
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int64_t, kStride0);
TILING_DATA_FIELD_DEF(int64_t, vStride0);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, attnWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, aftermaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, maskWorkspaceOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkFwdO, ChunkFwdOTilingData)

struct ChunkFwdOCompileInfo {};
} // namespace optiling
