/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_v2_tiling.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_V2_TILING_H
#define SCATTER_ND_UPDATE_V2_TILING_H
#include "register/tilingdata_base.h"
constexpr uint64_t MAX_DIM_NUM = 8;
namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterNdUpdateV2ScatterTiling)
TILING_DATA_FIELD_DEF(uint64_t, scatterLength)
TILING_DATA_FIELD_DEF(uint64_t, tailRow)
TILING_DATA_FIELD_DEF(uint64_t, frontRow)
TILING_DATA_FIELD_DEF(uint64_t, frontNum)
TILING_DATA_FIELD_DEF(uint64_t, tailNum)
TILING_DATA_FIELD_DEF(uint64_t, ubLengthForUpdates)
TILING_DATA_FIELD_DEF(uint64_t, scatterAlignLength)
TILING_DATA_FIELD_DEF(uint64_t, formDim)
TILING_DATA_FIELD_DEF(uint64_t, copyRow)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileNum)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileLength)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileTail)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileAlignLength)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateV2ScatterTilingOp, ScatterNdUpdateV2ScatterTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateV2LinearIndexTiling)
TILING_DATA_FIELD_DEF(uint64_t, coreNum)
TILING_DATA_FIELD_DEF(uint64_t, ubSize)
TILING_DATA_FIELD_DEF(uint64_t, indexDim)
TILING_DATA_FIELD_DEF(uint64_t, blockLength)
TILING_DATA_FIELD_DEF(uint64_t, blockNum)
TILING_DATA_FIELD_DEF(uint64_t, blockRemainLength)
TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, sortWorkspace)
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, indicesMask)
TILING_DATA_FIELD_DEF(uint64_t, isInt64Indices)
TILING_DATA_FIELD_DEF(uint64_t, needLargeIndexKernel)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateV2LinearIndexTilingOp, ScatterNdUpdateV2LinearIndexTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateV2ScatterTiling, scatterTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateV2LinearIndexTiling, linearIndexTiling)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateV2, ScatterNdUpdateV2TilingData)
REGISTER_TILING_DATA_CLASS(ScatterNdUpdateV2TilingDataOp, ScatterNdUpdateV2TilingData)

struct ScatterNdUpdateV2CompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif // SCATTER_ND_UPDATE_V2_TILING_H
