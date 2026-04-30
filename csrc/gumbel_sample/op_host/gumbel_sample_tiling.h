/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GUMBEL_SAMPLE_TILING_H
#define GUMBEL_SAMPLE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GumbelSampleTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, nRowsLarge);
    TILING_DATA_FIELD_DEF(uint32_t, nRowsSmall);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, numTiles);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, applyTemp);
END_TILING_DATA_DEF;

// CompileInfo：编译期缓存硬件核数，避免每次 Tiling 重新查询平台信息
struct GumbelSampleCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GumbelSample, GumbelSampleTilingData)

}  // namespace optiling

#endif  // GUMBEL_SAMPLE_TILING_H
