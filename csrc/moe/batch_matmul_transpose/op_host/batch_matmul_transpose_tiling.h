/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_transpose_tiling.h
 * \brief
 */
#ifndef BATCH_MATMUL_TRANSPOSE_TILING_H
#define BATCH_MATMUL_TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchMatmulTransposeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, m0);
TILING_DATA_FIELD_DEF(uint32_t, k0);
TILING_DATA_FIELD_DEF(uint32_t, n0);
TILING_DATA_FIELD_DEF(uint32_t, mLoop);
TILING_DATA_FIELD_DEF(uint32_t, kLoop);
TILING_DATA_FIELD_DEF(uint32_t, nLoop);
TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
TILING_DATA_FIELD_DEF(uint32_t, splitk);
TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
END_TILING_DATA_DEF;

struct BatchMatmulTransposeCompileInfo {
    uint32_t coreNumAic = 0;
    uint32_t coreNumAiv = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l2Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

REGISTER_TILING_DATA_CLASS(BatchMatmulTranspose, BatchMatmulTransposeTilingData)
} // namespace optiling

#endif // BATCH_MATMUL_TRANSPOSE_TILING_H
