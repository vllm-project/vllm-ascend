/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_mat_mul_v3_tiling_data.h
 * \brief
 */

#ifndef BATCH_MAT_MUL_V3_TILING_DATA_H
#define BATCH_MAT_MUL_V3_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#include "../../mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) Mc2MultiBatchInfo {
    uint32_t batchUsedCoreNum;
    uint32_t aBatchDimAll;
    uint32_t bBatchDimAll;
    uint32_t cBatchDimAll;
    uint32_t aBatchDim0;
    uint32_t bBatchDim0;
    uint32_t cBatchDim0;
    uint32_t aBatchDim1;
    uint32_t bBatchDim1;
    uint32_t cBatchDim1;
    uint32_t aBatchDim2;
    uint32_t bBatchDim2;
    uint32_t cBatchDim2;
    uint32_t aBatchDim3;
    uint32_t bBatchDim3;
    uint32_t cBatchDim3;
    uint32_t iterBatch;
    uint32_t biasWithBatch;
    uint32_t mOri;
    uint32_t batchTileBlock;
    uint32_t aBatch;
    uint32_t bBatch;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) Mc2BatchMatmulTilingData {
    Mc2MatmulV3TilingData matmulTiling;
    Mc2MultiBatchInfo Mc2multiBatchInfo;
};
#pragma pack(pop)

#endif
