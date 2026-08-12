/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file quant_batch_matmul_v3_tiling_data.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_TILING_DATA_H
#define MC2_QUANT_BATCH_MATMUL_V3_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace DequantBmm {
// Mc2QuantBatchMatmulV3Tiling set Mc2QuantBatchMatmulV3Params tilingData mc2 calls Mc2QuantBatchMatmulV3Tiling
// DoLibApiTiling
#pragma pack(push, 8)
struct Mc2QuantBatchMatmulV3DataParams {
    uint32_t batchA = 0;
    uint32_t batchB = 0;
    uint32_t batchC = 0;
    uint32_t batchA1 = 0;
    uint32_t batchA2 = 0;
    uint32_t batchA3 = 0;
    uint32_t batchA4 = 0;
    uint32_t batchB1 = 0;
    uint32_t batchB2 = 0;
    uint32_t batchB3 = 0;
    uint32_t batchB4 = 0;
    uint32_t batchC1 = 0;
    uint32_t batchC2 = 0;
    uint32_t batchC3 = 0;
    uint32_t batchC4 = 0;
    uint32_t singleCoreBatch = 0;
    uint32_t isPerTensor = 0;
    uint32_t isPertoken = 0;
    uint32_t isDoubleScale = 0;
    uint32_t biasThreeDim = 0;
    uint32_t ubCalcM = 0;
    uint32_t ubCalcN = 0;
    uint32_t needUbBuffer = 0;
    uint32_t realSingleCoreM = 0;
    uint32_t realSingleCoreN = 0;
    uint32_t biasDtype = 0; // 代替原来的isBiasBf16
    uint32_t ubSize = 0;
    uint32_t isMClash = 0;
    uint32_t isNClash = 0;
    uint32_t groupSizeM = 0;
    uint32_t groupSizeN = 0;
    uint32_t groupSizeK = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2L2cacheTileParams {
    uint32_t mTileCntL2 = 0;
    uint32_t nTileCntL2 = 0;
    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t calOrder = 0;
    uint32_t isBasicTiling = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2SlidingWindowParams {
    uint32_t mTailTile = 0;
    uint32_t nTailTile = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2QuantBatchMatmulV3TilingDataParams {
    Mc2QuantBatchMatmulV3DataParams params;
    TCubeTiling matmulTiling;
    Mc2L2cacheTileParams tileL2cacheTiling;
    Mc2SlidingWindowParams adaptiveSlidingWin;
};
#pragma pack(pop)
} // namespace DequantBmm
#endif // QUANT_BATCH_MATMUL_V3_TILING_DATA_H
