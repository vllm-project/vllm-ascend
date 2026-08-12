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
 * \file quant_batch_matmul_v3_tiling_data.h.h
 * \brief
 */

#ifndef quant_batch_matmul_v3_tiling_data_H
#define quant_batch_matmul_v3_tiling_data_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) Mc2QuantBatchMatmulV3Params {
    uint32_t batchA;
    uint32_t batchB;
    uint32_t batchC;
    uint32_t batchA1;
    uint32_t batchA2;
    uint32_t batchA3;
    uint32_t batchA4;
    uint32_t batchB1;
    uint32_t batchB2;
    uint32_t batchB3;
    uint32_t batchB4;
    uint32_t batchC1;
    uint32_t batchC2;
    uint32_t batchC3;
    uint32_t batchC4;
    uint32_t singleCoreBatch;
    uint32_t isPerTensor;
    uint32_t isPertoken;
    uint32_t isDoubleScale;
    uint32_t biasThreeDim;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
    uint32_t realSingleCoreM;
    uint32_t realSingleCoreN;
    uint32_t biasDtype;
    uint32_t ubSize;
    uint32_t isMClash;
    uint32_t isNClash;
    uint32_t groupSizeM;
    uint32_t groupSizeN;
    uint32_t groupSizeK;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2L2cacheTileParam {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
    uint32_t isBasicTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2SlidingWindowParam {
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2QuantBatchMatmulV3TilingData {
    Mc2QuantBatchMatmulV3Params params;
    TCubeTiling matmulTiling;
    Mc2L2cacheTileParam tileL2cacheTiling;
    Mc2SlidingWindowParam adaptiveSlidingWin;
};
#pragma pack(pop)

#endif
