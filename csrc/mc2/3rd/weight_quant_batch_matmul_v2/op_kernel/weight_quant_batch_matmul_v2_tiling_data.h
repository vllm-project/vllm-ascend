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
 * \file weight_quant_batch_matmul_v2_tiling_data.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData {
    uint8_t hasBias;
    uint8_t reverse1;
    uint16_t reverse2;
    uint8_t vecBlockDimN;
    uint8_t vecBlockDimK;
    uint8_t cubeBlockDimN;
    uint8_t cubeBlockDimK;
    uint16_t postSingleN;
    uint16_t postSingleM;
    uint32_t vecSingleN;
    uint32_t singleK;
    uint32_t cubeSingleM;
    uint32_t cubeSingleN;
    uint32_t cubeBaseK;
    uint64_t postSingleCoreN;
    uint64_t cubeSingleCoreNLoop;
    uint64_t cubeSingleCoreNTailLoop;
    uint64_t singleCoreKLoop;
    uint64_t singleCoreKTailLoop;
    uint64_t vecSingleCoreNLoop;
    uint64_t vecSingleCoreNTailLoop;
    uint64_t cubeSingleCoreN;
    uint64_t cubeSingleCoreNTail;
    uint64_t cubeSingleCoreNOriTail;
    uint64_t singleCoreK;
    uint64_t singleCoreKTail;
    uint64_t singleCoreKOriTail;
    uint64_t vectorSingleCoreN;
    uint64_t vectorSingleCoreNTail;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t nSizeAlign;
    uint64_t kSizeAlign;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2TilingData {
    uint8_t vecBlockDimN;
    uint8_t vecBlockDimK;
    uint8_t cubeBlockDimN;
    uint8_t cubeBlockDimM;
    uint8_t cubeBlockDimK;
    uint8_t kPadSize;
    uint8_t nPadSize;
    uint8_t haveBatchA;
    uint8_t haveBatchB;
    uint8_t reserve1;
    uint8_t reserve2;
    uint8_t reserve3;
    uint16_t vecSingleKGroupNum;
    uint16_t vecSingleKTailGroupNum;
    uint16_t AL1Pingpong;
    uint16_t BL1Pingpong;
    uint32_t vecSingleK;
    uint32_t vecSingleN;
    uint32_t vec2SingleM;
    uint32_t vecSingleKTail;
    uint32_t vecSingleNTail;
    uint32_t wInQueueSize;
    uint32_t offsetInQueueSize;
    uint32_t scaleInQueueSize;
    uint32_t wOutQueueSize;
    uint32_t antiQuantTmpBufferSize;
    uint32_t vecCubeNRatio;
    uint32_t vecCubeTailNRatio;
    uint32_t vecCubeKRatio;
    uint32_t vecCubeTailKRatio;
    uint32_t cubeTailM;
    uint32_t cubeTailN;
    uint32_t cubeSingleNLoop;
    uint32_t cubeSingleNTailLoop;
    uint32_t repeatAxisMax;
    uint64_t vecSingleKLoop;
    uint64_t vecSingleNLoop;
    uint64_t vecSingleKTailLoop;
    uint64_t vecSingleNTailLoop;
    uint64_t vec2SingleMLoop;
    uint64_t kAlign;
    uint64_t nAlign;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t groupSize;
    uint64_t mSize;
    uint64_t blockBatch;
    uint64_t shapeBatch;
    uint64_t mAubSize;
    uint64_t kAubSize;
    uint64_t nBubSize;
    uint64_t kBubSize;
    uint64_t mCubSize;
    uint64_t nCubSize;
    uint64_t mAL1Size;
    uint64_t kAL1Size;
    uint64_t nBL1Size;
    uint64_t kBL1Size;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2FixpipeTilingData {
    uint8_t hasBias;
    uint8_t nBlockNum;
    uint16_t baseK;
    uint16_t baseM;
    uint16_t baseN;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2MsdGroupTilingData {
    uint8_t vecBlockDimN;
    uint8_t cubeBlockDimK;
    uint8_t cubeBlockDimN;
    uint8_t vec1SingleCoreM;
    uint8_t hasBias;
    uint8_t reserve1;
    uint16_t reserve2;
    uint32_t reserve3;
    uint32_t singleCoreK;
    uint32_t vecSingleCoreN;
    uint32_t singleCoreGroup;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t groupSize;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2MsdTilingData {
    uint8_t cubeBlockDimN;
    uint8_t cubeBlockDimM;
    uint8_t cubeBlockDimK;
    uint8_t hasBias;
    uint16_t v1BaseM;
    uint16_t preloadTimes;
    uint16_t taskNSize;
    uint16_t taskSingleCoreNSize;
    uint16_t postProcessBaseM;
    uint16_t postProcessSingleCoreM;
    uint32_t preProcessUsedVecNum;
    uint32_t v1BaseK;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t groupPack;
    uint64_t groupSize;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2NzTilingData {
    uint8_t cubeBlockDimN;
    uint8_t cubeBlockDimM;
    uint16_t AL1Pingpong;
    uint16_t BL1Pingpong;
    uint64_t kAlign;
    uint64_t nAlign;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t mAubSize;
    uint64_t kAubSize;
    uint64_t nBubSize;
    uint64_t kBubSize;
    uint64_t mCubSize;
    uint64_t nCubSize;
    uint64_t mAL1Size;
    uint64_t kAL1Size;
    uint64_t nBL1Size;
    uint64_t kBL1Size;
    uint64_t groupSize;
    uint64_t batchX0;
    uint64_t batchX1;
    uint64_t batchX2;
    uint64_t batchX3;
    uint64_t batchWeight0;
    uint64_t batchWeight1;
    uint64_t batchWeight2;
    uint64_t batchWeight3;
    uint64_t batchY0;
    uint64_t batchY1;
    uint64_t batchY2;
    uint64_t batchY3;
    uint32_t biasWithBatch;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2RegBaseTilingData {
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimM = 0;
    uint8_t vecCoreParallel = 0;
    uint8_t reserve1 = 0;

    uint16_t AL1Pingpong = 0;
    uint16_t BL1Pingpong = 0;

    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupSize = 0;
    uint64_t mSize = 0;
    uint64_t nBubSize = 0;
    uint64_t kBubSize = 0;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2WeightQuantBatchMatmulV2ASTilingData {
    uint32_t cubeBlockDimM = 0;
    uint32_t cubeBlockDimN = 0;
    uint32_t hasBias = 0;
    uint32_t firstTailBlockCount = 0;
    uint32_t secondTailBlockCount = 0;
    uint32_t weightL2Cacheable = 0;
    uint32_t mainBlockL1Size = 0;
    uint32_t firstTailBlockL1Size = 0;
    uint32_t secondTailBlockL1Size = 0;
    uint32_t aPreloadSize = 0;
    uint64_t groupSize = 0;
    uint64_t mainBlockCount = 0;
    uint64_t mSize = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) WeightQuantBatchMatmulV2ASWTilingData {
    uint32_t hasBias = 0;
    uint32_t mTailTile = 0;
    uint32_t nTailTile = 0;
    uint32_t reserved1 = 0;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#endif
