/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON_HPP
#define KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace SparseAttn {

constexpr uint32_t SASA_FD_MAX_AIC = 32;
constexpr uint32_t SASA_FD_MAX_BASE_TASK = 32;

struct SparseAttentionScoreTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t topK;
    uint32_t maxBlocksPerBatch;
    uint32_t totalQTokens;
    uint32_t totalTaskNum;
    uint32_t firstBatchTaskNum;
    float scaleValue;
    uint32_t innerPrecise;
    uint32_t maxQSeqlen;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
    uint32_t groupSize;
    // BaseTileInfo
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
    // MmPhaseL1TileInfo
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
    uint32_t fdLseSubStride;
    // FD core-range metadata.
    uint32_t fdCorePerCoreTaskNum;
    uint32_t fdCoreTaskStart[SASA_FD_MAX_AIC];
    uint32_t fdCoreTaskEnd[SASA_FD_MAX_AIC];
    // FD combine-range metadata.
    uint32_t fdCombineTaskNum;
    uint32_t fdCombineBaseTask[SASA_FD_MAX_BASE_TASK];
    uint32_t fdPartialStartByBase[SASA_FD_MAX_BASE_TASK];
    uint32_t fdPartialCountByBase[SASA_FD_MAX_BASE_TASK];
    uint64_t fdIdentityOffset;
    uint64_t fdPartialLseOffset;
    uint64_t fdPartialOOffset;
};

}  // namespace SparseAttn

#endif  // KERNEL_COMMON_HPP
