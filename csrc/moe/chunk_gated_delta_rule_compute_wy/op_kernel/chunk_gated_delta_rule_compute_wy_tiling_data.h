/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleComputeWyTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t numChunks;
    int64_t groupSize;
    int64_t totalTasks;
    uint32_t localWorkspaceSize;
    uint32_t perCoreWorkspaceBytes;
    uint32_t usedCoreNum;
    uint32_t reserved0;
    TCubeTiling mmAttn;
    TCubeTiling mmSquare;
    TCubeTiling mmApplyU;
    TCubeTiling mmApplyW;
};
#pragma pack(pop)

#endif // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
