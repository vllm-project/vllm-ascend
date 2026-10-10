/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

// MUST be pack(1). The host serializer (register/tilingdata_base.h::FieldHandler) lays
// fields out with no alignment padding at all, so any padding here shifts every field
// after it. With pack(8) the uint64 workspaceOffset below lands at offset 96 instead of
// 92 and all four TCubeTiling structs at 104 instead of 100, giving bogus GM staging
// pointers and garbage matmul tiling ("The DDR address of the MTE instruction is out of
// range" on 310P).
#pragma pack(push, 1)
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
    uint64_t workspaceOffset;
    TCubeTiling mmAttn;
    TCubeTiling mmSquare;
    TCubeTiling mmApplyU;
    TCubeTiling mmApplyW;
};
#pragma pack(pop)

#endif // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
