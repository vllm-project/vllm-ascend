/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file chunk_gated_delta_rule_tiling_data.h
 * \brief
 */

#ifndef CHUNK_GATED_DELTA_RULE_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRule {
    constexpr uint64_t STRUCT_ALIGNAS = 8;
    #pragma pack(push, 8)
    struct alignas(STRUCT_ALIGNAS) ChunkGatedDeltaRuleTilingData {
        int64_t aiCoreNum;
        int64_t t;
        int64_t nk;
        int64_t dk;
        int64_t nv;
        int64_t dv;
        int64_t b;
        int64_t hasGamma;
        int64_t chunkSize;
        int64_t maxGroupLength;    // maxGroupLength = p * chunkSize
        int64_t interWorkspaceSz;
        int64_t stageWorkspaceSz;
        int64_t stageOneParaNum;
        float scale;
        AscendC::tiling::TCubeTiling matmulTilingFp32;    // BF16 C matmul tiling
        AscendC::tiling::TCubeTiling matmulTilingFp32C;   // FP32 C matmul tiling (for FP32 state path)
        int64_t stateIsFp32;
        int64_t stateStride0;
        int64_t stateStride1;
    };
    #pragma pack(pop)

    struct ChunkGroup {
        int64_t startPos = 0;    // 该ChunkGroup在T上的起始位置
        int64_t length = 0;      // 该ChunkGroup的长度
        int64_t chunkSize = 0;   // 每个chunk的长度
        int64_t coreStart = 0;   // 预留
        int64_t coreEnd = 0;     // 预留
    };
}  // ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H