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
 * \file compress_norm_rope_tiling_data.h
 * \brief CompressNormRope tiling data（A2/A3 两阶段重构版，扁平结构，kernel/host 共用）
 *
 * C4（coff=2）按"组"均分任务；C128（coff=1）按"组×dChunk"均分任务 + workspace 中转两阶段。
 */

#ifndef COMPRESS_NORM_ROPE_TILING_DATA_H
#define COMPRESS_NORM_ROPE_TILING_DATA_H
#include <cstdint>

namespace optiling {

struct CompressNormRopeTilingData {
    uint32_t batchSize = 0;             // B = cu_seqlens.dim(0) - 1
    uint32_t tokenSize = 0;             // T = mm_kv.dim(0)
    uint32_t headDim = 0;               // D = mm_kv.dim(1) / coff
    uint32_t cmpRatio = 0;              // r: 4 (coff=2) | 128 (coff=1)
    uint32_t usedCoreNum = 0;           // blockDim = aivNum
    uint32_t blockSize = 0;             // state_cache 分页块行数（c4=8, c128=32）
    uint32_t maxBlockNumPerBatch = 0;   // state_block_table.dim(1)
    uint32_t stateCacheStrideDim0 = 0;  // state_cache 第 0 维 stride（元素数）；与 ref 一致用 uint32 避免 uint64 跨编译器对齐不一致
    uint32_t mmKvStrideDim0 = 0;        // mm_kv 第 0 维 stride（元素数），支持 fused GEMM 输出切片
    uint32_t mmScoreStrideDim0 = 0;     // mm_score 第 0 维 stride（元素数），支持 fused GEMM 输出切片
    uint32_t dChunkSize = 0;            // c128 的 d 分块列数（64）；c4 = headDim
    uint32_t dChunkNum = 0;             // headDim / dChunkSize（c4 = 1）
    uint32_t maxGroupTaskNum = 0;       // 组数上界 = T/r + 2B（start_pos 未对齐跨组）
    uint32_t maxTaskNum = 0;            // maxGroupTaskNum * dChunkNum
    uint32_t taskPerCore = 0;           // maxTaskNum / usedCoreNum
    uint32_t taskRem = 0;               // maxTaskNum % usedCoreNum
    uint32_t ropeHeadDim = 0;           // rope 维度（生产=64），作用于行尾 [headDim-ropeHeadDim, headDim)
    uint32_t rotaryMode = 0;            // 1=HALF | 2=INTERLEAVE（生产=2）
    float normEps = 1e-6f;              // rms_norm eps
    uint32_t maxScNum = 0;              // 压缩行数上界 = min(T, T/r + B)（c128 workspace 行数/二阶段分行）
};

} // namespace optiling

#endif // COMPRESS_NORM_ROPE_TILING_DATA_H
