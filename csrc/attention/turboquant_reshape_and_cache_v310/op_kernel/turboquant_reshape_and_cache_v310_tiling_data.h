/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_DATA_H
#define TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

/*
 * Scenario selection:
 *   bit-width  -> TILING KEY (200 + bits). Compile-time: pack shift amounts must
 *                 be constants to unroll, and LEVELS sizes the codebook buffer.
 *   variant    -> runtime field below (MSE vs MSE+QJL)
 *   codebook   -> runtime field below (uniform affine vs Lloyd-Max LUT)
 * Three kernel instantiations cover all twelve A/B scenarios.
 */
struct TurboquantReshapeAndCacheV310TilingData {
    uint32_t numTokens;        // tokens in this launch
    uint32_t numKvHeads;       // 4 for Qwen3.5-4B
    uint32_t headDim;          // 256
    uint32_t blockSize;        // paged-cache block (64 / 128)
    uint32_t numBlocks;        // cache capacity in blocks
    uint32_t c1;               // NZ outer: (numKvHeads * packedHalves) / 16
    uint32_t packedHalves;     // headDim * bits / 16  (fp16 slots per vector)
    uint32_t vectorCoreNum;    // cores actually used

    uint32_t variant;          // TQ_VARIANT_MSE | TQ_VARIANT_PROD
    uint32_t codebookMode;     // TQ_CB_UNIFORM | TQ_CB_LUT
    uint32_t tokensPerCore;    // ceil(numTokens / vectorCoreNum)
    uint32_t reserved;         // pad to 16B

    // Host-computed constants. AscendC's Sqrt is tensor-only -- there is no
    // scalar overload -- and pulling libm into a kernel is not worth it for
    // values that are fixed for the whole launch.
    float sqrtHeadDim;         // sqrt(headDim)
    float invSqrtHeadDim;      // 1 / sqrt(headDim)
};

#endif  // TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_DATA_H
