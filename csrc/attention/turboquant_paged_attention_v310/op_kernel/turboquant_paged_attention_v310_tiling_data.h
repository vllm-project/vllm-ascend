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
#ifndef TURBOQUANT_PAGED_ATTENTION_V310_TILING_DATA_H
#define TURBOQUANT_PAGED_ATTENTION_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

/*
 * Same scenario split as the write path: bit-width is compile-time (tiling key
 * 210 + bits), variant + codebook are runtime fields.
 *
 * GQA is handled natively here. npu_incre_flash_attention has no GQA on 310P
 * (it fails in the tiling function), and the Tier-0 workaround -- expanding K/V
 * with repeat_interleave -- inflated the dense temporary 4x and was a direct
 * cause of the OOM kills at 6k context. This kernel loops queries within a KV
 * group instead, so K/V are read once per group.
 */
struct TurboquantPagedAttentionV310TilingData {
    uint32_t batch;
    uint32_t numHeads;         // query heads (16)
    uint32_t numKvHeads;       // kv heads (4)
    uint32_t headDim;          // 256
    uint32_t blockSize;        // paged-cache block (64 / 128)
    uint32_t maxBlocksPerSeq;  // block_table stride
    uint32_t c1;               // NZ outer dim of the cache
    uint32_t packedHalves;     // headDim * bits / 16

    uint32_t variant;          // TQ_VARIANT_MSE | TQ_VARIANT_PROD
    uint32_t codebookMode;     // TQ_CB_UNIFORM | TQ_CB_LUT
    uint32_t vectorCoreNum;
    uint32_t tasksPerCore;     // ceil(batch*numKvHeads / vectorCoreNum)

    float scale;               // 1/sqrt(headDim), folded into the QK product
    float sqrtHeadDim;         // host-computed; AscendC Sqrt is tensor-only
    float invSqrtHeadDim;
    uint32_t reserved;
};

#endif  // TURBOQUANT_PAGED_ATTENTION_V310_TILING_DATA_H
