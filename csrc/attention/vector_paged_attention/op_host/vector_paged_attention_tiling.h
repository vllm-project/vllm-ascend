/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vector_paged_attention_tiling.h
 * \brief Tiling data for VectorPagedAttention
 */
#ifndef VECTOR_PAGED_ATTENTION_TILING_H
#define VECTOR_PAGED_ATTENTION_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VectorPagedAttentionTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);       // requests in the step
    TILING_DATA_FIELD_DEF(uint32_t, numHeads);    // query heads
    TILING_DATA_FIELD_DEF(uint32_t, headDim);     // per-head width
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);   // KV rows per page
    TILING_DATA_FIELD_DEF(uint32_t, maxBlocks);   // block_table row stride
    TILING_DATA_FIELD_DEF(uint32_t, kvStride);    // elements per KV cache row
    TILING_DATA_FIELD_DEF(uint32_t, kvCapacity);  // blockSize * maxBlocks
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks);   // pages in the cache
    TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VectorPagedAttention, VectorPagedAttentionTilingData)

struct VectorPagedAttentionCompileInfo {
    uint32_t aivCoreNum;
};

}  // namespace optiling

#endif  // VECTOR_PAGED_ATTENTION_TILING_H
