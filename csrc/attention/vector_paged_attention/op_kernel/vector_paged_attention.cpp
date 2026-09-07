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
 * \file vector_paged_attention.cpp
 * \brief Kernel entry for VectorPagedAttention
 */
#include "vector_paged_attention.h"

extern "C" __global__ __aicore__ void vector_paged_attention(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_table,
    GM_ADDR seq_lens, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    VectorPagedAttentionOp::KernelVectorPagedAttention kernel;
    kernel.Init(query, key_cache, value_cache, block_table, seq_lens, attn_out, tilingData);
    kernel.Process();
}
