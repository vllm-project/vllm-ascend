/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_pa_kv_cache_vllm.cpp
 * \brief
 */

#include "gather_pa_kv_cache_vllm_nd.h"
#include "gather_pa_kv_cache_vllm_nz.h"
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void gather_pa_kv_cache_vllm(GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR blockTables,
                                                         GM_ADDR seqLens, GM_ADDR key, GM_ADDR value, GM_ADDR seqOffset,
                                                         GM_ADDR keyOut, GM_ADDR valueOut, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheVllmTilingData, tilingData, tiling);
#if (ORIG_DTYPE_KEY_CACHE == DT_INT8)
    if TILING_KEY_IS (618) {
        GatherPaKvCacheVllm::GatherPaKvCacheVllmNd<int8_t> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_KEY_CACHE == DT_FLOAT16 || ORIG_DTYPE_KEY_CACHE == DT_BF16)
    if TILING_KEY_IS (619) {
        GatherPaKvCacheVllm::GatherPaKvCacheVllmNd<half> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
#endif
    if TILING_KEY_IS (577) {
        GatherPaKvCacheVllm::GatherPaKvCacheVllmNz<int8_t> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
}