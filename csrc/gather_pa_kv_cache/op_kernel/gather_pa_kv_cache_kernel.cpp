/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_host/gather_pa_kv_cache_tiling.h"
#include "gather_pa_kv_cache_nd.h"
#include "gather_pa_kv_cache_nz.h"
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void gather_pa_kv_cache(GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR blockTables,
                                                         GM_ADDR seqLens, GM_ADDR key, GM_ADDR value, GM_ADDR seqOffset,
                                                         GM_ADDR keyOut, GM_ADDR valueOut, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    auto tilingDataGm = reinterpret_cast<__gm__ GatherPaKvCacheTilingData *>(tiling);
    GatherPaKvCacheTilingData tilingData = {
        .blockSize = tilingDataGm->blockSize,
        .numTokens = tilingDataGm->numTokens,
        .numblkTabCol = tilingDataGm->numblkTabCol,
        .tokenSizeK = tilingDataGm->tokenSizeK,
        .tokenSizeV = tilingDataGm->tokenSizeV,
        .typeByte = tilingDataGm->typeByte,
        .hasSeqStarts = tilingDataGm->hasSeqStarts,
        .isSeqLensCumsum = tilingDataGm->isSeqLensCumsum,
        .kCacheBlockStride = tilingDataGm->kCacheBlockStride,
        .vCacheBlockStride = tilingDataGm->vCacheBlockStride,
        .tilingKey = tilingDataGm->tilingKey,
    };

    // The direct-kernel launcher has no GE tiling-key channel, so dispatch on
    // the original key carried in the tiling payload.
    if (tilingData.tilingKey == TILING_KEY_ND_INT8) {
        GatherPaKvCache::GatherPaKvCacheNd<int8_t> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
    if (tilingData.tilingKey == TILING_KEY_ND_B16) {
        GatherPaKvCache::GatherPaKvCacheNd<half> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
    if (tilingData.tilingKey == TILING_KEY_NZ) {
        GatherPaKvCache::GatherPaKvCacheNz<int8_t> op(&pipe);
        op.Init(keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut, &tilingData);
        op.Process();
    }
}

namespace vllm_ascend {

extern void gather_pa_kv_cache_impl(void *stream, void *keyCache, void *valueCache, void *blockTables, void *seqLens,
                                    void *seqOffset, void *keyOut, void *valueOut, void *tiling,
                                    const uint32_t blockDim)
{
    gather_pa_kv_cache<<<blockDim, nullptr, stream>>>(keyCache, valueCache, blockTables, seqLens, keyOut, valueOut,
                                                      seqOffset, keyOut, valueOut, nullptr, tiling);
}

}  // namespace vllm_ascend
