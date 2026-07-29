/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "kvcache_scatter_copy_kernel.h"

using namespace AscendC;
using namespace KvcacheScatterCopyNs;

extern "C" __global__ __aicore__ void kvcache_scatter_copy(
    GM_ADDR hbmKRoPE, GM_ADDR hbmKvCache, GM_ADDR dramKRoPE, GM_ADDR dramKvCache,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR srcTokenIds, GM_ADDR dstSlots,
    GM_ADDR copyCounts, GM_ADDR hbmKRoPEOut, GM_ADDR hbmKvCacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KvcacheScatterCopyKernel<DTYPE_DRAM_K_ROPE> op(&pipe, &tilingData);
        op.Init(hbmKRoPE, hbmKvCache, dramKRoPE, dramKvCache, hbmBlockTable, dramBlockTable,
            srcTokenIds, dstSlots, copyCounts);
        op.Process();
    }
}
