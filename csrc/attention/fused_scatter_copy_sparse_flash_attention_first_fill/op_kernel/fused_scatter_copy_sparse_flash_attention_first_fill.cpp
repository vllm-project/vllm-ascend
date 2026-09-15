/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "kernel_operator.h"

using KvcacheScatterCopyTilingData =
    FusedScatterCopySparseFlashAttentionFirstFillTilingData;

#include "kvcache_scatter_copy_kernel.h"

using namespace AscendC;
using namespace KvcacheScatterCopyNs;

extern "C" __global__ __aicore__ void fused_scatter_copy_sparse_flash_attention_first_fill(
    GM_ADDR hbmKRoPE,
    GM_ADDR hbmKvCache,
    GM_ADDR dramKRoPE,
    GM_ADDR dramKvCache,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR missSourceIds,
    GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR cacheTokens,
    GM_ADDR hbmKRoPEOut,
    GM_ADDR hbmKvCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)hbmKRoPEOut;
    (void)hbmKvCacheOut;
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KvcacheScatterCopyKernel<DTYPE_DRAM_K_ROPE, true> op(
            &pipe, &tilingData);
        op.Init(
            hbmKRoPE, hbmKvCache, dramKRoPE, dramKvCache,
            hbmBlockTable, dramBlockTable, missSourceIds,
            missDstSlots, missCounts, cacheTokens);
        op.Process();
    }
}
