/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "sparse_kv_gather_kernel.h"
using namespace AscendC;
using namespace BaseApi;
extern "C" __global__ __aicore__ void sparse_kv_gather_group(
    __gm__ uint8_t *pagedCtkv0, __gm__ uint8_t *pagedKpe0, __gm__ uint8_t *pagedCtkv1, __gm__ uint8_t *pagedKpe1, __gm__ uint8_t *pagedCtkv2, __gm__ uint8_t *pagedKpe2,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndices, __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv0, __gm__ uint8_t *outKpe0, __gm__ uint8_t *outCtkv1, __gm__ uint8_t *outKpe1, __gm__ uint8_t *outCtkv2, __gm__ uint8_t *outKpe2,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC { return; }
    (void)workspace;
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
#define RUN_SKG_LAYER(ID) \
    SparseKvGatherKernel op##ID; \
    op##ID.Init(pagedCtkv##ID, pagedKpe##ID, blockTable, topkIndices, curPos, outCtkv##ID, outKpe##ID, tilingData.numBlocks, tilingData.maxBlocks, tilingData.topkN, tilingData.totalSlots, tilingData.slotsPerCore, tilingData.usedCoreNum, tilingData.blockTableType, tilingData.topkIndicesType, tilingData.curPosType, &pipe); \
    op##ID.Process()
    RUN_SKG_LAYER(0);
    if (tilingData.numCacheLayers > 1U) { RUN_SKG_LAYER(1); }
    if (tilingData.numCacheLayers > 2U) { RUN_SKG_LAYER(2); }
#undef RUN_SKG_LAYER
}
