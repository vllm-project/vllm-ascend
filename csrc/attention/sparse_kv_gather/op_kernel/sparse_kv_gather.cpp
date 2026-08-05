/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "sparse_kv_gather_kernel.h"

using namespace AscendC;
using namespace BaseApi;

extern "C" __global__ __aicore__ void sparse_kv_gather(
    __gm__ uint8_t *pagedCtkv,
    __gm__ uint8_t *pagedKpe,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *topkIndices,
    __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv,
    __gm__ uint8_t *outKpe,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    (void)workspace;

    TPipe pipe;

    GET_TILING_DATA(tilingData, tiling);

    SparseKvGatherKernel op;
    op.Init(
        pagedCtkv,
        pagedKpe,
        blockTable,
        topkIndices,
        curPos,
        outCtkv,
        outKpe,
        tilingData.numBlocks,
        tilingData.maxBlocks,
        tilingData.topkN,
        tilingData.totalSlots,
        tilingData.slotsPerCore,
        tilingData.usedCoreNum,
        tilingData.blockTableType,
        tilingData.topkIndicesType,
        tilingData.curPosType,
        &pipe);
    op.Process();
}
