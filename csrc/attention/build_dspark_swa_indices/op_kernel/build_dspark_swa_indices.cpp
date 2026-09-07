/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "build_dspark_swa_indices.h"

extern "C" __global__ __aicore__ void build_dspark_swa_indices(
    GM_ADDR kvBlockTable,
    GM_ADDR queryStartLoc,
    GM_ADDR seqLens,
    GM_ADDR perTokenSlots,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BuildDsparkSwaIndices::BuildDsparkSwaIndicesTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;

    // Pure int32 path — no TILING_KEY dtype dispatch needed (unlike
    // compressor_metadata which branches on float/half/bf16).
    if (TILING_KEY_IS(1)) {
        BuildDsparkSwaIndices::BuildDsparkSwaIndicesKernel op;
        op.Init(&tilingData, &pipe);
        op.Process(kvBlockTable, queryStartLoc, seqLens, perTokenSlots, workspace);
    }
}
