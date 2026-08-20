/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "sfa_remap_sparse_indices.h"

extern "C" __global__ __aicore__ void sfa_remap_sparse_indices(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SfaRemapSparseIndices::SfaRemapSparseIndicesTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    SfaRemapSparseIndices::SfaRemapSparseIndicesKernel op(&pipe);
    op.Init(input, output, &tilingData);
    op.Process();
}
