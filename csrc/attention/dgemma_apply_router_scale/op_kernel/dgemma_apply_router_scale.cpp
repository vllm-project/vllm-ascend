/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "dgemma_apply_router_scale.h"
#include "dgemma_apply_router_scale_tiling_data.h"
using namespace AscendC;
using namespace DgemmaApplyRouterScale;

extern "C" __global__ __aicore__ void
dgemma_apply_router_scale(GM_ADDR weights, GM_ADDR ids, GM_ADDR scale,
                          GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(DgemmaApplyRouterScaleTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelDgemmaApplyRouterScale op;
    op.Init(weights, ids, scale, out, &tilingData);
    op.Process();
}
