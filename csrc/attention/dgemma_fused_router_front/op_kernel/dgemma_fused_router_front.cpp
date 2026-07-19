/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front.cpp
 *  \brief MIX kernel entry: RMSNorm+root_size+scale (vector) -> GateLinear GEMM (cube).
 *  Vector runs FIRST (normalizes x into scratch), signals the cube, cube reads scratch
 *  and does scratch @ proj_weight.T -> router_logits. Reverse dependency vs the qkv MIX. */
#include "lib/matmul_intf.h"
#include <kernel_operator.h>
#include "dgemma_fused_router_front_aic_kernel.h"
#include "dgemma_fused_router_front_aiv_kernel.h"

using namespace AscendC;
using namespace DgemmaFusedRouterFront;

extern "C" __global__ __aicore__ void dgemma_fused_router_front(
    GM_ADDR x, GM_ADDR scale, GM_ADDR proj_weight, GM_ADDR norm_scratch,
    GM_ADDR logits_scratch, GM_ADDR per_expert_scale, GM_ADDR sync_scratch,
    GM_ADDR topk_weights, GM_ADDR topk_ids,
    GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(DgemmaFusedRouterFrontTilingData);
    GET_TILING_DATA(tiling_data, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const DgemmaFusedRouterFrontTilingData *td = &tiling_data;

    // Graph-capture-safe: the normed intermediate lives in a caller-provided persistent
    // buffer (fixed address across ACL graph capture/replay), NOT transient op workspace.
    GM_ADDR norm_ws = norm_scratch;
    GM_ADDR logits_ws = logits_scratch;

    if ASCEND_IS_AIV {
        // Vector runs first: RMSNorm + root_size + scale -> norm_ws, then signal cube.
        TPipe pipe;
        DgemmaFusedRouterFrontAivKernel<DTYPE_X> op;
        op.Init(x, scale, norm_ws, logits_ws, per_expert_scale, sync_scratch,
                topk_weights, topk_ids, td, &pipe);
        op.Process();
        return;
    }
    if ASCEND_IS_AIC {
        if (td->k == 0) return;
        // Cube waits for the vector's normed data, then GEMM norm_ws @ proj_weight.T.
        DgemmaFusedRouterFrontAicKernel<DTYPE_X> op;
        op.Init(norm_ws, proj_weight, logits_ws, sync_scratch, td);
        op.Process();
        return;
    }
}
