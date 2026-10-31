/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "arch20/compat_310p.h"
#endif
#include "arch20/compute_wy_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_compute_wy(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR q_kernel, GM_ADDR k_kernel, GM_ADDR w_kernel, GM_ADDR u_kernel, GM_ADDR g_kernel,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleComputeWyTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleComputeWyTilingData, tilingData, tiling);

    // MIX / default AiCore so Cube Matmul is available on Ascend 310P (not AIV_ONLY).
    TPipe pipe;
    ChunkGatedDeltaRuleComputeWy::KernelComputeWy op;
    op.Init(q, k, v, g, beta, q_kernel, k_kernel, w_kernel, u_kernel, g_kernel, workspace, &tilingData, &pipe);
    op.Process();
}
