/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe operator kernel function implementation file
 * Author: WANG Qiankun
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe operator kernel function implementation file
 */
#include "fused_deep_moe.h"
#include <kernel_operator.h>
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void fused_deep_moe(
    // input
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_permuted_weight, GM_ADDR gmm1_permuted_weight_scale,
    GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_smooth_scales, GM_ADDR expert_scales,
    // output
    GM_ADDR output, GM_ADDR outputRecvCount,
    // system
    GM_ADDR workspace, GM_ADDR tiling)
{
    icache_preload(8);
    // New output recvCount
    REGISTER_TILING_DEFAULT(FusedDeepMoeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1C2V
    GET_TILING_DATA(tiling_data, tiling);
    if constexpr (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        FusedDeepMoe<DTYPE_X, int32_t, false, TILING_KEY_VAR> op;
        op.Init(x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight, gmm2_weight_scale,
                expert_smooth_scales, expert_scales, output, outputRecvCount, workspace, nullptr, &tiling_data);
        op.Process();
    }
}
