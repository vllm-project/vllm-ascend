/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dispatch_gmm_combine_decode.h"
#include <kernel_operator.h>
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void dispatch_gmm_combine_decode(
    // input
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_permuted_weight, GM_ADDR gmm1_permuted_weight_scale,
    GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_smooth_scales, GM_ADDR expert_scales,
    // output
    GM_ADDR output, GM_ADDR outputRecvCount,
    // system
    GM_ADDR workspace, GM_ADDR tiling)
{
    icache_preload(8);
    REGISTER_TILING_DEFAULT(DispatchGmmCombineDecodeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1C2V
    GET_TILING_DATA(tiling_data, tiling);
    if constexpr (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        DispatchGmmCombineDecode<DTYPE_X, int32_t, false, TILING_KEY_VAR> op;
        op.Init(x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight, gmm2_weight_scale,
                expert_smooth_scales, expert_scales, output, outputRecvCount, workspace, nullptr, &tiling_data);
        op.Process();
    }
}
