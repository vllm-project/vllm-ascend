/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating.cpp
 * \brief AscendC kernel entry for FusedGdnGating.
 */

#include "kernel_operator.h"
#include "fused_gdn_gating_tiling_data.h"

#if __CCE_AICORE__ == 200
    #include "fused_gdn_gating_310.h"
#else
    #include "arch32/fused_gdn_gating_910.h"
#endif

using namespace AscendC;
// 引入 TilingData 所在的命名空间，确保宏能够正确解析
using namespace FusedGdnGating;

// NPU Device 侧算子全局主入口函数
extern "C" __global__ __aicore__ void fused_gdn_gating(
    GM_ADDR a_log, GM_ADDR a, GM_ADDR b, GM_ADDR dt_bias,
    GM_ADDR g, GM_ADDR beta_output,
    GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(FusedGdnGatingTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    
    auto tiling_ptr = reinterpret_cast<const FusedGdnGatingTilingData*>(&tilingData);
    
#if __CCE_AICORE__ == 200
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    
    if (TILING_KEY_IS(200000)) {
        optiling::KernelFusedGdnGating310 op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, tiling_ptr);
        op.Process();
    }
    
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        KernelFusedGdnGating<bfloat16_t, float> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelFusedGdnGating<half, float> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelFusedGdnGating<bfloat16_t, bfloat16_t> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelFusedGdnGating<half, bfloat16_t> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelFusedGdnGating<bfloat16_t, half> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KernelFusedGdnGating<half, half> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    }
#endif
}
