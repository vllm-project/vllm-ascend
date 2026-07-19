/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope.cpp
 *  \brief AscendC kernel entry for DgemmaFusedNormRope. */
#include "dgemma_fused_norm_rope.h"
#include "dgemma_fused_norm_rope_tiling_data.h"
using namespace AscendC;
using namespace DgemmaFusedNormRope;

extern "C" __global__ __aicore__ void
dgemma_fused_norm_rope(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR q_weight, GM_ADDR k_weight,
                       GM_ADDR cos, GM_ADDR sin,
                       GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
                       GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(DgemmaFusedNormRopeTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelDgemmaFusedNormRope<bfloat16_t> op;
        op.Init(q, k, v, q_weight, k_weight, cos, sin, q_out, k_out, v_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelDgemmaFusedNormRope<half> op;
        op.Init(q, k, v, q_weight, k_weight, cos, sin, q_out, k_out, v_out, &tilingData, &pipe);
        op.Process();
    }
}
