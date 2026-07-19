/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope.cpp
 *  \brief MIX kernel entry: qkv_proj GEMM (cube) + q/k/v RMSNorm + neox RoPE (vector). */
#include "lib/matmul_intf.h"
#include <kernel_operator.h>
#include "dgemma_fused_qkv_proj_norm_rope_aic_kernel.h"
#include "dgemma_fused_qkv_proj_norm_rope_aiv_kernel.h"

using namespace AscendC;
using namespace DgemmaFusedQkvProjNormRope;

extern "C" __global__ __aicore__ void dgemma_fused_qkv_proj_norm_rope(
    GM_ADDR hidden, GM_ADDR wqkv, GM_ADDR q_weight, GM_ADDR k_weight,
    GM_ADDR cos, GM_ADDR sin, GM_ADDR qkv_scratch,
    GM_ADDR qkv_scratch_out, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
    GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(DgemmaFusedQkvProjNormRopeTilingData);
    GET_TILING_DATA(tiling_data, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const DgemmaFusedQkvProjNormRopeTilingData *td = &tiling_data;

    // Intermediate qkv[m, n] lives in the graph-visible scratch output buffer.
    GM_ADDR qkv_ws = qkv_scratch_out;

    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);

    if ASCEND_IS_AIC {
        if (td->headDim == 0) return;
        DgemmaFusedQkvProjNormRopeAicKernel<DTYPE_HIDDEN> op;
        op.Init(hidden, wqkv, qkv_ws, td);
        op.Process();
        return;
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        DgemmaFusedQkvProjNormRopeAivKernel<DTYPE_HIDDEN> op;
        op.Init(qkv_ws, q_weight, k_weight, cos, sin, q_out, k_out, v_out, td, &pipe);
        op.Process();
        return;
    }
}
