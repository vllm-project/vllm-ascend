/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_gdn_decode.h"
#include "fused_gdn_decode_tiling_data.h"

using namespace AscendC;
using namespace FusedGdnDecode;

extern "C" __global__ __aicore__ void
fused_gdn_decode(GM_ADDR mixed_qkv, GM_ADDR a, GM_ADDR b, GM_ADDR a_log, GM_ADDR dt_bias,
                 GM_ADDR state, GM_ADDR ssm_state_indices, GM_ADDR out,
                 GM_ADDR state_out, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FusedGdnDecodeTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelFusedGdnDecode<bfloat16_t, float> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelFusedGdnDecode<half, float> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelFusedGdnDecode<bfloat16_t, bfloat16_t> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelFusedGdnDecode<half, half> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    }
}
