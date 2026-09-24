/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "arch22/attn_res_fwd_reload.h"
#include "arch22/attn_res_fwd_resident.h"
#include "attn_res_fwd_tiling_data.h"
#include "tiling_key_attn_res_fwd.h"

using namespace AscendC;
using namespace AttnResFwd;

extern "C" __global__ __aicore__ void attn_res_fwd_with_add(
    GM_ADDR prefixSum, GM_ADDR blockResidual, GM_ADDR projWeight, GM_ADDR normWeight,
    GM_ADDR addend, GM_ADDR hiddenStates, GM_ADDR prefixOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttnResFwdTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspaceGM;

    TPipe pipe;
    AttnResFwdInitParams initParams{prefixSum, blockResidual, projWeight, normWeight, hiddenStates, nullptr, nullptr,
        addend, prefixOut};
    if (TILING_KEY_IS(TILING_KEY_BF16_RELOAD)) {
        AttnResFwdReload<bfloat16_t, true> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_RESIDENT)) {
        AttnResFwdResident<bfloat16_t, true> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    }
}
