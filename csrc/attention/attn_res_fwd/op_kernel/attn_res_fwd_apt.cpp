/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_res_fwd_apt.cpp
 * \brief AttnResFwd A5(ascend950) kernel entry — same logic as A2, includes arch35/
 */
#include "arch35/attn_res_fwd_reload.h"
#include "arch35/attn_res_fwd_resident.h"
#include "attn_res_fwd_tiling_data.h"
#include "tiling_key_attn_res_fwd.h"

using namespace AscendC;
using namespace AttnResFwd;

extern "C" __global__ __aicore__ void attn_res_fwd(GM_ADDR prefixSum, GM_ADDR blockResidual, GM_ADDR projWeight,
                                                   GM_ADDR normWeight, GM_ADDR hiddenStates, GM_ADDR invRms,
                                                   GM_ADDR probs, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttnResFwdTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspaceGM; // 无用户 FP32-v Workspace；仅系统预留

    TPipe pipe;
    AttnResFwdInitParams initParams{prefixSum, blockResidual, projWeight, normWeight, hiddenStates, invRms, probs};

    if (TILING_KEY_IS(TILING_KEY_BF16_RELOAD)) {
        AttnResFwdReload<bfloat16_t> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_RESIDENT)) {
        AttnResFwdResident<bfloat16_t> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    }
}
