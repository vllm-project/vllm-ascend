/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sigmoid_gating_delta_rule_update.cpp
 * \brief
 */
#include "fused_sigmoid_gating_delta_rule_update.h"
#include "fused_sigmoid_gating_delta_rule_update_tiling_data.h"


using namespace AscendC;
using namespace matmul;
using namespace FusedSigmoidGatingDeltaRuleUpdate;


extern "C" __global__ __aicore__ void
fused_sigmoid_gating_delta_rule_update(GM_ADDR aLog, GM_ADDR a, GM_ADDR b, GM_ADDR dtBias, GM_ADDR query,
                           GM_ADDR key, GM_ADDR value, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices,
                           GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut, GM_ADDR workspaceGM,
                           GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(FusedSigmoidGatingDeltaRuleUpdateTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    RGDR<bfloat16_t, bfloat16_t, DTYPE_STATE> op(&tilingData);
    RGDRInitParams initParams{aLog, a, b, dtBias, query, key, value, state, cuSeqlens,
                              ssmStateIndices, numAcceptedTokens, out, stateOut};
    op.Init(initParams, &pipe);
    op.Process();
}
