/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_with_map_apt.cpp
 * \brief
 */

#include "moe_gating_top_k_with_map_regbase.h"
using namespace AscendC;
using namespace MoeGatingTopKWithMap;

#define TILING_KEY_REGBASE 10000

extern "C" __global__ __aicore__ void moe_gating_top_k_with_map(GM_ADDR x, GM_ADDR bias, GM_ADDR log2phy, GM_ADDR y, GM_ADDR expertIdx,
                                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKWithMapRegbaseTilingData, tiling_data_in, tiling);
    const MoeGatingTopKWithMapRegbaseTilingData *__restrict tilingData = &tiling_data_in;
    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_REGBASE)) {
        MoeGatingTopKWithMapRegbase<DTYPE_X> op;
        op.Init(x, bias, log2phy, y, expertIdx, out, userWS, tilingData, &tPipe);
        op.Process();
    }
}
