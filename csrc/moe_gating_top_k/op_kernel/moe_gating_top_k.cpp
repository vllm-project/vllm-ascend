/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k.cpp
 * \brief
 */

#include "moe_gating_top_k_e_k_fullload.h"
#include "moe_gating_top_k_without_group.h"
#include "moe_gating_top_k_generalized.h"
#include "error_log.h"

#define TILING_KEY_PER_GROUP_COUNT_32 0
#define TILING_KEY_WITHOUT_GROUP 1
#define TILING_KEY_GENERALIZED 2

using namespace AscendC;
using namespace MoeGatingTopK;
extern "C" __global__ __aicore__ void moe_gating_top_k(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{   
  
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }


    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKTilingData, tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
 
    const MoeGatingTopKTilingData *__restrict t = &tilingData;
    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_PER_GROUP_COUNT_32)) {
        MoeGatingTopKEKFullload<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP)) {
        MoeGatingTopKWithoutGroup<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GENERALIZED)) {
        MoeGatingTopKGenerlized<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    }

}
