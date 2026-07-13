/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif

#include "kernel_operator.h"

#ifdef ENABLE_TENSOR_API
#include "mega_moe.h"
#include "mega_moe_layered.h"
#endif

#include "mega_moe_tiling.h"
#include "mega_moe_tiling_key.h"

using namespace AscendC;
#ifdef ENABLE_TENSOR_API
using namespace MegaMoeImpl;
#endif

template<uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType, uint8_t CommModeType>
__global__ __aicore__ void mega_moe(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR xActiveMask,
    GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
#if defined(ENABLE_TENSOR_API) && \
    defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16) && \
    defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_BF16) && \
    defined(ORIG_DTYPE_WEIGHT1) && \
        ((ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E5M2) || \
         (ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E4M3FN) || \
         (ORIG_DTYPE_WEIGHT1 == DT_FLOAT4_E2M1)) && \
    defined(ORIG_DTYPE_WEIGHT2) && (ORIG_DTYPE_WEIGHT2 == ORIG_DTYPE_WEIGHT1)
    if constexpr (CommModeType == TILINGKEY_TPL_MTE) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1,
            DispatchQuantOutType, CombineQuantOutType> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
        }
    } else if constexpr (CommModeType == TILINGKEY_TPL_URMA) {
        // if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        //     MegaMoeLayered<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1,
        //     DispatchQuantOutType, CombineQuantOutType> op;
        //     op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
        //             scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
        //     op.Process();
        // }
    }
#endif
}