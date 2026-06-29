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
 * \brief Unified MegaMoe entry kernel.
 *        - A3 (910_93): W4A8 (int4 weight), INT8 (int8 weight), BF16/FP16 weight
 *        - A2 (910B):   INT8 (int8 weight), BF16/FP16 weight
 *        Note: W4A8 on A2 is reserved but not yet implemented.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "mega_moe_tiling_a2a3.h"
#include "mega_moe_tiling_key.h"
#include "mega_moe_a2.h"
#include "mega_moe.h"

using namespace AscendC;
using namespace MegaMoeImpl;
using namespace MegaMoeA2Impl;

template <bool TPL_IS_TRANSPOSE_W1, bool TPL_IS_TRANSPOSE_W2, int TPL_QUANT_MODE, int TPL_QUANT_OUT_TYPE, int TPL_ARCH>
__global__ __aicore__ void mega_moe(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                    GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1,
                                    GM_ADDR bias2, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR yOut,
                                    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr bool isNz = (FORMAT_WEIGHT1 == FORMAT_FRACTAL_NZ);
    constexpr bool isA2 = (TPL_ARCH == SOC_ASCEND910B);
    REGISTER_TILING_DEFAULT(MegaMoeTilingDataNonQuant);
    REGISTER_TILING_FOR_TILINGKEY("(QuantMode == MEGA_MOE_QUANT_MODE_PER_TENSOR)", MegaMoeTilingDataQuant);
    if constexpr (isA2) {
#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)
        if constexpr (TPL_QUANT_MODE == MEGA_MOE_QUANT_MODE_PER_TENSOR) {
            GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataQuant, tilingData, tilingGM);

            MegaMoeA2<DTYPE_X, DTYPE_WEIGHT1, DTYPE_Y, TPL_IS_TRANSPOSE_W1, TPL_IS_TRANSPOSE_W2, isNz, true> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, weightScales1, weightScales2, bias1, bias2,
                    xActiveMask, scales, yOut, expertTokenNumsOut, workspaceGM, tilingGM);
            op.Process();
        } else if constexpr (TPL_QUANT_MODE == MEGA_MOE_QUANT_MODE_NO_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataNonQuant, tilingData, tilingGM);

            MegaMoeA2<DTYPE_X, DTYPE_WEIGHT1, DTYPE_Y, TPL_IS_TRANSPOSE_W1, TPL_IS_TRANSPOSE_W2, isNz, true> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, weightScales1, weightScales2, bias1, bias2,
                    xActiveMask, scales, yOut, expertTokenNumsOut, workspaceGM, tilingGM);
            op.Process();
        }
#endif
    } else {
        if constexpr (TPL_QUANT_MODE == MEGA_MOE_QUANT_MODE_PER_TENSOR) {
            GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataQuant, tilingData, tilingGM);
            MegaMoe<DTYPE_X, DTYPE_WEIGHT1, DTYPE_Y, TPL_IS_TRANSPOSE_W1, TPL_IS_TRANSPOSE_W2, isNz, false> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, weightScales1, weightScales2, bias1, bias2,
                    xActiveMask, scales, yOut, expertTokenNumsOut, workspaceGM, tilingGM);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataNonQuant, tilingData, tilingGM);

            MegaMoe<DTYPE_X, DTYPE_WEIGHT1, DTYPE_Y, TPL_IS_TRANSPOSE_W1, TPL_IS_TRANSPOSE_W2, isNz, false> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, weightScales1, weightScales2, bias1, bias2,
                    xActiveMask, scales, yOut, expertTokenNumsOut, workspaceGM, tilingGM);
            op.Process();
        }
    }
}