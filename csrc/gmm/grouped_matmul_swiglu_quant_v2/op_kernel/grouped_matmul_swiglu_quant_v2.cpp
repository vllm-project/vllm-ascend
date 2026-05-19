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
 * \file grouped_matmul_swiglu_quant_v2.cpp
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_swiglu_quant_spilit_fusion.h"
#include "grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h"
#include "grouped_matmul_swiglu_quant_v2_a4w4_pipeline.h"
#include "grouped_matmul_swiglu_quant_v2_utils.h"
using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulDequantSwigluQuant;
extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList,
                                                                     GM_ADDR weight, GM_ADDR weightScale,
                                                                     GM_ADDR weightAssistanceMatrix, GM_ADDR bias,
                                                                     GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

#if defined(GMM_SWIGLU_QUANT_V2_A8W4_MSD)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2BaseParams_,
                               tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, mmTilingData, mmTilingData_, tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2, gmmSwiglu_, tiling);
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>;
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplTypeCustom<xType, weightType, yType>;
        matmulType::MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&mmTilingData_);
        }
        GMMSwigluQuantPipelineSchedule<matmulType> op(mm, &gmmSwigluQuantV2BaseParams_, &gmmSwiglu_, &tPipe);
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, userWorkspace);
        op.Process();
    }
#endif
#if defined(GMM_SWIGLU_QUANT_V2_A4W4)
    if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2BaseParams_,
                               tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, mmTilingData, mmTilingData_, tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2, gmmSwiglu_, tiling);
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>;
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplTypeCustom<xType, weightType, yType>;
        matmulType::MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&mmTilingData_);
        }
        GMMSwigluQuantPipelineSchedule<matmulType> op(mm, &gmmSwigluQuantV2BaseParams_, &gmmSwiglu_, &tPipe);
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, smoothScale, y, yScale, userWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2BaseParams_,
                               tiling);

        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, gmmSwigluQuantV2, gmmSwiglu_, tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingData, mmTilingData, mmTilingData_, tiling);
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, true>;
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplTypeCustom<xType, weightType, yType>;
        matmulType::MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&mmTilingData_);
        }
        GMMSwigluQuantPipelineSchedule<matmulType> op(mm, &gmmSwigluQuantV2BaseParams_, &gmmSwiglu_, &tPipe);
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, smoothScale, y, yScale, userWorkspace);
        op.Process();
    }
#endif
    if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(GMMSwigluQuantV2TilingFusionData, tilingData, tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingFusionData, matmulTiling, matmulTilingData, tiling);
        GroupedMatmulDequantSwigluQuantFusion op(&tPipe, &tilingData, &matmulTilingData);
        if ASCEND_IS_AIC {
            op.mm.SetSubBlockIdx(0);
            op.mm.Init(&matmulTilingData, &tPipe);
        }

        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, userWorkspace);
        op.Process();
    }
}
