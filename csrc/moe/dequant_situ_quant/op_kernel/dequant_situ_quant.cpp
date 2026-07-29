/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant.cpp
 * \brief Kernel entry point for DequantSituQuant operator.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dequant_situ_quant.h"

using namespace AscendC;

#define DSQ_STATIC_QUANT_ONE 10000
#define DSQ_STATIC_QUANT_VEC 10001
#define DSQ_STATIC_QUANT_ONE_BIAS 10002
#define DSQ_STATIC_QUANT_VEC_BIAS 10003
#define DSQ_DYNAMIC_QUANT_NO_SMOOTH 20000
#define DSQ_DYNAMIC_QUANT_SMOOTH 20001
#define DSQ_DYNAMIC_QUANT_NO_SMOOTH_BIAS 20002
#define DSQ_DYNAMIC_QUANT_SMOOTH_BIAS 20003
#define DSQ_INT32_DYNAMIC 30000
#define DSQ_BF16_DYNAMIC 40000

extern "C" __global__ __aicore__ void dequant_situ_quant(
    GM_ADDR xGM, GM_ADDR weightScaleGM, GM_ADDR activationScaleGM, GM_ADDR biasGM,
    GM_ADDR quantScaleGM, GM_ADDR quantOffsetGM, GM_ADDR groupIndexGM,
    GM_ADDR yGM, GM_ADDR scaleGM, GM_ADDR workspace, GM_ADDR tiling)
{
#if (ORIG_DTYPE_X == DT_INT8)
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userspace = GetUserWorkspace(workspace);
    if (userspace == nullptr) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(DequantSituQuantTilingData, tilingDataIn, tiling);
    const DequantSituQuantTilingData* __restrict__ tilingData = &tilingDataIn;

    if (TILING_KEY_IS(DSQ_STATIC_QUANT_ONE)) {
        DequantSituQuantOps::DequantSituQuantKernel<false> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_STATIC_QUANT_VEC)) {
        DequantSituQuantOps::DequantSituQuantKernel<false> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_STATIC_QUANT_ONE_BIAS)) {
        DequantSituQuantOps::DequantSituQuantKernel<true> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_STATIC_QUANT_VEC_BIAS)) {
        DequantSituQuantOps::DequantSituQuantKernel<true> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_DYNAMIC_QUANT_NO_SMOOTH)) {
        DequantSituQuantOps::DequantSituQuantKernel<false> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_DYNAMIC_QUANT_SMOOTH)) {
        DequantSituQuantOps::DequantSituQuantKernel<false> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_DYNAMIC_QUANT_NO_SMOOTH_BIAS)) {
        DequantSituQuantOps::DequantSituQuantKernel<true> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DSQ_DYNAMIC_QUANT_SMOOTH_BIAS)) {
        DequantSituQuantOps::DequantSituQuantKernel<true> op(&pipe);
        op.Init(xGM, weightScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace, tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_X == DT_INT32)
    if (TILING_KEY_IS(DSQ_INT32_DYNAMIC)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(DequantSituQuantTilingData, tilingDataIn, tiling);
        const DequantSituQuantTilingData* __restrict__ tilingData = &tilingDataIn;
        DequantSituQuantOps::DequantSituQuantK3Kernel<int32_t> op(&pipe);
        op.Init(xGM, weightScaleGM, activationScaleGM, biasGM, groupIndexGM, yGM, scaleGM, tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(DSQ_BF16_DYNAMIC)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(DequantSituQuantTilingData, tilingDataIn, tiling);
        const DequantSituQuantTilingData* __restrict__ tilingData = &tilingDataIn;
        DequantSituQuantOps::DequantSituQuantK3Kernel<bfloat16_t> op(&pipe);
        op.Init(xGM, weightScaleGM, activationScaleGM, biasGM, groupIndexGM, yGM, scaleGM, tilingData);
        op.Process();
    }
#endif
}
