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
 * \file swiglu_group_quant.cpp
 * \brief
 */

#include "swiglu_group_quant_perf.h"
#include "swiglu_mx_quant_perf.h"
#include "swiglu_fp8_quant_per_token.h"

#define GROUP_QUANT_TILING_KEY 1
#define MX_QUANT_TILING_KEY 2
#define FP8_QUANT_TILING_KEY 31
#define FP8_QUANT_YORIGIN_TILING_KEY 32
using namespace AscendC;

extern "C" __global__ __aicore__ void swiglu_group_quant(GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(GROUP_QUANT_TILING_KEY)) {
        SwigluGroupQuant::SwigluGroupQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_SCALE> op;
        op.Init(x, topkWeight, groupIndex, y, scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MX_QUANT_TILING_KEY)) {
        SwigluGroupQuant::SwigluMxQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_SCALE> op;
        op.Init(x, topkWeight, groupIndex, y, scale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FP8_QUANT_TILING_KEY)) {
        SwigluGroupQuant::SwigluFp8QuantPerToken<DTYPE_X, DTYPE_Y, DTYPE_SCALE, false> op;
        op.Init(x, topkWeight, groupIndex, y, scale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FP8_QUANT_YORIGIN_TILING_KEY)) {
        SwigluGroupQuant::SwigluFp8QuantPerToken<DTYPE_X, DTYPE_Y, DTYPE_SCALE, true> op;
        op.Init(x, topkWeight, groupIndex, y, scale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}