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
 * \file apply_top_k_top_p_custom.cpp
 * \brief
 */

#include "apply_top_k_top_p_custom.h"
#include "apply_top_p_custom.h"
using namespace AscendC;
using namespace ApplyTopKTopPCustomOp;
using namespace ApplyTopPCustomOp;

extern "C" __global__ __aicore__ void apply_top_k_top_p_custom(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPCustomOp::ApplyTopKTopPCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPCustomOp::ApplyTopKTopPCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.ProcessTopK();
    } else if (TILING_KEY_IS(2)) {
        ApplyTopPCustomOp::ApplyTopPCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out, workSpace);
        op.InitBuffer(&pipe);
        op.ProcessTopP();
    }
}