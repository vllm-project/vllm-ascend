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
 * \file apply_top_k_top_p_with_sorted.cpp
 * \brief Custom kernel entry point for ApplyTopKTopPWithSortedCustom operator.
 */

#include "apply_top_k_top_p_with_sorted.h"
#include "apply_top_p_with_sorted.h"
using namespace AscendC;
using namespace ApplyTopKTopPWithSortedCustomOp;
using namespace ApplyTopPWithSortedCustomOp;

extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted_custom(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPWithSortedCustomOp::ApplyTopKTopPWithSortedCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPWithSortedCustomOp::ApplyTopKTopPWithSortedCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.ProcessTopK();
    } else if (TILING_KEY_IS(2)) {
        ApplyTopPWithSortedCustomOp::ApplyTopPWithSortedCustom<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out, workSpace);
        op.InitBuffer(&pipe);
        op.ProcessTopP();
    }
}