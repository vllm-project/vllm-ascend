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

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "arch20/compat_310p.h"
#include "arch20/apply_top_k_top_p_custom.h"
#include "arch20/apply_top_p_custom.h"
#else
#include "apply_top_k_top_p_custom.h"
#include "apply_top_p_custom.h"
#endif
using namespace AscendC;
using namespace ApplyTopKTopPCustomOp;
using namespace ApplyTopPCustomOp;

// 310P has no bf16 hardware. The bf16 dtype variant of this op still gets
// registered (see apply_top_k_top_p_custom_def.cpp) and compiled to a kernel
// binary because the op is shared with 910B. On 310P we route the bf16
// dispatch to the fp16 (half) code path so the template is never instantiated
// with `bfloat16_t` — same trick as chunk_gated_delta_rule_fwd_h/arch20 uses
// via CATLASS_UNIFIED_CORE. Users pass fp16 tensors at runtime; the bf16
// kernel binary is never loaded on 310P in practice.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
using ActualOutT = std::conditional_t<std::is_same_v<DTYPE_OUT, bfloat16_t>, half, DTYPE_OUT>;
#else
using ActualOutT = DTYPE_OUT;
#endif

extern "C" __global__ __aicore__ void apply_top_k_top_p_custom(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPCustomOp::ApplyTopKTopPCustom<ActualOutT, float, ActualOutT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPCustomOp::ApplyTopKTopPCustom<ActualOutT, float, ActualOutT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out);
        op.InitBuffer(&pipe);
        op.ProcessTopK();
    } else if (TILING_KEY_IS(2)) {
        ApplyTopPCustomOp::ApplyTopPCustom<ActualOutT, float, ActualOutT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out, workSpace);
        op.InitBuffer(&pipe);
        op.ProcessTopP();
    }
}
