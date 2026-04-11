/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dispatch_gmm_combine_decode.h"
#include "dispatch_gmm_combine_decode_bf16_fp16.h"
#include <kernel_operator.h>
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void dispatch_gmm_combine_decode(
    // input
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_permuted_weight, GM_ADDR gmm1_permuted_weight_scale,
    GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales, GM_ADDR expert_smooth_scales,
    GM_ADDR x_active_mask,
    // output
    GM_ADDR output, GM_ADDR expertTokenNums,
    // system
    GM_ADDR workspace, GM_ADDR tiling)
{
    icache_preload(8);
    REGISTER_TILING_DEFAULT(DispatchGmmCombineDecodeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1C2V
    GET_TILING_DATA(tiling_data, tiling);

#if (ORIG_DTYPE_GMM1_PERMUTED_WEIGHT == DT_INT8)
    if constexpr (TILING_KEY_IS(0) || TILING_KEY_IS(1) || TILING_KEY_IS(2) || TILING_KEY_IS(3) || 
                  TILING_KEY_IS(4) || TILING_KEY_IS(5) || TILING_KEY_IS(6) || TILING_KEY_IS(7) ||
                  TILING_KEY_IS(8) || TILING_KEY_IS(9) || TILING_KEY_IS(10) || TILING_KEY_IS(11) || 
                  TILING_KEY_IS(12) || TILING_KEY_IS(13) || TILING_KEY_IS(14) || TILING_KEY_IS(15)) {
        DispatchGmmCombineDecodeImpl::DispatchGmmCombineDecode<
            DTYPE_X, DTYPE_GMM1_PERMUTED_WEIGHT_SCALE, DTYPE_GMM2_WEIGHT_SCALE, int8_t, int32_t, false, TILING_KEY_VAR> op;
        op.Init(x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight, gmm2_weight_scale,
                expert_scales, expert_smooth_scales, x_active_mask, output, expertTokenNums, workspace, nullptr, &tiling_data);
        op.Process();
    }
#elif (ORIG_DTYPE_GMM1_PERMUTED_WEIGHT == DT_BF16 || ORIG_DTYPE_GMM1_PERMUTED_WEIGHT == DT_FLOAT16)
    if constexpr (TILING_KEY_IS(0) || TILING_KEY_IS(1) || TILING_KEY_IS(2) || TILING_KEY_IS(3) || 
                  TILING_KEY_IS(4) || TILING_KEY_IS(5) || TILING_KEY_IS(6) || TILING_KEY_IS(7) ||
                  TILING_KEY_IS(8) || TILING_KEY_IS(9) || TILING_KEY_IS(10) || TILING_KEY_IS(11) || 
                  TILING_KEY_IS(12) || TILING_KEY_IS(13) || TILING_KEY_IS(14) || TILING_KEY_IS(15)) {
        DispatchGmmCombineDecodeBf16Fp16Impl::DispatchGmmCombineDecodeBf16Fp16<
            DTYPE_GMM1_PERMUTED_WEIGHT, DTYPE_GMM1_PERMUTED_WEIGHT_SCALE, DTYPE_GMM2_WEIGHT_SCALE, DTYPE_GMM1_PERMUTED_WEIGHT, int32_t, false, TILING_KEY_VAR> op;
        op.Init(x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight, gmm2_weight_scale,
                expert_scales, expert_smooth_scales, x_active_mask, output, expertTokenNums, workspace, nullptr, &tiling_data);
        op.Process();
    }
#endif
}
