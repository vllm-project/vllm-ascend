/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#if __has_include("../../hc_pre/op_kernel/hc_pre_m_k_split_core.h")
#include "../../hc_pre/op_kernel/hc_pre_m_k_split_core.h"
#else
#include "../hc_pre/hc_pre_m_k_split_core.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void hc_pre_coefficients(
    GM_ADDR x, GM_ADDR hc_fn, GM_ADDR hc_scale, GM_ADDR hc_base,
    GM_ADDR post, GM_ADDR comb_frag, GM_ADDR pre, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tilingData, tiling);
        GM_ADDR userWs = GetUserWorkspace(workspace);

        TPipe projectionPipe;
        HcPre::HcPreMembaseKSplitCorePart1<DTYPE_X> projection;
        projection.Init(x, hc_fn, userWs, &tilingData, &projectionPipe);
        projection.Process();
        projectionPipe.Destroy();

        TPipe coefficientPipe;
        HcPre::HcPreMembaseKSplitCorePremixPart2<DTYPE_X, false> coefficients;
        coefficients.Init(nullptr, hc_scale, hc_base, nullptr, post, comb_frag, nullptr, pre,
            userWs, &tilingData, &coefficientPipe);
        coefficients.Process();
        coefficientPipe.Destroy();
    }
}
