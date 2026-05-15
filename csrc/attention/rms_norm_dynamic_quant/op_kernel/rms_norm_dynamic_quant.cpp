/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant.cpp
 * \brief
 */
#include "rms_norm_dynamic_quant_normal_kernel.h"
#include "rms_norm_dynamic_quant_single_row_kernel.h"
#include "rms_norm_dynamic_quant_cut_d_kernel.h"

extern "C" __global__ __aicore__ void rms_norm_dynamic_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
    GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                                                \
    op.Init(x, gamma, smooth1, smooth2, beta, y1, y2, outScale1, outScale2, usrWorkspace, &tilingData); \
    op.Process()
    if (TILING_KEY_IS(0)) {
        // 0 Tiling, Do Nothing.
    } else if (TILING_KEY_IS(1)) {
        KernelAddRmsNormDynamicQuantNormal<DTYPE_X, DTYPE_Y1, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelAddRmsNormDynamicQuantSingleRow<DTYPE_X, DTYPE_Y1, 2> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormDynamicQuantSliceD<DTYPE_X, DTYPE_Y1, 3> op(&pipe);
        INIT_AND_PROCESS;
    }
}