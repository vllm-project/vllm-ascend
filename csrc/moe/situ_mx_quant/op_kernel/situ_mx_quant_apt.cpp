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
 * \file situ_mx_quant_apt.cpp
 * \brief Kernel entry point for Situ + MX quantization operator
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/situ_mx_quant_tiling_key.h"
#include "arch35/situ_mx_quant_tiling_data.h"
#include "arch35/situ_mx_quant_common.h"
#include "arch35/situ_mx_quant_axis_last.h"

using namespace AscendC;
using namespace SituMxQuantOp;

template <uint64_t hasLinearBeta, uint64_t dstTypeIndex>
__global__ __aicore__ void situ_mx_quant(
    GM_ADDR x, GM_ADDR groupList, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SituMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(SituMxQuantTilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (dstTypeIndex == TPL_DST_E4M3FN) {
        constexpr bool useLinearBeta = (hasLinearBeta == TPL_HAS_LINEAR_BETA);
        SituMxQuant::SituMxQuantAxisLast<bfloat16_t, fp8_e4m3fn_t, useLinearBeta> op;
        op.Init(x, y, mxscale, usrWorkspace, groupList, &tilingData, &pipe);
        op.Process();
    } else {
        constexpr bool useLinearBeta = (hasLinearBeta == TPL_HAS_LINEAR_BETA);
        SituMxQuant::SituMxQuantAxisLast<bfloat16_t, fp8_e5m2_t, useLinearBeta> op;
        op.Init(x, y, mxscale, usrWorkspace, groupList, &tilingData, &pipe);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
