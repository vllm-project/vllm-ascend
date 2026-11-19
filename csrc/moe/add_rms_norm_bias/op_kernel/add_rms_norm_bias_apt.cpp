/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from cann/ops-nn v9.2.0-beta.2, commit
// 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6.
// Source: norm/add_rms_norm/op_kernel/add_rms_norm_apt.cpp


/* !
 * \file add_rms_norm_bias_apt.cpp
 * \brief
 */

#include "arch35/add_rms_norm_bias_regbase.h"
#include "arch35/add_rms_norm_bias_regbase_split_d.h"

using namespace AscendC;
using namespace AddRmsNormBiasA5;

extern "C" __global__ __aicore__ void add_rms_norm_bias(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd,
    GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe aptPipe;
    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(AddRMSNormBiasRegbaseRFullLoadTilingData, aptTilingDataIn, tiling);
        KernelAddRmsNormBiasRegBase<DTYPE_X1> op(&aptPipe);
        op.Init(x1, x2, gamma, beta, y, rstd, x, &aptTilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(2000)) {
        GET_TILING_DATA_WITH_STRUCT(AddRMSNormBiasRegbaseTilingData, aptTilingDataIn, tiling);
        KernelAddRmsNormBiasRegBaseSplitD<DTYPE_X1> op(&aptPipe);
        op.Init(x1, x2, gamma, beta, y, rstd, x, &aptTilingDataIn);
        op.Process();
    }
}
