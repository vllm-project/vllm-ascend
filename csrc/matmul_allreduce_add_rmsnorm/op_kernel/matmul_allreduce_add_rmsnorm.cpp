/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lib/matmul_intf.h"
#include <kernel_operator.h>
#include "matmul_allreduce_add_rmsnorm_aic_kernel.h"
#include "matmul_allreduce_add_rmsnorm_aiv_kernel.h"
 
extern "C" __global__ __aicore__ void matmul_allreduce_add_rmsnorm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual,
    GM_ADDR gamma, GM_ADDR y, GM_ADDR add_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulAllreduceAddRmsnormTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAllreduceAddRmsnormTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    auto contextGM0 = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();

    hccl_.InitV2(contextGM0, &tiling_data);
    hccl_.SetCcTilingV2(offsetof(MatmulAllreduceAddRmsnormTilingData, mc2CcTiling));

    if ASCEND_IS_AIC {
        MatmulAllreduceAddRmsnormAicKernel<DTYPE_X1, DTYPE_Y> op;
        op.Init(x1, x2, residual, gamma, y, workspace, &tiling_data, hccl_);
        op.Process();
        return;
    }

    if ASCEND_IS_AIV {
        MatmulAllreduceAddRmsnormAivKernel<DTYPE_X1, DTYPE_Y> op;

        op.Init(x1, x2, residual, gamma, y, add_out, workspace, &tiling_data, hccl_);
        op.Process(&tiling_data);
        return;
    }
}
