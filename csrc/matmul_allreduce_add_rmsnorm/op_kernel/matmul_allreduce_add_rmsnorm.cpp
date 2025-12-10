/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    GET_TILING_DATA(tiling_data, tiling); 
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    auto tilingData = (__gm__ MatmulAllreduceAddRmsnormTilingData*)tiling;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tilingData->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tilingData->mc2CcTiling));
    auto contextGM0 = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();

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
