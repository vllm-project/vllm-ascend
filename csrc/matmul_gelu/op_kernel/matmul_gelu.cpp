/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "matmul_gelu_kernel.h"

using namespace MatmulGelu_Kernel;


extern "C" __global__ __aicore__ void matmul_gelu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MatmulGeluTilingData);
    GET_TILING_DATA(tiling_data, tiling);

    if (!tiling_data.transB) {
        MatmulGeluImpl<layout::RowMajor, half, 128, 256, 256, 64>(tiling_data, x, weight, bias, output, workspace);
    } else {
        MatmulGeluImpl<layout::ColumnMajor, half, 128, 256, 256, 64>(tiling_data, x, weight, bias, output, workspace);
    }

}
