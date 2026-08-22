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
 * \file mhc_post.cpp
 * \brief MhcPost kernel entry point
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_post.h"

using namespace AscendC;
using namespace MhcPost;

template <uint16_t usePermanentX>
__global__ __aicore__ void mhc_post(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcPostTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPostTilingData, tilingData, tiling);
    TPipe tPipe;

    MhcPostKernel<DTYPE_X, usePermanentX> op(&tPipe, &tilingData);
    op.Init(x, hRes, hOut, hPost, output, workspace);
    op.Process();
}