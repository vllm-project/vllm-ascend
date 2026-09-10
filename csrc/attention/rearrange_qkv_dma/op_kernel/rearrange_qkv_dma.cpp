// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#include "rearrange_qkv_dma.h"

using namespace rearrange_qkv_impl;

extern "C" __global__ __aicore__ void rearrange_qkv_dma(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    RearrangeQkvKernel op;
    op.Init(x, y, tilingData, &pipe);
    op.Process();
}
