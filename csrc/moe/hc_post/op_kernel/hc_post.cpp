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
 * \file hc_post_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#if defined(__DAV_C310__)
  #include "hc_post_float32.h"
  #include "hc_post_bfloat16.h"
#endif
#include "hc_post_d_split.h"

using namespace AscendC;
using namespace HcPost;
#if defined(__DAV_C310__)
    using namespace HcPostRegBase;
#endif

#define HC_POST_FLOAT 0
#define HC_POST_BFLOAT16 1

extern "C" __global__ __aicore__ void hc_post(GM_ADDR x, GM_ADDR residual, GM_ADDR post,
    GM_ADDR comb, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    #if defined(__DAV_C310__)
        GET_TILING_DATA_WITH_STRUCT(HcPostTilingData, tilingData, tiling);
        const HcPostTilingData *__restrict hcPostTilingData = &tilingData;
        if (TILING_KEY_IS(HC_POST_FLOAT)) {
            HcPostRegBaseFloat32<DTYPE_POST> op;
            op.Init(x, residual, post, comb, y, workspace, hcPostTilingData, &pipe);
            op.Process();
            return;
        } else if (TILING_KEY_IS(HC_POST_BFLOAT16)) {
            HcPostRegBaseBfloat16<DTYPE_X, DTYPE_POST> op;
            op.Init(x, residual, post, comb, y, workspace, hcPostTilingData, &pipe);
            op.Process();
            return;
        }
    #else
        GET_TILING_DATA_WITH_STRUCT(HcPostTilingData, tilingData, tiling);
        const HcPostTilingData *__restrict hcPostTilingData = &tilingData;
        HcPostKernelDSplit<DTYPE_X, DTYPE_POST> op;
        op.Init(x, residual, post, comb, y, workspace, hcPostTilingData, &pipe);
        op.Process();
        return;
    #endif
}