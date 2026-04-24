/**
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

/*!
 * \file hc_post_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#if defined(__DAV_C310__)
  #include "hc_post_float32.h"
  #include "hc_post_bfloat16.h"
    using namespace HcPostRegBase;
#endif
#include "hc_post_d_split.h"

using namespace AscendC;
using namespace HcPost;

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