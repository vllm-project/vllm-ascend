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
 * \file hc_pre.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

#if defined(__DAV_C310__)
    #include "hc_pre_m_k_split_core_arch35.h"
    #include "hc_pre_m_split_core_arch35.h"
    #include "hc_pre_base_arch35.h"
    using namespace HcPreNs;
#else
    #include "lib/matmul_intf.h"
    #include "hc_pre_m_k_split_core.h"
    #include "hc_pre_base.h"
    using namespace HcPre;
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void hc_pre(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR hc_scale, GM_ADDR hc_base,
                                             GM_ADDR y, GM_ADDR post, GM_ADDR comb_frag, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    TPipe pipe;

    // 950PR 950DT
    #if defined(__DAV_C310__)
        if (TILING_KEY_IS(1000)) {
            GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tiling_data_in, tiling);
            const HcPreTilingData *__restrict tilingData = &tiling_data_in;
            HcPreNs::HcPreMKSplitCorePart1<DTYPE_X> op;
            op.Init(x, hc_fn, userWs, tilingData, &pipe);
            op.Process();
            pipe.Destroy();

            TPipe pipeStage2;
            HcPreNs::HcPreMKSplitCorePart2<DTYPE_X> op2;
            op2.Init(x, hc_scale, hc_base, y, post, comb_frag, userWs, tilingData, &pipeStage2);
            op2.Process();
            pipeStage2.Destroy();
        } else if (TILING_KEY_IS(1001)) {
            GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tiling_data_in, tiling);
            const HcPreTilingData *__restrict tilingData = &tiling_data_in;
            HcPreNs::HcPreMSplitCorePart1<DTYPE_X> op;
            op.Init(x, hc_fn, hc_scale, hc_base, y, post, comb_frag, tilingData, &pipe);
            op.Process();
        }
    #else
      // A3
        if (TILING_KEY_IS(0)) {
            GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tiling_data_in, tiling);
            const HcPreTilingData *__restrict tilingData = &tiling_data_in;
            HcPre::HcPreMembaseKSplitCorePart1<DTYPE_X> op;
            op.Init(x, hc_fn, userWs, tilingData, &pipe);
            op.Process();

            pipe.Destroy();

            TPipe pipeStage2;
            HcPre::HcPreMembaseKSplitCorePart2<DTYPE_X> op2;
            op2.Init(x, hc_scale, hc_base, y, post, comb_frag, userWs, tilingData, &pipeStage2);
            op2.Process();

            pipeStage2.Destroy();
        }
    #endif
}