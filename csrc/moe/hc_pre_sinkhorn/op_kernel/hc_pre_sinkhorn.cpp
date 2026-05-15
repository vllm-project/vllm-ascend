/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hc_pre_sinkhorn.cpp
 * \brief
 */

#if defined(__DAV_C310__) 
    #include "hc_pre_sinkhorn_regbase_perf.h"
    #include "hc_pre_sinkhorn_regbase_base.h"
#else
    #include "hc_pre_sinkhorn_perf.h"
    #include "hc_pre_sinkhorn_base.h"
#endif

using namespace HcPreSinkhorn;

extern "C" __global__ __aicore__ void hc_pre_sinkhorn(GM_ADDR mixes, GM_ADDR rsqrt, GM_ADDR hcScale, GM_ADDR hcBase,
                                             GM_ADDR x, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        HcPreSinkhorn::HcPreSinkhornPerf<DTYPE_X> op;
        op.Init(mixes, rsqrt, hcScale, hcBase, x, y, post, combFrag, userWs, &tilingData, &pipe);
        op.Process();
    }
}