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
 * \file mhc_pre_clamp_sinkhorn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_clamp_sinkhorn_m_k_split_core.h"
#include "mhc_pre_clamp_sinkhorn_m_split_core.h"
#include "mhc_pre_clamp_sinkhorn_base.h"
using namespace MhcPreClampSinkhorn;

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_pre_clamp_sinkhorn(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                             GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
                                             GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                             GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR hResLogits,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
    const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        // -----------split m--------------
        TPipe pipe;
        MhcPreClampSinkhorn::MhcPreClampSinkhornStage1<DTYPE_X> op1;
        op1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        op1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        op1.Init(x, phi, invRms, hcBeforeNorm, userWs, tilingData, &pipe);
        op1.Process();
        pipe.Destroy();
            
        TPipe pipeStage2;
        MhcPreClampSinkhorn::MhcPreClampSinkhornStage2<DTYPE_X> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes,
                    hPre, hcBeforeNorm, invRms, sumOut, normOut, hResLogits,
                    userWs, tilingData, &pipeStage2);
        op2.Process(false);
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(1)) {
        // -----------split m k------------
        for (int64_t bsLoopIdx = 0; bsLoopIdx < tilingData->bsLoop; bsLoopIdx++) {
            int64_t curBsOffset = bsLoopIdx * tilingData->curBsSplit;
            int64_t curBs = (bsLoopIdx == tilingData->bsLoop - 1) ? tilingData->tailBs : tilingData->curBsSplit;
            bool isTailBsLoop = (bsLoopIdx == tilingData->bsLoop - 1) && (tilingData->tailBs != tilingData->curBsSplit);
            TPipe pipe;
            MhcPreClampSinkhorn::MhcPreClampSinkhornMembaseKSplitCorePart1<DTYPE_X> op;
            op.Init(x, phi, userWs, tilingData, &pipe, curBsOffset, curBs, isTailBsLoop);
            op.Process(curBsOffset, curBs, isTailBsLoop);
            pipe.Destroy();

            TPipe pipeStage2;
            MhcPreClampSinkhorn::MhcPreClampSinkhornMembaseKSplitCorePart2<DTYPE_X> op2;
            op2.Init(x, alpha, bias, hin, hPost, hRes,
                        hPre, hcBeforeNorm, invRms, sumOut, normOut, hResLogits,
                        userWs, tilingData, &pipeStage2, curBsOffset, isTailBsLoop);
            op2.Process(curBsOffset, curBs, isTailBsLoop);
            pipeStage2.Destroy();
            SyncAll<false>(); // cv全部同步
        }
    }
    // #endif
}