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
 * \file mhc_pre_clamp_sinkhorn_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

#include "arch35/mhc_pre_clamp_sinkhorn_m_k_split_core_arch35.h"
#include "arch35/mhc_pre_clamp_sinkhorn_m_split_core_arch35.h"
#include "arch35/mhc_pre_clamp_sinkhorn_m_k_split_core_gradout_arch35.h"
#include "arch35/mhc_pre_clamp_sinkhorn_m_split_core_gradout_arch35.h"
#include "arch35/mhc_pre_clamp_sinkhorn_base_arch35.h"
using namespace MhcPreClampSinkhornNs;

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_pre_clamp_sinkhorn(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                             GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
                                             GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                             GM_ADDR sumOut, GM_ADDR normOut,
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

    TPipe pipe;

    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
        const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMKSplitCorePart1<DTYPE_X> op;
        op.Init(x, phi, userWs, tilingData, &pipe);
        op.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMKSplitCorePart2<DTYPE_X> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, userWs, tilingData, &pipeStage2);
        op2.Process();
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(1001)) {
        GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
        const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMSplitCorePart1<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, hin, hPost, hRes, tilingData, &pipe);
        op.Process();

    } else if (TILING_KEY_IS(2000)) {  // K分核，Gradout
        GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
        const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMKSplitCorePart1Gradout<DTYPE_X> op;
        op.Init(x, phi, userWs, tilingData, &pipe);
        op.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMKSplitCorePart2Gradout<DTYPE_X> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut,
                 userWs, tilingData, &pipeStage2);
        op2.Process();
        pipeStage2.Destroy();

    } else if (TILING_KEY_IS(2001)) { // M分核，Gradout
        GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
        const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;
        MhcPreClampSinkhornNs::MhcPreClampSinkhornMSplitCorePart1Gradout<DTYPE_X> op;
        op.Init(x, phi, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut,
              tilingData, &pipe);
        op.Process();
    }
}