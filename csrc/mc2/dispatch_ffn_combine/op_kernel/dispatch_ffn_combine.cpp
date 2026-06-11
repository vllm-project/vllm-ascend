/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dispatch_ffn_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine.h"
#include "dispatch_ffn_combine_base.h"

using namespace AscendC;
using namespace DispatchFFNCombineImpl;
extern "C" __global__ __aicore__ void dispatch_ffn_combine(GM_ADDR x, GM_ADDR w1, GM_ADDR w2,  GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs,
    GM_ADDR xActiveMask, GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR profiling_data, GM_ADDR workspaceGM,  GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tilingData, tilingGM);

#if ENABLE_MOE_PROFILING
        int64_t profData[PROF_SIZE_PER_CORE];
        profData[0] = 1;
        profData[PROF_SIZE_PER_CORE - 1] = AscendC::GetSystemCycle();
        SetMoeProfilePtr(&profData[0]);
#endif

        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM, tilingGM);
        op.Process();

#if ENABLE_MOE_PROFILING
        if (profiling_data != nullptr) {
            AscendC::GlobalTensor<int64_t> profGlobal;
            profGlobal.SetGlobalBuffer((__gm__ int64_t *)(profiling_data));
            AscendC::GlobalTensor<int64_t> coreGlobal;
            if (g_coreType == AscendC::AIC) {
                coreGlobal = profGlobal[AscendC::GetBlockIdx() * PROF_SIZE_PER_CORE];
            } else {
                coreGlobal = profGlobal[(AscendC::GetBlockNum() + AscendC::GetBlockIdx()) * PROF_SIZE_PER_CORE];
            }
            for (unsigned i = 0; i < profData[0]; ++i) {
                coreGlobal(i) = profData[i];
                coreGlobal(PROF_SIZE_PER_CORE - i - 1) = profData[PROF_SIZE_PER_CORE - i - 1];
            }
        }
#endif
    }
}