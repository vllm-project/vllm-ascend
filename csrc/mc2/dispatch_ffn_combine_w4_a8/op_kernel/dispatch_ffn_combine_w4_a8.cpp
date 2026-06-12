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
 * \file dispatch_ffn_combine_w4a8.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_w4_a8_tiling.h"
#include "dispatch_ffn_combine_w4_a8.h"
#include "dispatch_ffn_combine_w4_a8_base.h"

using namespace AscendC;
using namespace DispatchFFNCombineW4A8Impl;

__aicore__ inline void SetMoeProfilePtr(__gm__ int64_t *profilePtr)
{
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef SPLIT_CORE_CUBE
    g_moeProfilePtrCube = profilePtr;
#elif defined(SPLIT_CORE_VEC)
    g_moeProfilePtrVec = profilePtr;
#else
    g_moeProfilePtr = profilePtr;
#endif
#else
    g_moeProfilePtr = profilePtr;
#endif
}

extern "C" __global__ __aicore__ void dispatch_ffn_combine_w4_a8(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId,
    GM_ADDR scale1, GM_ADDR scale2, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR probs,
    GM_ADDR xActiveMask, GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR profiling_data,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#if ENABLE_MOE_PROFILING
    __gm__ int64_t *profBase = (__gm__ int64_t *)(profiling_data);
    uint32_t slotOffset = 0;
    if (g_coreType == AscendC::AIC) {
        slotOffset = AscendC::GetBlockIdx() * PROF_SIZE_PER_CORE;
    } else {
        slotOffset = (AscendC::GetBlockNum() + AscendC::GetBlockIdx()) * PROF_SIZE_PER_CORE;
    }
    SetMoeProfilePtr(profBase + slotOffset);
    GetMoeProfilePtr()[0] = 1;
    GetMoeProfilePtr()[PROF_SIZE_PER_CORE - 1] = AscendC::GetSystemCycle();
#else
    (void)profiling_data;
#endif
    MoeTracing(TRACE_POINT("processing", "B"));

    REGISTER_TILING_DEFAULT(DispatchFFNCombineW4A8TilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineW4A8TilingData, tilingData, tilingGM);
        DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, bias1, bias2, probs, xActiveMask, c, expertTokenNums,
                workspaceGM, tilingGM);
        op.Process();
    }

    MoeTracing(TRACE_POINT("processing", "E"));

#if ENABLE_MOE_PROFILING
    AscendC::GlobalTensor<int64_t> coreGlobal;
    coreGlobal.SetGlobalBuffer(GetMoeProfilePtr(), PROF_SIZE_PER_CORE);
    uint32_t traceCount = GetMoeProfilePtr()[0];
    for (unsigned i = 0; i < traceCount; ++i) {
        if (i == 0 || (((uint64_t)coreGlobal[i].GetPhyAddr()) & 63) == 0) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(coreGlobal[i]);
        }
    }
#endif
}
