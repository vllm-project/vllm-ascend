/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_W4_A8_BASE_H
#define DISPATCH_FFN_COMBINE_W4_A8_BASE_H

#include "kernel_operator.h"

#define ENABLE_MOE_PROFILING 1
#define PROF_SIZE_PER_CORE 2048
#define ENABLE_MOE_PROFILING_BARRIER false

#ifndef TRACE_POINT
#define TRACE_POINT(label, event) 0
#endif

#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef SPLIT_CORE_CUBE
__BLOCK_LOCAL__ __inline__ __gm__ int64_t* g_moeProfilePtrCube;
#elif defined(SPLIT_CORE_VEC)
__BLOCK_LOCAL__ __inline__ __gm__ int64_t* g_moeProfilePtrVec;
#else
__BLOCK_LOCAL__ __inline__ __gm__ int64_t* g_moeProfilePtr;
#endif
#else
__BLOCK_LOCAL__ __inline__ __gm__ int64_t* g_moeProfilePtr;
#endif

__aicore__ inline __gm__ int64_t* GetMoeProfilePtr(uint32_t idx = 0)
{
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef SPLIT_CORE_CUBE
    return g_moeProfilePtrCube + idx;
#elif defined(SPLIT_CORE_VEC)
    return g_moeProfilePtrVec + idx;
#else
    return g_moeProfilePtr + idx;
#endif
#else
    return g_moeProfilePtr + idx;
#endif
}

template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracingWithCycle(int64_t data, int64_t cycle)
{
#if ENABLE_MOE_PROFILING
    if constexpr (sync) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __gm__ int64_t *profileData = GetMoeProfilePtr();
    uint32_t writeIdx = profileData[0];
    profileData[writeIdx++] = data;
    profileData[0] = writeIdx;
    profileData[PROF_SIZE_PER_CORE - writeIdx] = cycle;
#endif
}

template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data)
{
    MoeTracingWithCycle<sync>(data, AscendC::GetSystemCycle());
}

template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t index)
{
    MoeTracing<sync>(data | (int64_t)(((uint64_t)index) << 32));
}

template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t extraId, uint32_t index)
{
    MoeTracing<sync>(data, (extraId | (index << 8)));
}

#endif  // DISPATCH_FFN_COMBINE_W4_A8_BASE_H
