/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_base.h
 * \brief MoeTracing / profiling infrastructure for dispatch_ffn_combine operator.
 */

#ifndef DISPATCH_FFN_COMBINE_BASE_H
#define DISPATCH_FFN_COMBINE_BASE_H

#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// Profiling compile-time switches
// ---------------------------------------------------------------------------
#define ENABLE_MOE_PROFILING 1
#define PROF_SIZE_PER_CORE 2048
#define ENABLE_MOE_PROFILING_BARRIER true

// ---------------------------------------------------------------------------
// TRACE_POINT  macro — label + B/E tag, replaced by trace_preprocessor.py
// ---------------------------------------------------------------------------
#ifndef TRACE_POINT
#define TRACE_POINT(label, be) label "|" be
#endif

// ---------------------------------------------------------------------------
// Per-core profiling buffer pointer
// ---------------------------------------------------------------------------
__BLOCK_LOCAL__ __inline__ int64_t* g_moeProfilePtr;

__aicore__ inline int64_t* GetMoeProfilePtr(uint32_t idx = 0)
{
    return &g_moeProfilePtr[idx];
}

// ---------------------------------------------------------------------------
// MoeTracing — template function, NOT a macro
// ---------------------------------------------------------------------------
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracingWithCycle(int64_t data, int64_t cycle)
{
#if ENABLE_MOE_PROFILING
    if constexpr (sync) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    int64_t *profileData = GetMoeProfilePtr();
    profileData[profileData[0]++] = data;
    profileData[PROF_SIZE_PER_CORE - profileData[0]] = cycle;
#endif
}

// Basic call: record point_id + current cycle
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data)
{
    MoeTracingWithCycle<sync>(data, AscendC::GetSystemCycle());
}

// With index: encodes index into data upper 32 bits (for expert group / stage)
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t index)
{
    MoeTracing<sync>(data | (int64_t)(((uint64_t)index) << 32));
}

// With extraId + index: for simultaneous stageId and loop index
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t extraId, uint32_t index)
{
    MoeTracing<sync>(data, (extraId | (index << 8)));
}

// ---------------------------------------------------------------------------
// SetMoeProfilePtr — assigns the per-core buffer pointer
// ---------------------------------------------------------------------------
__aicore__ inline void SetMoeProfilePtr(int64_t *profilePtr)
{
    g_moeProfilePtr = profilePtr;
}

#endif // DISPATCH_FFN_COMBINE_BASE_H
