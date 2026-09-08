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
 * \file msa_index_score_debug.h
 * \brief Device-side debug helpers for MsaIndexScore (printf / DumpTensor).
 *
 * Enable by compiling with -DMSA_INDEX_SCORE_DEBUG=1 (and AscendC dump macros).
 * Dumps are limited to AIC/AIV block 0 and the first S-tile / first pass to
 * keep logs readable.
 */

#ifndef MSA_INDEX_SCORE_DEBUG_H
#define MSA_INDEX_SCORE_DEBUG_H

#include "kernel_operator.h"

#ifndef MSA_INDEX_SCORE_DEBUG
#define MSA_INDEX_SCORE_DEBUG 0
#endif

namespace MsaIndexScoreNs {

// DumpTensor 描述符：便于在日志中区分阶段。
constexpr uint32_t MSA_DUMP_DESC_Q = 1001;
constexpr uint32_t MSA_DUMP_DESC_K = 1002;
constexpr uint32_t MSA_DUMP_DESC_S_DOT = 2001;
constexpr uint32_t MSA_DUMP_DESC_S_MASK = 2002;
constexpr uint32_t MSA_DUMP_DESC_SCORE_MAX = 3001;
constexpr uint32_t MSA_DUMP_DESC_SCORE_FINAL = 3002;

// 常规采样长度；小尺寸用例 (D=16 / kvLen<=8) 时 DumpTensor 可覆盖完整向量。
constexpr uint32_t MSA_DEBUG_DUMP_ELEMS = 16;
// 边界 mask 窗口：打印 valid 尾部与紧随其后的 -inf，便于肉眼核对。
constexpr uint32_t MSA_DEBUG_MASK_WINDOW = 8;

#if MSA_INDEX_SCORE_DEBUG

__aicore__ inline bool MsaDebugPrimaryCore()
{
    // AIC: blockIdx == cube core；AIV: blockIdx == aivIdx（core0/sub0 -> 0）。
    return AscendC::GetBlockIdx() == 0;
}

template <typename T>
__aicore__ inline void MsaDumpGmSample(const AscendC::GlobalTensor<T> &g, uint32_t desc, uint32_t n)
{
    AscendC::DumpTensor(g, desc, n);
}

__aicore__ inline void MsaDumpUbSample(const AscendC::LocalTensor<float> &ub, uint32_t desc, uint32_t n)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DumpTensor(ub, desc, n);
}

__aicore__ inline void MsaPrintUbFloats(const AscendC::LocalTensor<float> &ub, uint32_t n)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint32_t lim = n < MSA_DEBUG_DUMP_ELEMS ? n : MSA_DEBUG_DUMP_ELEMS;
    for (uint32_t i = 0; i < lim; ++i) {
        AscendC::printf(" %f", ub.GetValue(i));
    }
    AscendC::printf("\n");
}

template <typename T>
__aicore__ inline void MsaPrintGmFloats(const AscendC::GlobalTensor<T> &g, uint32_t n)
{
    const uint32_t lim = n < MSA_DEBUG_DUMP_ELEMS ? n : MSA_DEBUG_DUMP_ELEMS;
    for (uint32_t i = 0; i < lim; ++i) {
        AscendC::printf(" %f", static_cast<float>(g.GetValue(i)));
    }
    AscendC::printf("\n");
}

#else // !MSA_INDEX_SCORE_DEBUG

__aicore__ inline bool MsaDebugPrimaryCore() { return false; }

template <typename T>
__aicore__ inline void MsaDumpGmSample(const AscendC::GlobalTensor<T> &, uint32_t, uint32_t)
{}

__aicore__ inline void MsaDumpUbSample(const AscendC::LocalTensor<float> &, uint32_t, uint32_t) {}

__aicore__ inline void MsaPrintUbFloats(const AscendC::LocalTensor<float> &, uint32_t) {}

template <typename T>
__aicore__ inline void MsaPrintGmFloats(const AscendC::GlobalTensor<T> &, uint32_t)
{}

#endif // MSA_INDEX_SCORE_DEBUG

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_DEBUG_H
