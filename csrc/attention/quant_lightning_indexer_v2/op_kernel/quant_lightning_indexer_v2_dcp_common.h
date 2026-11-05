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
 * \file quant_lightning_indexer_v2_dcp_common.h
 * \brief DCP (Decode Context Parallel) visible-length helper shared by arch35 kernel and metadata AICPU.
 *
 * The formula must stay identical to the arch22 implementation in
 * op_kernel/arch22/quant_lightning_indexer_v2_common_arch22.h (CalcDcpVisibleLen). arch22 itself is not
 * changed by the A5 feature; the shared golden tests keep both implementations aligned.
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_DCP_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_DCP_COMMON_H

#include <cstdint>

namespace QLIV2Dcp {

#ifdef __CCE_AICORE__
#define QLI_V2_DCP_INLINE __aicore__ inline
#else
#define QLI_V2_DCP_INLINE inline
#endif

QLI_V2_DCP_INLINE uint32_t MinU32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

QLI_V2_DCP_INLINE uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return (a > b) ? a : b;
}

// DCP: 由全局可见上界 u 算本地可见长度。
// u = max(C - Q_len + rowIdx + 1, 0)，即该 Q 行全局可见 token 的 exclusive 上界。
// 分片规则: 全局第 t 个 token 属于 rank (t / S) % P，本地序号按时间序排列。
QLI_V2_DCP_INLINE uint32_t CalcDcpVisibleLen(uint32_t u, uint32_t localS2Len, uint32_t cpWorld, uint32_t cpRank,
                                             uint32_t cpInterleaveSize)
{
    if (u == 0U || cpWorld == 0U || cpInterleaveSize == 0U) {
        return 0U;
    }
    uint64_t period = static_cast<uint64_t>(cpWorld) * static_cast<uint64_t>(cpInterleaveSize);
    uint64_t fullRounds = (static_cast<uint64_t>(u) / period) * static_cast<uint64_t>(cpInterleaveSize);
    int64_t rem = static_cast<int64_t>(static_cast<uint64_t>(u) % period) -
                  static_cast<int64_t>(cpRank) * static_cast<int64_t>(cpInterleaveSize);
    uint64_t remU = (rem <= 0) ? 0U : static_cast<uint64_t>(rem);
    uint64_t tail = (remU < static_cast<uint64_t>(cpInterleaveSize)) ? remU : static_cast<uint64_t>(cpInterleaveSize);
    uint64_t visible = fullRounds + tail;
    return (visible < static_cast<uint64_t>(localS2Len)) ? static_cast<uint32_t>(visible) : localS2Len;
}

// Kernel hot path: period / rankOffset / shift are precomputed once per launch.
// For power-of-two periods (the common DCP case, e.g. P=8/S=128 -> 1024) this
// replaces the 64-bit divide/modulo with shift and mask.
QLI_V2_DCP_INLINE uint32_t CalcDcpVisibleLenU32(uint32_t u, uint32_t localS2Len, uint32_t period,
                                                uint32_t rankOffset, uint32_t cpInterleaveSize, uint32_t periodShift,
                                                bool periodPow2)
{
    if (u == 0U || period == 0U || cpInterleaveSize == 0U) {
        return 0U;
    }
    uint32_t cycles;
    uint32_t rem;
    if (periodPow2) {
        cycles = u >> periodShift;
        rem = u & (period - 1U);
    } else {
        cycles = u / period;
        rem = u - cycles * period;
    }
    uint32_t fullRounds = cycles * cpInterleaveSize;
    int32_t diff = static_cast<int32_t>(rem) - static_cast<int32_t>(rankOffset);
    uint32_t remU = (diff <= 0) ? 0U : static_cast<uint32_t>(diff);
    uint32_t tail = (remU < cpInterleaveSize) ? remU : cpInterleaveSize;
    uint64_t visible = static_cast<uint64_t>(fullRounds) + static_cast<uint64_t>(tail);
    return (visible < static_cast<uint64_t>(localS2Len)) ? static_cast<uint32_t>(visible) : localS2Len;
}

#undef QLI_V2_DCP_INLINE

} // namespace QLIV2Dcp

#endif // QUANT_LIGHTNING_INDEXER_V2_DCP_COMMON_H
