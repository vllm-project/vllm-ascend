/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full license text.
 */
// A5-only physical-address preprocessing eligibility, shared by host and kernel.
#ifndef QSFA_A5_ADDRESS_PRECOMPUTE_H
#define QSFA_A5_ADDRESS_PRECOMPUTE_H
#include <cstdint>
namespace qsfa_a5 {
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline bool CanPrecomputeAddresses(int64_t blockSize, uint64_t tableWidth, uint64_t k, int64_t sparseMode,
    bool splitG, bool noRope)
{
    // Keep the scalar gather for split-G RoPE: the address stage did not amortize its cost in testing.
    if (splitG && !noRope) {
        return false;
    }
    if (sparseMode != 3 || blockSize <= 0 || (blockSize & (blockSize - 1)) != 0 || k == 0) {
        return false;
    }
    const uint64_t alignedK = (k + 127U) / 128U * 128U;
    const uint64_t bytes = (tableWidth * sizeof(int32_t) + 511U) / 512U * 512U +
        alignedK * (sizeof(int32_t) + sizeof(int64_t));
    return bytes <= 248U * 1024U;
}
} // namespace qsfa_a5
#endif
