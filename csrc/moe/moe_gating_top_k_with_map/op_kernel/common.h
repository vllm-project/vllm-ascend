/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_WITH_MAP_COMMON_H_V35
#define MOE_GATING_TOP_K_WITH_MAP_COMMON_H_V35

#include "kernel_operator.h"

namespace MoeGatingTopKWithMap {
using namespace AscendC;
constexpr int32_t MIN_FP32 = 0xFF800000;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T1 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

} // namespace MoeGatingTopKWithMap
#endif // MOE_GATING_TOP_K_WITH_MAP_COMMON_H_V35
