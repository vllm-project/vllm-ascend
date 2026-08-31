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
 * \file flash_mla_with_kvcache_tiling_utils.h
 * \brief
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TILING_UTILS_H
#define FLASH_MLA_WITH_KVCACHE_TILING_UTILS_H

#include "tiling/tiling_api.h"

namespace optiling {
namespace flash_mla_with_kvcache {

// 逐维比对strides与连续场景的期望stride, index返回首个非连续维, 全连续返回GRAPH_SUCCESS
inline ge::graphStatus CheckTensorContiguous(const uint32_t &tensorDimNum, const gert::Shape &inputShape,
                                             const gert::Stride *strides, int32_t index)
{
    if (strides == nullptr || strides->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t preStride = 1;
    for (index = static_cast<int32_t>(tensorDimNum) - 1; index >= 0; index--) {
        if (index == static_cast<int32_t>(tensorDimNum) - 1) {
            if (strides->GetStride(index) != preStride) {
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        uint64_t expected = inputShape.GetDim(index + 1) * preStride;
        if (strides->GetStride(index) != expected) {
            return ge::GRAPH_FAILED;
        }
        preStride = expected;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename T>
inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline auto IncreGcd(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    if (a % b == 0) {
        return b;
    }
    return IncreGcd(b, a % b);
}

} // namespace flash_mla_with_kvcache
} // namespace optiling

#endif // FLASH_MLA_WITH_KVCACHE_TILING_UTILS_H