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
 * \file ops_utils.h
 * \brief
 */
#ifndef ATVC_COMMON_OPS_UTILS_H
#define ATVC_COMMON_OPS_UTILS_H

#include <type_traits>

namespace OpsUtils {
template <typename T>
inline T Ceil(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
inline T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline T FloorDiv(T a, U b)
{
    if (b == 0) {
        return a;
    }
    return a / b;
}

template <typename T>
inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

/**
 * if align is 0, return 0
 */
template <typename T, typename U>
inline typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, U align)
{
    return align == 0 ? 0 : x / align * align;
}
} // namespace OpsUtils

#endif // ATVC_COMMON_OPS_UTILS_H
