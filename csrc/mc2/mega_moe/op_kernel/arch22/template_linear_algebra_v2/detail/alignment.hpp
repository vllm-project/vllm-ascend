/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ALIGNMENT_HPP
#define CATLASS_ALIGNMENT_HPP

#include "../../template_linear_algebra_v2/detail/macros.hpp"

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE
constexpr auto RoundUp(T const &val, U const &align)
{
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE
constexpr auto RoundDown(T const &val, U const &align)
{
    return val / align * align;
}

template <uint32_t DIVISOR, typename T>
CATLASS_HOST_DEVICE
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "DIVISOR must not be 0");
    return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
CATLASS_HOST_DEVICE
constexpr auto CeilDiv(T const &dividend, U const &divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <class T, class U>
CATLASS_HOST_DEVICE
constexpr auto Max(T const &a, U const &b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE
constexpr auto Min(T const &a, U const &b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

#endif  // CATLASS_ALIGNMENT_HPP
