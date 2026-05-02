/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ALIGNMENT_HPP
#define CATLASS_ALIGNMENT_HPP

#include "catlass/detail/macros.hpp"
#include "tla/numeric/integral_constant.hpp"

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
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = (T::value + U::value - 1) / U::value * U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return (T::value + align - 1) / align * align;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return (val + U::value - 1) / U::value * U::value;
    } else { // int, int
        return (val + align - 1) / align * align;
    }
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
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = T::value / U::value * U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return T::value / align * align;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return val / U::value * U::value;
    } else { // int, int
        return val / align * align;
    }
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
    if constexpr (tla::is_static<T>::value && tla::is_static<U>::value) { // Int, Int
        constexpr uint32_t res = (T::value + U::value - 1) / U::value;
        return tla::Int<res>{};
    } else if constexpr (tla::is_static<T>::value) { // Int, int
        return (T::value + divisor - 1) / divisor;
    } else if constexpr (tla::is_static<U>::value) { // int, Int
        return (dividend + U::value - 1) / U::value;
    } else { // int, int
        return (dividend + divisor - 1) / divisor;
    }
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
