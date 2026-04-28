/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file integral_constant.h
 * \brief
 */
#ifndef UTILS_INTEGRAL_CONSTANT_H
#define UTILS_INTEGRAL_CONSTANT_H
#include "kernel_operator.h"

namespace AscendC {
namespace Std {
template <typename...>
struct always_false : public false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;
} // namespace Std
} // namespace AscendC

namespace Act {
namespace Gemm {
template <int32_t t>
using Int = AscendC::Std::integral_constant<int32_t, t>;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _4 = Int<4>;
using _8 = Int<8>;
using _16 = Int<16>;
using _32 = Int<32>;
using _64 = Int<64>;
using _128 = Int<128>;
using _256 = Int<256>;
using _512 = Int<512>;
using _1024 = Int<1024>;
using _2048 = Int<2048>;
using _4096 = Int<4096>;
using _8192 = Int<8192>;
using _16384 = Int<16384>;
using _32768 = Int<32768>;
using _65536 = Int<65536>;

// Unary operator
template <auto t>
__host_aicore__ inline constexpr Int<+t> operator+(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<-t> operator-(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<~t> operator~(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<!t> operator!(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<*t> operator*(Int<t>)
{
    return {};
}

// Binary operator
template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t + u)> operator+(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t - u)> operator-(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t * u)> operator*(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t / u)> operator/(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t % u)> operator%(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t & u)> operator&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t | u)> operator|(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t ^ u)> operator^(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t << u)> operator<<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >> u)> operator>>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t && u)> operator&&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t || u)> operator||(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t == u)> operator==(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t != u)> operator!=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t > u)> operator>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t < u)> operator<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >= u)> operator>=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t <= u)> operator<=(Int<t>, Int<u>)
{
    return {};
}
} // namespace Gemm
} // namespace Act
#endif