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
 * \file tuple_utils.h
 * \brief
 */
#ifndef UTILS_TUPLE_UTILS_H
#define UTILS_TUPLE_UTILS_H

#include "lib/std/tuple.h"
#include "./integral_constant.h"

namespace Act {
namespace Gemm {
// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

template <size_t N, typename Tp>
__aicore__ constexpr inline decltype(auto) GetIntegralConstant()
{
    static_assert(AscendC::Std::is_tuple_v<Tp>, "Input must be a AscendC::Std::tuple type");
    return AscendC::Std::tuple_element<N, Tp>::type::value;
}
} // namespace Gemm
} // namespace Act
#endif