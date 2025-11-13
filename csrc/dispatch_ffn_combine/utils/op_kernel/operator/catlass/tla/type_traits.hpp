/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_UTIL_TYPE_TRAITS_HPP
#define TLA_UTIL_TYPE_TRAITS_HPP

#include <tuple>

#define __TLA_REQUIRES(...)   typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

namespace tla {

// using std::remove_cvref;
template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// using std::remove_cvref_t;
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;


// tuple_size, tuple_element
template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<size_t, std::tuple_size<T>::value> {};

template <class T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

} // end namespace tla

#endif // TLA_UTIL_TYPE_TRAITS_HPP