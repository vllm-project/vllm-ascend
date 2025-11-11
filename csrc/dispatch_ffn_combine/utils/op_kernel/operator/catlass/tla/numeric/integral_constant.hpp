/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_NUMERIC_INTEGER_CONSTANT_HPP
#define TLA_NUMERIC_INTEGER_CONSTANT_HPP

#include "catlass/catlass.hpp"
#include "tla/type_traits.hpp"
#include "tla/numeric/math.hpp"

namespace tla {

// A constant value: short name and type-deduction for fast compilation
template <auto v>
struct C {
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(v);
    CATLASS_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Deprecate
template <class T, T v>
using constant = C<v>;

template <bool b>
using bool_constant = C<b>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
using is_std_integral = std::is_integral<T>;

// A more std:: conforming integral_constant that enforces type but interops with C<v>
template <class T, T v>
struct integral_constant : C<v> {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Use tla::is_std_integral<T> to match built-in integral types (int, int64_t, unsigned, etc)
// Use tla::is_integral<T> to match both built-in integral types AND static integral types.

template <class T>
struct is_integral : bool_constant<is_std_integral<T>::value> {};
template <auto v>
struct is_integral<C<v>                  > : true_type {};
template <class T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};

// is_static detects if an (abstract) value is defined completely by its type (no members)
template <class T>
struct is_static : bool_constant<std::is_empty<remove_cvref_t<T>>::value> {};

// is_constant detects if a type is a static integral type and if v is equal to a value

template <auto n, class T>
struct is_constant : false_type {};
template <auto n, class T>
struct is_constant<n, T const > : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T const&> : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T      &> : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T     &&> : is_constant<n, T> {};
template <auto n, auto v>
struct is_constant<n, C<v>                  > : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, integral_constant<T, v>> : bool_constant<v == n> {};

//
// Specializations
//

template <int v>
using Int = C<v>;
using _64     = Int<64>;
using _128    = Int<128>;
using _256    = Int<256>;
using _512    = Int<512>;

/***************/
/** Operators **/
/***************/

#define TLA_LEFT_UNARY_OP(OP)                                       \
    template <auto t>                                               \
    CATLASS_HOST_DEVICE constexpr                                   \
    C<(OP t)> operator OP (C<t>) {                                  \
        return {};                                                  \
    }
#define TLA_BINARY_OP(OP)                                           \
    template <auto t, auto u>                                       \
    CATLASS_HOST_DEVICE constexpr                                   \
    C<(t OP u)> operator OP (C<t>, C<u>) {                          \
        return {};                                                  \
    }

TLA_LEFT_UNARY_OP(+);
TLA_LEFT_UNARY_OP(-);
TLA_LEFT_UNARY_OP(~);
TLA_LEFT_UNARY_OP(!);
TLA_LEFT_UNARY_OP(*);

TLA_BINARY_OP(+);
TLA_BINARY_OP(-);
TLA_BINARY_OP(*);
TLA_BINARY_OP(/);
TLA_BINARY_OP(%);
TLA_BINARY_OP(&);
TLA_BINARY_OP(|);
TLA_BINARY_OP(^);
TLA_BINARY_OP(<<);
TLA_BINARY_OP(>>);

#undef TLA_BINARY_OP
#undef TLA_LEFT_UNARY_OP
#undef TLA_RIGHT_UNARY_OP

//
// Named functions from math.hpp
//

#define TLA_NAMED_UNARY_FN(OP)                                          \
    template <auto t>                                                   \
    CATLASS_HOST_DEVICE constexpr                                       \
    auto OP (C<t>) {                                                    \
        return C<OP(t)>{};                                              \
    }
#define TLA_NAMED_BINARY_FN(OP)                                         \
    template <auto t, auto u>                                           \
    CATLASS_HOST_DEVICE constexpr                                       \
    auto OP (C<t>, C<u>) {                                              \
        return C<OP(t, u)>{};                                           \
    }                                                                   \
    template <auto t, class U,                                          \
              __TLA_REQUIRES(is_std_integral<U>::value)>                \
    CATLASS_HOST_DEVICE constexpr                                       \
    auto OP (C<t>, U u) {                                               \
        return OP(t, u);                                                \
    }                                                                   \
    template <class T, auto u,                                          \
              __TLA_REQUIRES(is_std_integral<T>::value)>                \
    CATLASS_HOST_DEVICE constexpr                                       \
    auto OP (T t, C<u>) {                                               \
        return OP(t, u);                                                \
    }

TLA_NAMED_BINARY_FN(max);
TLA_NAMED_BINARY_FN(min);

#undef TLA_NAMED_UNARY_FN
#undef TLA_NAMED_BINARY_FN


} // end namespace tla

#endif // TLA_NUMERIC_INTEGER_CONSTANT_HPP
