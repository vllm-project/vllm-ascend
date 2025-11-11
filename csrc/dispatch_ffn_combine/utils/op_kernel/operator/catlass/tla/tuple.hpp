/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_HPP
#define TLA_TUPLE_HPP

#include "tla/numeric/integral_constant.hpp"
#include "tla/numeric/integer_sequence.hpp"

namespace tla {

namespace detail {

// EBO stands for "empty base optimization."
template <size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

// Specialization for types T that are empty;
template <size_t N, class T>
struct EBO<N, T, true> {
    CATLASS_HOST_DEVICE constexpr
    EBO() {}

    CATLASS_HOST_DEVICE constexpr
    EBO(T const&) {}
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr
T getv(EBO<N, T, true> const&)
{
    return {};
}

// Specialization for types T that are not empty;
template <size_t N, class T>
struct EBO<N, T, false> {
    CATLASS_HOST_DEVICE constexpr
    EBO() : t_{} {}

    CATLASS_HOST_DEVICE constexpr
    EBO(T const& t) : t_{t} {}

    T t_;
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr
T const& getv(EBO<N, T, false> const& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr
T& getv(EBO<N, T, false>& x)
{
    return x.t_;
}

// TupleBase
template <class IdxSeq, class... T>
struct TupleBase;

template <size_t... I, class... T>
struct TupleBase<index_sequence<I...>, T...> : EBO<I, T>... {
    CATLASS_HOST_DEVICE constexpr
    TupleBase() {}

    CATLASS_HOST_DEVICE constexpr
    TupleBase(T const&... t) : EBO<I, T>(t)... {}
};

} // end namespace detail

// tla::tuple class.
template <class... T>
struct tuple : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...> {
    CATLASS_HOST_DEVICE constexpr
    tuple() {}

    CATLASS_HOST_DEVICE constexpr
    tuple(T const&... t) : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...>(t...) {}
};

// get for tla::tuple
template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr
decltype(auto) get(tuple<T...> const& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr
decltype(auto) get(tuple<T...>& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr
decltype(auto) get(tuple<T...>&& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(static_cast<tuple<T...>&&>(t));
}

namespace detail {

template <class T>
auto has_tuple_size(T*) -> bool_constant<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size((T*)0)) {};

template <class... T>
struct tuple_size<tla::tuple<T...>>
    : std::integral_constant<size_t, sizeof...(T)> {};

template <class... T>
struct tuple_size<const tla::tuple<T...>>
    : std::integral_constant<size_t, sizeof...(T)> {};

} // end namespace tla

#endif // TLA_TUPLE_HPP
