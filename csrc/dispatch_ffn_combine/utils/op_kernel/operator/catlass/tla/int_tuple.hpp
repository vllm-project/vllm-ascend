/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_INT_TUPLE_HPP
#define TLA_INT_TUPLE_HPP

#include "tla/type_traits.hpp"
#include "tla/tuple.hpp"
#include "tla/numeric/integral_constant.hpp"
#include "tla/numeric/integer_sequence.hpp"

namespace tla {
//
// Apply (Unpack)
// (t, f) => f(t_0,t_1,...,t_n)
//

namespace detail {
template <class T, class F, int... I>
CATLASS_HOST_DEVICE constexpr
auto apply(T&& t, F&& f, seq<I...>)
{
    return f(get<I>(static_cast<T&&>(t))...);
}

template <class T, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr
auto
tapply(T&& t, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(static_cast<T&&>(t)))...);
}

} // end namespace detail

template <class T, class F>
CATLASS_HOST_DEVICE constexpr
auto apply(T&& t, F&& f)
{
    return detail::apply(static_cast<T&&>(t), f, tuple_seq<T>{});
}

template <class T, class F, class G>
CATLASS_HOST_DEVICE constexpr
auto
transform_apply(T&& t, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::tapply(static_cast<T&&>(t), f, g, tuple_seq<T>{});
    } else {
        return g(f(static_cast<T&&>(t)));
    }
}

template <size_t I, class T,
          __TLA_REQUIRES(tla::is_integral<tla::remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr
decltype(auto) get(T&& t) noexcept
{
    static_assert(I == 0, "Index out of range");
    return static_cast<T&&>(t);
}

template <size_t I0, size_t I1, size_t... Is, class T>
CATLASS_HOST_DEVICE constexpr
decltype(auto) get(T&& t) noexcept
{
    return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}

// max
template <class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr
auto max(T0 const& t0, Ts const&... ts);

struct UnpackedMax {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr
    auto operator()(T const&... v) const {
        return tla::max(v...);
    }
};

template <class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr
auto max(T0 const& t0, Ts const&... ts)
{
    if constexpr (is_tuple<T0>::value) {
        return tla::max(tla::apply(t0, UnpackedMax{}), ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return tla::max(t0, tla::max(ts...));
    }
}

// rank
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr
auto rank(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<tuple_size<Tuple>::value>{};
        } else {
            return Int<1>{};
        }
    } else {
        return rank(get<Is...>(t));
    }
}

template <class Tuple>
using rank_t = decltype(rank(std::declval<Tuple>()));

template <class Tuple>
static constexpr auto rank_v = rank_t<Tuple>::value;

// depth
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr
auto depth(Tuple const& t);

struct UnpackedDepth {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr
    auto operator()(T const&... v) const {
        return tla::max(depth(v)...);
    }
};

template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr
auto depth(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<1>{} + tla::apply(t, UnpackedDepth{});
        } else {
            return Int<0>{};
        }
    } else {
        return depth(get<Is...>(t));
    }
}

template <class Tuple>
using depth_t = decltype(depth(std::declval<Tuple>()));

template <class Tuple>
static constexpr auto depth_v = depth_t<Tuple>::value;

struct MultipliesUnaryLfold {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr
    auto operator()(T const&... v) const {
        return (... * v);
    }
};

// Implementation of product as a function object
struct Product {
    template <class IntTuple>
    CATLASS_HOST_DEVICE constexpr
    auto
    operator()(IntTuple const& a) const
    {
        if constexpr (is_tuple<IntTuple>::value) {
        if constexpr (tuple_size<IntTuple>::value == 0) {
            return Int<1>{};
        } else {
            return tla::transform_apply(a, Product{}, MultipliesUnaryLfold{});
        }
        } else if constexpr (tla::is_integral<IntTuple>::value) {
        return a;
        }
    }
};

} // end namespace tla

#endif // TLA_INT_TUPLE_HPP