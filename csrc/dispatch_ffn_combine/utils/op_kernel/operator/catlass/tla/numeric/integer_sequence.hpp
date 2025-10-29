/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_NUMERIC_INTEGER_SEQUENCE_HPP
#define TLA_NUMERIC_INTEGER_SEQUENCE_HPP

#include "tla/numeric/integral_constant.hpp"
#include "tla/type_traits.hpp"

namespace tla {

template <typename T, T... Ns>
struct IntegerSequence {
    using value_type = T;
    static constexpr size_t size() { return sizeof...(Ns); }
};

template <typename Sequence, typename T, size_t N>
struct MakeIntegerSequenceImpl;

template <typename T, size_t... Ns>
struct MakeIntegerSequenceImpl<IntegerSequence<T, Ns...>, T, 0> {
    typedef IntegerSequence<T, Ns...> type;
};

template <typename T, size_t N, size_t... Ns>
struct MakeIntegerSequenceImpl<IntegerSequence<T, Ns...>, T, N> {
    typedef typename MakeIntegerSequenceImpl<IntegerSequence<T, N - 1, Ns...>, T, N - 1>::type type;
};

template <typename T, T N>
using MakeIntegerSequence = typename MakeIntegerSequenceImpl<IntegerSequence<T>, T, N>::type;


// index_sequence
template <size_t... Ints>
using index_sequence = IntegerSequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;

// int_sequence
template <int... Ints>
using int_sequence = IntegerSequence<int, Ints...>;

template <int N>
using make_int_sequence = MakeIntegerSequence<int, N>;

// Shortcuts
template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

} // end namespace tla

#endif  // TLA_NUMERIC_INTEGER_SEQUENCE_HPP