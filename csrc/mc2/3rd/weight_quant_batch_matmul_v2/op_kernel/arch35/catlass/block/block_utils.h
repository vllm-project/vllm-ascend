/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_BLOCK_BLOCK_UTILS_H
#define ARCH35_CATLASS_BLOCK_BLOCK_UTILS_H

#include "../utils/device_utils.h"
namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <size_t I, class Tuple>
struct deduce_optional_input {
private:
    static constexpr size_t validIndex =
        (I < AscendC::Std::tuple_size_v<Tuple> - 1) ? I : AscendC::Std::tuple_size_v<Tuple> - 1;

public:
    using type = AscendC::Std::conditional_t<(I < AscendC::Std::tuple_size_v<Tuple>),
                                             typename AscendC::Std::tuple_element<validIndex, Tuple>::type, void>;
};

namespace detail {

template <typename T>
DEVICE constexpr auto GetArgValue(T const &arg) -> decltype(arg)
{
    return arg;
}

template <typename T, T Value>
DEVICE constexpr T GetArgValue(AscendC::Std::integral_constant<T, Value>)
{
    return Value;
}

template <typename TupleT, typename ArgsTuple, size_t... I>
DEVICE auto Crd2idxHelper(TupleT const &t, ArgsTuple const &argsTuple, AscendC::Std::index_sequence<I...>)
{
    return (... +
            (GetArgValue<typename AscendC::Std::tuple_element<I, TupleT>::type>(AscendC::Std::get<I>(t)) *
             GetArgValue<typename AscendC::Std::tuple_element<I, ArgsTuple>::type>(AscendC::Std::get<I>(argsTuple))));
}
} // namespace detail
template <size_t I, class Tuple>
using deduce_optional_input_t = typename deduce_optional_input<I, Tuple>::type;

template <typename... TupleTypes, typename... Args>
DEVICE auto crd2idx(AscendC::Std::tuple<TupleTypes...> const &t, Args... args)
{
    static_assert(sizeof...(Args) == sizeof...(TupleTypes), "The number of arguments must match the tuple size");

    auto argsTuple = AscendC::Std::make_tuple(args...);
    return detail::Crd2idxHelper(t, argsTuple, AscendC::Std::make_index_sequence<sizeof...(TupleTypes)>{});
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
