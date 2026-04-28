/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TENSOR_HPP
#define TLA_TENSOR_HPP

#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"                     // tla::Shape
#include "tla/numeric/integral_constant.hpp"  // tla::is_integral
#include "tla/int_tuple.hpp"

namespace tla {
//
// Tensor
//

template <class BuiltinTensor, class Layout_, class Coord_, AscendC::TPosition Position>
struct Tensor {
    using Element = typename BuiltinTensor::PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendC::TPosition position = Position;

    CATLASS_HOST_DEVICE constexpr
    Tensor() {}

    CATLASS_HOST_DEVICE constexpr
    Tensor(BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord = {})
        : rep_(builtinTensor, layout, coord) {}

    //
    // Accessors
    //

    static constexpr int rank  = Layout::rank;

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) tensor() const
    {
        return *this;
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data() const
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data()
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) layout() const
    {
        return get<1>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) coord() const
    {
        return get<2>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) shape() const
    {
        return layout().shape();
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride() const
    {
        return layout().stride();
    }

    tla::tuple<BuiltinTensor, Layout, Coord> rep_;
};

template <class BuiltinTensor, class Layout, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, PositionType)
{
    using Coord = detail::MakeZeroTuple<Layout::rank>;
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout);
}

template <class BuiltinTensor, class Layout, class Coord, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord, PositionType)
{
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout, coord);
}

template <class Tensor, class Coord, class Shape>
CATLASS_DEVICE constexpr
auto GetTile(Tensor const& tensor, Coord const& coord, Shape const& shape)
{
    auto layout = tensor.layout();
    auto builtinTensor = tensor.data();
    auto layoutNew = MakeLayoutTile(layout, shape);
    auto coordNew = Add(tensor.coord(), coord);
    return MakeTensor(builtinTensor, layoutNew, coordNew, Catlass::Arch::PositionType<Tensor::position>{});
}

} // end namespace tla

#endif // TLA_TENSOR_HPP
