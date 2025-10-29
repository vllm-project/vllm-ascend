/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TENSOR_HPP
#define TLA_TENSOR_HPP

#include "tla/layout.hpp"                     // tla::Shape
#include "tla/numeric/integral_constant.hpp"  // tla::is_integral

namespace tla {
//
// Tensor
//

template <class BuiltinTensor, class Layout_, AscendC::TPosition Position>
struct Tensor {
    using Element = typename BuiltinTensor::PrimType;
    using Layout = Layout_;
    static constexpr AscendC::TPosition position = Position;

    CATLASS_HOST_DEVICE constexpr
    Tensor() {}

    CATLASS_HOST_DEVICE constexpr
    Tensor(BuiltinTensor const& builtinTensor, Layout const& layout)
        : rep_(builtinTensor, layout) {}

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
    decltype(auto) shape() const
    {
        return layout().shape();
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride() const
    {
        return layout().stride();
    }

    tla::tuple<BuiltinTensor, Layout> rep_;
};

template <class BuiltinTensor, class Layout, AscendC::TPosition Position>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout)
{
    return Tensor<BuiltinTensor, Layout, Position>(builtinTensor, layout);
}

template <class BuiltinTensor, class Layout, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, PositionType)
{
    return Tensor<BuiltinTensor, Layout, PositionType::POSITION>(builtinTensor, layout);
}

template <class Tensor, class Coord, class Shape>
CATLASS_DEVICE constexpr
auto GetTile(Tensor const& tensor, Coord const& coord, Shape const& shape)
{
    auto layout = tensor.layout();
    auto offset = layout(coord);
    auto builtinTensor = tensor.data();
    auto layoutNew = MakeLayoutTile(layout, shape);
    return MakeTensor<decltype(builtinTensor), decltype(layoutNew),
                      Tensor::position>(builtinTensor[offset], layoutNew);
}

} // end namespace tla

#endif // TLA_TENSOR_HPP
