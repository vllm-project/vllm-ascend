/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_LAYOUT_HPP
#define TLA_LAYOUT_HPP

#include "catlass/catlass.hpp"
#include "tla/numeric/integral_constant.hpp"
#include "tla/tuple.hpp"
#include "tla/int_tuple.hpp"
#include "catlass/layout/layout.hpp"

namespace tla {

// Aliases

template <class... Shapes>
using Shape = tla::tuple<Shapes...>;

template <class... Strides>
using Stride = tla::tuple<Strides...>;

template <class... Coords>
using Coord = tla::tuple<Coords...>;

template <class... Ts>
CATLASS_HOST_DEVICE constexpr
Shape<Ts...> MakeShape(Ts const&... t) {
    return {t...};
}
template <class... Ts>
CATLASS_HOST_DEVICE constexpr
Stride<Ts...> MakeStride(Ts const&... t) {
    return {t...};
}
template <class... Ts>
CATLASS_HOST_DEVICE constexpr
Coord<Ts...> MakeCoord(Ts const&... t) {
    return {t...};
}

//
// Layout
//

template <class Shape, class Stride>
struct Layout : private tla::tuple<Shape, Stride> {
    // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
    CATLASS_HOST_DEVICE constexpr
    Layout(Shape  const& shape  = {}, Stride const& stride = {})
        : tla::tuple<Shape, Stride>(shape, stride) {}

    //
    // Accessors
    //

    static constexpr int rank  = rank_v<Stride>;
    static constexpr int depth  = depth_v<Stride>;

    template <int... I>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) shape()
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) shape() const
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride> const&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride()
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride() const
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride> const&>(*this));
    }

    template <class Coord>
    CATLASS_HOST_DEVICE constexpr
    auto operator()(Coord const& coord) const
    {
        return crd2offset(coord, shape(), stride());
    }
};

// Layout construction

template <class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
auto MakeLayout(Shape const& shape, Stride const& stride)
{
    static_assert(is_tuple<Shape>::value || is_integral<Shape>::value);
    static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
    return Layout<Shape, Stride>(shape, stride);
}

// Convenience tags for common layouts

template <class LayoutTag>
CATLASS_HOST_DEVICE constexpr
auto MakeLayoutFromTag(LayoutTag const& tag)
{
    static_assert(std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                  std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                  std::is_same_v<LayoutTag, Catlass::layout::zN> ||
                  std::is_same_v<LayoutTag, Catlass::layout::nZ>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
        "Catlass::layout::ColumnMajor or Catlass::layout::zN or Catlass::layout::nZ");

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(tag.stride(0), Int<1>{}));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(Int<1>{}, tag.stride(1)));
    } else {  // zN or nZ
        return MakeLayout(MakeShape(MakeShape(tag.shape(0), tag.shape(1)), MakeShape(tag.shape(2), tag.shape(3))),
            MakeStride(MakeStride(tag.stride(0), tag.stride(1)), MakeStride(tag.stride(2), tag.stride(3))));
    }
}

// Return the shape of a mode
template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
decltype(auto) shape(Layout<Shape, Stride>& layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
decltype(auto) shape(Layout<Shape, Stride> const& layout)
{
    return layout.template shape<Is...>();
}

// Return the stride of a mode
template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
decltype(auto) stride(Layout<Shape, Stride>& layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
decltype(auto) stride(Layout<Shape, Stride> const& layout)
{
    return layout.template stride<Is...>();
}

// Return the rank of layout
template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
auto rank(Layout<Shape, Stride> const& layout)
{
    return rank(shape<Is...>(layout));
}

// Return the depth of the layout
template <int... Is, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
auto depth(Layout<Shape, Stride> const& layout)
{
    return depth(shape<Is...>(layout));
}

// Return the offset of coord
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
auto crd2offset(Coord  const& coord, Shape  const& shape, Stride const& stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
CATLASS_HOST_DEVICE constexpr
auto crd2offset_ttt(Coord  const& coord, Shape  const& shape, Stride const& stride, seq<Is...>)
{
    return (... + crd2offset(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
CATLASS_HOST_DEVICE constexpr
auto crd2offset_itt(CInt const& coord, STuple const& shape, DTuple const& stride, seq<I0, Is...>)
{
    if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
        return crd2offset(coord, get<I0>(shape), get<I0>(stride));
    } else if constexpr (is_constant<0, CInt>::value) {
        return crd2offset(_0{}, get<I0>(shape), get<I0>(stride)) +
               (_0{} + ... + crd2offset(_0{}, get<Is>(shape), get<Is>(stride)));
    } else {                             // General case
        return crd2offset(coord % Product{}(get<I0>(shape)), get<I0>(shape), get<I0>(stride)) +
               crd2offset_itt(coord / Product{}(get<I0>(shape)), shape, stride, seq<Is...>{});
    }
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr
auto crd2offset(Coord const& coord, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple<Coord>::value) {
        if constexpr (is_tuple<Shape>::value) {  // tuple tuple tuple
            static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
            static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_ttt(coord, shape, stride, tuple_seq<Coord>{});
        } else {  // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters");
        }
    } else {
        if constexpr (is_tuple<Shape>::value) {  // "int" tuple tuple
            static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_itt(coord, shape, stride, tuple_seq<Shape>{});
        } else {  // "int" "int" "int"
            return coord * stride;
        }
    }
}

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride>
struct is_layout<Layout<Shape, Stride>> : true_type {};

// Layout Check
namespace detail {

template <class Layout, class Enable = void>
struct isRowMajor {
    static bool const value = false;
};

template <class Layout>
struct isRowMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<1>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isColumnMajor {
    static bool const value = false;
};

template <class Layout>
struct isColumnMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable = void>
struct iszN {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszN<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);
    static bool const value = (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
                               shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
                               stride<1, 0>(Layout{}) == 1 &&
                               stride<0, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable = void>
struct iszZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);
    static bool const value = (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
                               shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
                               stride<1, 0>(Layout{}) == 1 &&
                               stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable = void>
struct isnZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct isnZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);
    static bool const value = (shape<0, 0>(Layout{}) == ELE_NUM_PER_C0 &&
                               shape<1, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
                               stride<0, 0>(Layout{}) == 1 &&
                               stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

} // end namespace detail

// Advanced Layout constructions
// Make a inner layout with Rows and Cols.
template <class Element, class LayoutTag, class T, class U>
CATLASS_HOST_DEVICE constexpr
auto MakeLayout(T const& rows, U const& cols)
{
    static_assert(std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                  std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                  std::is_same_v<LayoutTag, Catlass::layout::zN> ||
                  std::is_same_v<LayoutTag, Catlass::layout::nZ> ||
                  std::is_same_v<LayoutTag, Catlass::layout::zZ>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
        "Catlass::layout::ColumnMajor or Catlass::layout::zN or Catlass::layout::nZ or Catlass::layout::zZ");

    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BYTE_PER_FRACTAL / sizeof(Element);

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zN>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                           RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})));
    }
}

template <class Layout, class ShapeNew>
CATLASS_HOST_DEVICE constexpr
auto MakeLayoutTile(Layout const& layout, ShapeNew const& shapeNew)
{
    static_assert(is_tuple<ShapeNew>::value && depth_v<ShapeNew> == 1 && rank_v<ShapeNew> == 2);

    if constexpr (Layout::depth == 1 && Layout::rank == 2) {
        return MakeLayout(shapeNew, layout.stride());
    } else if constexpr (is_static<decltype(shape<0, 0>(layout))>::value &&
                         is_static<decltype(shape<1, 0>(layout))>::value) {
        const uint32_t rows = get<0>(shapeNew);
        const uint32_t cols = get<1>(shapeNew);
        constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
        constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
        return MakeLayout(
            MakeShape(MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                      MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
            layout.stride());
    } else {
        const uint32_t rows = get<0>(shapeNew);
        const uint32_t cols = get<1>(shapeNew);
        const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
        const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
        return MakeLayout(
            MakeShape(MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                      MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
            layout.stride());
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr
auto MakeLayoutL0C(T const& rows, U const& cols)
{
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
            MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
            MakeStride(
                Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * Catlass::C0_NUM_PER_FRACTAL)));
}

} // end namespace tla

# endif // TLA_LAYOUT_HPP