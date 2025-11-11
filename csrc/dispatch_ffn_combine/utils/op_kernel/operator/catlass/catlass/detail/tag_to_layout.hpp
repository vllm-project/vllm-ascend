/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_TAG_TO_LAYOUT_HPP
#define CATLASS_DETAIL_TAG_TO_LAYOUT_HPP

#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Catlass::detail {
////////////////////////////////////////////////////////////////////////////////////////////////////
// For each Catlass::layout, provides its corresponding tla layout types
template <class Element, class LayoutTag>
struct TagToLayout {
    using type = LayoutTag;
};

template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::ColumnMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<tla::Int<1>, int64_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>,
            tla::Stride<tla::Int<1>, int64_t>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>,
            tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element>
struct TagToLayout<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<1>, int64_t>,
            tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

// Convenience aliases
template <class Element, class LayoutTag>
using TagToLayout_t = typename TagToLayout<Element, LayoutTag>::type;

constexpr uint32_t ELE_NUM_PER_FRACTAL_L0C = 256;
using LayoutL0C = tla::Layout<
    tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
    tla::Stride<tla::Stride<tla::Int<C0_NUM_PER_FRACTAL>, tla::Int<ELE_NUM_PER_FRACTAL_L0C>>,
        tla::Stride<tla::Int<1>, int64_t>>>;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::detail

#endif // CATLASS_DETAIL_TAG_TO_LAYOUT_HPP