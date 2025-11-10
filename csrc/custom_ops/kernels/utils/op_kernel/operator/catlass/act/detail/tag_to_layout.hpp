/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef ACT_DETAIL_TAG_TO_LAYOUT_HPP
#define ACT_DETAIL_TAG_TO_LAYOUT_HPP

#include "../../act/layout/layout.hpp"
#include "../../tla/layout.hpp"

using namespace tla;
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Act::detail {
////////////////////////////////////////////////////////////////////////////////////////////////////
// For each Act::layout, provides its corresponding tla layout types
template <class Element, class LayoutTag>
struct TagToLayout {
    using type = LayoutTag;
};

template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = Layout<Shape<uint32_t, uint32_t>, Stride<int64_t, Int<1>>, Shape<uint32_t, uint32_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::ColumnMajor> {
    using type = Layout<Shape<uint32_t, uint32_t>, Stride<Int<1>, int64_t>, Shape<uint32_t, uint32_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = Layout<Shape<Shape<Int<C0_NUM_PER_FRACTAL>, uint32_t>, Shape<Int<ELE_NUM_PER_C0>, uint32_t>>,
                        Stride<Stride<Int<ELE_NUM_PER_C0>, Int<ELE_NUM_PER_FRACTAL>>, Stride<Int<1>, int64_t>>,
                        Shape<uint32_t, uint32_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = Layout<Shape<Shape<Int<C0_NUM_PER_FRACTAL>, uint32_t>, Shape<Int<ELE_NUM_PER_C0>, uint32_t>>,
                        Stride<Stride<Int<ELE_NUM_PER_C0>, int64_t>, Stride<Int<1>, Int<ELE_NUM_PER_FRACTAL>>>,
                        Shape<uint32_t, uint32_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    using type = Layout<Shape<Shape<Int<ELE_NUM_PER_C0>, uint32_t>, Shape<Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
                        Stride<Stride<Int<1>, int64_t>, Stride<Int<ELE_NUM_PER_C0>, Int<ELE_NUM_PER_FRACTAL>>>,
                        Shape<uint32_t, uint32_t>>;
};

// Convenience aliases
template <class Element, class LayoutTag>
using TagToLayout_t = typename TagToLayout<Element, LayoutTag>::type;

constexpr uint32_t ELE_NUM_PER_FRACTAL_L0C = 256;
using LayoutL0C = Layout<Shape<Shape<Int<C0_NUM_PER_FRACTAL>, uint32_t>, Shape<Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
                         Stride<Stride<Int<C0_NUM_PER_FRACTAL>, Int<ELE_NUM_PER_FRACTAL_L0C>>, Stride<Int<1>, int64_t>>,
                         Shape<uint32_t, uint32_t>>;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Act::detail

#endif  // ACT_DETAIL_TAG_TO_LAYOUT_HPP
