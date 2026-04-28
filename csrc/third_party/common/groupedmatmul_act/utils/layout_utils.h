/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layout_utils.h
 * \brief
 */

#ifndef UTILS_LAYOUT_UTILS_H
#define UTILS_LAYOUT_UTILS_H

#include "./integral_constant.h"

namespace Act {
namespace Gemm {
namespace layout {
struct RowMajor {};
struct RowMajorAlign {}; // ND_ALIGN align to 32
struct ColumnMajor {};
struct ColumnMajorAlign {};
struct Nz {};
struct Zn {};
} // namespace layout

constexpr int32_t C0_BYTE_SIZE = 32;
constexpr int32_t C0_SIZE_FP16 = C0_BYTE_SIZE / sizeof(half);       // 16
constexpr int32_t C0_SIZE_BF16 = C0_BYTE_SIZE / sizeof(bfloat16_t); // 16
constexpr int32_t C0_SIZE_FP32 = C0_BYTE_SIZE / sizeof(float);      // 8
constexpr int32_t C0_NUM_PER_FRACTAL = 16;
constexpr int32_t C0_SIZE_L0C = 16;

// TagToFormat
template <typename T>
struct TagToFormat {
    static_assert(AscendC::Std::always_false_v<T>, "TagToFormat is not implemented for this layout");
};

template <>
struct TagToFormat<layout::RowMajor> {
    using tag = layout::RowMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <>
struct TagToFormat<layout::RowMajorAlign> {
    using tag = layout::RowMajorAlign;
    static constexpr CubeFormat format = CubeFormat::ND_ALIGN;
};

template <>
struct TagToFormat<layout::ColumnMajor> {
    using tag = layout::ColumnMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <>
struct TagToFormat<layout::ColumnMajorAlign> {
    using tag = layout::ColumnMajorAlign;
    static constexpr CubeFormat format = CubeFormat::ND_ALIGN;
};

template <>
struct TagToFormat<layout::Zn> {
    using tag = layout::Zn;
    static constexpr CubeFormat format = CubeFormat::NZ;
};

template <>
struct TagToFormat<layout::Nz> {
    using tag = layout::Nz;
    static constexpr CubeFormat format = CubeFormat::NZ;
};

// TagToTrans
template <typename T>
struct TagToTrans {
    static_assert(AscendC::Std::always_false_v<T>, "TagToTrans is not implemented for this layout");
};

template <>
struct TagToTrans<layout::RowMajor> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::RowMajorAlign> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::ColumnMajor> {
    static constexpr bool value = true;
};

template <>
struct TagToTrans<layout::ColumnMajorAlign> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::Zn> {
    static constexpr bool value = true;
};

template <>
struct TagToTrans<layout::Nz> {
    static constexpr bool value = false;
};

// LayoutToFormat
template <typename T, typename Layout>
struct LayoutToFormat {
    static_assert(AscendC::Std::always_false_v<Layout>, "LayoutToFormat is not implemented for this layout");
};

template <typename T, typename IntType>
struct LayoutToFormat<T, AscendC::Layout<AscendC::Shape<IntType, IntType>, AscendC::Stride<IntType, _1>>> {
    static_assert(AscendC::Std::is_integral_v<IntType>, "Layout element type must be integral.");
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <typename IntType>
struct LayoutToFormat<
    half, AscendC::Layout<AscendC::Shape<AscendC::Shape<Int<C0_NUM_PER_FRACTAL>, IntType>,
                                         AscendC::Shape<Int<C0_SIZE_FP16>, IntType>>,
                          AscendC::Stride<AscendC::Stride<Int<C0_SIZE_FP16>, Int<C0_NUM_PER_FRACTAL * C0_SIZE_FP16>>,
                                          AscendC::Stride<_1, IntType>>>> {
    static_assert(AscendC::Std::is_integral_v<IntType>, "Layout element type must be integral.");
    static constexpr CubeFormat format = CubeFormat::NZ;
};

template <typename IntType>
struct LayoutToFormat<
    bfloat16_t,
    AscendC::Layout<
        AscendC::Shape<AscendC::Shape<Int<C0_NUM_PER_FRACTAL>, IntType>, AscendC::Shape<Int<C0_SIZE_BF16>, IntType>>,
        AscendC::Stride<AscendC::Stride<Int<C0_SIZE_BF16>, Int<C0_NUM_PER_FRACTAL * C0_SIZE_BF16>>,
                        AscendC::Stride<_1, IntType>>>> {
    static_assert(AscendC::Std::is_integral_v<IntType>, "Layout element type must be integral.");
    static constexpr CubeFormat format = CubeFormat::NZ;
};

template <typename IntType>
struct LayoutToFormat<
    float, AscendC::Layout<AscendC::Shape<AscendC::Shape<Int<C0_NUM_PER_FRACTAL>, IntType>,
                                          AscendC::Shape<Int<C0_SIZE_FP32>, IntType>>,
                           AscendC::Stride<AscendC::Stride<Int<C0_SIZE_FP32>, Int<C0_NUM_PER_FRACTAL * C0_SIZE_FP32>>,
                                           AscendC::Stride<_1, IntType>>>> {
    static_assert(AscendC::Std::is_integral_v<IntType>, "Layout element type must be integral.");
    static constexpr CubeFormat format = CubeFormat::NZ;
};

template <typename T, typename Layout>
inline constexpr CubeFormat LayoutToFormatV = LayoutToFormat<T, Layout>::format;

// FormatToLayout
template <typename T, CubeFormat format>
struct FormatToLayout {
    using type = CubeFormat;
};

template <typename T>
struct FormatToLayout<T, CubeFormat::ND> {
    using type = AscendC::Layout<AscendC::Shape<int, int>, AscendC::Stride<int, _1>>;
};

template <typename T>
struct FormatToLayout<T, CubeFormat::ND_ALIGN> {
    using type = AscendC::Layout<AscendC::Shape<int, int>, AscendC::Stride<int, _1>>;
};

template <typename T>
struct FormatToLayout<T, CubeFormat::NZ> {
    static constexpr int C0_SIZE = C0_BYTE_SIZE / sizeof(T);
    using type = AscendC::Layout<
        AscendC::Shape<AscendC::Shape<Int<C0_NUM_PER_FRACTAL>, int>, // Single fractal row dimension: 16 elements
                       AscendC::Shape<Int<C0_SIZE>, int>             // Column-wise size of a single fractal: C0_SIZE
                       >,
        AscendC::Stride<AscendC::Stride<Int<C0_SIZE>, Int<C0_NUM_PER_FRACTAL * C0_SIZE>>, AscendC::Stride<_1, int>>>;
};

template <typename T>
struct FormatToLayout<T, CubeFormat::ZN> {
    static constexpr int C0_SIZE = C0_BYTE_SIZE / sizeof(T);
    using type = AscendC::Layout<
        AscendC::Shape<AscendC::Shape<Int<C0_SIZE>, int>, AscendC::Shape<_16, int>>,
        AscendC::Stride<AscendC::Stride<_1, int>, AscendC::Stride<Int<C0_SIZE>, Int<C0_NUM_PER_FRACTAL * C0_SIZE>>>>;
};

template <typename T>
struct FormatToLayout<T, CubeFormat::ZZ> {
    static constexpr int C0_SIZE = C0_BYTE_SIZE / sizeof(T);
    using type = AscendC::Layout<
        AscendC::Shape<AscendC::Shape<_16, int>, AscendC::Shape<Int<C0_SIZE>, int>>,
        AscendC::Stride<AscendC::Stride<Int<C0_SIZE>, int>, AscendC::Stride<_1, Int<C0_NUM_PER_FRACTAL * C0_SIZE>>>>;
};

template <typename T, CubeFormat format>
using FormatToLayoutT = typename FormatToLayout<T, format>::type;

template <typename T, CubeFormat format>
__aicore__ constexpr inline decltype(auto) MakeLayoutByFormat(int row, int col)
{
    static_assert(format == CubeFormat::ND || format == CubeFormat::ND_ALIGN || format == CubeFormat::NZ ||
                      format == CubeFormat::ZN || format == CubeFormat::ZZ,
                  "Unsupport format");

    constexpr int c0Size = C0_BYTE_SIZE / sizeof(T);
    if constexpr (format == CubeFormat::ND || format == CubeFormat::ND_ALIGN) {
        return AscendC::MakeLayout(AscendC::MakeShape(row, col), AscendC::MakeStride(col, _1{}));
    } else if constexpr (format == CubeFormat::NZ) {
        return AscendC::MakeLayout(
            AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil(row, C0_NUM_PER_FRACTAL)),
                               AscendC::MakeShape(Int<c0Size>{}, AscendC::Ceil(col, c0Size))),
            AscendC::MakeStride(AscendC::MakeStride(Int<c0Size>{}, Int<C0_NUM_PER_FRACTAL * c0Size>{}),
                                AscendC::MakeStride(_1{}, AscendC::CeilAlign(row, C0_NUM_PER_FRACTAL) * c0Size)));
    } else if constexpr (format == CubeFormat::ZN) {
        return AscendC::MakeLayout(
            AscendC::MakeShape(AscendC::MakeShape(Int<c0Size>{}, AscendC::Ceil(row, c0Size)),
                               AscendC::MakeShape(_16{}, AscendC::Ceil(col, C0_NUM_PER_FRACTAL))),
            AscendC::MakeStride(AscendC::MakeStride(_1{}, AscendC::CeilAlign(col, C0_NUM_PER_FRACTAL) * c0Size),
                                AscendC::MakeStride(Int<c0Size>{}, Int<C0_NUM_PER_FRACTAL * c0Size>{})));
    } else { // CubeFormat:ZZ
        return AscendC::MakeLayout(AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil(row, C0_NUM_PER_FRACTAL)),
                                                   AscendC::MakeShape(Int<c0Size>{}, AscendC::Ceil(col, c0Size))),
                                AscendC::MakeStride(AscendC::MakeStride(Int<c0Size>{}, AscendC::CeilAlign(col, c0Size) *
                                                                                           C0_NUM_PER_FRACTAL),
                                                    AscendC::MakeStride(_1{}, Int<C0_NUM_PER_FRACTAL * c0Size>{})));
    }
}
} // namespace Gemm
} // namespace Act
#endif