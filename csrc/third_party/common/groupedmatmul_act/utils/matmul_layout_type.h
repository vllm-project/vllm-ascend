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
 * \file matmul_layout_type.h
 * \brief
 */

#ifndef UTILS_MATMUL_LAYOUT_TYPE_H
#define UTILS_MATMUL_LAYOUT_TYPE_H

#include "./integral_constant.h"
#include "./layout_utils.h"

namespace Act {
namespace Gemm {

template <typename LAYOUT, typename TYPE, AscendC::TPosition POSITION, bool ISTRANS = false>
struct MatmulLayoutType {
    using layout = LAYOUT;
    using T = TYPE;
    constexpr static AscendC::TPosition pos = POSITION;
    constexpr static bool isTrans = ISTRANS;
};

template <typename T, typename = void>
struct IsMatmulLayoutType : AscendC::Std::false_type {};

template <typename T>
struct IsMatmulLayoutType<T, std::void_t<typename T::layout, typename T::T, decltype(T::pos)>> : AscendC::Std::true_type {};

template <typename T>
constexpr bool IsMatmulLayoutTypeV = IsMatmulLayoutType<T>::value;

template <class LayoutT>
struct ToMatmulType {
    using Type = AscendC::MatmulType<LayoutT::pos, TagToFormat<typename LayoutT::layout>::format,
            typename LayoutT::T, TagToTrans<typename LayoutT::layout>::value>;
};

template <class LayoutT>
using ToMatmulTypeT = typename ToMatmulType<LayoutT>::Type;

} // namespace Gemm
} // namespace Act
#endif