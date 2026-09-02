/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "common_utils.h"
#include "integral_constant.h"
#include "tensor_traits.h"

namespace Cmct::Gemm {
enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

namespace detail {
// per-channel
//   n,k
//     layout (n,_1):(_1,_1) 由于无法知道stride是多大，所以不能用n,k表达
//   k,n
//     layout (n,_1):(_1,n)
// per-tensor
//   layout (_1,):(_1,)
// per-group shape (n,gn)
//   n,k
//     layout (n,!=_1):(gn,_1)
//   k,n
//     layout (n,!=_1):(_1, n)
template <typename T, typename U = void>
struct QuantTypeFrom;

template <typename U>
struct QuantTypeFrom<AscendC::Std::tuple<Cmct::Gemm::_1>, U> {
    static constexpr QuantType VALUE = QuantType::PER_TENSOR;
};

template <typename U>
struct QuantTypeFrom<AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>, U> {
    static constexpr QuantType VALUE = QuantType::PER_CHANNEL;
};

template <typename T0, typename T1, typename U>
struct QuantTypeFrom<AscendC::Std::tuple<T0, T1>, U>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, Cmct::Gemm::_1>> {
    static constexpr QuantType VALUE = QuantType::PER_GROUP;
};

template <typename T0, typename T1>
struct QuantTypeFrom<AscendC::Std::tuple<T0, T1>, AscendC::fp8_e8m0_t>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, Cmct::Gemm::_1>> {
    static constexpr QuantType VALUE = QuantType::MX;
};
} // namespace detail

template <typename Shape, typename ScaleType = void>
constexpr QuantType QUANT_TYPE = detail::QuantTypeFrom<typename AscendC::Std::remove_cvref_t<Shape>,
                                                       AscendC::Std::remove_cvref_t<ScaleType>>::VALUE;

constexpr uint32_t BYTE_PER_BLK = 32;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t FLAG_ID_MAX = 16;

namespace detail {
// (1, n) layout (n,_1):(_1,n)
// (n, 1) layout (n,_1):(_1,_1)
template <typename Layout>
struct Is2D1Dim1Impl : AscendC::Std::false_type {};

template <typename S0, typename S1, typename T0, typename T1>
struct Is2D1Dim1Impl<AscendC::Layout<AscendC::Std::tuple<S0, S1>, AscendC::Std::tuple<T0, T1>>>
    : AscendC::Std::bool_constant<AscendC::Std::is_same_v<S1, Cmct::Gemm::_1> &&
                                  AscendC::Std::is_same_v<T0, Cmct::Gemm::_1>> {};

template <typename Layout>
struct IsRowMajor2DImpl : AscendC::Std::false_type {};

template <typename Shape, typename T0>
struct IsRowMajor2DImpl<AscendC::Layout<Shape, AscendC::Std::tuple<T0, Cmct::Gemm::_1>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T0, Cmct::Gemm::_1>> {};

template <typename Layout>
struct IsColumnMajor2DImpl : AscendC::Std::false_type {};

// shape(t0, t1) stride(t0, t1)
template <typename T0, typename T1, typename U1>
struct IsColumnMajor2DImpl<AscendC::Layout<AscendC::Std::tuple<T0, T1>, AscendC::Std::tuple<Cmct::Gemm::_1, U1>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, Cmct::Gemm::_1> && !AscendC::Std::is_tuple_v<T1>> {};

template <typename Layout>
struct IsNz2DImpl : AscendC::Std::false_type {};
template <typename Shape, typename T>
struct IsNz2DImpl<AscendC::Layout<Shape, AscendC::Std::tuple<AscendC::Std::tuple<Cmct::Gemm::_16, Cmct::Gemm::_256>,
                                                             AscendC::Std::tuple<Cmct::Gemm::_1, T>>>>
    : AscendC::Std::true_type {};

// (a1,b1,b0,a0)
// layout ((a0,a1),(b0,b1)):((_1,u32),(_16,_256))
template <typename Layout>
struct IsZn2DImpl : AscendC::Std::false_type {};
template <typename Shape, typename T1>
struct IsZn2DImpl<AscendC::Layout<Shape, AscendC::Std::tuple<AscendC::Std::tuple<Cmct::Gemm::_1, T1>,
                                                             AscendC::Std::tuple<Cmct::Gemm::_16, Cmct::Gemm::_256>>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, Cmct::Gemm::_1>> {};

template <typename Layout>
struct IsRowMajor2DContiguousImpl : AscendC::Std::false_type {};
template <typename S0, typename S1>
struct IsRowMajor2DContiguousImpl<AscendC::Layout<AscendC::Std::tuple<S0, S1>, AscendC::Std::tuple<S1, Cmct::Gemm::_1>>>
    : AscendC::Std::bool_constant<is_static_v<S1>> {};
} // namespace detail

// Is2D1Dim1
//   (,_1):(_1,)
// IsRowMajor2D
//   :(!=_1,_1)
// IsColumnMajor2D
//   (,!=_1):(_1,)
// IsNz2D
//   :((_16,_256),(_1,))
// IsZn2D
//   :((_1,!=_1),(_16,_256))
// IsRowMajor2DContiguous
//   (,_x):(_x,_1)
template <typename Layout>
struct Is2D1Dim1 : detail::Is2D1Dim1Impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsColumnMajor2D : detail::IsColumnMajor2DImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsRowMajor2D : detail::IsRowMajor2DImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsNz2D : detail::IsNz2DImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsZn2D : detail::IsZn2DImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsRowMajor2DContiguous : detail::IsRowMajor2DContiguousImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

namespace detail {
template <typename T>
struct KbElemImpl;

template <>
struct KbElemImpl<AscendC::int4b_t> {
    static constexpr uint32_t VALUE = 2048;
};

template <>
struct KbElemImpl<int8_t> {
    static constexpr uint32_t VALUE = 1024;
};

template <>
struct KbElemImpl<float> {
    static constexpr uint32_t VALUE = 256;
};

template <>
struct KbElemImpl<half> {
    static constexpr uint32_t VALUE = 512;
};

template <>
struct KbElemImpl<bfloat16_t> {
    static constexpr uint32_t VALUE = 512;
};

template <>
struct KbElemImpl<AscendC::fp8_e8m0_t> {
    static constexpr uint32_t VALUE = 1024;
};
} // namespace detail

template <typename T>
constexpr uint32_t KB_ELEM = detail::KbElemImpl<typename AscendC::Std::remove_cvref_t<T>>::VALUE;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;

template <typename T>
constexpr uint64_t BLK_ELEM = 32 / sizeof(T);

template <typename T>
inline constexpr uint32_t C0 = 32 / sizeof(T);
} // namespace Cmct::Gemm
