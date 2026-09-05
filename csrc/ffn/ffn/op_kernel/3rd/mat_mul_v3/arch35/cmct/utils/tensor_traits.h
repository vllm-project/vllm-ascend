/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "integral_constant.h"

namespace Cmct::Gemm {
template <typename TensorTrait>
constexpr bool in_gm = TensorTrait::tPos == AscendC::TPosition::GM;

template <typename TensorTrait>
constexpr bool in_l1 = TensorTrait::tPos == AscendC::TPosition::TSCM || TensorTrait::tPos == AscendC::TPosition::A1 ||
                       TensorTrait::tPos == AscendC::TPosition::B1 || TensorTrait::tPos == AscendC::TPosition::C1;

template <typename TensorTrait>
constexpr bool in_l0a = TensorTrait::tPos == AscendC::TPosition::A2;

template <typename TensorTrait>
constexpr bool in_l0b = TensorTrait::tPos == AscendC::TPosition::B2;

template <typename TensorTrait>
constexpr bool in_bt = TensorTrait::tPos == AscendC::TPosition::C2;

template <typename TensorTrait>
constexpr bool in_l0c = TensorTrait::tPos == AscendC::TPosition::CO1;

template <typename TensorTrait>
constexpr bool in_ub = TensorTrait::tPos == AscendC::TPosition::VECIN ||
                       TensorTrait::tPos == AscendC::TPosition::VECOUT ||
                       TensorTrait::tPos == AscendC::TPosition::VECCALC;

template <bool isNz, typename T, AscendC::TPosition TPos>
struct TensorTraitL1 {};

template <typename T, AscendC::TPosition TPos>
struct TensorTraitL1<true, T, TPos> {
    // nz kn n1,k1,k0,n0
    using Type = AscendC::TensorTrait<
        T, TPos,
        AscendC::Layout<
            AscendC::Shape<AscendC::Shape<Cmct::Gemm::_16, uint64_t>, AscendC::Shape<Cmct::Gemm::_16, uint64_t>>,
            AscendC::Stride<AscendC::Stride<Cmct::Gemm::_1, uint64_t>,
                            AscendC::Stride<Cmct::Gemm::_16, Cmct::Gemm::_256>>>>;
};

template <typename T, AscendC::TPosition TPos>
struct TensorTraitL1<false, T, TPos> {
    // zn nk k1,n1,n0,k0
    using Type = AscendC::TensorTrait<
        T, TPos,
        AscendC::Layout<
            AscendC::Shape<AscendC::Shape<Cmct::Gemm::_16, uint64_t>, AscendC::Shape<Cmct::Gemm::_16, uint64_t>>,
            AscendC::Stride<AscendC::Stride<Cmct::Gemm::_16, Cmct::Gemm::_256>,
                            AscendC::Stride<Cmct::Gemm::_1, uint64_t>>>>;
};
} // namespace Cmct::Gemm
