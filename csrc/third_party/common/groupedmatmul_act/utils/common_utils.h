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
 * \file common_utils.h
 * \brief
 */

#ifndef UTILS_COMMON_UTILS_H
#define UTILS_COMMON_UTILS_H

#include "integral_constant.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
namespace Act {
namespace Gemm {
constexpr int64_t MATRIX_INNER_DIM_LIMIT_SIZE = 65536LL;
constexpr int32_t MATMUL_MNK_ALIGN = 16;
constexpr int32_t MATMUL_MNK_ALIGN_INT8 = 32;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr int64_t UB_FLOAT_ALIGN_NUM = 8LL;
constexpr int64_t L1_EVENT_ID_OFFSET = 2LL;
constexpr uint32_t UB_ALIGN_SIZE = 32U;
constexpr uint32_t UB_SUB_BANK_LEN = 256U; // SUB0: 256, SUB1: 256B
constexpr uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
constexpr uint32_t UB_SUB_BANK_ELEMS_B32 = 64U; // SUB0: 64, SUB1: 64
constexpr uint32_t UB_SUB_BANK_NUM = 2U;
constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr int MNK_B = 3;
constexpr int MNK_M0 = 4;
constexpr int MNK_N0 = 5;
constexpr static uint64_t A_FULL_LOAD_MODE = 1UL;
constexpr static uint64_t B_FULL_LOAD_MODE = 2UL;
constexpr static int64_t PER_BLOCK_SIZE = 128LL;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;

struct MatmulShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
};

__host_aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__host_aicore__ inline int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

/**
 * Get the aiv corenum from aic in different platforms
 */
__aicore__ inline uint32_t GetAicAivTaskRation()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3101)
    return 2U; // 2: aic:aiv = 1:2
#else
    return 1U;
#endif
}

template <typename CType, typename AType>
__aicore__ inline constexpr static bool IsQuantSenario()
{
    using L0cT = typename AscendC::GetMmDstType<AType>::Type;
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    if constexpr (!AscendC::IsTypeOneOfV<AType, int8_t, hifloat8_t, fp8_e4m3fn_t, fp8_e5m2_t, fp4x2_e2m1_t, fp4x2_e1m2_t> &&
                  AscendC::IsTypeOneOfV<CType, half, bfloat16_t>) {
        return false;
    }
    if constexpr (AscendC::IsTypeOneOfV<AType, hifloat8_t, fp8_e4m3fn_t, fp8_e5m2_t, fp4x2_e2m1_t, fp4x2_e1m2_t> &&
                  AscendC::IsTypeOneOfV<CType, hifloat8_t, fp8_e4m3fn_t, fp8_e5m2_t, half, bfloat16_t, float, fp4x2_e2m1_t, fp4x2_e1m2_t>) {
        return true;
    }
    if constexpr (AscendC::IsSameTypeV<L0cT, int32_t> && AscendC::IsSameTypeV<CType, bfloat16_t>) {
        return true;
    }
#endif
    if constexpr (AscendC::IsSameTypeV<L0cT, int32_t> && AscendC::IsTypeOneOfV<CType, half, int8_t, uint8_t>) {
        return true;
    } else if constexpr (AscendC::IsSameTypeV<L0cT, float> && AscendC::IsTypeOneOfV<CType, int8_t, uint8_t>) {
        return true;
    }
    return false;
}

template <class T>
struct is_static : AscendC::Std::bool_constant<std::is_empty<T>::value> {};

template <class T>
constexpr bool is_static_v = is_static<AscendC::Std::remove_cvref_t<T>>::value;

template <typename Stride>
struct is_2d_nz_c0_32_impl : AscendC::Std::false_type {};

template <typename T0, typename T1, typename U0, typename U1>
struct is_2d_nz_c0_32_impl<AscendC::Std::tuple<AscendC::Std::tuple<T0, T1>, AscendC::Std::tuple<U0, U1>>>
    : AscendC::Std::bool_constant<AscendC::Std::is_same_v<T0, Act::Gemm::_32> &&
                                  AscendC::Std::is_same_v<T1, Act::Gemm::_512> &&
                                  AscendC::Std::is_same_v<U0, Act::Gemm::_1> && !is_static_v<U1>> {};

template <class Stride>
struct is_2d_nz_c0_32 : is_2d_nz_c0_32_impl<typename AscendC::Std::remove_cvref_t<Stride>> {};
} // namespace Gemm
} // namespace Act
#endif