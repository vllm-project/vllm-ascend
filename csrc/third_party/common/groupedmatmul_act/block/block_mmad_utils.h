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
 * \file block_mmad_utils.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_UTILS_H
#define MATMUL_BLOCK_BLOCK_MMAD_UTILS_H

#include <type_traits>
#include "../utils/common_utils.h"
#include "../utils/integral_constant.h"
#include "../utils/tuple_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
 * @brief Check if all matrix types are F16 type
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if all matrix types are F16, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF16F16F16()
{
    return AscendC::IsSameTypeV<typename AType::T, half> && AscendC::IsSameTypeV<typename BType::T, half> &&
           AscendC::IsSameTypeV<typename CType::T, half>;
}

/**
 * @brief Check if matrix A and B are F16 and matrix C is F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are F16 and matrix C is F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF16F16F32()
{
    return AscendC::IsSameTypeV<typename AType::T, half> && AscendC::IsSameTypeV<typename BType::T, half> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

/**
 * @brief Check if all matrix types are Bf16
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if all matrix types are Bf16, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsBf16Bf16Bf16()
{
    return AscendC::IsSameTypeV<typename AType::T, bfloat16_t> && AscendC::IsSameTypeV<typename BType::T, bfloat16_t> &&
           AscendC::IsSameTypeV<typename CType::T, bfloat16_t>;
}

/**
 * @brief Check if matrix A and B are Bf16 and matrix C is F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are Bf16 and matrix C is F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsBf16Bf16F32()
{
    return AscendC::IsSameTypeV<typename AType::T, bfloat16_t> && AscendC::IsSameTypeV<typename BType::T, bfloat16_t> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

/**
 * @brief Check if matrix A and B are F16/Bf16 and matrix C is F16/Bf16/F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are F16/Bf16 and matrix C is F16/Bf16/F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF16OrBf16AB()
{
    return IsF16F16F16<AType, BType, CType>() || IsF16F16F32<AType, BType, CType>() ||
           IsBf16Bf16Bf16<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>();
}

/**
 * @brief Check if all matrix types are F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if all matrix types are F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF32F32F32()
{
    return AscendC::IsSameTypeV<typename AType::T, float> && AscendC::IsSameTypeV<typename BType::T, float> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

/**
 * @brief Check if matrix A and B are I8 and matrix C is I32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are I8 and matrix C is I32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsI8I8I32()
{
    return AscendC::IsSameTypeV<typename AType::T, int8_t> && AscendC::IsSameTypeV<typename BType::T, int8_t> &&
           AscendC::IsSameTypeV<typename CType::T, int32_t>;
}

/**
 * @brief Check if the matrix type is F4
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is F4, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsF4()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return AscendC::IsSameTypeV<typename MatmulType::T, fp4x2_e2m1_t> ||
           AscendC::IsSameTypeV<typename MatmulType::T, fp4x2_e1m2_t>;
#else
    return false;
#endif
}

/**
 * @brief Check if matrix A and B are Fp4 and matrix C is F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are Fp4 and matrix C is F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsFp4Fp4F32()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return IsF4<AType>() && IsF4<BType>() && AscendC::IsSameTypeV<typename CType::T, float>;
#else
    return false;
#endif
}

/**
 * @brief Check if the matrix type is F8
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is F8, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsF8()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return AscendC::IsSameTypeV<typename MatmulType::T, fp8_e5m2_t> ||
           AscendC::IsSameTypeV<typename MatmulType::T, fp8_e4m3fn_t>;
#else
    return false;
#endif
}

/**
 * @brief Check if matrix A and B are Fp8 and matrix C is F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are Fp8 and matrix C is F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsFp8Fp8F32()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return IsF8<AType>() && IsF8<BType>() && AscendC::IsSameTypeV<typename CType::T, float>;
#else
    return false;
#endif
}

/**
 * @brief Check if matrix A and B are HIF8 and matrix C is F32
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] CType: type of matrix C
 * @return Return true if matrix A and B are HIF8 and matrix C is F32, otherwise false
 */
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsHIF8HIF8F32()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return AscendC::IsSameTypeV<typename AType::T, hifloat8_t> && AscendC::IsSameTypeV<typename BType::T, hifloat8_t> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
#else
    return false;
#endif
}

/**
 * @brief Check if the given matrix type is ND format
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is ND format, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsND()
{
    return (MatmulType::format == CubeFormat::ND || MatmulType::format == CubeFormat::ND_ALIGN);
}

/**
 * @brief Check if the given matrix type is NZ format
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is NZ format, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsNz()
{
    return MatmulType::format == CubeFormat::NZ;
}

/**
 * @brief Get the Kb value of L1TileShape
 * @param [in] L1TileShape: L1TileShape type
 * @return Return the Kb value
 */
template <class L1TileShape>
__aicore__ inline constexpr auto GetL1Kb()
{
    static_assert(AscendC::Std::tuple_size_v<L1TileShape> >= 3, "L1TileShape must have at least 3 elements"); // 3: mnk
    if constexpr (AscendC::Std::tuple_size_v<L1TileShape> > 3) { // 3: MNKaKb Kb index
        return GetIntegralConstant<3, L1TileShape>();            // 3: MNKaKb Kb index
    } else {
        return GetIntegralConstant<MNK_K, L1TileShape>();
    }
}

/**
 * @brief Check if L1TileShape and L0TileShape are valid
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] L1TileShape: L1TileShape type
 * @param [in] L0TileShape: L0TileShape type
 * @param [in] l1BufferNum: l1 buffer count, default is DOUBLE_BUFFER_COUNT
 * @return Return true if L1TileShape and L0TileShape are valid, otherwise false
 */
template <class AType, class BType, class L1TileShape, class L0TileShape, int l1BufferNum = DOUBLE_BUFFER_COUNT>
__aicore__ inline constexpr bool IsTileShapeValid()
{
    constexpr auto l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    constexpr auto l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    constexpr auto l1Ka = GetIntegralConstant<MNK_K, L1TileShape>();
    constexpr auto l1Kb = GetL1Kb<L1TileShape>();

    constexpr auto l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    constexpr auto l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    constexpr auto l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    // Check L1 buffer L0 buffer
    if constexpr ((l1M * l1Ka * sizeof(typename AType::T) + l1N * l1Kb * sizeof(typename BType::T)) * l1BufferNum >
                  L1_SIZE) {
        return false;
    }
    if constexpr (l0M * l0K * sizeof(typename AType::T) > L0A_SIZE ||
                  l0N * l0K * sizeof(typename BType::T) > L0B_SIZE ||
                  l0M * l0N * sizeof(typename AscendC::GetMmDstType<typename AType::T>::Type) > L0C_SIZE) {
        return false;
    }
    // Check align
    if constexpr (!(l1M % MATMUL_MNK_ALIGN == 0 && l1N % MATMUL_MNK_ALIGN == 0 && l1Ka % MATMUL_MNK_ALIGN == 0 &&
                    l1Kb % MATMUL_MNK_ALIGN == 0) ||
                  !(l0M % MATMUL_MNK_ALIGN == 0 && l0N % MATMUL_MNK_ALIGN == 0 && l0K % MATMUL_MNK_ALIGN == 0)) {
        return false;
    }
    // Check L1 L0 shape
    return l1M == l0M && l1N == l0N && (l1Ka >= l0K && (l0K == 0 || l1Ka % l0K == 0)) &&
           (l1Kb >= l0K && (l0K == 0 || l1Kb % l0K == 0));
}

/**
 * @brief Check if L1 buffer is valid
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] L1TileShape: L1TileShape type
 * @param [in] bufferNum: buffer count, default is DOUBLE_BUFFER_COUNT
 * @return Return true if L1 buffer is valid, otherwise false
 */
template <class AType, class BType, class L1TileShape>
__aicore__ inline constexpr bool IsL1BufferValid(const int bufferNum = DOUBLE_BUFFER_COUNT)
{
    constexpr auto l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    constexpr auto l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    constexpr auto l1Ka = GetIntegralConstant<MNK_K, L1TileShape>();
    constexpr auto l1Kb = GetL1Kb<L1TileShape>();

    return (l1M * l1Ka * sizeof(typename AType::T) + l1N * l1Kb * sizeof(typename BType::T)) * bufferNum <= L1_SIZE;
}

/**
 * @brief Check if L0 buffer is valid
 * @param [in] AType: type of matrix A
 * @param [in] BType: type of matrix B
 * @param [in] L0TileShape: L0TileShape type
 * @param [in] bufferNum: buffer count, default is 1
 * @return Return true if L0 buffer is valid, otherwise false
 */
template <class AType, class BType, class L0TileShape>
__aicore__ inline constexpr bool IsL0BufferValid(const int bufferNum = 1) // L0 DB is optional
{
    constexpr auto l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    constexpr auto l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    constexpr auto l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    return l0M * l0K * sizeof(typename AType::T) * bufferNum <= L0A_SIZE &&
           l0N * l0K * sizeof(typename BType::T) * bufferNum <= L0B_SIZE &&
           l0M * l0N * sizeof(typename AscendC::GetMmDstType<typename AType::T>::Type) <= L0C_SIZE;
}

/**
 * @brief Get matrix shape parameters
 * @param [in] SingleShape: single shape
 * @param [in] L0TileShape: L0TileShape type
 * @return Return matrix shape parameters
 */
template <class SingleShape, class L0TileShape>
__aicore__ inline constexpr MatmulShapeParams GetMatmulShapeParams()
{
    return {GetIntegralConstant<MNK_M, SingleShape>(), GetIntegralConstant<MNK_N, SingleShape>(),
            GetIntegralConstant<MNK_K, SingleShape>(), GetIntegralConstant<MNK_M, L0TileShape>(),
            GetIntegralConstant<MNK_N, L0TileShape>(), GetIntegralConstant<MNK_K, L0TileShape>()};
}

/**
 * @brief Get function parameters
 * @param [in] intrinsicsCheck: whether to perform intrinsic checks
 * @return Return function parameters
 */
__aicore__ inline constexpr MatmulFuncParams GetFuncParams(bool intrinsicsCheck)
{
    MatmulFuncParams params{};
    params.intrinsicsCheck = intrinsicsCheck;
    return params;
}

/**
 * @brief Get bias parameters
 * @param [in] enableSetBias: whether to enable bias setting
 * @return Return bias parameters
 */
__aicore__ inline constexpr MatmulBiasParams GetBiasParams(bool enableSetBias)
{
    MatmulBiasParams params{};
    params.enableSetBias = enableSetBias;
    return params;
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
