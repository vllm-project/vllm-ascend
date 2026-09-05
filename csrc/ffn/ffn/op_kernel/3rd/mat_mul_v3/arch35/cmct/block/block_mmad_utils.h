/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#pragma once

#include "../../../inc/macro.h"
#include <type_traits>
#include "../utils/common_utils.h"
#include "../utils/integral_constant.h"
#include "../utils/tuple_utils.h"

namespace Cmct {
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
    return AscendC::IsSameType<typename AType::T, half>::value && AscendC::IsSameType<typename BType::T, half>::value &&
           AscendC::IsSameType<typename CType::T, half>::value;
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
    return AscendC::IsSameType<typename AType::T, half>::value && AscendC::IsSameType<typename BType::T, half>::value &&
           AscendC::IsSameType<typename CType::T, float>::value;
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
    return AscendC::IsSameType<typename AType::T, bfloat16_t>::value &&
           AscendC::IsSameType<typename BType::T, bfloat16_t>::value &&
           AscendC::IsSameType<typename CType::T, bfloat16_t>::value;
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
    return AscendC::IsSameType<typename AType::T, bfloat16_t>::value &&
           AscendC::IsSameType<typename BType::T, bfloat16_t>::value &&
           AscendC::IsSameType<typename CType::T, float>::value;
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
    return AscendC::IsSameType<typename AType::T, float>::value &&
           AscendC::IsSameType<typename BType::T, float>::value && AscendC::IsSameType<typename CType::T, float>::value;
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
    return AscendC::IsSameType<typename AType::T, int8_t>::value &&
           AscendC::IsSameType<typename BType::T, int8_t>::value &&
           AscendC::IsSameType<typename CType::T, int32_t>::value;
}

/**
 * @brief Check if the matrix type is F8
 * @param [in] MatmulType: matrix type
 * @return Return true if the matrix type is F8, otherwise false
 */
template <class MatmulType>
__aicore__ inline constexpr bool IsF8()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return AscendC::IsSameType<typename MatmulType::T, fp8_e5m2_t>::value ||
           AscendC::IsSameType<typename MatmulType::T, fp8_e4m3fn_t>::value;
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return IsF8<AType>() && IsF8<BType>() && AscendC::IsSameType<typename CType::T, float>::value;
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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return AscendC::IsSameType<typename AType::T, hifloat8_t>::value &&
           AscendC::IsSameType<typename BType::T, hifloat8_t>::value &&
           AscendC::IsSameType<typename CType::T, float>::value;
#else
    return false;
#endif
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
    if constexpr (AscendC::Std::tuple_size_v < L1TileShape >> 3) { // 3: MNKaKb Kb index
        return GetIntegralConstant<3, L1TileShape>();              // 3: MNKaKb Kb index
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
 * @brief Get the type of L0C and Bt
 */
struct GetL0CAndBtType {
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    using Type = int32_t;
#else
    using Type = float;
#endif
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
