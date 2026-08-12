/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file grouped_matmul_utils.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_UTILS_H
#define MC2_GROUPED_MATMUL_UTILS_H

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#if defined(ORIG_DTYPE_X) && defined(DT_INT8) && ORIG_DTYPE_X == DT_INT8
#define DTYPE_L0C_LOCAL int32_t
#else
#define DTYPE_L0C_LOCAL float
#endif
#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && defined(DT_FLOAT8_E5M2) && defined(DT_FLOAT8_E4M3FN) &&     \
    defined(DT_HIFLOAT8) && defined(DT_INT8) && defined(DT_FLOAT4_E2M1) && defined(DT_FLOAT4_E1M2) &&                  \
    defined(DT_INT4) &&                                                                                                \
    ((ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT8) ||                                                      \
     (ORIG_DTYPE_X == DT_HIFLOAT8 && ORIG_DTYPE_WEIGHT == DT_HIFLOAT8) ||                                              \
     ((ORIG_DTYPE_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) &&                                          \
      (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN)) ||                               \
     ((ORIG_DTYPE_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_X == DT_FLOAT4_E1M2) &&                                            \
      (ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_FLOAT4_E1M2)) ||                                 \
     (ORIG_DTYPE_X == DT_INT4 && ORIG_DTYPE_WEIGHT == DT_INT4))
#define V310_GMM_QUANT
#if defined(ORIG_DTYPE_SCALE) && defined(DT_FLOAT8_E8M0) && ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0
#define V310_GMM_QUANT_MX
#elif defined(ORIG_DTYPE_SCALE) && defined(DT_UINT64) && defined(DT_INT64) &&                                          \
    (ORIG_DTYPE_SCALE != DT_UINT64 && ORIG_DTYPE_SCALE != DT_INT64)
#define V310_GMM_QUANT_MIX
#define V310_GMM_QUANT_PERTENSOR_CUBE
#if (ORIG_DTYPE_X != DT_INT8 && ORIG_DTYPE_SCALE == DT_FLOAT)
#define V310_GMM_QUANT_PERTILE
#endif
#else
#define V310_GMM_QUANT_CUBE
#endif
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_X != ORIG_DTYPE_WEIGHT
#if ((ORIG_DTYPE_X == DT_FLOAT16 || ORIG_DTYPE_X == DT_BF16) &&                                                        \
     (ORIG_DTYPE_WEIGHT == DT_FLOAT8_E5M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT8_E4M3FN ||                                  \
      ORIG_DTYPE_WEIGHT == DT_HIFLOAT8 || ORIG_DTYPE_WEIGHT == DT_INT8 || ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 ||       \
      ORIG_DTYPE_WEIGHT == DT_FLOAT4_E1M2 || ORIG_DTYPE_WEIGHT == DT_FLOAT || ORIG_DTYPE_WEIGHT == DT_INT32 ||         \
      ORIG_DTYPE_WEIGHT == DT_INT4)) ||                                                                                \
    (ORIG_DTYPE_X == DT_INT8 && (ORIG_DTYPE_WEIGHT == DT_INT4 || ORIG_DTYPE_WEIGHT == DT_INT32)) ||                    \
    (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1 || ORIG_DTYPE_WEIGHT == DT_FLOAT))
#define V310_GMM_ANTI_QUANT
#endif
#endif
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && defined(ORIG_DTYPE_Y) && defined(DT_INT8) &&                \
    defined(DT_BF16) && defined(DT_INT4)
#if ORIG_DTYPE_X == ORIG_DTYPE_WEIGHT
#if ORIG_DTYPE_X == DT_INT8
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_QUANT_BF16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_FLOAT16
#define GMM_QUANT_FLOAT16
#define MM_DTYPE_Y int32_t
#elif ORIG_DTYPE_Y == DT_INT32
#define GMM_QUANT_INT32
#else
#define GMM_QUANT_INT8
#endif
#elif ORIG_DTYPE_X == DT_INT4
#define GMM_A4W4
#define MM_DTYPE_Y half
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_A4W4_BF16
#elif ORIG_DTYPE_Y == DT_FLOAT16
#define GMM_A4W4_FP16
#endif
#else
#define GMM_FLOAT
#endif
#else
#define GMM_ANTI_QUANT
#if ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT4
#define GMM_ANTI_QUANT_A8W4_MSD
#define GMM_ANTI_QUANT_A8W4
#if ORIG_DTYPE_Y == DT_BF16
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
#else
#define GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16
#endif
#define MM_DTYPE_Y int32_t
#else
#define GMM_ANTI_QUANT
#endif
#endif
#endif

#if defined(DTYPE_Y) && !defined(MM_DTYPE_Y)
#define MM_DTYPE_Y DTYPE_Y
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) ||                                                              \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
#undef DTYPE_SCALE
#define DTYPE_SCALE bfloat16_t
#elif defined(GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16)
#undef DTYPE_SCALE
#define DTYPE_SCALE float
#endif
#endif

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

#if defined(CONST_TILING)
#if defined(V310_GMM_ANTI_QUANT)
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)                                                   \
    GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, member, obj, tiling);                                             \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)                                                   \
    GET_TILING_DATA_MEMBER(tilingType, member, obj, tiling);                                                           \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#endif
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)                                                   \
    size_t offset##var = (size_t)(&((tilingType *)0)->member);                                                         \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((tiling) + (offset##var));
#endif

namespace MC2_GROUPED_MATMUL {
using namespace AscendC;

constexpr uint32_t INT8_BITS = 8;                                    // a int8 number has 8 bits
constexpr int32_t MKN_LIST_LEN = 128;                                // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;                          // 32: a block has 32 bytes data
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;                   // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) ||                                                             \
     (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))) &&                                     \
    defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT8
constexpr MatmulConfig NZ_CFG_MDL =
    GetMDLConfig(false, false, 0, true, false, false, true, true, true, false, false, true);
constexpr MatmulConfig matmulCFGUnitFlag{.doMultiDataLoad = true, .enUnitFlag = true, .enableKdimReorderLoad = true};
#else
constexpr MatmulConfig NZ_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false,
                                         false, 0,     0,    0, 0, 0, 0,     0,     true};
#endif

constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t SYNC_MODE2 = 2;
// used for static tiling template
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr int32_t STATIC_TILING_DEPTH_A1_B1 = 8;
constexpr int32_t STATIC_TILING_STEP_KA_KB = 4;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint32_t STATIC_TILING_MAX_K = 8192;
constexpr uint32_t STATIC_TILING_MAX_SINGLE_N = 1024;

template <class AT_, class BT_, class CT_, class BiasT_, const auto &MM_CFG = CFG_MDL>
struct MMType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

template <class AT_, class BT_, class CT_, class BiasT_, const auto &MM_CFG = CFG_MDL>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

enum class ActiveType : std::uint8_t {
    INVALID_TYPE = 0,
    RELU,
    GELU_TANH,
    GELU_ERR_FUNC,
    FASTGELU,
    SILU
};

template <typename T>
__aicore__ inline T GreatestCommonDivisor(T x, T b)
{
    T c = x;
    if (x < b) {
        x = b;
        b = c;
    }
    while (b != 0) {
        c = x;
        x = b;
        b = c % b;
    }
    return x;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
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

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
    // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
    // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
    // num = a + 3 also satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 7) & ~7; // & ~7: set last four bits of (a+7) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero}
}

template <typename T>
__aicore__ inline __gm__ T *GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(retPtr + index));
}

template <typename T>
__aicore__ inline constexpr uint32_t GetTypeBits()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 4; // 4: int4 bits number
    }
    return sizeof(T) * INT8_BITS;
}

__aicore__ static constexpr MatmulConfig GenGmmConf(bool isND2NZ)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = true,
        .basicM = BASIC_BLOCK_SIZE_128,
        .basicN = BASIC_BLOCK_SIZE_256,
        .basicK = BASIC_BLOCK_SIZE_128,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = isND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = BASIC_BLOCK_SIZE_128,
        .singleCoreN = STATIC_TILING_MAX_SINGLE_N,
        .singleCoreK = STATIC_TILING_MAX_K,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = true,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = false,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = false,
        .enableSetTail = true,
        .enableQuantVector = false,
        .enableSetDefineData = false,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isA2B2Shared = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
    };
}
} // namespace MC2_GROUPED_MATMUL

#endif // MC2_GROUPED_MATMUL_UTILS_H
