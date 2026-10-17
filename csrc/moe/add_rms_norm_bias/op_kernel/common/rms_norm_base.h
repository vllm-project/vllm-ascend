/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_BASE_H_
#define RMS_NORM_BASE_H_
#include "kernel_operator.h"

namespace RmsNorm {
using namespace AscendC;


/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220 && __CCE_AICORE__ != 310
#define bfloat16_t int16_t
#endif
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t UNROLL_NUM = 2;
constexpr int32_t NUM_PER_REP_FP32 = 64; // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr int32_t FLOAT_BTYPE_SIZE = 4;
constexpr int32_t NUM_PER_BLK_FP16 = 16;
constexpr int32_t CONTINUE_STRIDE = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr float MINUS_HALF = -0.5f;
constexpr uint32_t ZERO_UINT = 0;
constexpr uint32_t ONE_UINT = 1;
constexpr uint32_t TWO_UINT = 2;
constexpr uint32_t THREE_UINT = 3;
constexpr float ONE = 1;
constexpr int32_t SECOND_LOOP = 2;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t MAX_REAPEAT = 255;
constexpr int32_t DIM_NUM = 2;
constexpr int32_t NDDMA_DIM = 5;

constexpr uint32_t V_LENGTH = GetVRegSize() / sizeof(float);
constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr uint32_t SUM_COUNT = 2;
constexpr int32_t MOV_2 = 2;
constexpr int32_t MOV_4 = 4;
constexpr int32_t MOV_8 = 8;
constexpr int32_t MOV_16 = 16;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename T>
__aicore__ inline T Min(T left, T right)
{
    return (left < right ? left : right);
}

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T, typename T_GAMMA>
class KernelRmsNormBase {
#define IS_X_FP32 (is_same<T, float>::value)
#define IS_GAMMA_FP32 (is_same<T_GAMMA, float>::value)
#define IS_MIX_DTYPE ((!IS_X_FP32) && IS_GAMMA_FP32)
};

} // namespace RmsNorm
#endif // RMS_NORM_BASE_H_
