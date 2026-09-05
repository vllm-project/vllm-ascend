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
#include "device_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif

using namespace AscendC;

namespace Cmct {
template <typename T>
__aicore__ inline constexpr T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline constexpr T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline constexpr T FloorAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return a / b * b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? a : b;
#else
    return min(a, b);
#endif
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? b : a;
#else
    return max(a, b);
#endif
}

__aicore__ inline uint64_t CalcRealLen(uint64_t fullLen, uint64_t offset, uint64_t tileLen)
{
    return offset + tileLen > fullLen ? fullLen - offset : tileLen;
}

template <typename Dtype, typename Int>
__aicore__ inline Int ElemToByte(Int count)
{
    static_assert(AscendC::SupportType<Dtype, uint8_t, int8_t, half, bfloat16_t, AscendC::int4b_t, int4x2_t, fp8_e5m2_t,
                                       fp8_e4m3fn_t, hifloat8_t, fp8_e8m0_t, fp4x2_e2m1_t>(),
                  "not support this dtype");
    if (AscendC::Std::is_same_v<Dtype, AscendC::int4b_t> || AscendC::Std::is_same_v<Dtype, int4x2_t> ||
        AscendC::Std::is_same_v<Dtype, fp4x2_e2m1_t>) {
        // int4类型的元素数量必须是2的倍数
        return count >> 1;
    }
    return count * sizeof(Dtype);
}

template <typename Dtype, typename Int, Int Count>
__aicore__ inline Int ElemToByte(AscendC::Std::integral_constant<Int, Count>)
{
    static_assert(AscendC::SupportType<Dtype, uint8_t, int8_t, half, bfloat16_t, AscendC::int4b_t, int4x2_t, fp8_e5m2_t,
                                       fp8_e4m3fn_t, hifloat8_t, fp8_e8m0_t, fp4x2_e2m1_t>(),
                  "not support this dtype");
    if (AscendC::Std::is_same_v<Dtype, AscendC::int4b_t> || AscendC::Std::is_same_v<Dtype, int4x2_t> ||
        AscendC::Std::is_same_v<Dtype, fp4x2_e2m1_t>) {
        // int4类型的元素数量必须是2的倍数
        return Count >> 1;
    }
    return Count * sizeof(Dtype);
}
} // namespace Cmct
