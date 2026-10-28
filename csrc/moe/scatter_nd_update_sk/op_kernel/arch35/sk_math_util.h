/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_math_util.h
 * \brief Local substitutes for Ops::Base math/platform helpers used by the
 *        ported scatter_nd_update arch35 kernel headers (the toolkit headers
 *        "op_kernel/math_util.h" and "op_kernel/platform_util.h" are not
 *        available in this build environment).
 */

#ifndef SCATTER_ND_UPDATE_SK_MATH_UTIL_H_
#define SCATTER_ND_UPDATE_SK_MATH_UTIL_H_

#include "kernel_operator.h"

namespace Ops {
namespace Base {

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

template <typename T, typename U>
__aicore__ inline auto CeilDiv(T a, U b) -> decltype(a / b)
{
    return (a + b - 1) / b;
}

template <typename T, typename U>
__aicore__ inline auto CeilAlign(T a, U b) -> decltype(a / b)
{
    return (a + b - 1) / b * b;
}

template <typename T, typename U>
__aicore__ inline auto FloorAlign(T a, U b) -> decltype(a / b)
{
    return a / b * b;
}

} // namespace Base
} // namespace Ops

#endif // SCATTER_ND_UPDATE_SK_MATH_UTIL_H_
