/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_
#define OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_

namespace ops {

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

template <typename T>
__aicore__ inline T Ceil(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b;
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace ops
#endif // OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_