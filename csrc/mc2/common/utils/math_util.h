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
 * \file math_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_

#include "util/math_util.h"

namespace Ops {
namespace NN {
class MathUtil {
public:
    template <typename T>
    static T CeilDivision(T num1, T num2)
    {
        return Ops::Base::CeilDiv(num1, num2);
    }

    static int64_t CeilDivision(int64_t num1, int32_t num2)
    {
        return Ops::Base::CeilDiv(num1, static_cast<int64_t>(num2));
    }
    template <typename T>
    static T Align(T num1, T num2)
    {
        return Ops::Base::CeilAlign(num1, num2);
    }
};
} // namespace NN
} // namespace Ops

namespace ops {
template <typename T>
static T CeilAlign(T num1, T num2)
{
    return Ops::Base::CeilAlign(num1, num2);
}
template <typename T>
static T FloorAlign(T num1, T num2)
{
    return Ops::Base::FloorAlign(num1, num2);
}
template <typename T>
static T FloorDiv(T num1, T num2)
{
    return Ops::Base::FloorDiv(num1, num2);
}
template <typename T>
static T CeilDiv(T num1, T num2)
{
    return Ops::Base::CeilDiv(num1, num2);
}
} // namespace ops

#endif // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
