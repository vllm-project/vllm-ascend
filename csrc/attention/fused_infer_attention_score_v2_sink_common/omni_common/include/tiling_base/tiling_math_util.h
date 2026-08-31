/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_math_util.h
 * \brief Math utility functions for tiling operations
 */

#ifndef TILING_MATH_UTIL_H
#define TILING_MATH_UTIL_H

#include <cstdint>

namespace AiInfraOps {
namespace Transformer {
namespace Base {

// Ceiling division: returns ceil(a / b)
template <typename T1, typename T2>
static inline auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// Floor division: returns floor(a / b)
template <typename T1, typename T2>
static inline auto FloorDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return a / b;
}

// Ceiling alignment: returns the smallest multiple of b that is >= a
template <typename T1, typename T2>
static inline auto CeilAlign(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

} // namespace Base
} // namespace Transformer
} // namespace AiInfraOps

#endif
