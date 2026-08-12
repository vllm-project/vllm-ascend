/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_UTILS_MATH_UTILS_H
#define ARCH35_CATLASS_UTILS_MATH_UTILS_H

#include "device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <typename T>
DEVICE T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
DEVICE T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
DEVICE T Min(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? a : b;
#else
    return min(a, b);
#endif
}

template <typename T>
DEVICE T Max(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? b : a;
#else
    return max(a, b);
#endif
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
