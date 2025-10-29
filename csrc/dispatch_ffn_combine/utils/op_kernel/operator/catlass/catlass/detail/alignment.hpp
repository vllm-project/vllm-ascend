/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ALIGNMENT_HPP
#define CATLASS_ALIGNMENT_HPP

#include "catlass/detail/macros.hpp"

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T>
CATLASS_HOST_DEVICE
constexpr T RoundUp(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
CATLASS_HOST_DEVICE
constexpr T RoundDown(const T val, const T align)
{
    return val / align * align;
}

template <uint32_t DIVISOP, typename T>
CATLASS_HOST_DEVICE
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOP != 0, "DIVISOP must not be 0");
    return (dividend + DIVISOP - 1) / DIVISOP;
}

template <class T>
CATLASS_HOST_DEVICE
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

#endif  // CATLASS_ALIGNMENT_HPP
