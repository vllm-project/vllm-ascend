/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTN_INFRA_COMMON_HPP_
#define ATTN_INFRA_COMMON_HPP_

#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"

#define HOST_DEVICE __host_aicore__ inline

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

namespace NpuArch::Detail::Alignment
{

template <uint32_t ALIGN, typename T>
HOST_DEVICE
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T, class U>
HOST_DEVICE
constexpr auto RoundUp(T const &val, U const &align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
HOST_DEVICE
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0U, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
HOST_DEVICE
constexpr T RoundDown(const T val, const T align)
{
    if (align == 0) {
        return val;
    }
    return val / align * align;
}

template <uint32_t DIVISOR, typename T>
HOST_DEVICE
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "DIVISOR must not be 0");
    return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
HOST_DEVICE
constexpr auto CeilDiv(T const &dividend, U const &divisor)
{
    if (divisor == 0) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

}

/*!
 * \file base_defs.hpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#else
#endif

namespace NpuArch {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;

constexpr uint32_t STRIDE_LIMIT = 65536;

} // namespace NpuArch

#endif // ATTN_INFRA_COMMON_HPP_
