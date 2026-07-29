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
 * \file platform.h
 * \brief platform apator
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_
#define OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

#ifndef KERNEL_API
#define KERNEL_API extern "C" __global__ __aicore__
#endif

namespace platform {

#define MID_THREAD_NUM 1024

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

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

/**
 * Check whether the type is supported by atomic add for simd
 */
template <typename T>
__aicore__ inline constexpr bool IsSupportAtomicAddTypeSIMD()
{
#if __CCE_AICORE__ == 310
    return ops::IsSame<T, float>::value || ops::IsSame<T, half>::value || ops::IsSame<T, int16_t>::value ||
           ops::IsSame<T, int32_t>::value || ops::IsSame<T, int8_t>::value || ops::IsSame<T, bfloat16_t>::value;
#else
    return false;
#endif
}

} // namespace platform

namespace PlatformSocInfo {
__aicore__ inline constexpr bool IsDataCopyPadSupport() { return platform::IsDataCopyPadSupport(); }

} // namespace PlatformSocInfo

#endif // OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_