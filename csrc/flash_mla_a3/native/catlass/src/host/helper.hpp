/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_HELPER_HPP
#define EXAMPLES_COMMON_HELPER_HPP

#include <iostream>
#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include "tiling/platform/platform_ascendc.h"

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

int Gcd(int a, int b)
{
    return b == 0 ? a : Gcd(b, a % b);
}

int Lcm(int a, int b)
{
    return abs(a * b) / Gcd(a, b);
}

#endif  // EXAMPLES_COMMON_HELPER_HPP
