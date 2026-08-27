/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_CONSTANT_H
#define MATMUL_CONSTANT_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t CUBE_BLOCK = 16UL;

constexpr uint64_t FP8_C0_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_L0C = 4UL;

constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;

constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;

#endif
