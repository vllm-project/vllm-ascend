/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CATLASS_HPP
#define CATLASS_CATLASS_HPP

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "detail/alignment.hpp"
#include "detail/dependent_false.hpp"
#include "detail/macros.hpp"

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRCATLASSAL = 16;
constexpr uint32_t BYTE_PER_FRCATLASSAL = BYTE_PER_C0 * C0_NUM_PER_FRCATLASSAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRCATLASSAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRCATLASSAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRCATLASSAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

} // namespace Catlass

#endif // CATLASS_CATLASS_HPP
