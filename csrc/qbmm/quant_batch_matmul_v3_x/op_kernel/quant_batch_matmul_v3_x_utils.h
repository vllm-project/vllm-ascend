/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_x_utils.h
 * \brief Kernel-side scalar utilities (AlignUp, Min, scale-type tag) for the INT8 kernel.
 * \author Feodor Pisnitchenko
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_UTILS_H
#define QUANT_BATCH_MATMUL_V3_X_UTILS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace QBM {
using namespace AscendC;

// Scale-encoding template tag. Only u64 (VDEQ16-encoded) is
// instantiated; fp32 path is selected at runtime by comparing
// scaleType_ against this value.
constexpr int SCALE_UINT64 = 2;

// Mmad fragment: M0 x N0 with K0 contraction (int8).
constexpr uint32_t CUBE_M0      = 16;
constexpr uint32_t CUBE_N0      = 16;
constexpr uint32_t CUBE_K0_INT8 = 32;

// UB DataCopy block (32 B = 8 fp32 = 4 u64).
constexpr uint32_t UB_BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline T AlignUp(T a, T base) {
    return (a + base - 1) / base * base;
}

template <uint32_t base>
__aicore__ inline uint32_t AlignUp(uint32_t a) {
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T Min(T a, T b) {
    return a < b ? a : b;
}

}  // namespace QBM

#endif  // QUANT_BATCH_MATMUL_V3_X_UTILS_H
