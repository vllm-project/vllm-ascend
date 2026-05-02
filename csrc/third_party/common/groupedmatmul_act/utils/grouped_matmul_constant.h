/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_constant.h
 * \brief
 */
#ifndef UTILS_GROUPED_MATMUL_CONSTANT_H
#define UTILS_GROUPED_MATMUL_CONSTANT_H
namespace Act {
namespace Gemm {
namespace GroupedMatmul {
constexpr uint32_t GMM_BUFFER_NUM = 2;
constexpr uint16_t GMM_FLAG_ID_MAX = 16;
constexpr uint16_t GMM_AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t GMM_AIC_SYNC_AIV_FLAG = 8;
constexpr int32_t GMM_CUBE_SYNC_MTE1_FLAG = 3;
constexpr uint8_t GMM_AIC_SYNC_AIV_MODE = 4;
constexpr uint64_t GMM_MAX_STEP_SCALEA_K = 16;
constexpr uint32_t GMM_UB_ALIGN_SIZE = 32;
constexpr uint32_t GMM_MAX_REPEAT_TIMES = 255;

constexpr uint8_t GMM_SPLIT_M = 0;
constexpr uint8_t GMM_SPLIT_K = 2;
constexpr uint64_t GMM_CUBE_BLOCK = 16;
constexpr uint64_t GMM_INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cacheline 128
constexpr int32_t GMM_MKN_LIST_LEN = 128;              // 128: predefined array legnth

constexpr uint32_t GMM_BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_B8 = 32;
constexpr uint32_t GMM_k0_FLOAT16 = 16;
constexpr uint16_t GMM_DATA_BLOCK = 32;

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERGROUP_MODE = 0x1U << 4,
    PERBLOCK_MODE = 0x1U << 5,
};
} // namespace GroupedMatmul
} // namespace Gemm
} // namespace Act
#endif