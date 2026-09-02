/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_constant.h
 * \brief
 */
#pragma once
namespace Cmct {
namespace Gemm {
namespace QuantBatchMatmul {
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 4;
constexpr uint16_t DOUBLE_BUFFER = 2;
constexpr uint16_t FOUR_BUFFER = 4;
constexpr uint32_t QBMM_BUFFER_NUM = 2;
constexpr uint16_t QBMM_FLAG_ID_MAX = 16;
constexpr uint16_t QBMM_AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t QBMM_AIC_SYNC_AIV_FLAG = 8;
constexpr int32_t QBMM_CUBE_SYNC_MTE1_FLAG = 3;
constexpr uint8_t QBMM_AIC_SYNC_AIV_MODE = 4;
constexpr uint64_t QBMM_MAX_STEP_SCALEA_K = 16;
constexpr uint32_t QBMM_UB_ALIGN_SIZE = 32;

constexpr uint32_t QBMM_BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_B8 = 32;
constexpr uint32_t QBMM_k0_FLOAT16 = 16;
constexpr uint16_t QBMM_DATA_BLOCK = 32;

constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;

constexpr int32_t BT_SIZE = 4096;

constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr uint64_t B8_MIN_STEP = 2UL;
constexpr uint64_t B4_MIN_STEP = 4UL;

constexpr uint16_t INPUT_BUFFER_FLAG_0 = 0;
constexpr uint16_t INPUT_BUFFER_FLAG_1 = 1;
constexpr uint16_t INPUT_BUFFER_FLAG_2 = 2;
constexpr uint16_t INPUT_BUFFER_FLAG_3 = 3;

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};
} // namespace QuantBatchMatmul
} // namespace Gemm
} // namespace Cmct
