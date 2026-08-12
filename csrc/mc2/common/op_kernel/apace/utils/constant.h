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
 * \file constant.h
 * \brief Shared constants and helper types for matmul.
 */
#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Offsets inside the shared GM offset tuple used by quant matmul helpers.
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_SCALEA_OFFSET = 2UL;
constexpr uint64_t IDX_SCALEB_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 4UL;

// Packed block-shape slots: M/N extents followed by optional M/N split
// offsets returned by the scheduler.
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;

// Generic (m, n, k) tuple indices shared by host and device helpers.
constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

// MMAD accumulation mode selectors.
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr uint64_t B8_MIN_STEP = 2UL;

// Event identifiers used by the copy/compute pipeline.
constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint16_t SECOND_FLAG = 2;
constexpr uint16_t THIRD_FLAG = 3;
constexpr uint16_t SIXTH_FLAG = 6;
constexpr uint16_t SEVENTH_FLAG = 7;
constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM_MX = 6;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 4;
constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
constexpr uint16_t FLAG_ID_MAX = 16;

// Shared MX constants for the device-side kernel, block, tile, and utility
// helpers. Host tiling keeps its own prefixed names to avoid collisions when
// both header groups are included in the same translation unit.
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;

// Shared constants used by the host-side tiling engine.
//
// These values describe hardware granularity, cache-line alignment, buffering
// policy, and the search space limits used while selecting a tiling scheme.
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t FP4_C0_SIZE = 64UL;
constexpr uint64_t FP8_C0_SIZE = 32UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint64_t DATA_SIZE_L0C = 4UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t TILING_MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t TILING_MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr uint64_t STEPK_THERSHOLD = 4UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t SCALER_FACTOR_MIN = 1UL;
constexpr uint64_t SCALER_FACTOR_MAX = 127UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_FP16 = 2UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t CACHELINE = 512UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t RPC_WORKSIZE = 20UL;
constexpr uint16_t BLOCK_BASE_M = 256UL;
constexpr uint16_t BLOCK_BASE_N = 256UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;