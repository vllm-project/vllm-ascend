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
 * \file chunk_gated_delta_rule_utils.h
 * \brief Common constants for chunk_gated_delta_rule kernels
 */

#ifndef CHUNK_GATED_DELTA_RULE_UTILS_H__
#define CHUNK_GATED_DELTA_RULE_UTILS_H__

#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRule {
    // 同步信号
    constexpr uint64_t V_MTE3_EVENT = 0;
    constexpr uint64_t V_S_EVENT = 1;
    constexpr uint64_t MTE2_V_EVENT = 2;
    constexpr uint64_t S_V_EVENT = 3;
    constexpr uint64_t MTE3_MTE2_EVENT = 4;
    constexpr uint64_t MTE3_S_EVENT = 5;
    constexpr uint64_t FIX_MTE2_EVENT = 6;
    constexpr uint64_t S_MTE3_EVENT = 6;

    constexpr uint64_t NUM_ONE = 1;
    constexpr uint64_t BUFFER_NUM_ONE = 1;
    constexpr uint64_t BUFFER_NUM_TWO = 2;
    constexpr uint64_t TQUE_DEPTH_TWO = 2;
    constexpr uint64_t AIC_AIV_1_1 = 2;
    constexpr uint64_t BROADCAST_AXIS = 2;
    constexpr uint64_t TASK_RATIO = 2;
    constexpr uint64_t STAGE3_BUFFER_COUNT = 4;
    constexpr uint32_t MAX_L0_SIZE = 64 * 1024; // 64KB
    constexpr uint32_t BLOCK_SIZE = 32;         // copypad对齐块大小
    constexpr uint32_t BLOCK_FLOAT_NUM = 8;
    constexpr uint32_t BLOCK_BF16_NUM = 16;
    constexpr uint32_t TILE_LEN = 1024;   // 1024 = 1kb，经测试1kb和10kb性能差异很小
}  // ChunkGatedDeltaRule

#endif  // __CHUNK_GATED_DELTA_RULE_UTILS_H__
