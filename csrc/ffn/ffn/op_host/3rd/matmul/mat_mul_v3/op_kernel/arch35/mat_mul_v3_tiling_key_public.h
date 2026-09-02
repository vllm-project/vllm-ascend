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
 * \file mat_mul_v3_tiling_key_public.h
 * \brief
 */

#pragma once

#define MAT_MUL_TENSOR_LEVEL 2 // 数据类型定义
#define MAT_MUL_BASIC_LEVEL 1
#define MAT_MUL_HIGH_LEVEL 0

#define MAT_MUL_NO_TRANS 0 //  数据格式定义
#define MAT_MUL_TRANS 1

#define MAT_MUL_FOR_BATCH 0
#define MAT_MUL_ITER_BATCH_SINGLE_BIAS 1
#define MAT_MUL_BATCH_MATMUL_TO_MUL 2
#define MAT_MUL_MERGE_BATCH 3
#define MAT_MUL_FOR_FUSED_BATCH 4
#define MAT_MUL_BROADCAST_BATCH 5
#define MAT_MUL_ITER_BATCH_BROADCAST_A 6
#define MAT_MUL_ITER_BATCH_BROADCAST_B 7

#define MAT_MUL_BASIC 0
#define MAT_MUL_STREAM_K 1
#define MAT_MUL_K_EQUAL_ZERO 2
#define MAT_MUL_TO_MUL 3
#define MAT_MUL_TO_MULTI_MUL 4
#define MAT_MUL_SLICE 5
#define MAT_MUL_BASIC_SPLIT_K 6
#define MAT_MUL_SK_SPLIT_K 7

#define MAT_MUL_NO_FULL_LOAD 0
#define MAT_MUL_A_FULL_LOAD 1
#define MAT_MUL_B_FULL_LOAD 2
#define MAT_MUL_AB_FULL_LOAD 3

#define MAT_MUL_ON_THE_FLY 0
#define MAT_MUL_1V1_ND_ALIG_FIXPIPE 1
#define MAT_MUL_1V2_ND_ALIG_FIXPIPE 2

enum class MatMulV3ATrans : std::uint8_t { A_NO_TRANS = MAT_MUL_NO_TRANS, A_TRANS = MAT_MUL_TRANS };

enum class MatMulV3BTrans : std::uint8_t { B_NO_TRANS = MAT_MUL_NO_TRANS, B_TRANS = MAT_MUL_TRANS };

enum class MatMulV3ApiLevel : std::uint8_t {
    HIGH_LEVEL = MAT_MUL_HIGH_LEVEL,
    BASIC_LEVEL = MAT_MUL_BASIC_LEVEL,
    TENSOR_LEVEL = MAT_MUL_TENSOR_LEVEL
};

enum class MatMulV3BatchModel : std::uint8_t {
    BATCH_MODEL = MAT_MUL_FOR_BATCH,
    SINGLE_BIAS_MODEL = MAT_MUL_ITER_BATCH_SINGLE_BIAS,
    BATCH_MATMUL_TO_MUL = MAT_MUL_BATCH_MATMUL_TO_MUL,
    MERGE_BATCH_MODEL = MAT_MUL_MERGE_BATCH,
    FUSED_BATCH_MODEL = MAT_MUL_FOR_FUSED_BATCH,    // 只用于FusedMatMul的Bmm场景
    BROADCAST_BATCH_MODEL = MAT_MUL_BROADCAST_BATCH, // 只用于BatchMatMul广播场景
    ITER_BATCH_BROADCAST_A_MODEL = MAT_MUL_ITER_BATCH_BROADCAST_A,
    ITER_BATCH_BROADCAST_B_MODEL = MAT_MUL_ITER_BATCH_BROADCAST_B
};

enum class MatMulV3Model : std::uint8_t {
    BASIC = MAT_MUL_BASIC,
    STREAM_K = MAT_MUL_STREAM_K,
    K_EQUAL_ZERO = MAT_MUL_K_EQUAL_ZERO,
    TO_MUL = MAT_MUL_TO_MUL,
    TO_MULTI_MUL = MAT_MUL_TO_MULTI_MUL,
    SLICE = MAT_MUL_SLICE,
    BASIC_SPLIT_K = MAT_MUL_BASIC_SPLIT_K,
    SK_SPLIT_K = MAT_MUL_SK_SPLIT_K
};

enum class MatMulV3FullLoad : std::uint8_t {
    NONE_FULL_LOAD = MAT_MUL_NO_FULL_LOAD,
    A_FULL_LOAD = MAT_MUL_A_FULL_LOAD,
    B_FULL_LOAD = MAT_MUL_B_FULL_LOAD,
    AB_FULL_LOAD = MAT_MUL_AB_FULL_LOAD
};

enum class MatMulV3L0C2Out : std::uint8_t {
    ON_THE_FLY = MAT_MUL_ON_THE_FLY,
    ND_FIXPIPE_1_1 = MAT_MUL_1V1_ND_ALIG_FIXPIPE,
    ND_FIXPIPE_1_2 = MAT_MUL_1V2_ND_ALIG_FIXPIPE
};
