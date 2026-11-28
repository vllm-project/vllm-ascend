/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t TMP_SIZE = 65536;
constexpr uint32_t TMP_SIZE_DECODER = 32768;

constexpr int32_t TILING_BATCH = 0;
constexpr int32_t TILING_NUMHEADS = 1;
constexpr int32_t TILING_HEADDIM = 2;
constexpr int32_t TILING_NUMBLOKS = 3;
constexpr int32_t TILING_BLOCKSIZE = 4;
constexpr int32_t TILING_MAXBLOCKS = 5;
constexpr int32_t TILING_TOR = 6;
constexpr int32_t TILING_KVHEADS = 7;
constexpr int32_t TILING_HEADSIZE = 8;
constexpr int32_t TILING_PARASIZE = 9;
constexpr int32_t TILING_HEAD_SPLIT_SIZE = 10;
constexpr int32_t TILING_HEAD_SPLIT_NUM = 11;
constexpr int32_t TILING_HEADDIM_ROPE = 13;
constexpr int32_t TILING_MAX_KVSEQLEN = 14;
constexpr int32_t TILING_KVSPLIT = 15;
constexpr int32_t TILING_KVCORENUM = 16;
constexpr int32_t TILING_MAX_QSEQLEN = 17;
constexpr int32_t TILING_TOTAL_QTOKENS = 18;
constexpr int32_t TILING_FORMERTASKNUM = 19;
constexpr int32_t TILING_TAILTASKNUM = 20;
constexpr int32_t TILING_MASKTYPE = 12;
constexpr int32_t TILING_INPUT_STRIDE = 21;
constexpr int32_t TILING_BLOCKSIZE_CALC = 25;
constexpr int32_t TILING_HEADDIM_K_SPLIT = 38;
constexpr int32_t TILING_HEADDIM_V_SPLIT = 39;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
constexpr int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

#endif