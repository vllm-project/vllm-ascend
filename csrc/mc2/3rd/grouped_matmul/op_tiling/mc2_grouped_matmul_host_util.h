/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file mc2_grouped_matmul_host_util.h
 * \brief
 */

#ifndef MC2_GROUPED_MATMUL_HOST_UTIL_H
#define MC2_GROUPED_MATMUL_HOST_UTIL_H

#include <map>

namespace optiling {
namespace Mc2GroupedMatmul {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t SCALE_INDEX = 3;
constexpr uint32_t OFFSET_INDEX = 4;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 5;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 6;
constexpr uint32_t GROUPLIST_INDEX = 7;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 8;
constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t BEST_L1_PARTA = 256UL * 1024UL;
constexpr uint64_t BEST_L1_PARTB = 128UL * 1024UL;
constexpr uint64_t L1_PARTA_SIZE = 256UL * 1024UL;
constexpr int32_t SPLITK_BASEM_256 = 256;
constexpr int32_t SPLITK_BASEN_128 = 128;
constexpr int32_t SPLITK_BASEK_64 = 64;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t BEST_BASEN_A4W4 = 512;
constexpr int32_t BEST_BASEN_QUANT_ONE_GROUP = 128;
constexpr int32_t BEST_BASEM_QUANT_ONE_GROUP = 256;
constexpr int32_t BEST_BASEK_QUANT_ONE_GROUP = 128;
constexpr int32_t BEST_BASEN_MSD = 512;
constexpr int32_t BEST_UB_BASEK = 256;
constexpr int32_t BEST_UB_BASEN = 512;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t MIN_UB_BASEN = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t A16W8_MSD_STEP = 2;
constexpr uint32_t A16W8_MSD_KN_BASE_BLOCK = 128;
constexpr uint32_t A16W8_MSD_AVERAGE_TOKEN_NUM = 64;
constexpr uint32_t A16W8_MSD_MAX_K = 12U * 1024U;
constexpr uint32_t A16W8_MSD_MIN_N = 1024;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // 32: a block has 32 bytes data
constexpr uint32_t UB_ANTIQUANT_PER_BLOCK_ALIGN = 4U * 1024U;
constexpr uint32_t UB_A16W8_BLOCK_NUM_FP16 = 6; // 2 * sizeof(int8) + 2 * sizeof(half)
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_FP16 = 6;
constexpr uint32_t UB_A16W8_BLOCK_NUM_BF16 = 8; // tmpUb used 2 blks
constexpr uint32_t UB_A16W8_IO_USED_BLOCK_BF16 = 6;
constexpr uint32_t UB_A16W4_BLOCK_NUM_FP16 = 5; // 2 * sizeof(int4) + 2 * sizeof(half)
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_FP16 = 5;
constexpr uint32_t UB_A16W4_BLOCK_NUM_BF16 = 7; // tmpUb used 2 blks
constexpr uint32_t UB_A16W4_IO_USED_BLOCK_BF16 = 5;
constexpr uint32_t UB_A4W4_BLOCK_NUM = 16;
constexpr uint32_t UB_A4W4_IO_USED_BLOCK_HALF = 8; // 2 * sizeof(fp16) + 2 * sizeof(fp16/bf16)
constexpr uint32_t UB_A4W4_PER_BLOCK_ALIGN = 8U * 1024U;
constexpr uint32_t UB_DYNAMIC_QUANT_BLOCK_NUM = 28;
constexpr uint32_t UB_DUNAMIC_QUANT_IO_USED_BLOCK = 12;
constexpr uint32_t UB_QUANT_BLOCK_ALIGN = 2U * 1024U;
constexpr uint32_t UB_A16W8_MSD_BLOCK_NUM = 30;
constexpr uint32_t UB_A16W8_MSD_IO_USED_BLOCK = 6;
constexpr uint32_t UB_A16W8_MSD_BLOCK_ALIGN = 512;
constexpr uint32_t UB_STATIC_QUANT_BLOCK_NUM_BF16 = 20;
constexpr uint32_t UB_STATIC_QUANT_BLOCK_NUM_FP16 = 24;
constexpr uint32_t UB_STATIC_QUANT_IO_USED_BLOCK = 12;
constexpr uint32_t QUEUE_DOUBLE_BUFFER = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY = 0;
constexpr uint64_t TILING_KEY_TRANS_X = 1;
constexpr uint64_t TILING_KEY_TRANS_W = 2;
constexpr uint64_t TILING_KEY_ANTIQUANT_PERFORMANCE = 3;
constexpr uint64_t TILING_KEY_QUANT_2VECTOR = 4;
constexpr uint64_t TILING_KEY_QUANT_2VECTOR_TRANS_W = 5;
constexpr uint64_t TILING_KEY_A16W8_MSD = 6;
constexpr uint64_t TILING_KEY_A16W8_MSD_TRANS_W = 7;
constexpr uint64_t TILING_KEY_A8W4_MSD = 8;
constexpr uint64_t TILING_KEY_A8W4_MSD_NEW = 12;
constexpr uint64_t TILING_KEY_A8W4 = 18;           // per group
constexpr uint64_t TILING_KEY_A8W4_FAKE_A8W8 = 17; // per channel
constexpr uint64_t TILING_KEY_A8W4_AUTOTILING_A8W4 = 21;
constexpr uint64_t TILING_KEY_A8W8_SPARSE_M = 9;
constexpr uint64_t TILING_KEY_A8W8_SPARSE_M_TRANS_W = 10;
constexpr uint64_t TILING_KEY_STATIC_TILING_OFFSET = 13;
constexpr uint64_t ATTR_INDEX_SPLIT_ITEM = 0;
constexpr uint64_t ATTR_INDEX_TRANS_W = 2;
constexpr uint64_t ATTR_INDEX_TRANS_X = 3;
constexpr uint64_t ATTR_INDEX_GROUPTYPE = 4;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 5;
constexpr uint64_t ATTR_INDEX_ACT_TYPE = 6;
constexpr uint64_t ATTR_INDEX_TUNING_CONFIG = 7;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t GROUPLIST_TYPE_SPARSE_M = 2;
constexpr int32_t NO_SPLIT = -1;
constexpr int32_t SPLIT_M = 0;
constexpr int32_t SPLIT_K = 2;
// used for whether going into performance branch in antiquant case, by experiment
constexpr int64_t ANTIQUANT_PERFORMANCE_THRESHOLD = 5L * 1024L * 1024L;
constexpr int64_t ACT_TYPE_GELU = 2;
constexpr uint16_t MAX_TENSOR_CONT = 128;
constexpr int64_t FULL_K_SINGLE_N = 1280;       // used for fullload k case, by experiment
constexpr int64_t FULL_K_N_THRESHOLD = 2560;    // used for fullload k case, by experiment
constexpr int64_t FULL_K_M_THRESHOLD = 2048;    // used for fullload k case, by experiment
constexpr int64_t FULL_K_M_E_THRESHOLD = 256;   // used for fullload k case, by experiment
constexpr int64_t FULL_K_MAX_K_THRESHOLD = 384; // used for fullload k case, by experiment
constexpr int64_t FULL_K_MIN_K_THRESHOLD = 320; // used for fullload k case, by experiment
constexpr uint32_t MIN_NZ_DIM = 4;
constexpr uint32_t MIN_ND_DIM = 2;
constexpr uint32_t A3_AIC_NUM = 24;
constexpr uint32_t OFFSET_DIM_A8W4 = 3;
constexpr float INT4_DATA_TYPE_SIZE = 0.5;
// used for static tiling template
constexpr int32_t STATIC_TILING_DEPTH_A1_B1 = 8;
constexpr int32_t STATIC_TILING_STEP_KA_KB = 4;
constexpr int32_t STATIC_TILING_MAX_K = 8192;

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args>
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 2U * RecursiveSum(templateIds...);
}

const std::map<std::array<int64_t, 4>, std::array<int64_t, 2>> A8W8_PRETILING_WHITE_LIST = { // used for A8W8 preTiling,
                                                                                             // by experiment
    {{576, 7168, 4096, 0}, {128, 512}},
    {{576, 2048, 7168, 1}, {96, 1792}}};

const std::map<std::array<int64_t, 5>, int64_t> A8W4_PRETILING_WHITE_LIST = { // used for A8W4 preTiling, by experiment
    {{1, 16, 256, 512, 1}, 1},
    {{256, 1024, 512, 32768, 1}, 1}};

const std::map<std::array<int64_t, 2>, int64_t> A16W4_MSD_WHITE_LIST = { // used for A16W4 MSD, by experiment
    // K, N
    {{2048, 7168}, 1},
    {{7168, 4096}, 1}};

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
auto CeilAlign(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

/* *
 * if align is 0, return 0
 */
template <typename T>
auto FloorAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    return align == 0 ? 0 : x / align * align;
}
} // namespace Mc2GroupedMatmul
} // namespace optiling

#endif // MC2_GROUPED_MATMUL_HOST_UTIL_H
