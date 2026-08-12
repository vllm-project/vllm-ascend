/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_common_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__

#include <cstdint>
#include "graph/types.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
constexpr uint64_t L0A_SIZE_2 = 64 * 1024UL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t DB_OFF_SIZE = 1UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t NUM_FOUR = 4UL;
constexpr uint64_t CACHELINE = 512UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t DATA_SIZE_FP16 = 2UL;
constexpr uint64_t BASE_STEP = 1UL;
constexpr uint64_t ITER_COL_FIRST = 0UL;
constexpr uint64_t ITER_ROW_FIRST = 1UL;
constexpr uint64_t BASIC_L1_BUFFER_NUM = 4UL;
constexpr uint64_t RPC_WORKSIZE = 20UL;
constexpr uint64_t INIT_SPLIT_CNT = 1UL;
constexpr uint64_t INIT_SPLIT_VALUE = 0UL;

struct Mc2BatchMatMulV3RunInfo {
    uint64_t iterBatch = 0UL;
    uint64_t batchOutNum = 1UL;
};

struct Mc2MatMulV3BatchInfo {
    uint64_t batchA = 1UL;
    uint64_t batchA0 = 1UL;
    uint64_t batchA1 = 1UL;
    uint64_t batchA2 = 1UL;
    uint64_t batchA3 = 1UL;
    uint64_t batchB = 1UL;
    uint64_t batchB0 = 1UL;
    uint64_t batchB1 = 1UL;
    uint64_t batchB2 = 1UL;
    uint64_t batchB3 = 1UL;
    uint64_t batchC = 1UL;
    uint64_t batchC0 = 1UL;
    uint64_t batchC1 = 1UL;
    uint64_t batchC2 = 1UL;
    uint64_t batchC3 = 1UL;
    uint64_t batchBias = 1UL;
};

struct Mc2MatMulV3Args {
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint64_t mValue = 0UL;
    uint64_t mOriValue = 0UL;
    uint64_t nOriValue = 0UL;
    uint64_t kValue = 0UL;
    uint64_t nValue = 0UL;
    uint64_t aDtypeSize = 1UL;
    uint64_t bDtypeSize = 1UL;
    Mc2MatMulV3BatchInfo *batchInfo = nullptr;
};

struct Mc2MatMulV3TailInfo {
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t kCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;
};

struct Mc2MatMulV3MixInfo {
    uint64_t ubDB = 1UL;
};

struct Mc2MatMulV3RunInfo {
    uint64_t usedCoreNum = 1UL;
    uint64_t singleCoreM = 1UL;
    uint64_t singleCoreN = 1UL;
    uint64_t singleCoreK = 1UL;
    uint64_t baseM = 1UL;
    uint64_t baseN = 1UL;
    uint64_t baseK = 1UL;
    uint64_t stepKa = 1UL;
    uint64_t stepKb = 1UL;
    uint64_t depthA1 = 1UL;
    uint64_t depthB1 = 1UL;
    uint64_t stepM = 1UL;
    uint64_t stepN = 1UL;
    uint64_t iterateOrder = 0UL;
    uint64_t dbL0C = 0UL;
    uint64_t iterBatchL1 = 1UL;
    uint64_t iterBatchL0 = 1UL;
    uint64_t l1BufferNum = 2UL;
    uint64_t mBaseTailSplitCnt = 1UL;
    uint64_t nBaseTailSplitCnt = 1UL;
    double defaultBalance = 0.0; // 默认负载均衡率
    double redundantData = 0.0;  // 默认重复搬运量
    Mc2MatMulV3TailInfo tailInfo;
    Mc2BatchMatMulV3RunInfo bmmRunInfo;
    Mc2MatMulV3MixInfo mixInfo;
};

// 基本块维度表（X1,X2,X3,X4,X5），按X1,X2,X3从大到小排序,
// X4输入折算系数(由于base块调整后需求带宽变高，因此一些base块需要折算系数)，X5输出折算系数(当前未使用，等待fixp优化)
const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, double, double>> BLOCK_TABLE = {
    {336, 192, 48, 1, 1.126},     {336, 176, 48, 1, 1.126},     {336, 160, 48, 1, 1.126},
    {336, 144, 48, 1.073, 1.126}, {336, 128, 48, 1.167, 1.126}, {336, 112, 48, 1.287, 1.126},
    {320, 192, 48, 1, 1.126},     {320, 176, 48, 1, 1.126},     {320, 160, 48, 1.014, 1.126},
    {320, 144, 48, 1.089, 1.126}, {320, 128, 48, 1.183, 1.126}, {304, 208, 48, 1, 1.126},
    {304, 192, 48, 1, 1.126},     {304, 176, 48, 1, 1.126},     {304, 160, 48, 1.032, 1.126},
    {304, 144, 48, 1.107, 1.126}, {304, 128, 48, 1.201, 1.126}, {288, 224, 48, 1, 1.126},
    {288, 208, 48, 1, 1.126},     {288, 192, 48, 1, 1.126},     {288, 176, 48, 1, 1.126},
    {288, 160, 48, 1.051, 1.126}, {288, 144, 48, 1.126, 1.126}, {288, 128, 48, 1.220, 1.126},
    {272, 240, 48, 1, 1.126},     {272, 224, 48, 1, 1.126},     {272, 208, 48, 1, 1.126},
    {272, 192, 48, 1, 1.126},     {272, 176, 48, 1.012, 1.126}, {272, 160, 48, 1.073, 1.126},
    {272, 144, 48, 1.148, 1.126}, {272, 128, 48, 1.242, 1.126}, {256, 256, 64, 1, 1},
    {256, 240, 64, 1, 1},         {256, 224, 64, 1, 1},         {256, 208, 64, 1, 1},
    {256, 192, 64, 1, 1},         {256, 176, 64, 1.037, 1},     {256, 160, 64, 1.098, 1},
    {256, 144, 64, 1.173, 1},     {256, 128, 64, 1.267, 1},     {240, 272, 48, 1, 1.126},
    {240, 256, 64, 1, 1},         {240, 240, 64, 1, 1},         {240, 224, 64, 1, 1},
    {240, 208, 64, 1, 1},         {240, 192, 64, 1.014, 1},     {240, 176, 64, 1.065, 1},
    {240, 160, 64, 1.126, 1},     {240, 144, 64, 1.201, 1},     {240, 128, 64, 1.295, 1},
    {224, 288, 48, 1, 1.126},     {224, 272, 48, 1, 1.126},     {224, 256, 64, 1, 1},
    {224, 240, 64, 1, 1},         {224, 224, 64, 1, 1},         {224, 208, 64, 1.003, 1},
    {224, 192, 64, 1.046, 1},     {224, 176, 64, 1.097, 1},     {224, 160, 64, 1.159, 1},
    {224, 144, 64, 1.234, 1},     {208, 304, 48, 1, 1.126},     {208, 288, 48, 1, 1.126},
    {208, 272, 48, 1, 1.126},     {208, 256, 64, 1, 1},         {208, 240, 64, 1, 1},
    {208, 224, 64, 1.003, 1},     {208, 208, 64, 1.04, 1},      {208, 192, 64, 1.083, 1},
    {208, 176, 64, 1.134, 1},     {208, 160, 64, 1.196, 1},     {208, 144, 64, 1.271, 1},
    {192, 336, 48, 1, 1.126},     {192, 320, 48, 1, 1.126},     {192, 304, 48, 1, 1.126},
    {192, 288, 48, 1, 1.126},     {192, 272, 48, 1, 1.126},     {192, 256, 64, 1, 1},
    {192, 240, 64, 1.014, 1},     {192, 224, 64, 1.046, 1},     {192, 208, 64, 1.083, 1},
    {192, 192, 80, 1.126, 1},     {192, 176, 80, 1.178, 1},     {192, 160, 80, 1.239, 1},
    {176, 336, 48, 1, 1.126},     {176, 320, 48, 1, 1.126},     {176, 304, 48, 1, 1.126},
    {176, 288, 48, 1, 1.126},     {176, 272, 48, 1.012, 1.126}, {176, 256, 64, 1.037, 1},
    {176, 240, 64, 1.065, 1},     {176, 224, 64, 1.097, 1},     {176, 208, 64, 1.134, 1},
    {176, 192, 80, 1.178, 1},     {176, 176, 80, 1.229, 1},     {176, 160, 80, 1.290, 1},
    {160, 336, 48, 1, 1.126},     {160, 320, 48, 1.014, 1.126}, {160, 304, 48, 1.032, 1.126},
    {160, 288, 48, 1.051, 1.126}, {160, 272, 48, 1.073, 1.126}, {160, 256, 64, 1.098, 1},
    {160, 240, 64, 1.126, 1},     {160, 224, 64, 1.159, 1},     {160, 208, 64, 1.196, 1},
    {160, 192, 80, 1.239, 1},     {160, 176, 80, 1.290, 1},     {144, 336, 48, 1.073, 1.126},
    {144, 320, 48, 1.089, 1.126}, {144, 304, 48, 1.107, 1.126}, {144, 288, 48, 1.126, 1.126},
    {144, 272, 48, 1.148, 1.126}, {144, 256, 64, 1.173, 1},     {144, 240, 64, 1.201, 1},
    {144, 224, 64, 1.234, 1},     {144, 208, 64, 1.271, 1},     {128, 336, 48, 1.167, 1.126},
    {128, 320, 48, 1.183, 1.126}, {128, 304, 48, 1.201, 1.126}, {128, 288, 48, 1.220, 1.126},
    {128, 272, 48, 1.242, 1.126}, {128, 256, 64, 1.267, 1},     {128, 240, 64, 1.295, 1},
    {112, 336, 48, 1.287, 1.126}};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_COMMON_ADVANCED_H__
