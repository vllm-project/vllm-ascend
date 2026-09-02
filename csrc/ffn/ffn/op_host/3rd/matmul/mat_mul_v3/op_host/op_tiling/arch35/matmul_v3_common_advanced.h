/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#pragma once

#include <cstdint>
#include "graph/types.h"

namespace optiling {
namespace matmul_v3_advanced {
constexpr uint64_t L0A_SIZE_2 = 64 * 1024UL;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t DB_OFF_SIZE = 1UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32UL;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t BASIC_BLOCK_K_512_BYTE = 512UL;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t NUM_ONE = 1UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t NUM_FOUR = 4UL;
constexpr uint64_t NUM_FIVE = 5UL;
constexpr uint64_t NUM_EIGHT = 8UL;
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
constexpr uint64_t ALIGN_128 = 128UL;
constexpr uint64_t L1_SINGLE_SIZE_LIMIT = 48 * 1024UL;
constexpr uint64_t THOUSAND_NUM = 1000UL;
constexpr uint64_t KB_SIZE = 1024UL;
constexpr uint64_t MIN_TATL_BLOCK_SIZE = 4096UL;
constexpr double CUBE_BOUND_RATIO = 0.85;
constexpr double EPSILON = 1e-9;
constexpr int64_t FP32_SPLIT_K_THRESHOLD = 2048UL;
constexpr int64_t FP32_K_SWITCH_BASE = 268435456; // 1024 * 32 * 8192
constexpr uint64_t FP32_SPLIT_K_BASE1 = 1024UL;
constexpr uint64_t FP32_SPLIT_K_BASE2 = 8192UL;
constexpr uint64_t FP32_MIN_BASE_BLOCK = 64UL;

struct BatchMatMulV3RunInfo {
    uint64_t iterBatch = 0UL;
    uint64_t batchOutNum = 1UL;
    uint32_t broadcastAxisA = 4UL;
    uint32_t broadcastAxisB = 4UL;
};

struct MatMulV3BatchInfo {
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

struct MatMulV3Args {
    const char* opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    bool hasScale = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType x3Type = ge::DT_FLOAT16;
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
    uint64_t fusedOpType = 0UL;
    uint64_t batchX3 = 1UL;
    bool hasX3Input = false;
    bool isForceGrpAccForFp32 = false;
    bool isAvoidTensorApi = false;
    bool preferL0cDB2 = false; // FFN swiglu 单 matmul：偏好 L0C 双缓冲（db=2），避免 tile 占满 L0C 导致 fixpipe 串行
    bool preferL0cMSplitDB2 = false; // FFN swiglu 单 matmul：沿 M 拆半（保持 baseN）恢复 dbL0C=2，不损失 mmad N 宽度
    bool preferNoMSplit = false; // FFN down matmul：不沿 M 拆块（baseM=ceil(M,16)），避免 B 权重全量重复搬运
    bool preferUbDB2 = false; // FFN up 融合相：tile 超出单 UB 时沿 N 拆半压入半 UB，恢复 ubDB=2
                             // （AIV epilogue 与下一 tile 的 L0C→UB fixpipe 乒乓；沿 N 拆半使重复搬运
                             //  落在更小的 A 矩阵上，B 权重（K×N，通常更大）的 mCnt 重复不变）
    // FFN swiglu 单 matmul（N=2H，B 装载交错 gate/up）：
    // epilogue 半宽输出行须 32B 对齐（bf16 outWidth=baseN/2 个元素），故要求 baseN 为 32 的倍数；
    // 同时尾波只允许 M 方向拆分（N 拆分会产生非 32 对齐子 tile，破坏半 tile 配对）。
    bool swigluSingleNAlign32 = false;
    MatMulV3BatchInfo* batchInfo = nullptr;
};

struct MatMulV3TailInfo {
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t kCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;
};

struct MatMulV3MixInfo {
    uint64_t ubDB = 1UL;
};

struct BatchMatMulV3ToMulInfo {
    uint64_t m = 1UL;
    uint64_t n = 1UL;
    uint64_t b = 1UL;
    uint64_t usedCoreNum = 1UL;
    uint64_t singleCoreBatch = 1UL;
    uint64_t batchNum = 1UL;
    uint64_t batchNumLastRound = 1UL;
    uint64_t batchNumLastRoundTail = 1UL;
    uint64_t lastCoreNum = 1UL;
    uint64_t alignNum = 1UL;
};

struct MatMulV3ToMulInfo {
    uint64_t tileNum = 1UL;
    uint64_t baseMN = 1UL;
    uint64_t tailMN = 1UL;
    uint64_t baseK = 1UL;
    uint64_t tailK = 1UL;
    uint64_t loopK = 1UL;
    bool dataCopyMode = false;
};

struct MatMulV3RunInfo {
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
    uint64_t innerBatch = 0UL;
    uint64_t l1BufferNum = 2UL;
    uint64_t mBaseTailSplitCnt = 1UL;
    uint64_t nBaseTailSplitCnt = 1UL;
    double defaultBalance = 0.0; // 默认负载均衡率
    double redundantData = 0.0;  // 默认重复搬运量
    uint64_t totalDataAmount = 1UL;
    uint64_t mergeBatchAL1 = 1UL;
    uint64_t mergeBatchBL1 = 1UL;
    uint64_t mergeBatchL0 = 1UL;
    bool needNdDma = false;
    double cubeBoundParam = 0.0;
    double cubeBoundEdge = 0.0;
    MatMulV3TailInfo tailInfo;
    BatchMatMulV3RunInfo bmmRunInfo;
    MatMulV3MixInfo mixInfo;
    BatchMatMulV3ToMulInfo bmmToMulInfo;
    MatMulV3ToMulInfo mmToMulInfo;
};
} // namespace matmul_v3_advanced
} // namespace optiling
