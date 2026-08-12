/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_tiling_data.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_TILING_DATA_H__
#define __OP_KERNEL_MATMUL_TILING_DATA_H__

#include "../mat_mul_v3_tiling_data.h"
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint64_t TILINGDATA_OFFSET = 512;
constexpr uint64_t TILINGDATA_SPLIT_NUM = 2;

#pragma pack(push, 8)
struct Mc2MatMulV3TilingData {
    TCubeTiling tCubeTiling;
    // aswt滑窗最后一轮m或n方向的切分次数
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    // streamk 根据核数计算的k方向切分次数
    uint32_t kTailCnt = 0;
    // 负载均衡时，若一边需要均衡，则对应尾块均衡合并后的块数
    // -----------------------------------N=4673------------------------------------
    // -------------------------------------------nTailMain=240--nBaseTailSplitCnt=8
    // --------------------------------------------|-v------nBaseTail=1857----------
    // |256|256|256|256|256|256|256|256|256|256|256|240|240|240|240|240|240|240|177|
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 0;
    uint32_t nTailMain = 0;
    uint32_t isHf32 = 0;
    uint32_t aswWindowLen = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2MatMulV3TilingDataCopy {
    Mc2MatMulV3TilingData matMulTilingData;
    uint8_t reserved[TILINGDATA_OFFSET] = {}; // 申请一个空的512B大小的空间，用于tiling分块
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2BatchMatMulV3TilingData {
    Mc2MatMulV3TilingData matMulTilingData;
    uint32_t aBatchDimAll = 1;
    uint32_t bBatchDimAll = 1;
    uint32_t cBatchDimAll = 1;
    uint32_t biasBatchDimAll = 1;
    uint32_t aBatchDim0 = 1;
    uint32_t bBatchDim0 = 1;
    uint32_t cBatchDim0 = 1;
    uint32_t aBatchDim1 = 1;
    uint32_t bBatchDim1 = 1;
    uint32_t cBatchDim1 = 1;
    uint32_t aBatchDim2 = 1;
    uint32_t bBatchDim2 = 1;
    uint32_t cBatchDim2 = 1;
    uint32_t aBatchDim3 = 1;
    uint32_t bBatchDim3 = 1;
    uint32_t cBatchDim3 = 1;
    uint32_t iterBatch = 1;
    uint32_t batchOutNum = 1;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2MatMulV3BasicTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t skSingleCoreK = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 1;
    uint32_t nTailMain = 1;
    uint8_t isHf32 = 0;
    uint8_t l1BufferNum = 0;
    uint8_t l0cDB = 1; // 默认不开db为1
    uint8_t ubDB = 1;  // ub默认不开db为1
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2BatchMatMulV3BasicTilingData {
    Mc2MatMulV3BasicTilingData matMulTilingData;
    uint32_t batchDimAll = 1;
    uint32_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2BatchMatMulV3IterBatchBasicTilingData {
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    uint32_t b = 1;
    uint32_t iterBatchL1 = 1;
    uint32_t iterBatchL0 = 1;
    uint32_t isHf32 = 0;
    uint32_t baseM = 16;
    uint32_t baseN = 16;
    uint32_t baseK = 16;
};
#pragma pack(pop)

#endif // __OP_KERNEL_MATMUL_TILING_DATA_H__
