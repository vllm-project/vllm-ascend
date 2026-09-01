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
 * \file grouped_matmul_tiling_data_apt.h
 * \brief
 */
#ifndef GROUPED_MATMUL_TILING_DATA_H
#define GROUPED_MATMUL_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GroupedMatmulTilingData {
#pragma pack(push, 8)
struct GMMArray {
    // GroupedMatmul::MAX_TENSOR_CONT
    int32_t mList[128] = {0};
    int32_t kList[128] = {0};
    int32_t nList[128] = {0};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantBaseParams {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t singleWeight = 0;
    uint32_t singleX = 0;
    uint32_t singleY = 0;
    int32_t groupType = 0;
    uint32_t groupListType = 0;
    uint32_t hasBias = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t weightNoL2Cache = 0;
    uint32_t placeHolder = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantParams {
    uint32_t groupNum = 0;
    uint32_t activeType = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint8_t singleX = 0;
    uint8_t singleW = 0;
    uint8_t singleY = 0;
    int8_t groupType = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantBasicApiMMTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint16_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantParam {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint8_t singleX = 0;
    uint8_t singleWeight = 0;
    uint8_t singleY = 0;
    int8_t groupType = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint8_t cubeNumBlocksN = 0;
    uint8_t reserved = 0;
    uint32_t groupSize = 0;
    uint32_t mainBlockSize = 0;
    uint64_t mainBlockCount = 0;
    uint16_t firstTailBlockSize = 0;
    uint16_t secondTailBlockSize = 0;
    uint16_t firstTailBlockCount = 0;
    uint16_t secondTailBlockCount = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantTilingData {
    GMMQuantParams gmmQuantParams;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantBasicApiTilingData {
    GMMQuantParams gmmQuantParams;
    QuantBasicApiMMTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantTilingData {
    GMMNoQuantBaseParams gmmNoQuantParam;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

} // GroupedMatmulTilingData
#endif