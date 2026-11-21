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
 * \file basic_block_config.h
 * \brief
 */
#ifndef GMMSQ_BASIC_BLOCK_CONFIG_H
#define GMMSQ_BASIC_BLOCK_CONFIG_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"

using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    QuantType antiQuantType;
    bool hasAntiQuantOffset;
    QuantType quantType;
    CubeFormat weightFormat;
};

static constexpr WqmmConfig MXA8W4_NZNK = {false, true, QuantType::MX, false, QuantType::MX, CubeFormat::NZ};

struct BasicBlockControlParam {
    uint64_t processId;
    uint64_t mSize;
    uint64_t mL1Size;
    uint64_t curBasicBlockId;
    uint64_t basicBlockLimit;
    uint64_t mOffset;
    uint64_t nOffset;
};

struct BasicBlockOffsetParam {
    uint64_t mL1Size;
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t kAlign;
    uint64_t nAlign;

    GM_ADDR yGmAddr;
    GM_ADDR yScaleGmAddr;
};

struct VecAntiQuantConfig {
    uint64_t ubMte2BufferNum = 2;
    uint64_t ubMte2InnerSize = 512;
};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};

} // namespace GMMSQWeightQuant
#endif // GMMSQ_BASIC_BLOCK_CONFIG_H
