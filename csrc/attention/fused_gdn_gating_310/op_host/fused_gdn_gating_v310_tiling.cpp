/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310_tiling.cpp
 * \brief
 */
#include <cstdint>
#include <limits>
#include "fused_gdn_gating_v310_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static constexpr uint32_t HW_ALIGN_BYTES = 16;
static constexpr uint32_t DEFAULT_NUM_HEADS = 16;
static constexpr uint32_t DEFAULT_CORE_NUM = 1;
static constexpr uint32_t TARGET_CORE_NUM_SMALL = 1;
static constexpr uint32_t TARGET_CORE_NUM_MID = 4;
static constexpr uint32_t ELEMENT_THRESHOLD_SMALL = 2048;
static constexpr uint32_t ELEMENT_THRESHOLD_MID = 16384;
static constexpr uint32_t TARGET_TILE_ELEMENTS = 3072;
static constexpr float DEFAULT_BETA = 1.0f;
static constexpr float DEFAULT_THRESHOLD = 20.0f;
static constexpr int ATTR_INDEX_BETA = 0;
static constexpr int ATTR_INDEX_THRESHOLD = 1;
static constexpr int INPUT_INDEX_A = 0;

static uint32_t gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    return (a == 0) ? 1 : a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FusedGdnGatingV310TilingData tiling;
    auto shape = context->GetInputShape(INPUT_INDEX_A)->GetOriginShape();
    int64_t batch = (shape.GetDimNum() > 0) ? shape.GetDim(0) : 1;
    int64_t numHeads = (shape.GetDimNum() > 1) ? shape.GetDim(1) : DEFAULT_NUM_HEADS;
    uint32_t totalElements = static_cast<uint32_t>(batch * numHeads);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t hwCoreNum = ascendcPlatform.GetCoreNum();
    if (hwCoreNum == 0) {
        hwCoreNum = DEFAULT_CORE_NUM;
    }

    uint32_t baseBlock = (HW_ALIGN_BYTES * numHeads) / gcd(HW_ALIGN_BYTES, numHeads);

    uint32_t targetCoreNum = hwCoreNum;
    if (totalElements <= ELEMENT_THRESHOLD_SMALL) {
        targetCoreNum = TARGET_CORE_NUM_SMALL;
    } else if (totalElements <= ELEMENT_THRESHOLD_MID) {
        targetCoreNum = TARGET_CORE_NUM_MID;
    }

    if (totalElements <= baseBlock) {
        targetCoreNum = TARGET_CORE_NUM_SMALL;
    }

    uint32_t elementsPerCore = (totalElements + targetCoreNum - 1) / targetCoreNum;
    elementsPerCore = ((elementsPerCore + baseBlock - 1) / baseBlock) * baseBlock;
    if (elementsPerCore == 0) {
        elementsPerCore = baseBlock;
    }

    uint32_t usedCores = (totalElements + elementsPerCore - 1) / elementsPerCore;
    if (usedCores > targetCoreNum) {
        usedCores = targetCoreNum;
    }

    uint32_t tailElements = totalElements - (usedCores - 1) * elementsPerCore;

    uint32_t tileElements = TARGET_TILE_ELEMENTS;
    tileElements = (tileElements / baseBlock) * baseBlock;
    if (tileElements == 0) {
        tileElements = baseBlock;
    }
    if (tileElements > elementsPerCore) {
        tileElements = elementsPerCore;
    }

    float beta = DEFAULT_BETA;
    float threshold = DEFAULT_THRESHOLD;

    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* betaPtr = attrs->GetFloat(ATTR_INDEX_BETA);
        const float* thresholdPtr = attrs->GetFloat(ATTR_INDEX_THRESHOLD);
        if (betaPtr != nullptr) {
            beta = *betaPtr;
        }
        if (thresholdPtr != nullptr) {
            threshold = *thresholdPtr;
        }
    }

    tiling.set_usedCoreNum(usedCores);
    tiling.set_alignedLength(elementsPerCore);
    tiling.set_tailLength(tailElements);
    tiling.set_numHeads(static_cast<uint32_t>(numHeads));
    tiling.set_tileRows(tileElements);
    tiling.set_beta(beta);
    tiling.set_threshold(threshold);

    if (beta == 0.0f) {
        tiling.set_inv_beta(std::numeric_limits<float>::infinity());
    } else {
        tiling.set_inv_beta(1.0f / beta);
    }

    context->SetBlockDim(usedCores);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FusedGdnGatingV310)
    .Tiling(TilingFunc);
} // namespace optiling