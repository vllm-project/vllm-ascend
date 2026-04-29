/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "gumbel_sample_tiling.h"

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// vocab 维分块大小：4096 fp32 = 16KB，远低于 UB；4096/64=64 ≤ 255 满足 RepeatTime 约束
constexpr uint32_t GUMBEL_SAMPLE_BLOCK_SIZE = 4096;

static ge::graphStatus GumbelSampleTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 优先从 CompileInfo 取核数（编译期缓存），回退到实时查询
    uint32_t aivCoreNum = 0;
    auto ptrCompileInfo = reinterpret_cast<const GumbelSampleCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo != nullptr && ptrCompileInfo->totalCoreNum > 0) {
        aivCoreNum = ptrCompileInfo->totalCoreNum;
    } else {
        auto platformInfo = context->GetPlatformInfo();
        if (platformInfo == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    }
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto logitsShapeBundle = context->GetInputShape(0);
    if (logitsShapeBundle == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto logitsShape = logitsShapeBundle->GetStorageShape();
    if (logitsShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }

    int64_t numReqsI64   = logitsShape.GetDim(0);
    int64_t vocabSizeI64 = logitsShape.GetDim(1);
    if (numReqsI64 <= 0 || vocabSizeI64 <= 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numReqs   = static_cast<uint32_t>(numReqsI64);
    uint32_t vocabSize = static_cast<uint32_t>(vocabSizeI64);

    uint32_t usedCoreNum = (numReqs < aivCoreNum) ? numReqs : aivCoreNum;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t formerNum   = numReqs % usedCoreNum;
    uint32_t nRowsLarge  = CeilDivU32(numReqs, usedCoreNum);
    uint32_t nRowsSmall  = numReqs / usedCoreNum;
    uint32_t numTiles    = CeilDivU32(vocabSize, GUMBEL_SAMPLE_BLOCK_SIZE);
    uint32_t lastTileLen = vocabSize - (numTiles - 1) * GUMBEL_SAMPLE_BLOCK_SIZE;

    // apply_temperature 属性（bool，默认 true）
    uint32_t applyTemp = 1u;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* applyTempAttr = attrs->GetAttrPointer<bool>(0);
        if (applyTempAttr != nullptr) {
            applyTemp = (*applyTempAttr) ? 1u : 0u;
        }
    }

    GumbelSampleTilingData tiling;
    tiling.set_numReqs(numReqs);
    tiling.set_vocabSize(vocabSize);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_formerNum(formerNum);
    tiling.set_nRowsLarge(nRowsLarge);
    tiling.set_nRowsSmall(nRowsSmall);
    tiling.set_blockSize(GUMBEL_SAMPLE_BLOCK_SIZE);
    tiling.set_numTiles(numTiles);
    tiling.set_lastTileLen(lastTileLen);
    tiling.set_applyTemp(applyTemp);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(usedCoreNum);
    // TilingKey 与 op_kernel/gumbel_sample_tiling_key.h 中 ASCENDC_TPL_SEL 声明一致：0/1
    context->SetTilingKey(applyTemp);

    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GumbelSample(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GumbelSampleCompileInfo>();
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GumbelSample)
    .Tiling(GumbelSampleTilingFunc)
    .TilingParse<GumbelSampleCompileInfo>(TilingPrepare4GumbelSample);

}  // namespace optiling
