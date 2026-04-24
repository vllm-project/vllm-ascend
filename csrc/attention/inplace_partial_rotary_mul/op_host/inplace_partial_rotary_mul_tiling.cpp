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
 * \file rotary_position_embedding.cc
 * \brief
 */
#include "inplace_partial_rotary_mul_tiling.h"
#include "register/op_def_registry.h"
// #include "log/log.h"
#include "tiling/tiling_api.h"
// #include "tiling_base/tiling_templates_registry.h"
#include <vector>
namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        socVersion_ = ascendcPlatform.GetSocVersion();
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const RotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
    const uint32_t inputMode = *(attrs->GetAttrPointer<uint32_t>(MODE_ATTR_IDX));
    OPS_LOG_I(context_->GetNodeName(), "[mode]: %d", inputMode);
    inputMode_ = inputMode;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RotaryPositionEmbedding(gert::TilingContext *context)
{
    OPS_LOG_I(context, "Tiling4RotaryPositionEmbedding start");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4RotaryPositionEmbedding", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4RotaryPositionEmbedding", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95)
    {
        std::vector<std::unique_ptr<RopeRegBaseTilingClass>> regBaseTilingCases;
        regBaseTilingCases.push_back(std::unique_ptr<RopeRegBaseTilingClass>(new RopeRegBaseTilingClassAAndB(context)));
        regBaseTilingCases.push_back(std::unique_ptr<RopeRegBaseTilingClass>(new RopeRegBaseTilingClassAB(context)));
        regBaseTilingCases.push_back(std::unique_ptr<RopeRegBaseTilingClass>(new RopeRegBaseTilingClassABAAndBA(context)));
        regBaseTilingCases.push_back(std::unique_ptr<RopeRegBaseTilingClass>(new RopeRegBaseTilingClassBAB(context)));
        OPS_LOG_I(context, "Using arch35 tiling for ASCEND910_95");
        
        for (const auto& ptr : regBaseTilingCases)
        {
            if (ptr)
            {
                ge::graphStatus status = ptr->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID)
                {
                    OPS_LOG_I(context, "Do general op tiling success priority");
                    return status;
                }
                OPS_LOG_I(context, "Ignore general op tiling priority");
            }
        }
        OPS_LOG_I(context, "Using tiling for ASCEND910_71");
        RotaryPosEmbeddingMembaseTilingClass rotaryPosEmbeddingMembaseTilingClass(context);
        return rotaryPosEmbeddingMembaseTilingClass.DoOpTiling();
    } else {
        return Tiling4InplacePartialRotaryMul(context);
    }
}

ge::graphStatus TilingPrepareForRotaryPositionEmbedding(gert::TilingParseContext *context)
{
    OPS_LOG_I(context, "TilingPrepareForRotaryPositionEmbedding context success");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMul)
    .Tiling(Tiling4RotaryPositionEmbedding)
    .TilingParse<RotaryPositionEmbeddingCompileInfo>(TilingPrepareForRotaryPositionEmbedding);
} // namespace optiling

