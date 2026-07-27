/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling_arch35.cpp
 * \brief
 */
#include "recurrent_gated_delta_rule_tiling.h"

#include <array>

#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr uint64_t RGDR_ASCEND_950_TEMPLATE_PRIORITY = 1000;

constexpr size_t QUERY_INDEX = 0;
constexpr size_t KEY_INDEX = 1;
constexpr size_t VALUE_INDEX = 2;
constexpr size_t BETA_INDEX = 3;
constexpr size_t STATE_INDEX = 4;
constexpr size_t CUSEQLENS_INDEX = 5;
constexpr size_t SSM_STATE_INDICES_INDEX = 6;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;

class RecurrentGatedDeltaRuleTilingArch35 final : public RecurrentGatedDeltaRuleTiling {
public:
    explicit RecurrentGatedDeltaRuleTilingArch35(gert::TilingContext *context)
        : RecurrentGatedDeltaRuleTiling(context)
    {
    }

protected:
    bool IsCapable() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            return false;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        return ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeShapesArch35() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                    return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        OP_CHECK_IF(CalUbSizeArch35() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
                    return ge::GRAPH_FAILED);

        PrintTilingData();
        return ge::GRAPH_SUCCESS;
    }

private:
    ge::graphStatus AnalyzeShapesArch35()
    {
        const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
        const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
        const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
        const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
        const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
        const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
        const auto &ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();

        OP_CHECK_IF(CheckShapeDimAndRelation(queryShape, keyShape, valueShape, betaShape, stateShape, cuSeqlensShape,
                                             ssmStateShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "AnalyzeShapes rule failed: CheckShapeDimAndRelation"),
                    return ge::GRAPH_FAILED);

        tilingData_.t = queryShape.GetDim(DIM_0);
        tilingData_.nk = queryShape.GetDim(DIM_1);
        tilingData_.dk = queryShape.GetDim(DIM_2);
        tilingData_.nv = valueShape.GetDim(DIM_1);
        tilingData_.dv = valueShape.GetDim(DIM_2);
        tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
        tilingData_.b = cuSeqlensShape.GetDim(DIM_0) - 1;

        OP_CHECK_IF(CheckShapeValueRangeAndRule() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "AnalyzeShapes rule failed: CheckShapeValueRangeAndRule"),
                    return ge::GRAPH_FAILED);

        UpdateDynamicBlockDimByTaskUnits();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CalUbSizeArch35()
    {
        struct RuleItem {
            const char *name;
            HostRuleFn fn;
        };

        OP_CHECK_IF(RuleInitUbCalcContext() != ge::GRAPH_SUCCESS,
            OP_LOGE(inputParams_.opName, "CalUbSize rule failed: RuleInitUbCalcContext"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(RuleCalcFixedUbBytes() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalUbSize rule failed: RuleCalcFixedUbBytes"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(RuleCalcWorkingUbBytes() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalUbSize rule failed: RuleCalcWorkingUbBytes"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(RuleCalcVStepCoeff() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalUbSize rule failed: RuleCalcVStepCoeff"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(FinalizeVStepFromUbArch35() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalUbSize rule failed: FinalizeVStepFromUbArch35"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus FinalizeVStepFromUbArch35()
    {
        BufferProfile selected;
        const std::array<BufferProfile, 3> candidates = {{
            BufferProfile(1u, 1u, 0u, 0u, false),
            BufferProfile(1u, 2u, 0u, 0u, false),
            BufferProfile(2u, 2u, 0u, 0u, false),
        }};

        for (const auto &candidate : candidates) {
            BufferProfile profile;
            if (!EvaluateBufferProfile(ubCalcCtx_.ubSize, ubCalcCtx_.workingUbBytes, ubCalcCtx_.aDk,
                                       candidate.stateOutBufferNum, candidate.attnOutBufferNum, profile)) {
                continue;
            }
            if (IsBetterProfile(profile, selected)) {
                selected = profile;
            }
        }

        OP_LOGD(context_->GetNodeName(),
                "selected profile: stateOutBufferNum=[%u], attnOutBufferNum=[%u], vStep=[%u], repeatTime=[%u], "
                "valid=[%d]",
                selected.stateOutBufferNum, selected.attnOutBufferNum, selected.vStep, selected.repeatTime,
                selected.valid);

        if (!selected.valid) {
            OP_LOGE(context_->GetNodeName(), "vStep should be bigger than 8, shape is too big");
            return ge::GRAPH_FAILED;
        }

        auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
        int64_t stateDtypeSize = (stateDtype == ge::DT_FLOAT) ? 4 : 2;
        int64_t queueCoeff =
            (stateDtypeSize + static_cast<int64_t>(stateDtypeSize * selected.stateOutBufferNum)) * ubCalcCtx_.aDk +
            static_cast<int64_t>(4 * selected.attnOutBufferNum);
        int64_t ubRestBytes =
            ubCalcCtx_.ubSize - ubCalcCtx_.fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
        if (ubRestBytes < 0) {
            OP_LOGE(context_->GetNodeName(), "ubRestBytes should be non-negative, but got %ld", ubRestBytes);
            return ge::GRAPH_FAILED;
        }

        tilingData_.ubCalSize = compileInfo_.ubSize;
        tilingData_.vStep = selected.vStep;
        tilingData_.stateOutBufferNum = selected.stateOutBufferNum;
        tilingData_.attnOutBufferNum = selected.attnOutBufferNum;
        tilingData_.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule,
                             RecurrentGatedDeltaRuleTilingArch35,
                             RGDR_ASCEND_950_TEMPLATE_PRIORITY);

} // namespace optiling
