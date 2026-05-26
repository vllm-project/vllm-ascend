/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sigmoid_gating_delta_rule_update_tiling.cpp
 * \brief
 */
#include "fused_sigmoid_gating_delta_rule_update_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "math_util.h"
#include "error/ops_error.h"
#include <array>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(FusedSigmoidGatingDeltaRuleUpdate, FusedSigmoidGatingDeltaRuleUpdateTiling, 0);

const size_t A_LOG_INDEX = 0;
const size_t A_INDEX = 1;
const size_t B_GATE_INDEX = 2;
const size_t DT_BIAS_INDEX = 3;
const size_t QUERY_INDEX = 4;
const size_t KEY_INDEX = 5;
const size_t VALUE_INDEX = 6;
const size_t STATE_INDEX = 7;
const size_t CUSEQLENS_INDEX = 8;
const size_t SSM_STATE_INDICES_INDEX = 9;
const size_t ACC_TO_INDEX = 10;

const size_t QKV_DIM_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t HEAD_PARAM_DIM_NUM = 1;
const size_t STATE_DIM_NUM = 4;
const size_t CUSEQLENS_DIM_NUM = 1;
const size_t SSM_STATE_INDICES_DIM_NUM = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t MAX_MTP = 8;

template <typename T1, typename T2>
static T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type CeilAlign(T x, T align) {
    return CeilDiv(x, align) * align;
}

void FusedSigmoidGatingDeltaRuleUpdateTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = compileInfo_.aivNum;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::DoOpTiling()
{
    OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
};

uint64_t FusedSigmoidGatingDeltaRuleUpdateTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::GetWorkspaceSize()
{
    // system workspace size is 16 * 1024 * 1024 = 16M;
    constexpr int64_t sysWorkspaceSize = 16777216;
    workspaceSize_ = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(FusedSigmoidGatingDeltaRuleUpdateTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_LOG_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_LOG_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_GATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(B_GATE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(DT_BIAS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(DT_BIAS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CUSEQLENS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CUSEQLENS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSM_STATE_INDICES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(SSM_STATE_INDICES_INDEX));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16"),
                return ge::GRAPH_FAILED);

    auto aLogDtype = context_->GetInputDesc(A_LOG_INDEX)->GetDataType();
    auto aDtype = context_->GetInputDesc(A_INDEX)->GetDataType();
    auto bDtype = context_->GetInputDesc(B_GATE_INDEX)->GetDataType();
    auto dtBiasDtype = context_->GetInputDesc(DT_BIAS_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(aLogDtype != ge::DT_FLOAT || dtBiasDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "A_log and dt_bias dtype should be float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(aDtype != ge::DT_BF16 || bDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "a and b dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_FLOAT && stateDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "state dtype should be bfloat16 or float32"),
                return ge::GRAPH_FAILED);
    auto cuSeqlensDtype = context_->GetInputDesc(CUSEQLENS_INDEX)->GetDataType();
    auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    OP_CHECK_IF(cuSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "cuSeqlens dtype and ssmStateIndices dtype should be int32"),
                return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetDataType();
        OP_CHECK_IF(numAcceptedTokensDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "numAcceptedTokens dtype should be int32"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}


bool FusedSigmoidGatingDeltaRuleUpdateTiling::CheckDimEqual(const gert::Shape a, const int64_t dimA, gert::Shape b, const int64_t dimB,
                                                  const std::string &nameA, const std::string &nameB,
                                                  const std::string &dimDesc)
{
    if (a.GetDim(dimA) != b.GetDim(dimB)) {
        OP_LOGE(context_->GetNodeName(), "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
                dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                b.GetDim(dimB));
        return false;
    }
    return true;
}

bool FusedSigmoidGatingDeltaRuleUpdateTiling::CheckDim(const gert::Shape shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu",
                dimDesc.c_str(), dim, shape.GetDimNum());
        return false;
    }
    return true;
}

// Split shape checks/fill/scheduling decisions to improve readability and maintenance.
ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::CheckShapeDimAndRelation(const gert::Shape &queryShape,
                                                                         const gert::Shape &keyShape,
                                                                         const gert::Shape &valueShape,
                                                                         const gert::Shape &aLogShape,
                                                                         const gert::Shape &aShape,
                                                                         const gert::Shape &bShape,
                                                                         const gert::Shape &dtBiasShape,
                                                                         const gert::Shape &stateShape,
                                                                         const gert::Shape &cuSeqlensShape,
                                                                         const gert::Shape &ssmStateShape)
{
    if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
        !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(aLogShape, HEAD_PARAM_DIM_NUM, "A_log") ||
        !CheckDim(aShape, BETA_DIM_NUM, "a") || !CheckDim(bShape, BETA_DIM_NUM, "b") ||
        !CheckDim(dtBiasShape, HEAD_PARAM_DIM_NUM, "dt_bias") ||
        !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
        !CheckDim(cuSeqlensShape, CUSEQLENS_DIM_NUM, "actual_seq_lengths") ||
        !CheckDim(ssmStateShape, SSM_STATE_INDICES_DIM_NUM, "ssm_state_indices")) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckDimEqual(queryShape, DIM_0, keyShape, DIM_0, "query", "key", "T dimension") ||
        !CheckDimEqual(queryShape, DIM_1, keyShape, DIM_1, "query", "key", "Nk dimension") ||
        !CheckDimEqual(queryShape, DIM_2, keyShape, DIM_2, "query", "key", "Dk dimension") ||
        !CheckDimEqual(stateShape, DIM_1, valueShape, DIM_1, "state", "value", "Nv dimension") ||
        !CheckDimEqual(stateShape, DIM_2, valueShape, DIM_2, "state", "value", "Dv dimension") ||
        !CheckDimEqual(valueShape, DIM_0, queryShape, DIM_0, "value", "query", "T dimension") ||
        !CheckDimEqual(aShape, DIM_0, queryShape, DIM_0, "a", "query", "T dimension") ||
        !CheckDimEqual(aShape, DIM_1, valueShape, DIM_1, "a", "value", "Nv dimension") ||
        !CheckDimEqual(bShape, DIM_0, queryShape, DIM_0, "b", "query", "T dimension") ||
        !CheckDimEqual(bShape, DIM_1, valueShape, DIM_1, "b", "value", "Nv dimension") ||
        !CheckDimEqual(aLogShape, DIM_0, valueShape, DIM_1, "A_log", "value", "Nv dimension") ||
        !CheckDimEqual(dtBiasShape, DIM_0, valueShape, DIM_1, "dt_bias", "value", "Nv dimension") ||
        !CheckDimEqual(stateShape, DIM_3, queryShape, DIM_2, "state", "query", "Dk dimension")) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FusedSigmoidGatingDeltaRuleUpdateTiling::FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape,
                                                         const gert::Shape &stateShape,
                                                         const gert::Shape &cuSeqlensShape)
{
    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
    tilingData_.b = cuSeqlensShape.GetDim(DIM_0) - 1;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::CheckShapeValueRangeAndRule()
{
    OP_CHECK_IF(tilingData_.nk > 256 || tilingData_.nv > 256 || tilingData_.dk > 512 || tilingData_.dv > 512,
                OP_LOGE(inputParams_.opName,
                        "nk and nv should no bigger than 256, dk and dv should no bigger than 512, but nk is %u, nv is "
                        "%u, dk is %u, dv is %u",
                        tilingData_.nk, tilingData_.nv, tilingData_.dk, tilingData_.dv),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.nv % tilingData_.nk != 0,
                OP_LOGE(inputParams_.opName,
                        "nv should be an integer multiple of nk, but nv is %u, nk is %u",
                        tilingData_.nv, tilingData_.nk),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FusedSigmoidGatingDeltaRuleUpdateTiling::UpdateDynamicBlockDimByTaskUnits()
{
    // Dynamic blockDim: do not launch more cores than effective (batch, head) task units.
    uint64_t taskUnits = static_cast<uint64_t>(tilingData_.b) * static_cast<uint64_t>(tilingData_.nv);
    if (taskUnits == 0) {
        taskUnits = 1;
    }
    uint64_t maxCoreNum = (compileInfo_.aivNum > 0) ? compileInfo_.aivNum : 1;
    uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
    tilingData_.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);
    OP_LOGD(context_->GetNodeName(), "taskUnits: [%llu], selected vectorCoreNum: [%u]",
            static_cast<unsigned long long>(taskUnits), tilingData_.vectorCoreNum);
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCheckShapeDimAndRelation()
{
    const auto &aLogShape = context_->GetInputShape(A_LOG_INDEX)->GetOriginShape();
    const auto &aShape = context_->GetInputShape(A_INDEX)->GetOriginShape();
    const auto &bShape = context_->GetInputShape(B_GATE_INDEX)->GetOriginShape();
    const auto &dtBiasShape = context_->GetInputShape(DT_BIAS_INDEX)->GetOriginShape();
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    const auto &ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
    return CheckShapeDimAndRelation(queryShape, keyShape, valueShape, aLogShape, aShape, bShape, dtBiasShape,
                                    stateShape, cuSeqlensShape, ssmStateShape);
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleFillTilingShapeData()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    FillTilingShapeData(queryShape, valueShape, stateShape, cuSeqlensShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCheckShapeValueRangeAndRule()
{
    return CheckShapeValueRangeAndRule();
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleUpdateDynamicBlockDimByTaskUnits()
{
    UpdateDynamicBlockDimByTaskUnits();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleInitUbCalcContext()
{
    ubCalcCtx_.ubSize = compileInfo_.ubSize;
    ubCalcCtx_.aNv = CeilAlign(tilingData_.nv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    ubCalcCtx_.aDv = CeilAlign(tilingData_.dv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    ubCalcCtx_.aDk = CeilAlign(tilingData_.dk, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcFixedUbBytes()
{
    ubCalcCtx_.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx_.aNv, ubCalcCtx_.aDv, ubCalcCtx_.aDk);
    tilingData_.ubRestBytes = ubCalcCtx_.ubSize - ubCalcCtx_.fixedUbBytes;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcWorkingUbBytes()
{
    ubCalcCtx_.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx_.aNv, ubCalcCtx_.aDv, ubCalcCtx_.aDk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcVStepCoeff()
{
    ubCalcCtx_.coeff = CalcVStepCoeff(ubCalcCtx_.aDk, 1, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::RuleFinalizeVStepFromUb()
{
    return FinalizeVStepFromUb(ubCalcCtx_.ubSize, ubCalcCtx_.workingUbBytes, ubCalcCtx_.coeff);
}

// AnalyzeShapes now executes a deterministic rule-chain, easier to extend/maintain.
ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::AnalyzeShapes()
{
    struct RuleItem {
        const char *name;
        HostRuleFn fn;
    };
    const std::array<RuleItem, 4> shapeRules = {{
        {"RuleCheckShapeDimAndRelation", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCheckShapeDimAndRelation},
        {"RuleFillTilingShapeData", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleFillTilingShapeData},
        {"RuleCheckShapeValueRangeAndRule", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCheckShapeValueRangeAndRule},
        {"RuleUpdateDynamicBlockDimByTaskUnits", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleUpdateDynamicBlockDimByTaskUnits},
    }};
    for (const auto &rule : shapeRules) {
        OP_CHECK_IF((this->*(rule.fn))() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "AnalyzeShapes rule failed: %s", rule.name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}


bool FusedSigmoidGatingDeltaRuleUpdateTiling::CheckFormat(ge::Format format, const std::string &Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
        return false;
    }
    return true;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(A_LOG_INDEX)->GetStorageFormat(), "A_log") ||
        !CheckFormat(context_->GetInputDesc(A_INDEX)->GetStorageFormat(), "a") ||
        !CheckFormat(context_->GetInputDesc(B_GATE_INDEX)->GetStorageFormat(), "b") ||
        !CheckFormat(context_->GetInputDesc(DT_BIAS_INDEX)->GetStorageFormat(), "dt_bias") ||
        !CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
        !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "actual_seq_lengths") ||
        !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "ssm_state_indices")) {
        return ge::GRAPH_FAILED;
    }

    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetStorageFormat();
        OP_CHECK_IF(numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_->GetNodeName(), "numAcceptedTokens format not support NZ"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    float scaleValue = *attrs->GetAttrPointer<float>(0);
    float softplusBeta = *attrs->GetAttrPointer<float>(1);
    float softplusThreshold = *attrs->GetAttrPointer<float>(2);
    tilingData_.scale = scaleValue;
    tilingData_.softplusBeta = softplusBeta;
    tilingData_.softplusThreshold = softplusThreshold;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::GetOptionalInput()
{
    tilingData_.hasGama = 1;
    tilingData_.hasGamaK = 0;
    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) == nullptr) {
        tilingData_.hasAcceptedTokens = 0;
    } else {
        tilingData_.hasAcceptedTokens = 1;
    }

    return ge::GRAPH_SUCCESS;
}

void FusedSigmoidGatingDeltaRuleUpdateTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "softplusBeta: [%f]", tilingData_.softplusBeta);
    OP_LOGD(context_->GetNodeName(), "softplusThreshold: [%f]", tilingData_.softplusThreshold);
    OP_LOGD(context_->GetNodeName(), "hasGama: [%u]", tilingData_.hasGama);
    OP_LOGD(context_->GetNodeName(), "hasGamaK: [%u]", tilingData_.hasGamaK);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
}

int64_t FusedSigmoidGatingDeltaRuleUpdateTiling::CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const
{
    int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv); // 4 for qInQueue_ & kInQueue_, 2 for vInQueue_
    usedUbBytes += 128;                                  // reserve 128 Bytes
    usedUbBytes += MAX_MTP * 2 * aNv; // aInQueue_
    usedUbBytes += MAX_MTP * 2 * aNv; // bInQueue_
    usedUbBytes += MAX_MTP * 4 * aNv; // fused gamaInQueue_
    usedUbBytes += 2 * 4 * aNv;       // A_log and dt_bias queues
    return usedUbBytes;
}

int64_t FusedSigmoidGatingDeltaRuleUpdateTiling::CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const
{
    int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
    usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv); // 8 for qk in ub, 4 for v in ub, 4 for beta in ub
    return usedUbBytes;
}

int64_t FusedSigmoidGatingDeltaRuleUpdateTiling::CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum,
                                                       uint32_t attnOutBufferNum) const
{
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    int64_t stateDtypeSize = (stateDtype == ge::DT_FLOAT) ? 4 : 2;
    int64_t coeff = (stateDtypeSize + static_cast<int64_t>(stateDtypeSize * stateOutBufferNum)) * aDk +
                    static_cast<int64_t>(4 * attnOutBufferNum); // stateIn/stateOut/attnOut queues
    coeff += (4 + 4) * aDk + 4 + 4;                             // qInUb/kInUb/vInUb/deltaInUb/attnInUb
    return coeff;
}

bool FusedSigmoidGatingDeltaRuleUpdateTiling::EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk,
                                                           uint32_t stateOutBufferNum, uint32_t attnOutBufferNum,
                                                           BufferProfile &profile) const
{
    int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8; // 8 * sizeof(float) = 32
    if (vStep < 8) {
        return false;
    }
    int64_t repeatTime = CeilDiv(tilingData_.dv, static_cast<uint32_t>(vStep));
    vStep = CeilAlign(CeilDiv(tilingData_.dv, static_cast<uint32_t>(repeatTime)),
                                 static_cast<uint32_t>(8));
    if (vStep < 8) {
        return false;
    }
    profile.stateOutBufferNum = stateOutBufferNum;
    profile.attnOutBufferNum = attnOutBufferNum;
    profile.vStep = static_cast<uint32_t>(vStep);
    profile.repeatTime = static_cast<uint32_t>(repeatTime);
    profile.valid = true;
    return true;
}

bool FusedSigmoidGatingDeltaRuleUpdateTiling::IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) const
{
    if (!current.valid) {
        return true;
    }
    if (candidate.repeatTime != current.repeatTime) {
        return candidate.repeatTime < current.repeatTime;
    }
    uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
    uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
    if (candidateDepth != currentDepth) {
        return candidateDepth > currentDepth;
    }
    return candidate.vStep > current.vStep;
}

ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::FinalizeVStepFromUb(int64_t ubSize, int64_t usedUbBytes, int64_t coeff)
{
    (void)coeff;
    int64_t aDk = CeilAlign(tilingData_.dk, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    BufferProfile selected;
    const std::array<BufferProfile, 4> candidates = {{
        BufferProfile(1u, 1u, 0u, 0u, false),
        BufferProfile(1u, 2u, 0u, 0u, false),
        BufferProfile(2u, 2u, 0u, 0u, false),
        BufferProfile(3u, 3u, 0u, 0u, false)
    }};
    for (const auto &candidate : candidates) {
        BufferProfile profile;
        if (!EvaluateBufferProfile(ubSize, usedUbBytes, aDk, candidate.stateOutBufferNum, candidate.attnOutBufferNum,
                                   profile)) {
            continue;
        }
        if (IsBetterProfile(profile, selected)) {
            selected = profile;
        }
    }

    OP_LOGD(context_->GetNodeName(), "selected profile: stateOutBufferNum=[%u], attnOutBufferNum=[%u], vStep=[%u], repeatTime=[%u], valid=[%d]",
            selected.stateOutBufferNum, selected.attnOutBufferNum, selected.vStep, selected.repeatTime, selected.valid);

    if (!selected.valid) {
        OP_LOGE(context_->GetNodeName(), "vStep should be bigger than 8, shape is too big");
        return ge::GRAPH_FAILED;
    }
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();

    int64_t stateDtypeSize = (stateDtype == ge::DT_FLOAT) ? 4 : 2;

    int64_t queueCoeff = (stateDtypeSize + static_cast<int64_t>(stateDtypeSize * selected.stateOutBufferNum)) * aDk +
                         static_cast<int64_t>(4 * selected.attnOutBufferNum);
    int64_t ubRestBytes = ubSize - ubCalcCtx_.fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
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

// CalUbSize now runs an ordered UB rule-chain with explicit intermediate states.
ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTiling::CalUbSize()
{
    struct RuleItem {
        const char *name;
        HostRuleFn fn;
    };
    const std::array<RuleItem, 5> ubRules = {{
        {"RuleInitUbCalcContext", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleInitUbCalcContext},
        {"RuleCalcFixedUbBytes", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcFixedUbBytes},
        {"RuleCalcWorkingUbBytes", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcWorkingUbBytes},
        {"RuleCalcVStepCoeff", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleCalcVStepCoeff},
        {"RuleFinalizeVStepFromUb", &FusedSigmoidGatingDeltaRuleUpdateTiling::RuleFinalizeVStepFromUb},
    }};
    for (const auto &rule : ubRules) {
        OP_CHECK_IF((this->*(rule.fn))() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "CalUbSize rule failed: %s", rule.name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedSigmoidGatingDeltaRuleUpdateTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("FusedSigmoidGatingDeltaRuleUpdate", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForFusedSigmoidGatingDeltaRuleUpdate(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("FusedSigmoidGatingDeltaRuleUpdate", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<FusedSigmoidGatingDeltaRuleUpdateCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedSigmoidGatingDeltaRuleUpdate)
    .Tiling(FusedSigmoidGatingDeltaRuleUpdateTilingFunc)
    .TilingParse<FusedSigmoidGatingDeltaRuleUpdateCompileInfo>(TilingPrepareForFusedSigmoidGatingDeltaRuleUpdate);
} // namespace optiling
