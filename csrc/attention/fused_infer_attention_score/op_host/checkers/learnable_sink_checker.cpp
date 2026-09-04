/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file learnable_sink_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "learnable_sink_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// check sink dtype
ge::graphStatus LearnableSinkChecker::CheckSinkDtypeSupport(const FiaTilingInfo &fiaInfo)
{
    const gert::CompileTimeTensorDesc *learnableSinkDesc = fiaInfo.opParamInfo.learnableSink.desc;
    if (learnableSinkDesc != nullptr) {
        if (fiaInfo.npuArch == NpuArch::DAV_3510) {
            if (learnableSinkDesc->GetDataType() != ge::DT_FLOAT16 &&
                learnableSinkDesc->GetDataType() != ge::DT_BF16) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "learnable_sink",
                    ToString(learnableSinkDesc->GetDataType()).c_str(),
                    "The dtype of learnable_sink must be BF16 or FLOAT16 when learnable sink is enabled");
                return ge::GRAPH_FAILED;
            }

            if (learnableSinkDesc->GetDataType() != fiaInfo.inputQType) {
                std::string reason = "The dtype of learnable_sink must be the same as the dtype of query(" +
                    ToString(fiaInfo.inputQType) + ") when learnable sink is enabled";
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "learnable_sink",
                    ToString(learnableSinkDesc->GetDataType()).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            if (learnableSinkDesc->GetDataType() != ge::DT_BF16) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "learnable_sink",
                    ToString(learnableSinkDesc->GetDataType()).c_str(),
                    "The dtype of learnable_sink must be BF16 when learnable sink is enabled");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus LearnableSinkChecker::CheckFeatureSupport(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }
    // sink 不支持左padding
    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "leftPadding",
            "leftPadding must be empty when learnable sink is enabled"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when learnable sink is enabled"),
        return ge::GRAPH_FAILED);

    if (fiaInfo.enableAlibiPse) {
        std::string reason = "The value of pse_type cannot be 2/3 when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type",
            "2/3", reason.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
            "pseShift must be empty when learnable sink is enabled"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.isOutQuantEnable,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "quantScale2",
            "quantScale2 must be empty when learnable sink is enabled"),
        return ge::GRAPH_FAILED);

    if (fiaInfo.innerPrecise != HIGH_PRECISION) {
        std::string reason = "The value of inner_precise must be " + std::to_string(HIGH_PRECISION) +
            " when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inner_precise",
            std::to_string(fiaInfo.innerPrecise).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    // 仅支持GQA 非量化
    OP_CHECK_IF(!(enableNonQuant_ && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "learnableSink",
            "learnableSink must be empty when not in the no-quantized GQA mode"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check sink shape
ge::graphStatus LearnableSinkChecker::CheckSinkShapeSupport(const FiaTilingInfo &fiaInfo)
{
    uint32_t sinkDimNum = fiaInfo.opParamInfo.learnableSink.tensor->GetStorageShape().GetDimNum();
    uint32_t sinkDim0 = fiaInfo.opParamInfo.learnableSink.tensor->GetStorageShape().GetDim(0);
    if (sinkDimNum != 1) {
        std::string reason = "The shape dim of learnable_sink must be equal to 1 when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "learnable_sink",
            std::to_string(sinkDimNum).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (sinkDim0 != fiaInfo.n1Size) {
        std::string reason = "The shape of learnable_sink must be equal to axis N(" + std::to_string(fiaInfo.n1Size) +
            ") of query when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "learnable_sink",
            ToStringRaw(fiaInfo.opParamInfo.learnableSink.tensor->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check axis
ge::graphStatus LearnableSinkChecker::CheckAxisSupport(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.qkHeadDim != NUM_192 && fiaInfo.qkHeadDim != NUM_128 && fiaInfo.qkHeadDim != NUM_64) {
        std::string reason = "D of query must be in the range of {64, 128, 192} when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.vHeadDim != NUM_128 && fiaInfo.vHeadDim != NUM_64) {
        std::string reason = "D of value must be in the range of {64, 128} when learnable sink is enabled";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
            ToStringRaw(fiaInfo.opParamInfo.value.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckSinkDtypeSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSinkShapeSupport(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if ((ge::GRAPH_SUCCESS != CheckFeatureSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckAxisSupport(fiaInfo)) && fiaInfo.npuArch == NpuArch::DAV_3510) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
