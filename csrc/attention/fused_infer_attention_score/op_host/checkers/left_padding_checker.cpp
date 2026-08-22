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
 * \file left_padding_checker.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "left_padding_checker.h"

namespace optiling {
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// Utility functions for common checkers.
ge::graphStatus LeftPaddingChecker::CheckShapeSupport(const gert::Tensor *tensor,
                                                      const std::vector<int64_t> &expectShapeList) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeList.begin(), expectShapeList.end(), tensor->GetShapeSize()) == expectShapeList.end()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// CheckSinglePara
ge::graphStatus LeftPaddingChecker::CheckShapeAndDim(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value,
    // the Shape size and Dim number corresponding to the Padding size must both be 1.
    if (fiaInfo.qPaddingSizeFlag) {
        const std::vector<int64_t> querypaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckShapeSupport(fiaInfo.opParamInfo.queryPaddingSize.tensor,
            querypaddingsizeShapeNumList)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(fiaInfo.opName, "query_padding_size",
                std::to_string(fiaInfo.opParamInfo.queryPaddingSize.tensor->GetShapeSize()).c_str(), "1");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(fiaInfo.opParamInfo.queryPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(fiaInfo.opName, "The dim number of query paddingsize is not 1!"), return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag) {
        const std::vector<int64_t> kvpaddingsizeShapeNumList = {SHAPE_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckShapeSupport(fiaInfo.opParamInfo.kvPaddingSize.tensor,
            kvpaddingsizeShapeNumList)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(fiaInfo.opName, "kv_padding_size",
                std::to_string(fiaInfo.opParamInfo.kvPaddingSize.tensor->GetShapeSize()).c_str(), "1");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(fiaInfo.opParamInfo.kvPaddingSize.tensor->GetStorageShape().GetDimNum() != DIM_NUM_1,
                    OP_LOGE(fiaInfo.opName, "The dim number of key/value paddingsize is not 1!"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// CheckExistence
ge::graphStatus LeftPaddingChecker::CheckExistenceDesc(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.qPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.opParamInfo.queryPaddingSize.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                        "queryPaddingSize", "desc of queryPaddingSize cannot be empty"),
                    return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.opParamInfo.kvPaddingSize.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                        "kvPaddingSize", "desc of kvPaddingSize cannot be empty"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus LeftPaddingChecker::CheckFeatureActualLen(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, the actual sequence length information must be provided.
    if (fiaInfo.qPaddingSizeFlag && !fiaInfo.isMaxWorkspace) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengthsQ.tensor == nullptr ||
                        fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize() == 0U,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsQ",
                        "actualSeqLengthsQ cannot be empty when query_padding_size exists"),
                    return ge::GRAPH_FAILED);
    }

    if (fiaInfo.kvPaddingSizeFlag && !fiaInfo.isMaxWorkspace) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
                        fiaInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0U,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsKv",
                        "actualSeqLengthsKv cannot be empty when kv_padding_size exists"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureLayout(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD scenarios are not supported.
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        if (layoutStr == "BSH_BNSD") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "BSH_BNSD",
                "The value of input_layout cannot be BSH_BNSD when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
        if (layoutStr == "BSND_BNSD") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "BSND_BNSD",
                "The value of input_layout cannot be BSND_BNSD when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
        if (layoutStr == "TND") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "TND",
                "The value of input_layout cannot be TND when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
        if (layoutStr == "NTD") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "NTD",
                "The value of input_layout cannot be NTD when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
        if (layoutStr == "NTD_TND") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "NTD_TND",
                "The value of input_layout cannot be NTD_TND when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
        if (layoutStr == "TND_NTD") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "TND_NTD",
                "The value of input_layout cannot be TND_NTD when leftpadding is enabled");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureAlibiPse(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, PseType = 2/3 is not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.enableAlibiPse,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "leftPadding",
                        "leftPadding must be empty when pseType = 2/3"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeaturePageAttention(const FiaTilingInfo &fiaInfo)
{
    // When left-padding is enabled for Query and Key/Value, PageAttention scenarios are not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "leftPadding",
                        "leftPadding must be empty when page attention is enabled"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckFeatureQueryS(const FiaTilingInfo &fiaInfo)
{
    // When antiquantMode is 0 or 1 and data type of key/value is int8 scenario, leftpadding is not supported.
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        if (fiaInfo.s1Size > 1) {
            int64_t keyAntiquantMode = 0;
            if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
                keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
            }
            OP_CHECK_IF(
                (keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                    fiaInfo.inputKvType == ge::DT_INT8,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "leftPadding",
                    "leftPadding must be empty in keyAntiquant/valueAntiquant split mode and data type of "
                    "key/value is int8 scenario, when keyAntiquantMode/valueAntiquantMode is 0 or 1"),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckShapeAndDim(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceDesc(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureActualLen(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureLayout(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureAlibiPse(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeaturePageAttention(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureQueryS(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag) {
        OP_CHECK_IF(
            fiaInfo.isQKVDDifferent && fiaInfo.ropeMode == RopeMode::NO_ROPE,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "leftPadding",
                "leftPadding must be empty in the GQA scenario, when D of query and key is not equal to D of value"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeftPaddingChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
