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
 * \file pse_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "pse_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

constexpr int64_t PSE_OUTER_MUL_ADD_TYPE = 0;
constexpr int64_t PSE_OUTER_ADD_MUL_TYPE = 1;

// singlepara
ge::graphStatus PSEChecker::CheckPseType(const FiaTilingInfo &fiaInfo)
{
    // 校验pseType的合法性
    if (fiaInfo.opParamInfo.pseType == nullptr) {
        // 若pseType不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t pseType = *fiaInfo.opParamInfo.pseType;
    // pseType支持范围为0
    if (pseType != PSE_OUTER_MUL_ADD_TYPE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type",
            std::to_string(pseType).c_str(), "The value of pse_type must be 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftDataType(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的数据类型
    if (!fiaInfo.pseShiftFlag) {
        // 若pseType或者pseShift的数据类型不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    auto &pseShiftDesc = fiaInfo.opParamInfo.pseShift.desc;
    // query的datatype为FLOAT16、INT8时，pseShift的datatype应为FLOAT16
    if (fiaInfo.inputQType == ge::DT_FLOAT16 || fiaInfo.inputQType == ge::DT_INT8) {
        if (pseShiftDesc->GetDataType() != ge::DT_FLOAT16) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                ToString(pseShiftDesc->GetDataType()).c_str(),
                "The dtype of pse_shift must be FLOAT16 when the dtype of query "
                "is FLOAT16 or INT8 and pse_type is 0");
            return ge::GRAPH_FAILED;
        }
    }
    // query的datatype为BFLOAT16时，pseShift的datatype应为BFLOAT16
    if (fiaInfo.inputQType == ge::DT_BF16) {
        if (pseShiftDesc->GetDataType() != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                ToString(pseShiftDesc->GetDataType()).c_str(),
                "The dtype of pse_shift must be BFLOAT16 when the dtype of query is BFLOAT16 and pseType is 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftShape(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的shape
    auto &pseShiftTensor = fiaInfo.opParamInfo.pseShift.tensor;
    if (pseShiftTensor == nullptr) {
        // 若pseShiftTensor不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    auto pseShiftShape = pseShiftTensor->GetStorageShape();
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t n1Size = fiaInfo.n1Size;
    uint32_t s1Size = fiaInfo.s1Size;
    int64_t s2Size = fiaInfo.s2Size;
    // pseShift的维度必须为4
    uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
    if (pseShiftDimNum != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "pse_shift",
            std::to_string(pseShiftDimNum).c_str(), "The shape dim of pse_shift must be 4 when pse_type is 0");
        return ge::GRAPH_FAILED;
    }
    uint32_t pseShiftBatch = pseShiftShape.GetDim(DIM_NUM_0);
    uint32_t pseShiftN = pseShiftShape.GetDim(DIM_NUM_1);
    uint32_t pseShiftS1 = pseShiftShape.GetDim(DIM_NUM_2);
    uint32_t pseShiftS2 = pseShiftShape.GetDim(DIM_NUM_3);
    uint32_t actualSharedPrefixLen = 0;
    if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor != nullptr &&
        fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0) {
        if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>() != nullptr) {
            actualSharedPrefixLen = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>()[0];
        }
    }
    if (pseShiftS1 > 1) {
        // P_S1 > 1分支
        if ((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) ||
            (pseShiftS1 < s1Size) || (pseShiftS2 < s2Size + actualSharedPrefixLen)) {
            std::string reason = "The shape of pse_shift must be [1 or " + std::to_string(batchSize) + ", " +
                std::to_string(n1Size) + ", >=" + std::to_string(s1Size) + ", >=" +
                std::to_string(s2Size + actualSharedPrefixLen) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "pse_shift", ToStringRaw(pseShiftShape).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        // P_S1 = 1分支
        if ((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) ||
            (pseShiftS1 != 1) || (pseShiftS2 < s2Size + actualSharedPrefixLen)) {
            std::string reason = "The shape of pse_shift must be [1 or " + std::to_string(batchSize) + ", " +
                std::to_string(n1Size) + ", 1, >=" + std::to_string(s2Size + actualSharedPrefixLen) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "pse_shift", ToStringRaw(pseShiftShape).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus PSEChecker::CheckPseShiftExistence(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift的desc的存在性，若pseShift存在，其desc也必须存在
    if (fiaInfo.opParamInfo.pseShift.tensor != nullptr) {
        OP_CHECK_IF((fiaInfo.opParamInfo.pseShift.desc == nullptr),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                        "pseShift", "dtype of pseShift cannot be empty"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus PSEChecker::CheckFeatureDConsistency(const FiaTilingInfo &fiaInfo)
{
    // 校验pseShift使能时，D不等长不支持
    OP_CHECK_IF(fiaInfo.pseShiftFlag && fiaInfo.isQKVDDifferent,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
            "pseShift must be empty when D of query and key is not equal to D of value"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckerFeatureCrossover(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    // 校验pseShift使能时与其他特性交叉的约束
    if (fiaInfo.pseShiftFlag) {
        // D不等长时，不支持pse
        OP_CHECK_IF(fiaInfo.isQKVDDifferent,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                        "pseShift must be empty when D of query and key is not equal to D of value"),
                    return ge::GRAPH_FAILED);
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        // MLA场景，不支持pse
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128 || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
                        fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                        "pseShift must be empty in the MLA scenario"), return ge::GRAPH_FAILED);
        // pse使能时，若inputLayout为BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD，不支持pse
        OP_CHECK_IF(layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD" || layoutStr == "TND" || layoutStr == "NTD" ||
                        layoutStr == "NTD_TND" || layoutStr == "TND_NTD",
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                        "pseShift must be empty when the inputLayout is BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus PSEChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPseType(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPseShiftExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureDConsistency(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckerFeatureCrossover(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPseShiftDataType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckPseShiftShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling