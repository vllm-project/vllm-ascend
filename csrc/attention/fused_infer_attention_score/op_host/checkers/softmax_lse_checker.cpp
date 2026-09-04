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
 * \file softmax_lse_checker.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "softmax_lse_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// CheckSinglePara
ge::graphStatus SoftmaxLSEChecker::CheckSingleDtype(const FiaTilingInfo &fiaInfo)
{
    // SoftmaxLse only supports output FP32
    if (fiaInfo.softmaxLseFlag) {
        OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.lseOut.desc, SOFTMAX_LSE_NAME),
            OP_LOGE_FOR_INVALID_DTYPE(fiaInfo.opName, "softmax_lse",
                ToString(fiaInfo.opParamInfo.lseOut.desc->GetDataType()).c_str(),
                ToString(ge::DT_FLOAT).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus SoftmaxLSEChecker::CheckExistenceShapeAndDesc(const FiaTilingInfo &fiaInfo) const
{
    // When softmaxLseFlag is true, both the SoftmaxLse shape and the lseOut tensor must not be null.
    if (fiaInfo.softmaxLseFlag) {
        const gert::StorageShape *lseShape = fiaInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseShape == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "softmax_lse",
                "When softmaxLseFlag is true, softmaxLse cannot be empty"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.lseOut.desc == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(fiaInfo.opName, "TensorDesc of softmax_lse"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCrossFeature
ge::graphStatus SoftmaxLSEChecker::CheckFeatureDimAndShape(const FiaTilingInfo &fiaInfo) const
{
    // When softmaxLseFlag is true and emptyTensorFlag is false:
    // -If the LayOut is TND, the shape must have 3 dimensions (matching TN1).
    // -Otherwise, the shape must have 4 dimensions (matching [B,N,S1,1]).
    if (fiaInfo.softmaxLseFlag && (!fiaInfo.emptyTensorFlag)) {
        const gert::StorageShape *lseShape = fiaInfo.opParamInfo.lseOut.shape;
        std::shared_ptr<FiaTilingShapeCompare> softmaxLseShapeCmp_ = nullptr;
        if (fiaInfo.outLayout == FiaLayout::TND || fiaInfo.outLayout == FiaLayout::NTD) {
            if (lseShape->GetStorageShape().GetDimNum() != DIM_NUM_3) {
                std::string dimStr = std::to_string(lseShape->GetStorageShape().GetDimNum()) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    fiaInfo.opName, "softmax_lse", dimStr.c_str(),
                    "TND/NTD The shape of softmax_lse must be 3D!");
                return ge::GRAPH_FAILED;
            }
            if (lseShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.qTSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != fiaInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != SHAPE_PARAMS_CONST) {
                std::string shapeStr = ToStringRaw(lseShape->GetStorageShape());
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "softmax_lse", shapeStr.c_str(),
                    "The shape of softmax_lse must be [T, N, 1]");
                return ge::GRAPH_FAILED;
            }
        } else {
            if (lseShape->GetStorageShape().GetDimNum() != DIM_NUM_4) {
                std::string dimStr = std::to_string(lseShape->GetStorageShape().GetDimNum()) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "softmax_lse", dimStr.c_str(), "4D");
                return ge::GRAPH_FAILED;
            }
            if (lseShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.bSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != fiaInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != fiaInfo.s1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_3) != SHAPE_PARAMS_CONST) {
                std::string shapeStr = ToStringRaw(lseShape->GetStorageShape());
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "softmax_lse", shapeStr.c_str(),
                                                      "The shape of softmax_lse must be [B, N, Q_S, 1]");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// enableAntiQuant 相关校验函数
ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceShapeAndDesc(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureDimAndShape(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        fiaInfo.isQKVDDifferent && fiaInfo.ropeMode == RopeMode::NO_ROPE && fiaInfo.softmaxLseFlag,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "softmax_lse_flag", "True",
            "In GQA scenario, when the headDim of query, key and value are different, softmax_lse_flag must be False"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
