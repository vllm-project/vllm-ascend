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
 * \file post_quant_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "post_quant_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

// CheckSingle
ge::graphStatus PostQuantChecker::CheckSingleDtype(const FiaTilingInfo &fiaInfo) const
{
    // QuantScale2 and quantOffset2 only support bf16/fp32 data type.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantScale2.desc, QUANT_SCALE2_NAME)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale2",
            ToString(fiaInfo.opParamInfo.quantScale2.desc->GetDataType()).c_str(),
            "The datatype of quant_scale2 must be bf16 or fp32");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantOffset2.desc, QUANT_OFFSET2_NAME)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_offset2",
            ToString(fiaInfo.opParamInfo.quantOffset2.desc->GetDataType()).c_str(),
            "The datatype of quant_offset2 must be bf16 or fp32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus PostQuantChecker::CheckExistenceQuantScale2(const FiaTilingInfo &fiaInfo) const
{
    // Post-quantization scenarios must include quantScale2.
    if (fiaInfo.isOutQuantEnable) {
        OP_CHECK_IF(fiaInfo.opParamInfo.quantScale2.tensor == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "quant_scale2", "empty",
                "In post quant scenario, quant_scale2 cannot be empty"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.quantScale2.desc == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "quant_scale2", "empty",
                "In post quant scenario, the TensorDesc of quant_scale2 cannot be empty"),
            return ge::GRAPH_FAILED);
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        if (quantScale2ShapeSize <= 0) {
            std::string reasonMsg = "Shape size of quant_scale2 must be positive in post quant scenario!";
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(fiaInfo.opName, "quant_scale2",
                std::to_string(quantScale2ShapeSize).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus PostQuantChecker::CheckFeatureAttenOut(const FiaTilingInfo &fiaInfo) const
{
    // post-quantization scenarios only support int8/fp8_e4m3fn/hifloat8 output data type
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        if (outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8) {
            std::string reasonMsg = "The datatype of attention_out must be within the range "
                "{int8, fp8_e4m3fn, hifloat8} when quantScale2 is not empty";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(outputType).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureQueryDType(const FiaTilingInfo &fiaInfo) const
{
    // Post-quantization scale dtype must be FP32. BF16 is allowed only if the query is BF16
    if (fiaInfo.isOutQuantEnable) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        OP_CHECK_IF(fiaInfo.inputQType != ge::DT_BF16 && quantScale2Type != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale2",
                ToString(quantScale2Type).c_str(),
                "When the datatype of query is not bf16, the dtype of quant_scale2 must be float32"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureLayout(const FiaTilingInfo &fiaInfo) const
{
    // For post quant per-tensor, quant scale/offset only support [1].
    // For post quant per-channel, quant scale/offset dim multiply result only support qN * vD.
    if (fiaInfo.isOutQuantEnable) {
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();
        std::string layoutString = fiaInfo.opParamInfo.layOut;
        bool isSupportedLayout = layoutString == "BSH" || layoutString == "BSND" || layoutString == "BNSD" ||
                                 layoutString == "BNSD_BSND";
        uint32_t numHeads = *(fiaInfo.opParamInfo.numHeads);
        uint64_t quantScale2ShapeSizePerChannel =
            static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(fiaInfo.vHeadDim);
        // per-tensor or per-channel verification
        if (quantScale2Dim == 1) {
            if (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel || !isSupportedLayout) {
                if ((static_cast<uint64_t>(quantScale2ShapeSize) != 1U)) {
                    std::string reasonMsg =
                        "For post quant per-tensor, the shape size of quant scale/offset should be equal to [1]";
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(fiaInfo.opName, "quant_scale2",
                        std::to_string(quantScale2ShapeSize).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            }
        } else {
            if (isSupportedLayout) {
                if ((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel)) {
                    std::string quantScale2SizeStr = std::to_string(quantScale2ShapeSize);
                    std::string reasonMsg = "In post quant per-channel scenario, when layout is " + layoutString + ", "
                        "the total element count of quantScale2/quantOffset2 should be numHeads * vHeadDim";
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(fiaInfo.opName, "quant_scale2",
                        quantScale2SizeStr.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape() !=
                    gert::Shape({numHeads, fiaInfo.vHeadDim})) {
                    std::string quantScale2ShapeStr =
                        ToStringRaw(fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape());
                    std::string reasonMsg = "In post quant per-channel scenario, when layout is " + layoutString + ", "
                        "the shape of quantScale2/quantOffset2 should be [numHeads, vHeadDim]";
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "quant_scale2",
                        quantScale2ShapeStr.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureOutputEqual(const FiaTilingInfo &fiaInfo) const
{
    // In the Anti-quantization scenario,
    // only KV input and post-quantization outputs with the same data type are supported.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        if (outputType != fiaInfo.inputKvType) {
            std::string reasonMsg = "When quantScale2 is not empty, "
                "the datatype of attenOut must be the same as the datatype of key and value";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(outputType).c_str(), reasonMsg);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeaturePrefix(const FiaTilingInfo &fiaInfo) const
{
    // When prefix exists, post-quantization scenarios only support int8 output data types.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(fiaInfo.sysPrefixFlag && outputType != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
            ToString(outputType).c_str(),
            "When keySharedPrefix and valueSharedPrefix are both not empty, the datatype of attentionOut must be int8"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureRowValid(const FiaTilingInfo &fiaInfo) const
{
    if (!fiaInfo.isOutQuantEnable) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.s1Size > 1) {
        bool checkPostQuantOffset =
            (fiaInfo.outputType == ge::DT_INT8) &&
            (fiaInfo.opParamInfo.quantOffset2.tensor != nullptr && fiaInfo.opParamInfo.quantOffset2.desc != nullptr) &&
            (fiaInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetShapeSize() != 0);
        if (!fiaInfo.isMaxWorkspace) {
            std::vector<int64_t> actualSeqLengthsKV{};
            std::vector<int64_t> actualSeqLengths{};
            actualSeqLengthsKV.resize(fiaInfo.bSize);
            actualSeqLengths.resize(fiaInfo.bSize);

            const gert::Tensor *tempData = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
            const gert::Tensor *tempDataKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
            const int64_t preTokens = fiaInfo.opParamInfo.preToken == nullptr ? 0 : *fiaInfo.opParamInfo.preToken;
            const int64_t nextTokens = fiaInfo.opParamInfo.nextToken == nullptr ? 0 : *fiaInfo.opParamInfo.nextToken;
            uint32_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
            uint32_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
            for (uint32_t i = 0; i < fiaInfo.bSize; i++) {
                if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
                    actualSeqLengths[i] = fiaInfo.s1Size;
                } else {
                    actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i]) :
                                                                static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
                }
                if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) ||
                    (tempDataKV->GetData<int64_t>() == nullptr)) { // The user did not input act_seq_kv
                    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
                        actualSeqLengthsKV[i] = fiaInfo.s2Size;
                    } else {
                        actualSeqLengthsKV[i] = fiaInfo.kvListSeqLens[i];
                    }
                } else {
                    actualSeqLengthsKV[i] = (actualLenDimsKV > 1) ?
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]) :
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
                }
                int64_t preTokensPerbatch = 0;
                int64_t nextTokensPerbatch = 0;
                if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
                    preTokensPerbatch = static_cast<int64_t>(SPARSE_MODE_INT_MAX);
                    nextTokensPerbatch =
                        actualSeqLengthsKV[i] + static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i];
                } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
                    preTokensPerbatch = fiaInfo.preToken - actualSeqLengthsKV[i] -
                                        static_cast<int64_t>(fiaInfo.systemPrefixLen) + actualSeqLengths[i];
                    nextTokensPerbatch = fiaInfo.nextToken + actualSeqLengthsKV[i] +
                                         static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i];
                } else {
                    preTokensPerbatch = fiaInfo.preToken;
                    nextTokensPerbatch = fiaInfo.nextToken;
                }
                OP_CHECK_IF((checkPostQuantOffset &&
                             ((preTokensPerbatch + actualSeqLengthsKV[i] +
                                   static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i] <
                               0) ||
                              (nextTokensPerbatch < 0))),
                            OPS_REPORT_VECTOR_INNER_ERR(
                                fiaInfo.opName,
                                "When sparse mode = %d, output dtype is int8, the output's dequant offset "
                                "is not null or empty tensor, "
                                "preTokens = %ld and nextTokens = %ld, some rows of the matrix do not "
                                "participate in the calculation, "
                                "the accuracy of the final result will be incorrect. Please see the "
                                "documentation for more details.",
                                fiaInfo.sparseMode, preTokens, nextTokens),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckAntiquantNotSupport(const FiaTilingInfo &fiaInfo) const
{
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        if (fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN &&
            (fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16)) {
            std::string reasonMsg = "When keyAntiquantMode=1 and valueAntiquantMode=1, "
                "and the datatype of key is FLOAT8_E4M3FN, the datatype of attentionOut must be BF16 or FLOAT16";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(fiaInfo.outputType).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
        }
    }

    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        if (fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN &&
            (fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16)) {
            std::string reasonMsg = "When keyAntiquantMode=4 and valueAntiquantMode=4, "
                "and the datatype of key is FLOAT8_E4M3FN, the datatype of attentionOut must be BF16 or FLOAT16";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(fiaInfo.outputType).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCheckMultiPara
ge::graphStatus PostQuantChecker::CheckMultiParaQuantOffset2(const FiaTilingInfo &fiaInfo) const
{
    // Scale2 and offset2 should have same dtype and shape
    if (fiaInfo.isOutQuantEnable && fiaInfo.opParamInfo.quantOffset2.tensor != nullptr &&
        fiaInfo.opParamInfo.quantOffset2.desc != nullptr) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();

        const ge::DataType quantOffset2Datatype = fiaInfo.opParamInfo.quantOffset2.tensor->GetDataType();
        int64_t quantOffset2ShapeSize = fiaInfo.opParamInfo.quantOffset2.tensor->GetShapeSize();
        size_t quantOffset2Dim = fiaInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetDimNum();

        OP_CHECK_IF(quantOffset2Datatype != quantScale2Type,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_offset2",
                        ToString(quantOffset2Datatype).c_str(),
                        "The datatype of QuantScale2 and quantOffset2 must be the same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2Dim != quantScale2Dim,
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "quant_scale2 and quant_offset2",
                (std::to_string(quantOffset2Dim) + "D and " + std::to_string(quantScale2Dim) + "D").c_str(),
                "The shape dims of quant_scale2 and quant_offset2 must be the same"),
                    return ge::GRAPH_FAILED);
        if (quantOffset2ShapeSize != quantScale2ShapeSize) {
            std::string shapeMsg = std::to_string(quantScale2ShapeSize) + ", " + std::to_string(quantOffset2ShapeSize);
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                fiaInfo.opName, "quant_scale2 and quant_offset2", shapeMsg.c_str(),
                "The shape sizes of quant_scale2 and quant_offset2 must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceQuantScale2(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAttenOut(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureQueryDType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeaturePrefix(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            if (ge::GRAPH_SUCCESS != CheckFeatureRowValid(fiaInfo)) {
                 return ge::GRAPH_FAILED;
            }
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureOutputEqual(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckAntiquantNotSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckMultiParaQuantOffset2(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
