/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gmm_qbmm_tiling.cpp
 * \brief Copied from gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
 */
#include <alog_pub.h>
#include "gmm_qbmm_tiling.h"

#include "mc2_log_compat.h"
#include "log/error_code.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "../op_kernel/gqmm_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
using namespace optiling::Mc2GroupedMatmulTiling;
using namespace Mc2GroupedMatmulTilingData;
using namespace optiling::Mc2GroupedMatmul;
using namespace optiling::Mc2GroupedMatmulTiling::GmmConstant;

namespace optiling {
namespace Mc2GroupedMatmulTiling {
bool Mc2GroupedQbmmTiling::IsCapable()
{
    return true;
}

void Mc2GroupedQbmmTiling::Reset()
{
    tilingData_ = GMMQuantTilingData();
}

ge::graphStatus Mc2GroupedQbmmTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<GMMCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "CompileInfo"),
                    return ge::GRAPH_FAILED);

        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0ASize;
        aicoreParams_.l0bSize = compileInfoPtr->l0BSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0CSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_LOGI(context_, "Platform info: aicNum(%lu) ubSize(%lu) l1Size(%lu) l0aSize(%lu) l0bSize(%lu) l0cSize(%lu).",
            aicoreParams_.aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0aSize,
            aicoreParams_.l0bSize, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

bool Mc2GroupedQbmmTiling::IsMicroScaling() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
}

bool Mc2GroupedQbmmTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs) {
        OP_CHECK_IF(
            attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
            OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "attrs", std::to_string(attrs->GetAttrNum()).c_str(),
                                      (std::string("should be > ") + std::to_string(ATTR_INDEX_ACT_TYPE + 1)).c_str()),
            return false);
        const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
        const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
        const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
        const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);

        inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
        inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
        inputParams_.groupType = groupTypePtr != nullptr ? *groupTypePtr : inputParams_.groupType;
        inputParams_.splitItem = splitItemPtr != nullptr ? *splitItemPtr : inputParams_.splitItem;
        inputParams_.actType = actTypePtr != nullptr ? *actTypePtr : inputParams_.actType;
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    OP_CHECK_IF(inputParams_.groupType != SPLIT_M && inputParams_.groupType != SPLIT_K,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "groupType",
                                          std::to_string(inputParams_.groupType).c_str(),
                                          "only support 0 (SPLIT_M) or 2 (SPLIT_K)"),
                return false);
    OP_CHECK_IF((inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
                    inputParams_.groupType != SPLIT_M,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "groupType",
                                          std::to_string(inputParams_.groupType).c_str(),
                                          "only support 0 (SPLIT_M) for FLOAT4"),
                return false);
    if (inputParams_.groupType == SPLIT_M) {
        OP_CHECK_IF(inputParams_.transA,
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "transA", "true", "false when groupType == SPLIT_M"),
                    return false);
    } else {
        OP_CHECK_IF(!inputParams_.transA,
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "transA", "false", "true when groupType == SPLIT_K"),
                    return false);
        OP_CHECK_IF(inputParams_.transB,
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "transB", "true", "false when groupType == SPLIT_K"),
                    return false);
    }

    inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
    inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    inputParams_.isSingleY = (inputParams_.splitItem == 2 || inputParams_.splitItem == 3);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckBiasDtype() const
{
    if ((inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2)) {
        OP_CHECK_IF(
            inputParams_.biasDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                                  "If x is FLOAT4, the dtype of bias must be FLOAT."),
            return false);
    } else if (inputParams_.aDtype == ge::DT_INT8) {
        if (inputParams_.cDtype == ge::DT_BF16) {
            OP_CHECK_IF(inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_BF16 &&
                            inputParams_.biasDtype != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "bias",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                            "If x is INT8 and output is BF16, the dtype of bias must be INT32, BF16 or FLOAT."),
                        return false);
        } else if (inputParams_.cDtype == ge::DT_FLOAT16) {
            OP_CHECK_IF(inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_FLOAT16 &&
                            inputParams_.biasDtype != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "bias",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                            "If x is INT8 and output is FLOAT16, the dtype of bias must be INT32, FLOAT16 or FLOAT."),
                        return false);
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                                  "If x is INT8, the dtype of output must be BF16 or FLOAT16.");
            return false;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                              "The dtype of bias must be compatible with the dtype of x.");
        return false;
    }
    return true;
}

bool Mc2GroupedQbmmTiling::CheckDtypeForWeightNz(bool isPertokenScaleNull) const
{
    OP_CHECK_IF(inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, weight",
                    (std::string("[") + ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype) + "]")
                        .c_str(),
                    "The dtypes of x, weight must be INT8 for Nz format weight."),
                return false);
    if (!isPertokenScaleNull) {
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "perTokenScale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "The dtype of perTokenScale must be FLOAT for Nz format weight."),
                    return false);
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_BF16 && inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "scale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                        "The dtype of scale must be BF16 or FLOAT for Nz format weight."),
                    return false);
    } else {
        static const std::vector<ge::DataType> legalScaleDtypes = {ge::DT_UINT64, ge::DT_INT64, ge::DT_FLOAT,
                                                                   ge::DT_BF16};
        OP_CHECK_IF(
            std::find(legalScaleDtypes.begin(), legalScaleDtypes.end(), inputParams_.scaleDtype) ==
                legalScaleDtypes.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                "The dtype of scale must be within the range {UINT64, INT64, FLOAT, BF16} for Nz format weight."),
            return false);
    }
    return true;
}

bool Mc2GroupedQbmmTiling::AnalyzeDtype()
{
    static const std::vector<ge::DataType> legalInputDtypes = {
        ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                              "The dtype of x must be within the range {INT8, HIFLOAT8, FLOAT8_E4M3, "
                                              "FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2}."),
        return false);
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "weight",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                              "The dtype of weight must be within the range {INT8, HIFLOAT8, "
                                              "FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2}."),
        return false);
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(inputParams_.opName, "Bias from tensor is not nullptr, but bias from desc is nullptr."),
                return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : inputParams_.biasDtype;
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    isWeightNz_ = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    if (isWeightNz_) {
        OP_CHECK_IF(!CheckDtypeForWeightNz(nullptr == pertokenScaleDesc),
                    OP_LOGE(inputParams_.opName, "CheckDtypeForWeightNz failed."), return false);
    }
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "y"), return false);
    inputParams_.cDtype = yDesc->GetDataType();
    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckBiasDtype(), OP_LOGE(inputParams_.opName, "CheckBiasDtype failed."), return false);
    }
    return true;
}

bool Mc2GroupedQbmmTiling::CheckQuantParamsForMXTypeM(const gert::Shape &xScaleShape,
                                                      const gert::Shape &wScaleShape) const
{
    auto xScaleDimNum = xScaleShape.GetDimNum();
    auto wScaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(wScaleDimNum != MXFP_TYPE_M_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                         (std::to_string(wScaleDimNum) + "D").c_str(),
                                                         "The shape dim of scale must be 4D in mx quant split m mode."),
                return false);
    OP_CHECK_IF(xScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    inputParams_.opName, "pertokenScale", (std::to_string(xScaleDimNum) + "D").c_str(),
                    "The shape dim of pertokenScale must be 3D in mx quant split m mode."),
                return false);
    auto wScaleEDim = static_cast<uint64_t>(wScaleShape.GetDim(0));
    auto wScaleNDim = static_cast<uint64_t>(
        inputParams_.transB ? wScaleShape.GetDim(1) : wScaleShape.GetDim(2)); // 2 is index for the third dim
    auto wScaleKDim =
        static_cast<uint64_t>(inputParams_.transB ? wScaleShape.GetDim(2) : // 2 is index for the third dim
                                                    wScaleShape.GetDim(1));
    auto xScaleMDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(1) : xScaleShape.GetDim(0));
    auto xScaleKDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(0) : xScaleShape.GetDim(1));
    auto wScaleLastDim = static_cast<uint64_t>(wScaleShape.GetDim(wScaleDimNum - 1));
    auto xScaleLastDim = static_cast<uint64_t>(xScaleShape.GetDim(xScaleDimNum - 1));
    auto expectedKDimValue = CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR);
    OP_CHECK_IF(wScaleEDim != inputParams_.groupNum || wScaleKDim != expectedKDimValue ||
                    wScaleNDim != inputParams_.nSize || wScaleLastDim != MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "scale",
                    (std::string("[") + std::to_string(wScaleEDim) + ", " + std::to_string(wScaleNDim) + ", " +
                     std::to_string(wScaleKDim) + ", " + std::to_string(wScaleLastDim) + "]")
                        .c_str(),
                    (std::string("The shape of scale must be [") + std::to_string(inputParams_.groupNum) + ", " +
                     std::to_string(inputParams_.nSize) + ", " + std::to_string(expectedKDimValue) +
                     ", 2] in mx quant split m mode.")
                        .c_str()),
                return false);
    OP_CHECK_IF(xScaleMDim != inputParams_.mSize || xScaleKDim != expectedKDimValue ||
                    xScaleLastDim != MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "pertokenScale",
                    (std::string("[") + std::to_string(xScaleMDim) + ", " + std::to_string(xScaleKDim) + ", " +
                     std::to_string(xScaleLastDim) + "]")
                        .c_str(),
                    (std::string("The shape of pertokenScale must be [") + std::to_string(inputParams_.mSize) + ", " +
                     std::to_string(expectedKDimValue) + ", 2] in mx quant split m mode.")
                        .c_str()),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckQuantParamsForMXTypeK(const gert::Shape &xScaleShape,
                                                      const gert::Shape &wScaleShape) const
{
    auto xScaleDimNum = xScaleShape.GetDimNum();
    auto wScaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(wScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                         (std::to_string(wScaleDimNum) + "D").c_str(),
                                                         "The shape dim of scale must be 3D in mx quant split k mode."),
                return false);
    OP_CHECK_IF(xScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    inputParams_.opName, "pertokenScale", (std::to_string(xScaleDimNum) + "D").c_str(),
                    "The shape dim of pertokenScale must be 3D in mx quant split k mode."),
                return false);
    auto xScaleLastDim = static_cast<uint64_t>(xScaleShape.GetDim(xScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(
        inputParams_.transA ? xScaleShape.GetDim(0) : xScaleShape.GetDim(xScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto xScaleMDim = static_cast<uint64_t>(
        inputParams_.transA ? xScaleShape.GetDim(xScaleDimNum - LAST_SECOND_DIM_INDEX) : xScaleShape.GetDim(0));
    auto wScaleLastDim = static_cast<uint64_t>(wScaleShape.GetDim(wScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(
        inputParams_.transB ? wScaleShape.GetDim(0) : wScaleShape.GetDim(wScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(
        inputParams_.transB ? wScaleShape.GetDim(wScaleDimNum - LAST_SECOND_DIM_INDEX) : wScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams_.kSize / MXFP_BASEK_FACTOR + inputParams_.groupNum;
    OP_CHECK_IF(!inputParams_.transA || inputParams_.transB,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    inputParams_.opName, "transA, transB",
                    (std::string("[") + (inputParams_.transA ? "true" : "false") + ", " +
                     (inputParams_.transB ? "true" : "false") + "]")
                        .c_str(),
                    "The values of transA, transB must be within the range {true, false} for split k mx quant mode."),
                return false);
    OP_CHECK_IF(xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue ||
                    xScaleMDim != inputParams_.mSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "pertokenScale",
                    (std::string("[") + std::to_string(xScaleKDim) + ", " + std::to_string(xScaleMDim) + ", " +
                     std::to_string(xScaleLastDim) + "]")
                        .c_str(),
                    (std::string("The shape of pertokenScale must be [") + std::to_string(expectedKDimValue) + ", " +
                     std::to_string(inputParams_.mSize) + ", 2] for split k mx quant mode.")
                        .c_str()),
                return false);
    OP_CHECK_IF(wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue ||
                    wScaleNDim != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "scale",
                    (std::string("[") + std::to_string(wScaleKDim) + ", " + std::to_string(wScaleNDim) + ", " +
                     std::to_string(wScaleLastDim) + "]")
                        .c_str(),
                    (std::string("The shape of scale must be [") + std::to_string(expectedKDimValue) + ", " +
                     std::to_string(inputParams_.nSize) + ", 2] for split k mx quant mode.")
                        .c_str()),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckQuantParamsForMxQuantMode(const gert::StorageShape *xScaleStorageShape,
                                                          const gert::Shape &wScaleShape) const
{
    // 多数参数在CheckQuantParamsForMxQuantMode函数调用前已有非空校验
    OP_CHECK_IF(xScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "xScale"),
                return false);
    auto &xScaleShape = xScaleStorageShape->GetStorageShape();
    if (inputParams_.groupType == SPLIT_M) {
        OP_CHECK_IF(!CheckQuantParamsForMXTypeM(xScaleShape, wScaleShape),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForMXTypeM failed."), return false);
    } else {
        OP_CHECK_IF(!CheckQuantParamsForMXTypeK(xScaleShape, wScaleShape),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForMXTypeK failed."), return false);
    }
    return true;
}


bool Mc2GroupedQbmmTiling::CheckQuantParamsForNonKGroupQuantMode(const gert::Shape &wScaleShape) const
{
    auto wScaleDimNum = wScaleShape.GetDimNum();
    // dim num 1 for the shape (g,), dim num 2 for the shape (g,1) or (g,n)
    OP_CHECK_IF(wScaleDimNum != 1 && wScaleDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    inputParams_.opName, "scale", (std::to_string(wScaleDimNum) + "D").c_str(),
                    "The shape dim of scale must be 1D or 2D in non k axis group quant mode."),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckFp4Shape() const
{
    OP_CHECK_IF(inputParams_.kSize % EVEN_FACTOR != 0,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "kSize", std::to_string(inputParams_.kSize).c_str(),
                                          "should be even number when x is FLOAT4"),
                return false);
    // 2: mxfp4场景下不支持K轴为2
    OP_CHECK_IF(inputParams_.kSize == 2,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "kSize", "2", "should not be 2 when x is FLOAT4"),
                return false);
    if (!inputParams_.transB) {
        OP_CHECK_IF(inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "nSize", std::to_string(inputParams_.nSize).c_str(),
                                              "should be even number when x is FLOAT4 and weight is not transposed"),
                    return false);
    }
    return true;
}

bool Mc2GroupedQbmmTiling::CheckBiasShape(const gert::StorageShape *biasStorageShape) const
{
    auto &biasShape = biasStorageShape->GetStorageShape();
    OP_CHECK_IF(biasStorageShape->GetStorageShape().GetDimNum() != BIAS_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    inputParams_.opName, "bias",
                    (std::to_string(biasStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                    "The shape dim of bias must be 2D."),
                return false);
    auto biasEDim = static_cast<uint64_t>(biasShape.GetDim(0));
    auto biasNDim = static_cast<uint64_t>(biasShape.GetDim(1));
    OP_CHECK_IF(biasEDim != inputParams_.groupNum || biasNDim != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "bias",
                    (std::string("[") + std::to_string(biasEDim) + ", " + std::to_string(biasNDim) + "]").c_str(),
                    (std::string("The shape of bias must be [") + std::to_string(inputParams_.groupNum) + ", " +
                     std::to_string(inputParams_.nSize) + "].")
                        .c_str()),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckQuantParams(const gert::StorageShape *xScaleStorageShape,
                                            const gert::Shape &wScaleShape) const
{
    // 非k分组量化校验
    if (inputParams_.bQuantMode != QuantMode::MX_PERGROUP_MODE && inputParams_.bQuantMode != QuantMode::PERGROUP_MODE &&
        inputParams_.bQuantMode != QuantMode::PERBLOCK_MODE) {
        OP_CHECK_IF(!CheckQuantParamsForNonKGroupQuantMode(wScaleShape),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForNonKGroupQuantMode failed."), return false);
    }
    // mx量化校验
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        OP_CHECK_IF(!CheckQuantParamsForMxQuantMode(xScaleStorageShape, wScaleShape),
                    OP_LOGE(inputParams_.opName, "CheckParamsForMxQuantMode failed."), return false);
    }

    return true;
}

bool Mc2GroupedQbmmTiling::CheckShapeForWeightNz(const gert::Shape &wShape) const
{
    auto wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(wDimNum != WEIGHTNZ_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "weight",
                                                         (std::to_string(wDimNum) + "D").c_str(),
                                                         "The shape dim of weight must be 5D for Nz format."),
                return false);
    OP_CHECK_IF(wShape[WEIGHTNZ_FIFTH_DIM] != WEIGHTNZ_N0_32,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "weight",
                    (std::string("dim4=") + std::to_string(wShape[WEIGHTNZ_FIFTH_DIM])).c_str(),
                    "The shape [dim4] of weight must be 32 for Nz format."),
                return false);
    OP_CHECK_IF(wShape[WEIGHTNZ_FORTH_DIM] != WEIGHTNZ_K0_16,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "weight",
                    (std::string("dim3=") + std::to_string(wShape[WEIGHTNZ_FORTH_DIM])).c_str(),
                    "The shape [dim3] of weight must be 16 for Nz format."),
                return false);
    auto wShapeDimThird = static_cast<uint64_t>(wShape[WEIGHTNZ_THIRD_DIM]);
    auto wShapeDimSecond = static_cast<uint64_t>(wShape[WEIGHTNZ_SECOND_DIM]);
    if (!inputParams_.transB) {
        OP_CHECK_IF(wShapeDimThird != CeilDiv(inputParams_.kSize, WEIGHTNZ_K0_16),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "weight",
                        (std::string("dim2=") + std::to_string(wShapeDimThird)).c_str(),
                        (std::string("The shape [dim2] of weight must be equal to ceil(kSize/16) = ") +
                         std::to_string(CeilDiv(inputParams_.kSize, WEIGHTNZ_K0_16)))
                            .c_str()),
                    return false);
        OP_CHECK_IF(wShapeDimSecond != CeilDiv(inputParams_.nSize, WEIGHTNZ_N0_32),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "weight",
                        (std::string("dim1=") + std::to_string(wShapeDimSecond)).c_str(),
                        (std::string("The shape [dim1] of weight must be equal to ceil(nSize/32) = ") +
                         std::to_string(CeilDiv(inputParams_.nSize, WEIGHTNZ_N0_32)))
                            .c_str()),
                    return false);
    } else {
        OP_CHECK_IF(wShapeDimThird != CeilDiv(inputParams_.nSize, WEIGHTNZ_N0_16),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "weight",
                        (std::string("dim2=") + std::to_string(wShapeDimThird)).c_str(),
                        (std::string("The shape [dim2] of weight must be equal to ceil(nSize/16) = ") +
                         std::to_string(CeilDiv(inputParams_.nSize, WEIGHTNZ_N0_16)))
                            .c_str()),
                    return false);
        OP_CHECK_IF(wShapeDimSecond != CeilDiv(inputParams_.kSize, WEIGHTNZ_K0_32),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "weight",
                        (std::string("dim1=") + std::to_string(wShapeDimSecond)).c_str(),
                        (std::string("The shape [dim1] of weight must be equal to ceil(kSize/32) = ") +
                         std::to_string(CeilDiv(inputParams_.kSize, WEIGHTNZ_K0_32)))
                            .c_str()),
                    return false);
    }
    OP_CHECK_IF(1 == inputParams_.kSize || 1 == inputParams_.nSize,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "nSize/kSize",
                                          (std::string("kSize=") + std::to_string(inputParams_.kSize) +
                                           ", nSize=" + std::to_string(inputParams_.nSize))
                                              .c_str(),
                                          "cannot be 1 for Nz format weight"),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::CheckActiveModeDtype(const gert::StorageShape *xScaleStorageShape) const
{
    OP_CHECK_IF((inputParams_.aDtype != ge::DT_INT8) || (inputParams_.bDtype != ge::DT_INT8),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, weight",
                    (std::string("[") + ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype) + "]")
                        .c_str(),
                    "If activation function is enabled, the dtypes of x, weight must be DT_INT8."),
                return false);
    if (inputParams_.cDtype == ge::DT_BF16) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_BF16 && inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                        "If output is DT_BF16, the dtype of scale must be DT_BF16 or DT_FLOAT."),
                    return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT && xScaleStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "perTokenScale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "If output is DT_BF16, the dtype of perTokenScale must be DT_FLOAT."),
                    return false);
    } else if (inputParams_.cDtype == ge::DT_FLOAT16) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                        "If output is DT_FLOAT16, the dtype of scale must be DT_FLOAT."),
                    return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT && xScaleStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "perTokenScale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "If output is DT_FLOAT16, the dtype of perTokenScale must be DT_FLOAT."),
                    return false);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "output", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
            "If activation function is enabled, the dtype of output must be DT_BF16 or DT_FLOAT16.");
        return false;
    }
    return true;
}

bool Mc2GroupedQbmmTiling::CheckActiveMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape)
{
    OP_CHECK_IF(inputParams_.actType == GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "actType",
                                          std::to_string(inputParams_.actType).c_str(), "GELU_ERR_FUNC not supported"),
                return false);
    OP_CHECK_IF(
        inputParams_.actType > GMMActType::GMM_ACT_TYPE_SILU || inputParams_.actType < GMMActType::GMM_ACT_TYPE_NONE,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "actType", std::to_string(inputParams_.actType).c_str(),
                                  "only supports RELU/GELU_TANH/FASTGELU/SILU"),
        return false);
    OP_CHECK_IF(!CheckActiveModeDtype(xScaleStorageShape),
                OP_LOGE(context_->GetNodeName(), "CheckActiveModeDtype failed."), return false);
    auto wScaleDims = wScaleShape.GetDimNum();
    if (xScaleStorageShape != nullptr) {
        auto &xScaleShape = xScaleStorageShape->GetStorageShape();
        auto xScaleDims = xScaleShape.GetDimNum();
        OP_CHECK_IF(xScaleDims != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "perTokenScale", (std::to_string(xScaleDims) + "D").c_str(),
                        "The shape dim of perTokenScale must be 1D or nullptr when activation function is enabled."),
                    return false);
        OP_CHECK_IF(static_cast<uint64_t>(xScaleShape[0]) != inputParams_.mSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "perTokenScale",
                        (std::string("dim0=") + std::to_string(static_cast<uint64_t>(xScaleShape[0]))).c_str(),
                        (std::string("The shape [dim0] of perTokenScale must be ") +
                         std::to_string(inputParams_.mSize) + " when activation function is enabled.")
                            .c_str()),
                    return false);
    }
    OP_CHECK_IF(wScaleDims != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "scale", (std::to_string(wScaleDims) + "D").c_str(),
                    "The shape dim of scale must be 2D when activation function is enabled."),
                return false);
    OP_CHECK_IF(static_cast<uint64_t>(wScaleShape[0]) != inputParams_.groupNum ||
                    static_cast<uint64_t>(wScaleShape[1]) != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "scale",
                    (std::string("[") + std::to_string(static_cast<uint64_t>(wScaleShape[0])) + ", " +
                     std::to_string(static_cast<uint64_t>(wScaleShape[1])) + "]")
                        .c_str(),
                    (std::string("The shape of scale must be [") + std::to_string(inputParams_.groupNum) + ", " +
                     std::to_string(inputParams_.nSize) + "] when activation function is enabled.")
                        .c_str()),
                return false);
    if (inputParams_.nSize == 1) {
        inputParams_.bQuantMode = QuantMode::PERCHANNEL_MODE;
    }
    return true;
}

bool Mc2GroupedQbmmTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);

    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    const gert::Shape &weightNzStorageShape = wStorageShape->GetStorageShape();

    // 全量化scale必须有值，目前无输出int32等不需要scale的场景
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "scale"),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                         (std::to_string(scaleDimNum) + "D").c_str(),
                                                         "The shape dim of scale must be a positive integer."),
                return false);
    auto xScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetMKNList(), OP_LOGE(inputParams_.opName, "SetMKNList failed."), return false);
    OP_CHECK_IF(!SetQuantMode(wScaleShape, xScaleStorageShape, wShape),
                OP_LOGE(inputParams_.opName, "SetQuantMode failed."), return false);
    OP_CHECK_IF(!CheckQuantParams(xScaleStorageShape, wScaleShape),
                OP_LOGE(inputParams_.opName, "CheckQuantParams failed."), return false);

    if (isWeightNz_) {
        OP_CHECK_IF(!CheckShapeForWeightNz(weightNzStorageShape),
                    OP_LOGE(context_->GetNodeName(), "CheckShapeForWeightNz failed."), return false);
    }
    if (inputParams_.actType != GMMActType::GMM_ACT_TYPE_NONE) {
        OP_CHECK_IF(!CheckActiveMode(wScaleShape, xScaleStorageShape),
                    OP_LOGE(context_->GetNodeName(), "CheckActiveMode failed."), return false);
    }
    if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(!CheckFp4Shape(), OP_LOGE(inputParams_.opName, "CheckFp4Shape failed."), return false);
        if (inputParams_.hasBias) {
            auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
            OP_CHECK_IF(!CheckBiasShape(biasStorageShape), OP_LOGE(inputParams_.opName, "CheckBiasShape failed."),
                        return false);
        }
    }
    SetKernelType();
    return true;
}

bool Mc2GroupedQbmmTiling::SetQuantMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape,
                                        const gert::Shape &wShape)
{
    auto wScaleDims = wScaleShape.GetDimNum();
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    // scale pertensor: (g,1) 2维或（g,）1维, perchannel:（g, N), 2维
    if (wScaleDims == 2 && static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - 1)) == inputParams_.nSize &&
        inputParams_.nSize != 1UL) {
        inputParams_.bQuantMode = QuantMode::PERCHANNEL_MODE;
    } else if ((wScaleDims == 2 && wScaleShape[wScaleDims - 1] == 1) || // 2:（g,1) 2维
               (wScaleDims == 1 && static_cast<uint64_t>(wScaleShape[0]) == inputParams_.groupNum)) {
        inputParams_.bQuantMode = QuantMode::PERTENSOR_MODE;
    }
    if (xScaleStorageShape != nullptr) {
        // split_m: pertoken (M,), pertensor（g,1) 2维或（g,）1维;
        // split_k: pertoken (g, M), pertensor（g,1) 2维或（g,) 1维
        auto &xScaleShape = xScaleStorageShape->GetStorageShape();
        auto xScaleDims = xScaleShape.GetDimNum();
        if (inputParams_.aDtype != ge::DT_INT8 &&
            ((xScaleDims == 2 && xScaleShape[xScaleDims - 1] == 1) || // 2:（g,1) 2维
             (xScaleDims == 1 && static_cast<uint64_t>(xScaleShape[0]) == inputParams_.groupNum &&
              inputParams_.groupNum != inputParams_.mSize))) {
            inputParams_.aQuantMode = QuantMode::PERTENSOR_MODE;
        } else {
            inputParams_.aQuantMode = QuantMode::PERTOKEN_MODE;
        }
        SetPerGroupQuantMode(xScaleShape, wScaleShape, wShape);
    }
    return true;
}

void Mc2GroupedQbmmTiling::SetPerGroupQuantMode(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape,
                                                const gert::Shape &wShape)
{
    if (inputParams_.aDtype == ge::DT_INT8) {
        return;
    }
    auto xScaleDims = xScaleShape.GetDimNum();
    if (xScaleDims < X_DIMS) {
        return;
    }
    auto wScaleDims = wScaleShape.GetDimNum();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    if (wDimNum != wScaleDims || (inputParams_.groupType == SPLIT_M && wScaleDims < SPLIT_M_W_DIMS) ||
        (inputParams_.groupType == SPLIT_K && wScaleDims < SPLIT_K_W_DIMS)) {
        return;
    }
    QuantMode aQuantMode = QuantMode::DEFAULT;
    QuantMode bQuantMode = QuantMode::DEFAULT;
    if (inputParams_.groupType == SPLIT_M) {
        for (uint64_t i = 1; i < wScaleDims; ++i) {
            if (wScaleShape.GetDim(i) != CeilDiv(wShape.GetDim(i), PER_BLOCK_GROUP_SIZE)) {
                return;
            }
        }
        bQuantMode = QuantMode::PERBLOCK_MODE;

        uint64_t scaleKPerBlock = CeilDiv(inputParams_.kSize, PER_BLOCK_GROUP_SIZE);
        if (static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_FIRST_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_SECOND_DIM_INDEX)) == inputParams_.mSize) {
            aQuantMode = QuantMode::PERGROUP_MODE;
        }
    }
    if (inputParams_.groupType == SPLIT_K) {
        uint64_t scaleKPerBlock = inputParams_.kSize / PER_BLOCK_GROUP_SIZE + inputParams_.groupNum;
        uint64_t scaleNPerBlock = CeilDiv(inputParams_.nSize, PER_BLOCK_GROUP_SIZE);
        if (static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_SECOND_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_FIRST_DIM_INDEX)) == inputParams_.mSize) {
            aQuantMode = QuantMode::PERGROUP_MODE;
        }
        if (static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - LAST_SECOND_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - LAST_FIRST_DIM_INDEX)) == scaleNPerBlock) {
            bQuantMode = QuantMode::PERBLOCK_MODE;
        }
    }
    if (aQuantMode == QuantMode::PERGROUP_MODE && bQuantMode == QuantMode::PERBLOCK_MODE) {
        inputParams_.aQuantMode = QuantMode::PERGROUP_MODE;
        inputParams_.bQuantMode = QuantMode::PERBLOCK_MODE;
    }
}

bool Mc2GroupedQbmmTiling::SetGroupNum(uint32_t groupListIndex)
{
    auto groupListStorageShape = context_->GetOptionalInputShape(groupListIndex);
    OP_CHECK_IF(groupListStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "groupList"),
                return false);
    const gert::Shape &groupListShape = groupListStorageShape->GetStorageShape();
    OP_CHECK_IF(groupListShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "groupList",
                                                         (std::to_string(groupListShape.GetDimNum()) + "D").c_str(),
                                                         "The shape dim of groupList must be 1D."),
                return false);
    inputParams_.groupNum = groupListShape.GetDim(0);
    OP_CHECK_IF(inputParams_.groupNum > GMM_MAX_GROUP_LIST_SIZE,
                OP_LOGE_FOR_INVALID_VALUE(
                    inputParams_.opName, "groupNum", std::to_string(inputParams_.groupNum).c_str(),
                    (std::string("should not be greater than ") + std::to_string(GMM_MAX_GROUP_LIST_SIZE)).c_str()),
                return false);
    return true;
}

bool Mc2GroupedQbmmTiling::SetMKN(const gert::Shape &xShape, const gert::Shape &wShape)
{
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    OP_CHECK_IF(wDimNum < MIN_ND_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "weight",
                                                         (std::to_string(wDimNum) + "D").c_str(),
                                                         "The shape dim of weight must be at least 2D."),
                return false);
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    OP_CHECK_IF(xDimNum < MIN_ND_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x",
                                                         (std::to_string(xDimNum) + "D").c_str(),
                                                         "The shape dim of x must be at least 2D."),
                return false);
    auto mSize = inputParams_.transA ? xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX) :
                                       xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    auto kSize = inputParams_.transA ? xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX) :
                                       xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);
    auto nSize = inputParams_.transB ? wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX) :
                                       wShape.GetDim(wDimNum - LAST_FIRST_DIM_INDEX);
    // In static graph mode, when M or N is 0, the framework will skip tiling check and return empty tensors, but when
    // K = 0, tiling should be able to intercept.
    OP_CHECK_IF(
        xShape.GetShapeSize() == 0 || wShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "x/weight",
            (std::string("x=") + std::to_string(xShape.GetShapeSize()) + ", w=" + std::to_string(wShape.GetShapeSize()))
                .c_str(),
            "The shape of x/weight cannot be an empty tensor, check k dimension."),
        return false);
    inputParams_.mSize = mSize;
    inputParams_.kSize = kSize;
    inputParams_.nSize = nSize;
    return true;
}

bool Mc2GroupedQbmmTiling::SetMKNList()
{
    if (inputParams_.groupType == SPLIT_M) {
        mList_[0] = -1;
        kList_[0] = static_cast<int32_t>(inputParams_.kSize);
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    } else {
        mList_[0] = static_cast<int32_t>(inputParams_.mSize);
        kList_[0] = -1;
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    }
    return true;
}

ge::graphStatus Mc2GroupedQbmmTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
                OP_LOGE(inputParams_.opName, "Failed to analyze context_ info."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2GroupedQbmmTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    if (retM != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retM);
        return ge::GRAPH_FAILED;
    }
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    if (retK != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retK);
        return ge::GRAPH_FAILED;
    }
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    if (retN != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retN);
        return ge::GRAPH_FAILED;
    }
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2GroupedQbmmTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.mmTilingData.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.mmTilingData.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.mmTilingData.depthA1 = basicTiling_.depthA1;
    tilingData_.mmTilingData.depthB1 = basicTiling_.depthB1;
    tilingData_.mmTilingData.stepM = basicTiling_.stepM;
    tilingData_.mmTilingData.stepN = basicTiling_.stepN;
    tilingData_.mmTilingData.stepKa = basicTiling_.stepKa;
    tilingData_.mmTilingData.stepKb = basicTiling_.stepKb;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.mmTilingData.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.mmTilingData.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.mmTilingData.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void Mc2GroupedQbmmTiling::SetKernelType()
{
    // 以选择主模板设置kernelType, 0: dequant fixp随路（包含K轴分组）；1：dequant vector计算；2：perGroup-perBlock
    inputParams_.kernelType = 0UL;
    // mx K轴分组当前是独立的模板，后续归一
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        return;
    }
    // perGroup-perBlock(GB)有独立pertile模板
    if (inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE) {
        inputParams_.kernelType = 2UL;
        return;
    }
    // pertensor-pertensor且没有后处理的bias，都可以走dequant fixp随路
    bool isPertensorCube =
        inputParams_.aQuantMode <= QuantMode::PERTENSOR_MODE && inputParams_.bQuantMode == QuantMode::PERTENSOR_MODE;
    bool isBiasEpilogue =
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias && inputParams_.biasDtype != ge::DT_INT32;
    // 如果bias bf16/fp16/fp32，需mix模板进行后处理
    if (isPertensorCube && !isBiasEpilogue) {
        return;
    }
    bool isScaleEpilogue = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64);
    // 后处理的bias和（scale非64bits && ！isPertensorCube）需要走dequant vec模板
    if (isBiasEpilogue || isScaleEpilogue) {
        inputParams_.kernelType = 1UL;
    }
}

uint64_t Mc2GroupedQbmmTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(inputParams_.kernelType));
}

ge::graphStatus Mc2GroupedQbmmTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2GroupedQbmmTiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(sizeof(tilingData_) % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", sizeof(tilingData_)),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

void Mc2GroupedQbmmTiling::PrintQuantParams()
{
    GMMQuantParams &params = tilingData_.gmmQuantParams;
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.groupNum << ", activeType = " << params.activeType
        << ", aQuantMode = " << params.aQuantMode << ", bQuantMode = " << params.bQuantMode
        << ", singleX=" << static_cast<int32_t>(params.singleX)
        << ", singleW = " << static_cast<int32_t>(params.singleW)
        << ", singleY = " << static_cast<int32_t>(params.singleY)
        << ", groupType = " << static_cast<int32_t>(params.groupType)
        << ", groupListType = " << static_cast<uint32_t>(params.groupListType)
        << ", hasBias = " << static_cast<int32_t>(params.hasBias);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
}

void Mc2GroupedQbmmTiling::CalBasicBlock()
{
    bool isGBQuantMode =
        inputParams_.aQuantMode == QuantMode::PERGROUP_MODE && inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE;
    basicTiling_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseM = !inputParams_.transA ?
                             CeilAlign(basicTiling_.baseM, CUBE_BLOCK) :
                             CeilAlign(basicTiling_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
    if (isGBQuantMode) {
        // 不管M/K轴分组，单单单场景下，N不变，可以确定baseN
        if (inputParams_.nSize <= PER_BLOCK_GROUP_SIZE || basicTiling_.baseM > PER_BLOCK_GROUP_SIZE) {
            basicTiling_.baseN = PER_BLOCK_GROUP_SIZE;
        } else {
            basicTiling_.baseN = GmmConstant::BASIC_BLOCK_SIZE_256;
        }
        basicTiling_.baseK = PER_BLOCK_GROUP_SIZE;
        return;
    }
    basicTiling_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseN = inputParams_.transB ?
                             CeilAlign(basicTiling_.baseN, CUBE_BLOCK) :
                             CeilAlign(basicTiling_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
    basicTiling_.baseK = CeilAlign(
        std::min(GetShapeWithDataType(GmmConstant::BASIC_BLOCK_SIZE_128, inputParams_.aDtype), inputParams_.kSize),
        GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));

    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        basicTiling_.baseK = CeilAlign(basicTiling_.baseK, MXFP_BASEK_FACTOR); // mx_mmad requires basek align to 64
        bool isFp4Input = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
        if (isFp4Input && !inputParams_.transB) {
            // 64: mx_mmad requires the inner axis to align to 64
            basicTiling_.baseN = CeilAlign(basicTiling_.baseN, static_cast<uint64_t>(64));
        }
    }

    if (basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C > aicoreParams_.l0cSize) {
        uint64_t maxBaseN = aicoreParams_.l0cSize / (basicTiling_.baseM * DATA_SIZE_L0C);
        basicTiling_.baseN = std::min(basicTiling_.baseN, maxBaseN);
        basicTiling_.baseN =
            !inputParams_.transB ?
                FloorAlign(basicTiling_.baseN, CUBE_BLOCK) :
                FloorAlign(basicTiling_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
    }
}

bool Mc2GroupedQbmmTiling::IsBiasInL1() const
{
    // 目前仅int8进bias int32需要进L1
    return inputParams_.hasBias && inputParams_.biasDtype == ge::DT_INT32;
}

ge::graphStatus Mc2GroupedQbmmTiling::CalL1Tiling()
{
    basicTiling_.stepM = 1UL;
    basicTiling_.stepN = 1UL;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, basicTiling_.baseM);
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, basicTiling_.baseN);
    basicTiling_.singleCoreK = inputParams_.kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t totalL1Size = aicoreParams_.l1Size;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        (basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) ? DB_SIZE : 1;
    uint64_t singleCoreBiasSize = IsBiasInL1() ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize =
        inputParams_.bQuantMode == QuantMode::PERCHANNEL_MODE && inputParams_.kernelType == 0 ?
            basicTiling_.baseN * scaleDtypeSize :
            0;
    uint64_t usedSize = singleCoreBiasSize + singleCoreScaleSize;
    OP_CHECK_IF(
        totalL1Size <= usedSize,
        OP_LOGE(context_->GetNodeName(), "L1 space overflow. L1Size: %lu, used space: %lu", totalL1Size, usedSize),
        return ge::GRAPH_FAILED);
    uint64_t leftL1Size = totalL1Size - usedSize;
    return CalL1Depth(leftL1Size);
}

ge::graphStatus Mc2GroupedQbmmTiling::CalL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        if (inputParams_.groupType == SPLIT_M) {
            baseScaleASize =
                GetSizeWithDataType(CeilAlign(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), 2UL) * basicTiling_.baseM,
                                    inputParams_.perTokenScaleDtype);
            baseScaleBSize =
                GetSizeWithDataType(CeilAlign(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), 2UL) * basicTiling_.baseN,
                                    inputParams_.scaleDtype);
        } else {
            baseScaleASize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseM, // 2 is dim value of last scale dim
                inputParams_.perTokenScaleDtype);
            baseScaleBSize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseN, // 2 is dim value of last pertokenScale dim
                inputParams_.scaleDtype);
        }
    }
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
                OP_LOGE(context_->GetNodeName(), "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                        baseL1Size),
                return ge::GRAPH_FAILED);
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    basicTiling_.depthA1 = std::max(depthASec, depthBSec);
    basicTiling_.depthB1 = basicTiling_.depthA1;
    if (basicTiling_.depthA1 * baseL1Size > leftL1Size) {
        basicTiling_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        basicTiling_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }
    CalStepKs();
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        CalScaleFactors();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2GroupedQbmmTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V120) {
        return depthInit;
    }
    if (perDepthSize == 0) {
        perDepthSize = 1;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
        while ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
               (depthScale * baseKSize) > GmmConstant::BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
            (depthScale * baseKSize) >= GmmConstant::BASIC_BLOCK_SIZE_256) {
            if (baseKSize == 0) {
                baseKSize = 1;
            }
            depthScale = GmmConstant::BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

void Mc2GroupedQbmmTiling::CalStepKs()
{
    // depthA,depthB 为1时，stepka, stepkb 只能是1.
    basicTiling_.stepKa = basicTiling_.depthA1 == 1UL ? 1UL : basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 == 1UL ? 1UL : basicTiling_.depthB1 / DB_SIZE;

    if (basicTiling_.stepKa * basicTiling_.baseK > inputParams_.kSize) {
        basicTiling_.stepKa = CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }

    if (basicTiling_.stepKb * basicTiling_.baseK >= inputParams_.kSize) {
        basicTiling_.stepKb = CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }
    // G-B量化场景下，限制stepK最大为4, 防止issue queue阻塞
    if (inputParams_.aQuantMode == QuantMode::PERGROUP_MODE && inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE) {
        basicTiling_.stepKa = std::min(basicTiling_.stepKa, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
        basicTiling_.stepKb = std::min(basicTiling_.stepKb, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
    }
    if (basicTiling_.stepKa >= basicTiling_.stepKb && basicTiling_.stepKa * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa && basicTiling_.stepKb * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }

    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void Mc2GroupedQbmmTiling::CalScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                                                  inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize =
        GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1Size / (basicTiling_.depthA1 * baseScaleASize + basicTiling_.depthB1 * baseScaleBSize));

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    if (baseScaleASize == 0) {
        baseScaleASize = 1;
    }
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    if (baseScaleBSize == 0) {
        baseScaleBSize = 1;
    }
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    if (basicTiling_.scaleFactorA <= scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorA * basicTiling_.depthA1 * baseScaleASize);
        basicTiling_.scaleFactorB = std::min(
            static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)), basicTiling_.scaleFactorB);
    } else if (basicTiling_.scaleFactorB <= scaleInit && basicTiling_.scaleFactorA > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorB * basicTiling_.depthB1 * baseScaleBSize);
        basicTiling_.scaleFactorA = std::min(
            static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)), basicTiling_.scaleFactorA);
    } else if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -=
            (scaleInit * basicTiling_.depthB1 * baseScaleBSize + scaleInit * basicTiling_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)),
                                      basicTiling_.scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)),
                                      basicTiling_.scaleFactorB - scaleInit);
        basicTiling_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        basicTiling_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

uint64_t Mc2GroupedQbmmTiling::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        // 2: 判断是否是偶数
        OP_CHECK_IF(shapeSize % 2 != 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "shapeSize", std::to_string(shapeSize).c_str(),
                                              "must be even when dtype is FLOAT4/INT4"),
                    return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2UL;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t Mc2GroupedQbmmTiling::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}
} // namespace Mc2GroupedMatmulTiling
} // namespace optiling
