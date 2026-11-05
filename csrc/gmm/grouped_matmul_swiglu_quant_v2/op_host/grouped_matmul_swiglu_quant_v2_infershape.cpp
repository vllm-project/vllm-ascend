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
 * \file grouped_matmul_swiglu_quant_proto.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace ops {
const int64_t X_INDEX = 0;
const int64_t WEIGHT_INDEX = 3;
const int64_t WEIGHTSCALE_DIM_PERTOKEN = 2;
const int64_t WEIGHTSCALE_INDEX = 4;
const int64_t M_DIM_INDEX = 0;
const int64_t DIM_LEN = 2;
const int64_t SPLIT_RATIO = 2;
const int64_t OUT_DIM_LEN = 3;
const int64_t N_SPLIT_RATIO = 128;
constexpr size_t GMMSQ_INDEX_ATTR_QUANT_DTYPE = 3UL;
constexpr size_t GMMSQ_INDEX_ATTR_QUANT_MODE = 2UL;
constexpr size_t QUANT_MODE_MX_TYPE = 2;
constexpr size_t QUANT_MODE_PERTOKEN_TYPE = 0;
constexpr int64_t DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE = -1;

static std::set<std::string> GmmDavidSupportSoc = {"Ascend950"};
static const std::unordered_set<ge::DataType> DavidSupportedInputDtypes = {
    ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT8_E4M3FN,
    ge::DataType::DT_FLOAT4_E2M1, ge::DataType::DT_INT8, ge::DataType::DT_HIFLOAT8};
bool  isSupportedInputDtypeForDavid(ge::DataType dtype)
{
    return DavidSupportedInputDtypes.find(dtype) != DavidSupportedInputDtypes.end();
}

static ge::graphStatus InferShape4GroupedMatmulSwigluQuantV2(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *weightScaleShape = context->GetDynamicInputShape(WEIGHTSCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShape);
    int64_t m = xShape->GetDim(M_DIM_INDEX);
    int64_t nDimIndex = weightScaleShape->GetDimNum() - 1;
    auto outScaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScaleShape);
    if (nDimIndex == OUT_DIM_LEN) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const bool *transposeWeightPtr = attrs->GetBool(WEIGHTSCALE_INDEX);
        const bool transposeWeight = (transposeWeightPtr != nullptr ? *transposeWeightPtr : false);
        nDimIndex = transposeWeight ? weightScaleShape->GetDimNum() - OUT_DIM_LEN :
                                        weightScaleShape->GetDimNum() - DIM_LEN;
        int64_t dimValue = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex));
        int64_t n = 0;
        if (dimValue == DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE) {
            n = dimValue;
        } else {
            n = static_cast<int64_t>(Ops::Base::CeilDiv(weightScaleShape->GetDim(nDimIndex), N_SPLIT_RATIO));
        }
        outScaleShape->SetDimNum(OUT_DIM_LEN);
        outScaleShape->SetDim(0, m);
        outScaleShape->SetDim(1, n);
        outScaleShape->SetDim(2, SPLIT_RATIO); // 设置outScaleShape的第2维度
    } else {
        outScaleShape->SetDimNum(1);
        outScaleShape->SetDim(0, m);
    }

    int64_t dimValue = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex));
    int64_t n = 0;
    if (dimValue == DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE) {
        n = dimValue;
    } else {
        n = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex) / SPLIT_RATIO);
        if (weightScaleShape->GetDimNum() == WEIGHTSCALE_DIM_PERTOKEN) {
            n = static_cast<int64_t>(weightScaleShape->GetDim(1) / SPLIT_RATIO);
        }
    }
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(DIM_LEN);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmulSwigluQuantV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outDtype = attrs->GetInt(GMMSQ_INDEX_ATTR_QUANT_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDtype);
    const int64_t* quantMode = attrs->GetInt(GMMSQ_INDEX_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        auto xDtype = context->GetInputDataType(X_INDEX);
        auto weightDtype = context->GetDynamicInputDataType(WEIGHT_INDEX, 0);
        OP_CHECK_IF(!isSupportedInputDtypeForDavid(xDtype) || !isSupportedInputDtypeForDavid(weightDtype),
                OP_LOGE(context->GetNodeName(), "Invalid Input on this platform, expected FLOAT8_E4M3,"
                            "FLOAT8_E5M2, FLOAT4_E2M1, INT_8, HIFLOAT8, but actual value of x is %s, weight is %s.",
                            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str()), return GRAPH_FAILED);

        OP_CHECK_IF(*quantMode != QUANT_MODE_MX_TYPE && *quantMode != QUANT_MODE_PERTOKEN_TYPE,
                OP_LOGE(context->GetNodeName(), "On this platform, quantMode should be 0(Pertoken) or 2(MX),"
                            " but actual value is %ld.", *quantMode), return GRAPH_FAILED);
    }
    auto weightScaleDtype = context->GetDynamicInputDataType(WEIGHTSCALE_INDEX, 0);
    if (*quantMode == QUANT_MODE_MX_TYPE) {
        if (weightScaleDtype == ge::DataType::DT_FLOAT8_E8M0) {
            context->SetOutputDataType(1, DataType::DT_FLOAT8_E8M0);
        }  else {
            OP_LOGE(context->GetNodeName(), "In mx quant mode, quantMode should be 2, but actual value is %ld.", *quantMode);
            return GRAPH_FAILED;
        }
    } else if (*quantMode == QUANT_MODE_PERTOKEN_TYPE) {
        context->SetOutputDataType(1, DataType::DT_FLOAT);
    }
    context->SetOutputDataType(0, static_cast<ge::DataType>(*outDtype));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSwigluQuantV2)
    .InferShape(InferShape4GroupedMatmulSwigluQuantV2)
    .InferDataType(InferDataType4GroupedMatmulSwigluQuantV2);
} // namespace ops
