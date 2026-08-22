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
 * \file dequant_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "dequant_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// check Q, KV, OUT 类型 全量化
ge::graphStatus DequantChecker::CheckDataTypeSupportFullquant(const FiaTilingInfo &fiaInfo)
{
    // {Q, KV, attenOut}
    const std::vector<std::tuple<ge::DataType, ge::DataType, ge::DataType>> fullQuantDtypeSupported = {
        {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_BF16}};

    std::tuple<ge::DataType, ge::DataType, ge::DataType> inOutDtypeTuple = {fiaInfo.inputQType, fiaInfo.inputKvType,
                                                                            fiaInfo.outputType};
    if (ge::GRAPH_SUCCESS != CheckValueSupport(inOutDtypeTuple, fullQuantDtypeSupported)) {
        std::string paramNames = "query, key/value, attention_out";
        std::string incorrectDtypes = std::string(ToString(fiaInfo.inputQType).c_str()) + ", " +
                                      std::string(ToString(fiaInfo.inputKvType).c_str()) + ", " +
                                      std::string(ToString(fiaInfo.outputType).c_str());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(), incorrectDtypes.c_str(),
            "The dtypes of these parameters support only the following combinations: "
            "[(INT8, INT8, BF16), (FLOAT8_E4M3FN, FLOAT8_E4M3FN, FLOAT16), "
            "(FLOAT8_E4M3FN, FLOAT8_E4M3FN, BF16), (HIFLOAT8, HIFLOAT8, FLOAT16), "
            "(HIFLOAT8, HIFLOAT8, BF16)]");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(fiaInfo.inputKvType != fiaInfo.opParamInfo.value.desc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value",
            ToString(fiaInfo.opParamInfo.value.desc->GetDataType()).c_str(),
            "In fullquant scenario, the datatype of value must be the same as the datatype of key"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// MLA dequantscale dtype:fp32
ge::graphStatus DequantChecker::CheckDequantScaleDtypeMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT ||
                    fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT) {
        std::string paramNames = "dequant_scale_query, key_antiquant_scale and value_antiquant_scale";
        std::string incorrectDtypes =
            std::string(ToString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str()) + ", " +
            std::string(ToString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str()) + ", " +
            std::string(ToString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(), incorrectDtypes.c_str(),
            "The datatype of dequant_scale_query, key_antiquant_scale and value_antiquant_scale"
            " must be FLOAT32 in MLA fullquant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// GQA perblock dequantscale dtype:fp32
ge::graphStatus DequantChecker::CheckDequantScaleDtypeGQAPerblock(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT) {
        std::string paramNames = "dequant_scale_query, key_antiquant_scale, value_antiquant_scale";
        std::string incorrectDtypes =
            std::string(ToString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str()) + ", " +
            std::string(ToString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str()) + ", " +
            std::string(ToString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(), incorrectDtypes.c_str(),
            "In per-block quant scenario, the datatypes of dequantScaleQuery, keyAntiquantScale "
            "and valueAntiquantScale must be float32");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(fiaInfo.opParamInfo.quantScale1.tensor != nullptr &&
        fiaInfo.opParamInfo.quantScale1.desc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale1",
            ToString(fiaInfo.opParamInfo.quantScale1.desc->GetDataType()).c_str(),
            "In per-block quant scenario, the datatype of quant_scale1 must be float32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleDtypeGQAPertensor(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *quantScale1Desc = fiaInfo.opParamInfo.quantScale1.desc;
    const gert::CompileTimeTensorDesc *deqScale1Desc = fiaInfo.opParamInfo.deqScale1.desc;
    const gert::CompileTimeTensorDesc *deqScale2Desc = fiaInfo.opParamInfo.deqScale2.desc;
    if (quantScale1Desc == nullptr || deqScale1Desc == nullptr || deqScale2Desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(quantScale1Desc, QUANT_SCALE1_NAME),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale1",
                    ToString(quantScale1Desc->GetDataType()).c_str(),
                    "In per-tensor quant scenario, the datatype of quant_scale1 must be FLOAT"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(deqScale1Desc, DEQUANT_SCALE1_NAME),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "dequant_scale1",
                    ToString(deqScale1Desc->GetDataType()).c_str(),
                    "In per-tensor quant scenario, "
                    "the datatype of dequant_scale1 must be within the range {UINT64, FLOAT}"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(deqScale2Desc, DEQUANT_SCALE2_NAME),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "dequant_scale2",
                    ToString(deqScale2Desc->GetDataType()).c_str(),
                    "In per-tensor quant scenario, "
                    "the datatype of dequant_scale2 must be within the range {UINT64, FLOAT}"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// MXFP8 dequantscale dtype:fp8_e8m0
ge::graphStatus DequantChecker::CheckDequantScaleDtypeMXFP8Fullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                    ToString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str(),
                    "In MXFP8 fullquant scenario, the datatype of dequant_scale_query must be FLOAT8_E8M0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                    ToString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str(),
                    "In MXFP8 fullquant scenario, the datatype of key_antiquant_scale must be FLOAT8_E8M0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                    ToString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str(),
                    "In MXFP8 fullquant scenario, the datatype of value_antiquant_scale must be FLOAT8_E8M0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.quantScale1.tensor != nullptr &&
                fiaInfo.opParamInfo.quantScale1.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale1",
                    ToString(fiaInfo.opParamInfo.quantScale1.desc->GetDataType()).c_str(),
                    "In MXFP8 fullquant scenario, the datatype of quant_scale1 must be FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// FP8 GQA fullquant dequantscale dtype:fp32
ge::graphStatus DequantChecker::CheckDequantScaleDtypeFP8GQAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.opParamInfo.dequantScaleQuery.desc == nullptr ||
        fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr ||
        fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                    DataTypeToSerialString(fiaInfo.opParamInfo.dequantScaleQuery.desc->GetDataType()).c_str(),
                    "In FP8 GQA fullquant scenario, the datatype of dequant_scale_query must be FLOAT32"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                    DataTypeToSerialString(fiaInfo.opParamInfo.keyAntiquantScale.desc->GetDataType()).c_str(),
                    "In FP8 GQA fullquant scenario, the datatype of key_antiquant_scale must be FLOAT32"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                    DataTypeToSerialString(fiaInfo.opParamInfo.valueAntiquantScale.desc->GetDataType()).c_str(),
                    "In FP8 GQA fullquant scenario, the datatype of value_antiquant_scale must be FLOAT32"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.opParamInfo.quantScale1.tensor != nullptr &&
                fiaInfo.opParamInfo.quantScale1.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "quant_scale1",
                    DataTypeToSerialString(fiaInfo.opParamInfo.quantScale1.desc->GetDataType()).c_str(),
                    "In FP8 GQA fullquant scenario, the datatype of quant_scale1 must be FLOAT32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleDtypeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleDtypeMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeGQAPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeGQAPertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeMXFP8Fullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleDtypeFP8GQAFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// MLA全量化 Q: per-token-head (3), KV: per-tensor (0)
ge::graphStatus DequantChecker::CheckDequantModeMLAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        *fiaInfo.opParamInfo.queryQuantMode != PER_TOKEN_HEAD_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "query_quant_mode",
            std::to_string(*fiaInfo.opParamInfo.queryQuantMode).c_str(),
            "In MLA fullquant scenario, query_quant_mode must be per-token-head(3)"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*fiaInfo.opParamInfo.keyAntiquantMode != PER_CHANNEL_MODE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_antiquant_mode",
            std::to_string(*fiaInfo.opParamInfo.keyAntiquantMode).c_str(),
            "In MLA fullquant scenario, key_antiquant_mode must be per-tensor(0)"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *fiaInfo.opParamInfo.valueAntiquantMode != PER_CHANNEL_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "value_antiquant_mode",
            std::to_string(*fiaInfo.opParamInfo.valueAntiquantMode).c_str(),
            "In MLA fullquant scenario, value_antiquant_mode must be per-tensor(0)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// per-block qkv antiquantMode(7)
ge::graphStatus DequantChecker::CheckDequantModeGQAPerblock(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((*fiaInfo.opParamInfo.keyAntiquantMode != PER_BLOCK_MODE ||
                 *fiaInfo.opParamInfo.valueAntiquantMode != PER_BLOCK_MODE ||
                 *fiaInfo.opParamInfo.queryQuantMode != PER_BLOCK_MODE),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName,
            "query_quant_mode, key_antiquant_mode and value_antiquant_mode",
            (std::to_string(*fiaInfo.opParamInfo.queryQuantMode) + ", "
                + std::to_string(*fiaInfo.opParamInfo.keyAntiquantMode) + " and "
                + std::to_string(*fiaInfo.opParamInfo.valueAntiquantMode)).c_str(),
            "In per-block scenario, query_quant_mode, key_antiquant_mode "
            "and value_antiquant_mode must be per-block(7)"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// per-tensor qkv antiquantMode(0)
ge::graphStatus DequantChecker::CheckDequantModeGQAPertensor(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((*fiaInfo.opParamInfo.keyAntiquantMode != PER_CHANNEL_MODE ||
                 *fiaInfo.opParamInfo.valueAntiquantMode != PER_CHANNEL_MODE ||
                 *fiaInfo.opParamInfo.queryQuantMode != PER_CHANNEL_MODE),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName,
            "query_quant_mode, key_antiquant_mode and value_antiquant_mode",
            (std::to_string(*fiaInfo.opParamInfo.queryQuantMode) + ", "
                + std::to_string(*fiaInfo.opParamInfo.keyAntiquantMode) + " and "
                + std::to_string(*fiaInfo.opParamInfo.valueAntiquantMode)).c_str(),
            "In per-tensor scenario, query_quant_mode, key_antiquant_mode "
            "and value_antiquant_mode must be per-tensor(0)"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// mxfp9 qk:per-token-group v:per-channel-group
ge::graphStatus DequantChecker::CheckDequantModeMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        *fiaInfo.opParamInfo.queryQuantMode != PER_TOKEN_GROUP_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "query_quant_mode",
            std::to_string(*fiaInfo.opParamInfo.queryQuantMode).c_str(),
            "In MXFP8 fullquant scenario, query_quant_mode must be per-token-group(6)"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*fiaInfo.opParamInfo.keyAntiquantMode != PER_TOKEN_GROUP_MODE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_antiquant_mode",
            std::to_string(*fiaInfo.opParamInfo.keyAntiquantMode).c_str(),
            "In MXFP8 fullquant scenario, key_antiquant_mode must be per-token-group(6)"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *fiaInfo.opParamInfo.valueAntiquantMode != PER_CHANNEL_GROUP_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "value_antiquant_mode",
            std::to_string(*fiaInfo.opParamInfo.valueAntiquantMode).c_str(),
            "In MXFP8 fullquant scenario, value_antiquant_mode must be per-channel-group(8)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// fp8 gqa qk:per-token-head v:per-tensor-head
ge::graphStatus DequantChecker::CheckDequantModeFP8GQAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        *fiaInfo.opParamInfo.queryQuantMode != PER_TOKEN_HEAD_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "query_quant_mode",
            std::to_string(*fiaInfo.opParamInfo.queryQuantMode).c_str(),
            "In FP8 GQA fullquant scenario, query_quant_mode must be per-token-head(3)"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*fiaInfo.opParamInfo.keyAntiquantMode != PER_TOKEN_HEAD_MODE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_antiquant_mode",
                    std::to_string(*fiaInfo.opParamInfo.keyAntiquantMode).c_str(),
                    "In FP8 GQA fullquant scenario, key_antiquant_mode must be per-token-head(3)"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *fiaInfo.opParamInfo.valueAntiquantMode != PER_TENSOR_HEAD_MODE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "value_antiquant_mode",
            std::to_string(*fiaInfo.opParamInfo.valueAntiquantMode).c_str(),
            "In FP8 GQA fullquant scenario, value_antiquant_mode must be per-tensor-head(0)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// check dequant scale Mode
ge::graphStatus DequantChecker::CheckDequantModeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantModeGQAPertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeGQAPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeMXFP8Fullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantModeFP8GQAFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckTensorExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
                                                          const std::string &quantModeName,
                                                          const std::string &inputName) const
{
    OP_CHECK_IF(
        tensor == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, inputName.c_str(), "empty",
            ("In " + quantModeName + " scenario, " + inputName.c_str() + " cannot be empty").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckTensorNotExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
                                                             const std::string &quantModeName,
                                                             const std::string &inputName) const
{
    OP_CHECK_IF(tensor != nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, inputName.c_str(), "not empty",
                ("In " + quantModeName + " scenario, " + inputName.c_str() + " must be empty").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus DequantChecker::CheckExistenceNoquant(const FiaTilingInfo &fiaInfo) const
{
    struct ParamInfo {
        const char* paraName;
        const gert::Tensor *tensor;
    };
    
    std::vector<ParamInfo> quantParams = {
        {"deq_scale1", fiaInfo.opParamInfo.deqScale1.tensor},
        {"quant_scale1", fiaInfo.opParamInfo.quantScale1.tensor},
        {"deq_scale2", fiaInfo.opParamInfo.deqScale2.tensor},
        {"dequant_scale_query", fiaInfo.opParamInfo.dequantScaleQuery.tensor},
        {"antiquant_scale", fiaInfo.opParamInfo.antiquantScale.tensor},
        {"antiquant_offset", fiaInfo.opParamInfo.antiquantOffset.tensor},
        {"key_antiquant_scale", fiaInfo.opParamInfo.keyAntiquantScale.tensor},
        {"key_antiquant_offset", fiaInfo.opParamInfo.keyAntiquantOffset.tensor},
        {"value_antiquant_scale", fiaInfo.opParamInfo.valueAntiquantScale.tensor},
        {"value_antiquant_offset", fiaInfo.opParamInfo.valueAntiquantOffset.tensor}
    };
    
    for (const auto& param : quantParams) {
        if (param.tensor != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, param.paraName,
                ("In no quant scenario, quant related param: " +
                    std::string(param.paraName) + " must be empty").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckExistencePertensorFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "per-tensor quant";
    const gert::Tensor *quantScale1Tensor = fiaInfo.opParamInfo.quantScale1.tensor;
    const gert::Tensor *deqScale1Tensor = fiaInfo.opParamInfo.deqScale1.tensor;
    const gert::Tensor *deqScale2Tensor = fiaInfo.opParamInfo.deqScale2.tensor;
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, quantScale1Tensor, quantModeName, "quantScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, deqScale1Tensor, quantModeName, "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, deqScale2Tensor, quantModeName, "deqScale2")) {
        return ge::GRAPH_FAILED;
    }

    // shapesize != 0
    OP_CHECK_IF(
        (quantScale1Tensor->GetStorageShape().GetShapeSize() == 0 ||
         deqScale1Tensor->GetStorageShape().GetShapeSize() == 0 ||
         deqScale2Tensor->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "dequant_scale1, quant_scale1 and dequant_scale2",
            ("In " + quantModeName + " scenario, these scale tensors should not be empty").c_str()),
        return ge::GRAPH_FAILED);

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                          quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                          quantModeName, "valueAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                          quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckExistenceMLAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "MLA fullquant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantScale1.tensor,
                                                          quantModeName, "quantScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckExistencePerblockFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "per-block quant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// mxfp8
ge::graphStatus DequantChecker::CheckExistenceMXFP8Fullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "mxfp8 quant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantScale2.tensor,
                                                          quantModeName, "quantScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantOffset2.tensor,
                                                          quantModeName, "quantOffset2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// fp8 gqa
ge::graphStatus DequantChecker::CheckExistenceFP8GQAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    string quantModeName = "fp8 gqa quant";
    // Q/K/V antiquantScale
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.dequantScaleQuery.tensor,
                                                       quantModeName, "dequantScaleQuery") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantScale.tensor,
                                                       quantModeName, "keyAntiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantScale.tensor,
                                                       quantModeName, "valueAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // 不支持offset
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantOffset.tensor,
                                                          quantModeName, "antiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyAntiquantOffset.tensor,
                                                          quantModeName, "keyAntiquantOffset") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.valueAntiquantOffset.tensor,
                                                          quantModeName, "valueAntiquantOffset")) {
        return ge::GRAPH_FAILED;
    }

    // 其他量化方式参数不能存在
    if (ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale1.tensor, quantModeName,
                                                          "deqScale1") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.deqScale2.tensor, quantModeName,
                                                          "deqScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantScale2.tensor,
                                                          quantModeName, "quantScale2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.quantOffset2.tensor,
                                                          quantModeName, "quantOffset2") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.antiquantScale.tensor,
                                                          quantModeName, "antiquantScale") ||
        ge::GRAPH_SUCCESS != CheckTensorNotExistFullquant(fiaInfo, fiaInfo.opParamInfo.keyRopeAntiquantScale.tensor,
                                                          quantModeName, "keyRopeAntiquantScale")) {
        return ge::GRAPH_FAILED;
    }

    // fp8 gqa 仅支持PA场景
    if (ge::GRAPH_SUCCESS != CheckTensorExistFullquant(fiaInfo, fiaInfo.opParamInfo.blockTable.tensor,
                                                       quantModeName, "block_table")) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// GQA per-tensor
ge::graphStatus DequantChecker::CheckFeaturePertensorFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    
    // 不支持后量化
    OP_CHECK_IF(fiaInfo.isOutQuantEnable,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                    ToString(fiaInfo.outputType).c_str(),
                "In per-tensor quant scenario, postQuant is not supported"),
                return ge::GRAPH_FAILED);

    // 不支持 rope
    OP_CHECK_IF(fiaInfo.mlaMode != MlaMode::NO_MLA,
                OP_LOGE(fiaInfo.opName, "In per-tensor quant scenario, rope is not supported."),
                return ge::GRAPH_FAILED);

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2 or 3",
                "In per-tensor quant scenario, pseType cannot be 2 or 3"),
                return ge::GRAPH_FAILED);

    // 不支持PA_NZ
    const uint32_t keyDim = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDim == DIM_NUM_5, OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key", "5D",
                "In per-tensor quant scenario, PA_NZ is not supported"),
                return ge::GRAPH_FAILED);

    // 不支持QS=1
    std::string queryShapeStr = ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape());
    OP_CHECK_IF(fiaInfo.s1Size == 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query", queryShapeStr.c_str(),
                "In per-tensor quant scenario, the S axis of query cannot be 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GQA per-block
ge::graphStatus DequantChecker::CheckFeaturePerblockFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2/3",
                    "In per-block quant scenario, pse_type cannot be 2/3"),
                return ge::GRAPH_FAILED);
    // 不支持pse
    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                    "In per-block quant scenario, pse_shift must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_padding_size and kv_padding_size",
            "In per-block quant scenario, query_padding_size and kv_padding_size must be empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
            "In per-block quant scenario, key_shared_prefix and value_shared_prefix must be empty"),
        return ge::GRAPH_FAILED);
                
    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key and value",
            "In per-block quant scenario, key/value storage mode cannot be tensor list"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "block_table",
                    "In per-block quant scenario, block_table must be empty"),
                return ge::GRAPH_FAILED);

    // 不支持 softmaxlse
    OP_CHECK_IF(fiaInfo.softmaxLseFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "softmax_lse",
                    "In per-block quant scenario, softmax_lse must be empty"),
                return ge::GRAPH_FAILED);

    // 不支持mask
    OP_CHECK_IF(fiaInfo.attenMaskFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "atten_mask",
                    "In per-block quant scenario, atten_mask must be empty"),
                return ge::GRAPH_FAILED);

    // innerPrecise 仅支持 0/1
    OP_CHECK_IF((fiaInfo.innerPrecise != HIGH_PRECISION) && (fiaInfo.innerPrecise != HIGH_PERFORMANCE),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inner_precise",
                    std::to_string(fiaInfo.innerPrecise).c_str(),
                    "In per-block quant scenario, inner_precise must be 0 or 1"),
                return ge::GRAPH_FAILED);

    // 不支持 rope
    OP_CHECK_IF(fiaInfo.mlaMode != MlaMode::NO_MLA,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_rope and key_rope",
                    "In per-block quant scenario, rope is not supported"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 不支持左padding、tensorlist、prefix、pse、alibipse
ge::graphStatus DequantChecker::CheckFeatureMLAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2/3",
                    "In MLA fullquant scenario, pse_type cannot be 2/3"),
                return ge::GRAPH_FAILED);
    
    // 不支持pse
    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                    "In MLA fullquant scenario, pse_shift must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_padding_size and kv_padding_size",
            "In MLA fullquant scenario, query_padding_size and kv_padding_size must be empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
            "In MLA fullquant scenario, key_shared_prefix and value_shared_prefix must be empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key and value",
                    "In MLA fullquant scenario, key/value storage mode cannot be tensor list"),
                return ge::GRAPH_FAILED);
    
    // MLA int8全量化仅支持PA_NZ
    const uint32_t keyDim = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 &&
        !(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && keyDim == DIM_NUM_5),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "PA_BnBsH",
                    "In MLA fullquant scenario when the datatype of query is INT8, input_layout must be PA_NZ"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 不支持左padding、tensorlist、prefix、pse、alibipse
ge::graphStatus DequantChecker::CheckFeatureMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    // 不支持alibipse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2/3",
                    "In MXFP8 fullquant scenario, pse_type cannot be 2/3"),
                return ge::GRAPH_FAILED);
    
    // 不支持pse
    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                    "In MXFP8 fullquant scenario, pse_shift must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                    "query_padding_size and kv_padding_size",
                    "In MXFP8 fullquant scenario, query_padding_size and kv_padding_size must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                    "key_shared_prefix and value_shared_prefix",
                    "In MXFP8 fullquant scenario, key_shared_prefix and value_shared_prefix must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key and value",
                    "In MXFP8 fullquant scenario, key/value storage mode cannot be tensor list"),
                return ge::GRAPH_FAILED);
    
    // innerPrecise 仅支持 0/1
    OP_CHECK_IF((fiaInfo.innerPrecise != HIGH_PRECISION) && (fiaInfo.innerPrecise != HIGH_PERFORMANCE),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inner_precise",
                    std::to_string(fiaInfo.innerPrecise).c_str(),
                    "In MXFP8 fullquant scenario, inner_precise must be 0 or 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 不支持rope、pse、alibipse、左padding、tensorlist、prefix；仅支持page attention
ge::graphStatus DequantChecker::CheckFeatureFP8GQAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2 or 3",
                    "In FP8 GQA fullquant scenario, pse_type should not be 2/3"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                    "In FP8 GQA fullquant scenario, pse is not supported"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                    "query_padding_size and kv_padding_size",
                    "In FP8 GQA fullquant scenario, left padding is not supported"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                    "key_shared_prefix and value_shared_prefix",
                    "In FP8 GQA fullquant scenario, system prefix is not supported"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key and value",
                    "In FP8 GQA fullquant scenario, key/value tensorlist is not supported"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.ropeMode != RopeMode::NO_ROPE,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_rope and key_rope",
                    "In FP8 GQA fullquant scenario, rope is not supported"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 全量化不同量化方式 不支持的特性
ge::graphStatus DequantChecker::CheckFeatureSupportFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeaturePertensorFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePerblockFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureMXFP8Fullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureFP8GQAFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check scale shape
ge::graphStatus DequantChecker::CheckDequantScaleKVMLAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    // kv: [1]
    if ((fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr) &&
        (fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape().GetDimNum() != NUM1 ||
         fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape());
        std::string reasonMsg = "In MLA fullquant scenario, the shape of keyAntiquantScale must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr) &&
        (fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape().GetDimNum() != NUM1 ||
         fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape());
        std::string reasonMsg = "In MLA fullquant scenario, the shape of valueAntiquantScale must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleQueryMLAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    // query
    const gert::Shape queryInputShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const uint32_t queryDimNum = queryInputShape.GetDimNum();
    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const uint32_t dequantScaleQueryDimNum = dequantScaleQueryShape.GetDimNum();

    if (fiaInfo.qLayout == FiaLayout::BSH) {
        if ((queryDimNum != dequantScaleQueryDimNum)) {
            std::string dimStr =
                std::to_string(dequantScaleQueryDimNum) + "D" + " and " + std::to_string(queryDimNum) + "D" ;
            std::string reasonMsg =
                "In MLA fullquant scenario, the shape dim of dequantScaleQuery and query must be the same";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "dequant_scale_query and query",
                dimStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        if ((queryInputShape.GetDim(DIM_NUM_0) != dequantScaleQueryShape.GetDim(DIM_NUM_0)) ||
            (queryInputShape.GetDim(DIM_NUM_1) != dequantScaleQueryShape.GetDim(DIM_NUM_1)) ||
            (*fiaInfo.opParamInfo.numHeads != dequantScaleQueryShape.GetDim(DIM_NUM_2))) {
            std::string shapeStr = ToStringRaw(dequantScaleQueryShape);
            std::string reasonMsg = "In MLA fullquant scenario, the shape of dequantScaleQuery "
                "should be equal to the shape of query (with the 2nd dim replaced by numHeads)";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.qLayout == FiaLayout::BSND || fiaInfo.qLayout == FiaLayout::BNSD ||
               fiaInfo.qLayout == FiaLayout::TND) {
        if ((dequantScaleQueryDimNum != queryDimNum - NUM1)) {
            std::string dimStr = std::to_string(dequantScaleQueryDimNum) + "D";
            std::string reasonMsg =
                "In MLA fullquant scenario, the shape dim of dequantScaleQuery should be equal to queryDimNum - 1";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                dimStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        for (uint32_t i = 0; i < dequantScaleQueryDimNum; i++) {
            if ((queryInputShape.GetDim(i) != dequantScaleQueryShape.GetDim(i))) {
                std::string shapeStr = ToStringRaw(dequantScaleQueryShape);
                std::string reasonMsg = "In MLA fullquant scenario, the " + std::to_string(i) +
                    "th dim of dequantScaleQuery should be equal to the " + std::to_string(i) + "th dim of query";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                    shapeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapePertensor(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    // shape:[1]
    if ((fiaInfo.opParamInfo.quantScale1.tensor != nullptr) &&
        (fiaInfo.opParamInfo.quantScale1.tensor->GetStorageShape().GetDimNum() != NUM1 ||
         fiaInfo.opParamInfo.quantScale1.tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.quantScale1.tensor->GetStorageShape());
        std::string reasonMsg = "In per-tensor quant scenario, the shape of quantScale1 must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "quantScale1", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.opParamInfo.deqScale1.tensor != nullptr) &&
        (fiaInfo.opParamInfo.deqScale1.tensor->GetStorageShape().GetDimNum() != NUM1 ||
         fiaInfo.opParamInfo.deqScale1.tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.deqScale1.tensor->GetStorageShape());
        std::string reasonMsg = "In per-tensor quant scenario, the shape of dequantScale1 must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequantScale1", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.opParamInfo.deqScale2.tensor != nullptr) &&
        (fiaInfo.opParamInfo.deqScale2.tensor->GetStorageShape().GetDimNum() != NUM1 ||
         fiaInfo.opParamInfo.deqScale2.tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.deqScale2.tensor->GetStorageShape());
        std::string reasonMsg = "In per-tensor quant scenario, the shape of dequantScale2 must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequantScale2", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleBnNBsDShapeMXFP8(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.kvLayout != FiaLayout::BnNBsD) {
        return ge::GRAPH_SUCCESS;
    }
    // key
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const uint32_t keyAntiquantScaleDimNum = keyAntiquantScaleShape.GetDimNum();
    // value
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
    const uint32_t valueAntiquantScaleDimNum = valueAntiquantScaleShape.GetDimNum();
    const uint32_t mxfp8BlockSize = 64;

    // pa场景 BnNBsD支持0/1轴非连续
    int32_t dimIndex = 0;
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyAntiquantScaleDimNum, keyAntiquantScaleShape, fiaInfo.kScaleStrides, dimIndex)) &&
            (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "keyAntiquantScale",
            "In MXFP8 Fullquant BnNBsD scenario, only 0th and 1st axis of keyAntiquantScale can be non-contiguous, the "
            + std::to_string(dimIndex) + "th axis of keyAntiquantScale must be contiguous"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(valueAntiquantScaleDimNum, valueAntiquantScaleShape, fiaInfo.vScaleStrides, dimIndex)) &&
            (dimIndex != 0 && dimIndex != 1)),
         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "valueAntiquantScale",
            std::string("In MXFP8 Fullquant BnNBsD scenario, ") +
            "only 0th and 1st axis of valueAntiquantScale can be non-contiguous, the" +
            std::to_string(dimIndex) + "th axis of valueAntiquantScale must be contiguous"),
        return ge::GRAPH_FAILED);

    // BnNBsD pa key --[blocknum, kv_n, blocksize, k_D/64, 2] // [fzj] 
    if ((fiaInfo.totalBlockNum != keyAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
        (fiaInfo.n2Size != keyAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
        (fiaInfo.blockSize != keyAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
        (CeilDivision(fiaInfo.qkHeadDim, mxfp8BlockSize) != keyAntiquantScaleShape.GetDim(DIM_NUM_3)) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_4) != 2)) {
        std::string shapeStr = ToStringRaw(keyAntiquantScaleShape);
        std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of key is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
            ", the shape of keyAntiquantScale should be [totalBlockNum, n2Size, blockSize, qkHeadDim/64, 2]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    // BnNBsD pa value --[blocknum, kv_n, blocksize/64, v_D, 2]
    if ((fiaInfo.totalBlockNum != valueAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
        (fiaInfo.n2Size != valueAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
        (CeilDivision(fiaInfo.blockSize, static_cast<int32_t>(mxfp8BlockSize)) !=
            valueAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
        (fiaInfo.vHeadDim) != valueAntiquantScaleShape.GetDim(DIM_NUM_3) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_4) != 2)) {
        std::string shapeStr = ToStringRaw(valueAntiquantScaleShape);
        std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of value is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
            ", the shape of valueAntiquantScale should be [totalBlockNum, n2Size, blockSize/64, vHeadDim, 2]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleNZShapeMXFP8(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.kvLayout != FiaLayout::NZ) {
        return ge::GRAPH_SUCCESS;
    }
    // key
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const uint32_t keyAntiquantScaleDimNum = keyAntiquantScaleShape.GetDimNum();
    // value
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
    const uint32_t valueAntiquantScaleDimNum = valueAntiquantScaleShape.GetDimNum();

    const uint32_t mxfp8BlockSize = 64;
    const uint32_t d0 = 16;
    const int32_t d0Temp = 16;

    // pa场景 NZ支持0/1轴非连续
    int32_t dimIndex = 0;
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyAntiquantScaleDimNum, keyAntiquantScaleShape, fiaInfo.kScaleStrides, dimIndex)) &&
            (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "keyAntiquantScale",
            "In MXFP8 Fullquant Nz scenario, only 0th and 1st axis of keyAntiquantScale can be non-contiguous, the "
            + std::to_string(dimIndex) + "th axis of keyAntiquantScale must be contiguous"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(valueAntiquantScaleDimNum, valueAntiquantScaleShape, fiaInfo.vScaleStrides, dimIndex)) &&
            (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "valueAntiquantScale",
            "In MXFP8 Fullquant Nz scenario, only 0th and 1st axis of valueAntiquantScale can be non-contiguous, the "
            + std::to_string(dimIndex) + "th axis of valueAntiquantScale must be contiguous"),
        return ge::GRAPH_FAILED);

    // NZ pa key --[blocknum, kv_n, blocksize/16, k_D/64, 16, 2]
    if ((fiaInfo.totalBlockNum != keyAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
        (fiaInfo.n2Size != keyAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
        (CeilDivision(fiaInfo.blockSize, d0Temp) != keyAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
        (CeilDivision(fiaInfo.qkHeadDim, mxfp8BlockSize) != keyAntiquantScaleShape.GetDim(DIM_NUM_3)) ||
        (d0 != keyAntiquantScaleShape.GetDim(DIM_NUM_4)) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_5) != 2)) {
        std::string shapeStr = ToStringRaw(keyAntiquantScaleShape);
        std::string reasonMsg = "In MXFP8 fullquant scenario when layout of key is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
            ", the shape of keyAntiquantScale should be [totalBlockNum, n2Size, blockSize/d0Temp, qkHeadDim/64, d0, 2]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    // NZ pa value --[blocknum, kv_n, v_D/16, blocksize/64, D0, 2]
    if ((fiaInfo.totalBlockNum != valueAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
        (fiaInfo.n2Size != valueAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
        (CeilDivision(fiaInfo.vHeadDim, d0) != valueAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
        (CeilDivision(fiaInfo.blockSize, static_cast<int32_t>(mxfp8BlockSize)) !=
            valueAntiquantScaleShape.GetDim(DIM_NUM_3)) ||
        (d0 != valueAntiquantScaleShape.GetDim(DIM_NUM_4)) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_5) != 2)) {
        std::string shapeStr = ToStringRaw(valueAntiquantScaleShape);
        std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of value is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
            ", the shape of valueAntiquantScale should be [totalBlockNum, n2Size, vHeadDim/d0, blockSize/64, d0, 2]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapeMXFP8(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    // query
    const gert::Shape queryInputShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const uint32_t queryDimNum = queryInputShape.GetDimNum();
    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const uint32_t dequantScaleQueryDimNum = dequantScaleQueryShape.GetDimNum();
    // key
    const gert::Shape keyInputShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const uint32_t keyDimNum = keyInputShape.GetDimNum();
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const uint32_t keyAntiquantScaleDimNum = keyAntiquantScaleShape.GetDimNum();
    // value
    const gert::Shape valueInputShape = fiaInfo.opParamInfo.value.shape->GetStorageShape();
    const uint32_t valueDimNum = valueInputShape.GetDimNum();
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
    const uint32_t valueAntiquantScaleDimNum = valueAntiquantScaleShape.GetDimNum();

    const int64_t mxfp8BlockSize = 64;
    uint32_t scaleKVDim = fiaInfo.pageAttentionFlag ? ((fiaInfo.kvLayout == FiaLayout::NZ) ? 6 : 5) : 4;
    bool enableMxfp8Decode = (fiaInfo.gSize * fiaInfo.s1Size <= 80);

    // 非pa场景 不支持kv kvscale不连续
    int32_t dimIndex = 0;  // 占位，非pa场景用不到
    OP_CHECK_IF((!fiaInfo.pageAttentionFlag && 
        ((CheckTensorContiguous(keyDimNum, keyInputShape, fiaInfo.keyStrides, dimIndex) != ge::GRAPH_SUCCESS) ||
        (CheckTensorContiguous(valueDimNum, valueInputShape, fiaInfo.valueStrides, dimIndex) != ge::GRAPH_SUCCESS) ||
        (CheckTensorContiguous(keyAntiquantScaleDimNum, keyAntiquantScaleShape, fiaInfo.kScaleStrides, dimIndex) != ge::GRAPH_SUCCESS) ||
        (CheckTensorContiguous(valueAntiquantScaleDimNum, valueAntiquantScaleShape, fiaInfo.vScaleStrides, dimIndex) != ge::GRAPH_SUCCESS))),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key/value/keyAntiquantScale/valueAntiquantScale",
            "In non-PA scenarios, MXFP8 full quantization does not support non-contiguous tensors"),
            return ge::GRAPH_FAILED); 
    
    // qscale dim支持4维[T, N, D//64, 2]和5维[N2, T, G, D//64, 2]
    if ((dequantScaleQueryDimNum != 4 && dequantScaleQueryDimNum != 5)) {
        std::string dimStr = std::to_string(dequantScaleQueryDimNum) + "D";
        std::string reasonMsg = "In MXFP8 fullquant scenario, when the layout of query is " +
            std::string(LayoutToSerialString(fiaInfo.qLayout)) + " and the layout of key/value is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
            ", the shape dim of dequantScaleQuery should be 4 or 5";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
            dimStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    // kvscale dim
    if ((keyAntiquantScaleDimNum != scaleKVDim || valueAntiquantScaleDimNum != scaleKVDim)) {
        std::string dimStr = std::to_string(keyAntiquantScaleDimNum) + "D and " +
            std::to_string(valueAntiquantScaleDimNum) + "D";
        std::string reasonMsg = "In MXFP8 fullquant scenario, when the layout of query is " +
            std::string(LayoutToSerialString(fiaInfo.qLayout)) + " and the layout of key/value is " +
            std::string(LayoutToSerialString(fiaInfo.kvLayout)) + ", the shape dim of keyAntiquantScale and "
            "valueAntiquantScale should both be " + std::to_string(scaleKVDim) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "key_antiquant_scale and value_antiquant_scale",
            dimStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // TND prefill/decode qkv scale
    // prefill query -- [T, N, D//64, 2]; decode query -- [N2, T, G, D//64, 2]
    if (dequantScaleQueryDimNum == 4) {
        if (((queryInputShape.GetDim(DIM_NUM_0) != dequantScaleQueryShape.GetDim(DIM_NUM_0)) ||
            (fiaInfo.n1Size != dequantScaleQueryShape.GetDim(DIM_NUM_1)) ||
            (CeilDivision(queryInputShape.GetDim(DIM_NUM_2), mxfp8BlockSize) !=
            dequantScaleQueryShape.GetDim(DIM_NUM_2)) ||
            (2 != dequantScaleQueryShape.GetDim(DIM_NUM_3)))) {
            std::string shapeStr = ToStringRaw(dequantScaleQueryShape);
            std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of query is " +
                std::string(LayoutToSerialString(fiaInfo.qLayout)) +
                ", the shape of dequantScaleQuery should be [T, N, D//64, 2]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (((fiaInfo.n2Size != dequantScaleQueryShape.GetDim(DIM_NUM_0)) ||
            (queryInputShape.GetDim(DIM_NUM_0) != dequantScaleQueryShape.GetDim(DIM_NUM_1)) ||
            (fiaInfo.gSize != dequantScaleQueryShape.GetDim(DIM_NUM_2)) ||
            (CeilDivision(queryInputShape.GetDim(DIM_NUM_2), mxfp8BlockSize) !=
            dequantScaleQueryShape.GetDim(DIM_NUM_3)) ||
            (2 != dequantScaleQueryShape.GetDim(DIM_NUM_4)))) {
            std::string shapeStr = ToStringRaw(dequantScaleQueryShape);
            std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of query is " +
                std::string(LayoutToSerialString(fiaInfo.qLayout)) +
                ", the shape of dequantScaleQuery should be [N2, T, G, D//64, 2]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    // shape告警
    if (!enableMxfp8Decode && dequantScaleQueryDimNum == DIM_NUM_5) {
        OP_LOGW(fiaInfo.opName, "In the mxfp8 prefill scenario, to achieve better performance, "
                "the query scale is recommended to use the shape [T, N, D//64, 2], "
                "which corresponds to [%ld, %ld, %ld, %ld] in this case.", queryInputShape.GetDim(DIM_NUM_0),
                fiaInfo.n1Size, CeilDivision(queryInputShape.GetDim(DIM_NUM_2), mxfp8BlockSize), 2);
    }
    if (enableMxfp8Decode && dequantScaleQueryDimNum == DIM_NUM_4) {
        OP_LOGW(fiaInfo.opName, "In the mxfp8 decode scenario, to achieve better performance, "
                "the query scale is recommended to use the shape [N2, T, G, D//64, 2], "
                "which corresponds to [%ld, %ld, %ld, %ld, %ld] in this case.", fiaInfo.n2Size,
                queryInputShape.GetDim(DIM_NUM_0), fiaInfo.gSize,
                CeilDivision(queryInputShape.GetDim(DIM_NUM_2), mxfp8BlockSize), 2);
    }
    // kv scale
    if (fiaInfo.pageAttentionFlag) {
        OP_CHECK_IF((fiaInfo.kvLayout != FiaLayout::BnNBsD && fiaInfo.kvLayout != FiaLayout::NZ),
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(fiaInfo.opName, "key",
                    LayoutToSerialString(fiaInfo.kvLayout).c_str(),
                    "In MXFP8 fullquant scenario, the layout of key only supports BnNBsD or NZ when PA is enabled"),
                return ge::GRAPH_FAILED);
        // pa场景 BNBD/NZ支持0/1轴非连续
        OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyDimNum, keyInputShape, fiaInfo.keyStrides, dimIndex)) &&
                    (dimIndex != 0 && dimIndex != 1)),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key",
                    "In MXFP8 Fullquant BnNBsD/NZ scenario, only 0th and 1st axis of key can be non-contiguous, the "
                    + std::to_string(dimIndex) + "th axis of key must be contiguous"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(valueDimNum,
                        valueInputShape, fiaInfo.valueStrides, dimIndex)) &&
                        (dimIndex != 0 && dimIndex != 1)),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "value",
                    "In MXFP8 Fullquant BnNBsD/NZ scenario, only 0th and 1st axis of value can be non-contiguous, the "
                    + std::to_string(dimIndex) + "th axis of value must be contiguous"),
                return ge::GRAPH_FAILED);
        if (ge::GRAPH_SUCCESS != CheckDequantScaleBnNBsDShapeMXFP8(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (ge::GRAPH_SUCCESS != CheckDequantScaleNZShapeMXFP8(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // key -- [T, N, D/64, 2]
        if ((keyInputShape.GetDim(DIM_NUM_0) != keyAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
            (fiaInfo.n2Size != keyAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
            (CeilDivision(keyInputShape.GetDim(DIM_NUM_2), mxfp8BlockSize) != keyAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
            (keyAntiquantScaleShape.GetDim(DIM_NUM_3) != 2)) {
            std::string shapeStr = ToStringRaw(keyAntiquantScaleShape);
            std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of key is " +
                std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
                ", the shape of keyAntiquantScale should be [T, N, D/64, 2]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        // value -- [T/64, N, D, 2]
        if (fiaInfo.isMaxWorkspace && fiaInfo.qLayout == FiaLayout::TND) {
            return ge::GRAPH_SUCCESS;
        }
        if ((GetValueScaleActualKVlens4TNDNoPa(fiaInfo) != valueAntiquantScaleShape.GetDim(DIM_NUM_0)) ||
            (fiaInfo.n2Size != valueAntiquantScaleShape.GetDim(DIM_NUM_1)) ||
            (valueInputShape.GetDim(DIM_NUM_2) != valueAntiquantScaleShape.GetDim(DIM_NUM_2)) ||
            (valueAntiquantScaleShape.GetDim(DIM_NUM_3) != 2)) {
            std::string shapeStr = ToStringRaw(valueAntiquantScaleShape);
            std::string reasonMsg = "In MXFP8 fullquant scenario, when layout of value is " +
                std::string(LayoutToSerialString(fiaInfo.kvLayout)) +
                ", the shape of valueAntiquantScale should be [T/64, N, D, 2]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

int64_t DequantChecker::GetValueScaleActualKVlens4TNDNoPa(const FiaTilingInfo &fiaInfo) const
{
    int64_t valueScaleKVlens = 0;
    if (fiaInfo.isMaxWorkspace) {
        return valueScaleKVlens;
    }
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        return valueScaleKVlens;
    }
    const int64_t mxfp8BlockSize = 64;
    for (int64_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        if (actualSeqLengthsKvTensor->GetData<int64_t>() == nullptr) {
            return valueScaleKVlens;
        }
        if (bIdx == 0) {
            valueScaleKVlens = CeilDivision(actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx], mxfp8BlockSize);
        } else {
            int64_t curSeqLengthData = (actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx] -
            actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1]);
            valueScaleKVlens += CeilDivision(curSeqLengthData, mxfp8BlockSize);
        }
    }
    return valueScaleKVlens;
}

ge::graphStatus DequantChecker::CheckQuantScale1ShapeMXFP8(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Tensor *quantScale1Tensor = fiaInfo.opParamInfo.quantScale1.tensor;
    // shape:[1]
    if ((quantScale1Tensor != nullptr) &&
        (quantScale1Tensor->GetStorageShape().GetDimNum() != NUM1 || quantScale1Tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(quantScale1Tensor->GetStorageShape());
        std::string reasonMsg = "In MxFP8 fullquant scenario, the shape of quantScale1 must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "quantScale1", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckQuantScale1ShapePerblock(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Tensor *quantScale1Tensor = fiaInfo.opParamInfo.quantScale1.tensor;
    // shape:[1]
    if ((quantScale1Tensor != nullptr) &&
        (quantScale1Tensor->GetStorageShape().GetDimNum() != NUM1 || quantScale1Tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(quantScale1Tensor->GetStorageShape());
        std::string reasonMsg = "In per-block quant scenario, the shape of quantScale1 must be [1]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "quantScale1", shapeStr.c_str(), reasonMsg.c_str());

        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// GQA antiquantscale
ge::graphStatus DequantChecker::CheckDequantScaleShapePerblockNTD(const FiaTilingInfo &fiaInfo,
    const gert::Shape &dequantScaleQueryShape, const gert::Shape &keyAntiquantScaleShape,
    const gert::Shape &valueAntiquantScaleShape, uint32_t fp8QBlockSize, uint32_t fp8KVBlockSize) const
{
    if ((dequantScaleQueryShape.GetDimNum() != DIM_NUM_3) ||
        (keyAntiquantScaleShape.GetDimNum() != DIM_NUM_3) ||
        (valueAntiquantScaleShape.GetDimNum() != DIM_NUM_3)) {
        std::string dimStr = std::to_string(dequantScaleQueryShape.GetDimNum()) + "D, " +
                             std::to_string(keyAntiquantScaleShape.GetDimNum()) + "D and " +
                             std::to_string(valueAntiquantScaleShape.GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            fiaInfo.opName, "dequant_scale_query, key_antiquant_scale and value_antiquant_scale",
            dimStr.c_str(),
            "In per-block quant scenario, when layout is NTD_TND, the shape dims of "
            "dequantScaleQuery, keyAntiquantScale and valueAntiquantScale must all be 3D");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // dequantScaleQuery [N, T/128+B, D/256]
    if ((dequantScaleQueryShape.GetDim(DIM_NUM_0) != fiaInfo.n1Size) ||
        (dequantScaleQueryShape.GetDim(DIM_NUM_1) != fiaInfo.qTSize / fp8QBlockSize + fiaInfo.bSize) ||
        (dequantScaleQueryShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "dequant_scale_query",
            ToStringRaw(dequantScaleQueryShape).c_str(),
            "In per-block quant scenario, when layout is NTD_TND, the shape of dequantScaleQuery "
            "should be [n1Size, qTSize/128+bSize, qkHeadDim/256]");
        return ge::GRAPH_FAILED;
    }
    // keyAntiquantScale/valueAntiquantScale [N, T/256+B, D/256]
    if ((keyAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.n2Size) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.qkHeadDim, fp8KVBlockSize))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "key_antiquant_scale",
            ToStringRaw(keyAntiquantScaleShape).c_str(),
            "In per-block quant scenario, when layout is NTD_TND, the shape of keyAntiquantScale "
            "should be [n2Size, kTSize/256+bSize, qkHeadDim/256]");
        return ge::GRAPH_FAILED;
    }

    if ((valueAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.n2Size) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.kTSize / fp8KVBlockSize + fiaInfo.bSize) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.vHeadDim, fp8KVBlockSize))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "value_antiquant_scale",
            ToStringRaw(valueAntiquantScaleShape).c_str(),
            "In per-block quant scenario, when layout is NTD_TND, the shape of valueAntiquantScale "
            "should be [n2Size, kTSize/256+bSize, vHeadDim/256]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapePerblockNonNTD(const FiaTilingInfo &fiaInfo,
    const gert::Shape &dequantScaleQueryShape, const gert::Shape &keyAntiquantScaleShape,
    const gert::Shape &valueAntiquantScaleShape, uint32_t fp8QBlockSize, uint32_t fp8KVBlockSize) const
{
    if ((dequantScaleQueryShape.GetDimNum() != DIM_NUM_4) ||
        (keyAntiquantScaleShape.GetDimNum() != DIM_NUM_4) ||
        (valueAntiquantScaleShape.GetDimNum() != DIM_NUM_4)) {
        std::string dimStr = std::to_string(dequantScaleQueryShape.GetDimNum()) + "D, " +
                             std::to_string(keyAntiquantScaleShape.GetDimNum()) + "D and " +
                             std::to_string(valueAntiquantScaleShape.GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            fiaInfo.opName, "dequant_scale_query, key_antiquant_scale and value_antiquant_scale",
            dimStr.c_str(),
            "In per-block quant scenario, when layout is not NTD_TND, the shape dims of "
            "dequantScaleQuery, keyAntiquantScale and valueAntiquantScale must all be 4D");
        return ge::GRAPH_FAILED;
    }

    // dequantScaleQuery [B, N, S/128, 1]
    if ((dequantScaleQueryShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
        (dequantScaleQueryShape.GetDim(DIM_NUM_1) != fiaInfo.n1Size) ||
        (dequantScaleQueryShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s1Size, fp8QBlockSize)) ||
        (dequantScaleQueryShape.GetDim(DIM_NUM_3) != NUM1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "dequant_scale_query",
            ToStringRaw(dequantScaleQueryShape).c_str(),
            "In per-block quant scenario, when layout is not NTD_TND, the shape of dequantScaleQuery "
            "should be [bSize, n1Size, s1Size/128, 1]");
        return ge::GRAPH_FAILED;
    }
    // keyAntiquantScale/valueAntiquantScale [B, N, S/256, 1]
    if ((keyAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.n2Size) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize))) ||
        (keyAntiquantScaleShape.GetDim(DIM_NUM_3) != NUM1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "key_antiquant_scale",
            ToStringRaw(keyAntiquantScaleShape).c_str(),
            "In per-block quant scenario, when layout is not NTD_TND, the shape of keyAntiquantScale "
            "should be [bSize, n2Size, s2Size/256, 1]");
        return ge::GRAPH_FAILED;
    }
    if ((valueAntiquantScaleShape.GetDim(DIM_NUM_0) != fiaInfo.bSize) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_1) != fiaInfo.n2Size) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_2) != CeilDivision(fiaInfo.s2Size, int64_t(fp8KVBlockSize))) ||
        (valueAntiquantScaleShape.GetDim(DIM_NUM_3) != NUM1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "value_antiquant_scale",
            ToStringRaw(valueAntiquantScaleShape).c_str(),
            "In per-block quant scenario, when layout is not NTD_TND, the shape of valueAntiquantScale "
            "should be [bSize, n2Size, s2Size/256, 1]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapePerblock(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();

    constexpr uint32_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint32_t fp8KVBlockSize = 256U; // 256 is SInnerSize

    // NTD_TND格式 scale dim = 3
    if (fiaInfo.qLayout == FiaLayout::NTD) {
        return CheckDequantScaleShapePerblockNTD(fiaInfo, dequantScaleQueryShape,
            keyAntiquantScaleShape, valueAntiquantScaleShape, fp8QBlockSize, fp8KVBlockSize);
    } else {
        return CheckDequantScaleShapePerblockNonNTD(fiaInfo, dequantScaleQueryShape,
            keyAntiquantScaleShape, valueAntiquantScaleShape, fp8QBlockSize, fp8KVBlockSize);
    }
}

ge::graphStatus DequantChecker::CheckQuantScale1ShapeFP8GQA(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Tensor *quantScale1Tensor = fiaInfo.opParamInfo.quantScale1.tensor;
    // shape:[1]
    if ((quantScale1Tensor != nullptr) &&
        (quantScale1Tensor->GetStorageShape().GetDimNum() != NUM1 ||
        quantScale1Tensor->GetShapeSize() != NUM1)) {
        std::string shapeStr = ToStringRaw(quantScale1Tensor->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "quant_scale1",
            shapeStr.c_str(),
            "In FP8 GQA fullquant scenario, the shape of quant_scale1 must be [1]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// fp8 gqa q_scale: NT, k_scale: BnNBs, v_scale: N
ge::graphStatus DequantChecker::CheckDequantScaleShapeFP8GQA(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }

    // query scale: NT layout [n1Size, T]
    const gert::Shape queryInputShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const gert::Shape dequantScaleQueryShape = fiaInfo.opParamInfo.dequantScaleQuery.tensor->GetStorageShape();
    const uint32_t dequantScaleQueryDimNum = dequantScaleQueryShape.GetDimNum();
    OP_CHECK_IF(dequantScaleQueryDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
                    std::to_string(dequantScaleQueryDimNum).c_str(),
                    "In FP8 GQA fullquant scenario, the dim num of dequant_scale_query should be 2 (NT layout)"),
                return ge::GRAPH_FAILED);
    if (fiaInfo.n1Size != dequantScaleQueryShape.GetDim(DIM_NUM_0) ||
        queryInputShape.GetDim(DIM_NUM_1) != dequantScaleQueryShape.GetDim(DIM_NUM_1)) {
        std::string actualShape =
            "[" + std::to_string(dequantScaleQueryShape.GetDim(DIM_NUM_0)) + ", " +
            std::to_string(dequantScaleQueryShape.GetDim(DIM_NUM_1)) + "]";
        std::string expectedShape =
            "[" + std::to_string(fiaInfo.n1Size) + ", " +
            std::to_string(queryInputShape.GetDim(DIM_NUM_1)) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "dequant_scale_query",
            actualShape.c_str(),
            ("In FP8 GQA fullquant scenario, "
             "the shape of dequant_scale_query should be " + expectedShape + " (NT layout).").c_str());
        return ge::GRAPH_FAILED;
    }

    // key scale: BnNBs layout [totalBlockNum, n2Size, blockSize]
    const gert::Shape keyInputShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const uint32_t keyDimNum = keyInputShape.GetDimNum();
    const gert::Shape keyAntiquantScaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
    const uint32_t keyAntiquantScaleDimNum = keyAntiquantScaleShape.GetDimNum();

    OP_CHECK_IF(keyAntiquantScaleDimNum != DIM_NUM_3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                    std::to_string(keyAntiquantScaleDimNum).c_str(),
                    "In FP8 GQA fullquant scenario, the dim num of key_antiquant_scale should be 3 (BnNBs layout)"),
                return ge::GRAPH_FAILED);

    if (fiaInfo.totalBlockNum != keyAntiquantScaleShape.GetDim(DIM_NUM_0) ||
        fiaInfo.n2Size != keyAntiquantScaleShape.GetDim(DIM_NUM_1) ||
        fiaInfo.blockSize != keyAntiquantScaleShape.GetDim(DIM_NUM_2)) {
        std::string actualShape =
            "[" + std::to_string(keyAntiquantScaleShape.GetDim(DIM_NUM_0)) + ", " +
            std::to_string(keyAntiquantScaleShape.GetDim(DIM_NUM_1)) + ", " +
            std::to_string(keyAntiquantScaleShape.GetDim(DIM_NUM_2)) + "]";
        std::string expectedShape =
            "[" + std::to_string(fiaInfo.totalBlockNum) + ", " +
            std::to_string(fiaInfo.n2Size) + ", " +
            std::to_string(fiaInfo.blockSize) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
            actualShape.c_str(),
            ("In FP8 GQA fullquant scenario, "
             "the shape of key_antiquant_scale should be " + expectedShape + " (BnNBs layout).").c_str());
        return ge::GRAPH_FAILED;
    }

    // pa场景 key kscale BnNBs支持0/1轴非连续
    int32_t dimIndex = 0;
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyAntiquantScaleDimNum, keyAntiquantScaleShape,
                fiaInfo.kScaleStrides, dimIndex)) && (dimIndex != 0 && dimIndex != 1)),
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant BnNBsD scenarios, "
                        "kscale only supports non-contiguous tensors in dimensions 0 or 1, "
                        "but currently the non-contiguous dimension is dimension %d", dimIndex),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyDimNum, keyInputShape, fiaInfo.keyStrides,
                dimIndex)) && (dimIndex != 0 && dimIndex != 1)),
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant BnNBsD scenarios, "
                        "key only supports non-contiguous tensors in dimensions 0 or 1, "
                        "but currently the non-contiguous dimension is dimension %d", dimIndex),
                return ge::GRAPH_FAILED);

    // value scale: N layout [n2Size]
    const gert::Shape valueInputShape = fiaInfo.opParamInfo.value.shape->GetStorageShape();
    const uint32_t valueDimNum = valueInputShape.GetDimNum();
    const gert::Shape valueAntiquantScaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
    const uint32_t valueAntiquantScaleDimNum = valueAntiquantScaleShape.GetDimNum();
    OP_CHECK_IF(valueAntiquantScaleDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                    std::to_string(valueAntiquantScaleDimNum).c_str(),
                    "In FP8 GQA fullquant scenario, the dim num of value_antiquant_scale should be 1 (N layout)"),
                return ge::GRAPH_FAILED);
    if (fiaInfo.n2Size != valueAntiquantScaleShape.GetDim(DIM_NUM_0)) {
        std::string actualShape = "[" + std::to_string(valueAntiquantScaleShape.GetDim(DIM_NUM_0)) + "]";
        std::string expectedShape = "[" + std::to_string(fiaInfo.n2Size) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
            actualShape.c_str(),
            ("In FP8 GQA fullquant scenario, "
             "the shape of value_antiquant_scale should be " + expectedShape + " (N layout).").c_str());
        return ge::GRAPH_FAILED;
    }

    // pa场景 value BnNBsD支持0/1轴非连续
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(valueDimNum, valueInputShape, fiaInfo.valueStrides,
                dimIndex)) && (dimIndex != 0 && dimIndex != 1)),
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant BnNBsD scenarios, "
                        "value only supports non-contiguous tensors in dimensions 0 or 1, "
                        "but currently the non-contiguous dimension is dimension %d", dimIndex),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GQA FP8 Fullquant stride 校验
ge::graphStatus DequantChecker::CheckStrideFP8GQAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.isTensorV1) {
        return ge::GRAPH_SUCCESS;
    }
  
    const uint64_t scaleRows = (static_cast<uint64_t>(fiaInfo.blockSize) * sizeof(float)) /
        (static_cast<uint64_t>(fiaInfo.qkHeadDim) * sizeof(uint8_t));
    const uint64_t kCacheBlockRows = static_cast<uint64_t>(fiaInfo.blockSize) + scaleRows;
    const uint64_t vCacheBlockRows = static_cast<uint64_t>(fiaInfo.blockSize) + scaleRows;

    // key_antiquant_scale BnNBs
    OP_CHECK_IF(fiaInfo.kScaleStrides == nullptr,
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant scenario, key_antiquant_scale stride is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t kScaleStride1 = kCacheBlockRows * static_cast<uint64_t>(fiaInfo.qkHeadDim) / sizeof(float);
    uint64_t kScaleStride0 = static_cast<uint64_t>(fiaInfo.n2Size) * kScaleStride1;
    if (fiaInfo.kScaleStrides->GetDimNum() != 3 ||
        fiaInfo.kScaleStrides->GetStride(2) != 1 ||
        fiaInfo.kScaleStrides->GetStride(1) != kScaleStride1 ||
        fiaInfo.kScaleStrides->GetStride(0) != kScaleStride0) {
        std::string expStr = "(" + std::to_string(kScaleStride0) + ", " +
            std::to_string(kScaleStride1) + ", 1)";
        std::string actStr = "(" +
            std::to_string(fiaInfo.kScaleStrides->GetStride(0)) + ", " +
            std::to_string(fiaInfo.kScaleStrides->GetStride(1)) + ", " +
            std::to_string(fiaInfo.kScaleStrides->GetStride(2)) + ")";
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
            ("stride is " + actStr + ", should be " + expStr).c_str());
        return ge::GRAPH_FAILED;
    }

    // key BNBD
    OP_CHECK_IF(fiaInfo.keyStrides == nullptr,
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant scenario, key stride is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t keyStride1 = kCacheBlockRows * static_cast<uint64_t>(fiaInfo.qkHeadDim);
    uint64_t keyStride0 = static_cast<uint64_t>(fiaInfo.n2Size) * keyStride1;
    if (fiaInfo.keyStrides->GetDimNum() != 4 ||
        fiaInfo.keyStrides->GetStride(3) != 1 ||
        fiaInfo.keyStrides->GetStride(2) != static_cast<uint64_t>(fiaInfo.qkHeadDim) ||
        fiaInfo.keyStrides->GetStride(1) != keyStride1 ||
        fiaInfo.keyStrides->GetStride(0) != keyStride0) {
        std::string expStr = "(" + std::to_string(keyStride0) + ", " +
            std::to_string(keyStride1) + ", " +
            std::to_string(fiaInfo.qkHeadDim) + ", 1)";
        std::string actStr = "(" +
            std::to_string(fiaInfo.keyStrides->GetStride(0)) + ", " +
            std::to_string(fiaInfo.keyStrides->GetStride(1)) + ", " +
            std::to_string(fiaInfo.keyStrides->GetStride(2)) + ", " +
            std::to_string(fiaInfo.keyStrides->GetStride(3)) + ")";
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key",
            ("stride is " + actStr + ", should be " + expStr).c_str());
        return ge::GRAPH_FAILED;
    }

    // value BNBD
    OP_CHECK_IF(fiaInfo.valueStrides == nullptr,
                OP_LOGE(fiaInfo.opName,
                        "In FP8 GQA fullquant scenario, value stride is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t valStride1 = vCacheBlockRows * static_cast<uint64_t>(fiaInfo.vHeadDim);
    uint64_t valStride0 = static_cast<uint64_t>(fiaInfo.n2Size) * valStride1;
    if (fiaInfo.valueStrides->GetDimNum() != 4 ||
        fiaInfo.valueStrides->GetStride(3) != 1 ||
        fiaInfo.valueStrides->GetStride(2) != static_cast<uint64_t>(fiaInfo.vHeadDim) ||
        fiaInfo.valueStrides->GetStride(1) != valStride1 ||
        fiaInfo.valueStrides->GetStride(0) != valStride0) {
        std::string expStr = "(" + std::to_string(valStride0) + ", " +
            std::to_string(valStride1) + ", " +
            std::to_string(fiaInfo.vHeadDim) + ", 1)";
        std::string actStr = "(" +
            std::to_string(fiaInfo.valueStrides->GetStride(0)) + ", " +
            std::to_string(fiaInfo.valueStrides->GetStride(1)) + ", " +
            std::to_string(fiaInfo.valueStrides->GetStride(2)) + ", " +
            std::to_string(fiaInfo.valueStrides->GetStride(3)) + ")";
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "value",
            ("stride is " + actStr + ", should be " + expStr).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleKVMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapePertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQuantScale1ShapePerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQuantScale1ShapeMXFP8(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQuantScale1ShapeFP8GQA(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDequantScaleShapeCrossFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleQueryMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapePerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapeMXFP8(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDequantScaleShapeFP8GQA(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check 不同量化方式 支持的数据类型
ge::graphStatus DequantChecker::CheckInputDTypeFullquant(const FiaTilingInfo &fiaInfo)
{
    if (enableQPerTokenHeadKVPerTensor_) {  // MLA 全量化 QKV : fp8_e4m3/int8
        if (!(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN || fiaInfo.inputQType == ge::DT_INT8 ||
              fiaInfo.inputQType == ge::DT_HIFLOAT8)) {
            std::string paramNames = "query, key and value";
            std::string incorrectDtypes = ToString(fiaInfo.inputQType) + ", " +
                                          ToString(fiaInfo.inputKvType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(),
                incorrectDtypes.c_str(),
                "In MLA fullquant scenario, the datatype of query, key and value must be "
                "FLOAT8_E4M3FN, INT8 or HIFLOAT8");
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(fiaInfo.outputType != ge::DT_BF16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(fiaInfo.outputType).c_str(),
                "In MLA fullquant scenario, the datatype of attention_out must be BF16"),
            return ge::GRAPH_FAILED);

        if (!(fiaInfo.inputQRopeType == ge::DT_BF16 && fiaInfo.inputKRopeType == ge::DT_BF16)) {
            std::string paramNames = "query_rope and key_rope";
            std::string incorrectDtypes = ToString(fiaInfo.inputQRopeType) + ", " +
                                          ToString(fiaInfo.inputKRopeType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(),
                incorrectDtypes.c_str(),
                "In MLA fullquant scenario, the datatype of query_rope and key_rope must be BF16");
            return ge::GRAPH_FAILED;
        }
    } else if (enableQKVPertensorQuant_) {  // GQA Pertensor QKV:int8
        OP_CHECK_IF((fiaInfo.inputQType != ge::DT_INT8),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "query",
                ToString(fiaInfo.inputQType).c_str(),
                "In per-tensor quant scenario, the datatype of query must be INT8"),
            return ge::GRAPH_FAILED);
    } else if (enableQKVPerblockQuant_) {  // GQA perblock fp8_e4m3/hifp8
        OP_CHECK_IF((fiaInfo.inputQType != ge::DT_FLOAT8_E4M3FN && fiaInfo.inputQType != ge::DT_HIFLOAT8),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "query",
                ToString(fiaInfo.inputQType).c_str(),
                "In per-block quant scenario, the datatype of query must be FLOAT8_E4M3FN or HIFLOAT8"),
            return ge::GRAPH_FAILED);
    } else if (enableQKVMxfp8FullQuant_) {
        if (!(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN)) {
            std::string paramNames = "query and key_value";
            std::string incorrectDtypes = ToString(fiaInfo.inputQType) + ", " +
                                          ToString(fiaInfo.inputKvType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(),
                incorrectDtypes.c_str(),
                "In MXFP8 fullquant scenario, the datatype of query and key_value must be FLOAT8_E4M3FN");
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(fiaInfo.outputType).c_str(),
                "In MXFP8 fullquant scenario, the datatype of attention_out must be BF16 or FLOAT16"),
            return ge::GRAPH_FAILED);
        // rope
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            if (!(fiaInfo.inputQRopeType == ge::DT_BF16 && fiaInfo.inputKRopeType == ge::DT_BF16)) {
                std::string paramNames = "query_rope and key_rope";
                std::string incorrectDtypes = ToString(fiaInfo.inputQRopeType) + ", " +
                                              ToString(fiaInfo.inputKRopeType);
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, paramNames.c_str(),
                    incorrectDtypes.c_str(),
                    "In MXFP8 fullquant scenario, when rope is split, the datatype of "
                    "query_rope and key_rope must be BF16");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(fiaInfo.outputType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                    ToString(fiaInfo.outputType).c_str(),
                    "In MXFP8 fullquant scenario, when rope is split, the datatype of attention_out must be BF16"),
                return ge::GRAPH_FAILED);
        }
    } else if (enableQKPerTokenHeadVPerHead_) {
        std::string qkvDtypes = std::string(DataTypeToSerialString(fiaInfo.inputQType).c_str()) + ", " +
                                std::string(DataTypeToSerialString(fiaInfo.inputKvType).c_str());
        OP_CHECK_IF(!(fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "query, key/value",
                        qkvDtypes.c_str(),
                        "In FP8 GQA fullquant scenario, query and key/value datatype should be FLOAT8_E4M3FN"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                        DataTypeToSerialString(fiaInfo.outputType).c_str(),
                        "In FP8 GQA fullquant scenario, attention_out datatype should be BF16 or FLOAT16"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check per-block layout
ge::graphStatus DequantChecker::CheckInputLayoutPerblock(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPerblockQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<std::string> unsupportedLayoutList = {
        "BNSD_NBSD", "BSH_NBSD", "BSH_BNSD", "BSND_BNSD", "BSND_NBSD", "NTD", "TND", "TND_NTD"};
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;

    if (std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), inputLayout) !=
        unsupportedLayoutList.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            inputLayout.c_str(),
            "In per-block quant scenario, input_layout cannot be within the range "
            "{BNSD_NBSD, BSH_NBSD, BSH_BNSD, BSND_BNSD, BSND_NBSD, NTD, TND, TND_NTD}");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// check per-tensor layout
ge::graphStatus DequantChecker::CheckInputLayoutPertensor(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVPertensorQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> unsupportedLayoutList = {
        "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "BSH_BNSD", "BSND_BNSD", "TND", "NTD", "NTD_TND", "TND_NTD"};

    if (std::find(unsupportedLayoutList.begin(), unsupportedLayoutList.end(), inputLayout) !=
        unsupportedLayoutList.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            inputLayout.c_str(),
            "In per-tensor quant scenario, input_layout cannot be within the range"
            " {BNSD_NBSD, BSND_NBSD, BSH_NBSD, BSH_BNSD, BSND_BNSD, TND, NTD, NTD_TND, TND_NTD}");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// check mla fullqunat layout
ge::graphStatus DequantChecker::CheckInputLayoutMLAFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQPerTokenHeadKVPerTensor_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> supportedLayoutListFP8 = {
        "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"};
    const std::vector<std::string> supportedLayoutListINT8 = {
        "BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"};
    
    if ((fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN || fiaInfo.inputQType == ge::DT_HIFLOAT8) &&
        (std::find(supportedLayoutListFP8.begin(), supportedLayoutListFP8.end(), inputLayout) ==
        supportedLayoutListFP8.end())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            fiaInfo.opParamInfo.layOut,
            "In MLA fullquant scenario, when the datatype of query is FLOAT8_E4M3FN or HIFLOAT8, "
            "layout must be BSH/BSND/BNSD/TND/BSH_NBSD/BSND_NBSD/BNSD_NBSD/TND_NTD");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.inputQType == ge::DT_INT8 &&
        (std::find(supportedLayoutListINT8.begin(), supportedLayoutListINT8.end(), inputLayout) ==
        supportedLayoutListINT8.end())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            fiaInfo.opParamInfo.layOut,
            "In MLA fullquant scenario, when the datatype of query is INT8, "
            "layout must be BSH/BSND/TND/BSH_NBSD/BSND_NBSD/TND_NTD");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// check mxfp8 fullqunat layout
ge::graphStatus DequantChecker::CheckInputLayoutMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const
{
    if (!enableQKVMxfp8FullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> supportedLayoutList = {"TND"};
    
    OP_CHECK_IF((std::find(supportedLayoutList.begin(), supportedLayoutList.end(), inputLayout) ==
                supportedLayoutList.end()),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                fiaInfo.opParamInfo.layOut,
                "In MXFP8 fullquant scenario, input layout must be TND"),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// check fp8 gqa fullquant layout: input layout仅支持NTD_TND, q layout仅支持NTD
ge::graphStatus DequantChecker::CheckInputLayoutFP8GQAFullquant(const FiaTilingInfo &fiaInfo)
{
    if (!enableQKPerTokenHeadVPerHead_) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> supportedLayoutList = {"NTD_TND"};

    OP_CHECK_IF((std::find(supportedLayoutList.begin(), supportedLayoutList.end(), inputLayout) ==
                supportedLayoutList.end()),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                fiaInfo.opParamInfo.layOut,
                "In FP8 GQA fullquant scenario, input layout must be NTD_TND"),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.qLayout != FiaLayout::NTD,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                    LayoutToSerialString(fiaInfo.qLayout),
                    "In FP8 GQA fullquant scenario, query layout only support NTD"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvLayout != FiaLayout::BnNBsD,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                    LayoutToSerialString(fiaInfo.kvLayout),
                    "In FP8 GQA fullquant scenario, key/value layout only support BnNBsD"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check layout
ge::graphStatus DequantChecker::CheckInputLayoutFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckInputLayoutPerblock(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutPertensor(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutMLAFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutMXFP8Fullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputLayoutFP8GQAFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check n1 size
ge::graphStatus DequantChecker::CheckN1SizeFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (enableQPerTokenHeadKVPerTensor_) {
        static const std::set<uint32_t> supportNumHeadForMLAFullQuant = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
        OP_CHECK_IF((supportNumHeadForMLAFullQuant.find(fiaInfo.n1Size) == supportNumHeadForMLAFullQuant.end()),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_heads",
                std::to_string(fiaInfo.n1Size).c_str(),
                "In MLA fullquant scenario, num_heads must be in {1, 2, 4, 8, 16, 32, 64, 128}"),
            return ge::GRAPH_FAILED);
    } else {  // QN <= 256
        OP_CHECK_IF((fiaInfo.n1Size > N1_LIMIT || fiaInfo.n1Size < NUM1),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_heads",
                std::to_string(fiaInfo.n1Size).c_str(),
                "In GQA fullquant scenario, num_heads must be within the range [1, 256]"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((fiaInfo.n1Size % fiaInfo.n2Size != 0),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "num_heads and num_key_value_heads",
            (std::to_string(fiaInfo.n1Size) + " and " + std::to_string(fiaInfo.n2Size)).c_str(),
            "In fullquant scenario, num_heads must be a multiple of num_key_value_heads"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check n2 size
ge::graphStatus DequantChecker::CheckN2SizeFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (enableQPerTokenHeadKVPerTensor_) {
        OP_CHECK_IF((fiaInfo.n2Size != NUM1),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_key_value_heads",
                std::to_string(fiaInfo.n2Size).c_str(),
                "In MLA fullquant scenario, num_key_value_heads must be 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((fiaInfo.n2Size > N2_LIMIT || fiaInfo.n2Size < NUM1),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_key_value_heads",
                std::to_string(fiaInfo.n2Size).c_str(),
                "In GQA fullquant scenario, num_key_value_heads must be within the range [1, 256]"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check g size
ge::graphStatus DequantChecker::CheckGSizeFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (enableQPerTokenHeadKVPerTensor_) { // G <= 128
        if (fiaInfo.gSize < NUM1 || fiaInfo.gSize > NUM_128) {
            std::string qShape = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape());
            std::string kShape = ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            std::string shapesStr = qShape + " and " + kShape;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and key",
                shapesStr.c_str(),
                "In MLA fullquant scenario, the axis G must be within the range [1, 128]");
            return ge::GRAPH_FAILED;
        }
    } else {  // G <= 64
        if (fiaInfo.gSize < NUM1 || fiaInfo.gSize > G_LIMIT) {
            std::string qShape = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape());
            std::string kShape = ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            std::string shapesStr = qShape + " and " + kShape;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and key",
                shapesStr.c_str(),
                "In GQA fullquant scenario, the axis G must be within the range [1, 64]");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// check d size
ge::graphStatus DequantChecker::CheckDSizeFullquant(const FiaTilingInfo &fiaInfo) const
{
    if (enableQPerTokenHeadKVPerTensor_) {
        if (fiaInfo.qkHeadDim != NUM_512) {
            std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape());
            std::string reasonMsg = "In MLA fullquant scenario, the axis D of query must be 512";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query", shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (enableQKVPerblockQuant_) {
        if (fiaInfo.qkHeadDim > NUM_128 || fiaInfo.qkHeadDim < 1) {
            std::string shapeMsg = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and key", shapeMsg.c_str(),
                "In per-block quant scenario, the axis D of query and key must be >=1 and <=128");
            return ge::GRAPH_FAILED;
        }

        if (fiaInfo.vHeadDim > NUM_128 || fiaInfo.vHeadDim < 1) {
            std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.value.shape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value", shapeStr.c_str(),
                "In per-block quant scenario, the axis D of value must be >=1 and <=128");
            return ge::GRAPH_FAILED;
        }
    } else if (enableQKVMxfp8FullQuant_) {
        if (!((fiaInfo.qkHeadDim == NUM_128 && fiaInfo.vHeadDim == NUM_128) ||
              (fiaInfo.qkHeadDim == NUM_64 && fiaInfo.vHeadDim == NUM_64))) {
            std::string shapeMsg = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.value.shape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and value", shapeMsg.c_str(),
                "In the MXFP8 full quant scenario, the axis D of query and value must both be 64 or both be 128");
            return ge::GRAPH_FAILED;
        }
    } else if (enableQKPerTokenHeadVPerHead_) {
        std::string dimValues = "qkHeadDim=" + std::to_string(fiaInfo.qkHeadDim) +
                                ", vHeadDim=" + std::to_string(fiaInfo.vHeadDim);
        OP_CHECK_IF(
            (fiaInfo.qkHeadDim != NUM_128 || fiaInfo.vHeadDim != NUM_128),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "query/key/value",
                dimValues.c_str(),
                "In FP8 GQA fullquant scenario, the D axis of query/key and value are only support 128"),
            return ge::GRAPH_FAILED);
    } else {
        if (fiaInfo.qkHeadDim > D_LIMIT || fiaInfo.qkHeadDim < NUM1) {
            std::string shapeMsg = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and key", shapeMsg.c_str(),
                "In per-tensor quant scenario, the axis D of query and key must be >=1 and <=8192");
            return ge::GRAPH_FAILED;
        }

        if (fiaInfo.qkHeadDim != fiaInfo.vHeadDim) {
            std::string shapeMsg = ToString(fiaInfo.opParamInfo.value.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            std::string reasonMsg = "In per-tensor quant scenario, the axis D of value(" +
                std::to_string(fiaInfo.vHeadDim) + ") must be equal to the axis D of key(" +
                std::to_string(fiaInfo.qkHeadDim) + ")";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                fiaInfo.opName, "value and key", shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// check axis
ge::graphStatus DequantChecker::CheckInputAxisFullquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckN1SizeFullquant(fiaInfo) || ge::GRAPH_SUCCESS != CheckN2SizeFullquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckGSizeFullquant(fiaInfo) || ge::GRAPH_SUCCESS != CheckDSizeFullquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// enableAntiQuant 相关校验函数
// singlePara
ge::graphStatus DequantChecker::CheckSingleParaForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckAntiquantModeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckInputKVTypeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckAntiquantModeForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // antiquantMode合法性校验
    // keyAntiquantMode
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    OP_LOGI(fiaInfo.opName, "keyAntiquantMode is %ld.", keyAntiquantMode);
    OP_LOGI(fiaInfo.opName, "valueAntiquantMode is %ld.", valueAntiquantMode);
    OP_CHECK_IF(((keyAntiquantMode != PER_CHANNEL_MODE) && (keyAntiquantMode != PER_TOKEN_MODE) &&
                 (keyAntiquantMode != PER_TENSOR_HEAD_MODE) && (keyAntiquantMode != PER_TOKEN_HEAD_MODE) &&
                 (keyAntiquantMode != PER_TOKEN_PA_MODE) && (keyAntiquantMode != PER_TOKEN_HEAD_PA_MODE) &&
                 (keyAntiquantMode != PER_TOKEN_GROUP_MODE)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_antiquant_mode",
            std::to_string(keyAntiquantMode).c_str(),
            "keyAntiquantMode must be within the range {0, 1, 2, 3, 4, 5, 6}"),
                return ge::GRAPH_FAILED);
    // valueAntiquantMode
    OP_CHECK_IF(((valueAntiquantMode != PER_CHANNEL_MODE) && (valueAntiquantMode != PER_TOKEN_MODE) &&
                 (valueAntiquantMode != PER_TENSOR_HEAD_MODE) && (valueAntiquantMode != PER_TOKEN_HEAD_MODE) &&
                 (valueAntiquantMode != PER_TOKEN_PA_MODE) && (valueAntiquantMode != PER_TOKEN_HEAD_PA_MODE) &&
                 (valueAntiquantMode != PER_TOKEN_GROUP_MODE)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "value_antiquant_mode",
            std::to_string(valueAntiquantMode).c_str(),
            "valueAntiquantMode must be within the range {0, 1, 2, 3, 4, 5, 6}"),
                return ge::GRAPH_FAILED);
    bool kPerChnVPerTokFlag = (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE);
    // KV分离场景
    OP_CHECK_IF((!kPerChnVPerTokFlag && (keyAntiquantMode != valueAntiquantMode)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "key_antiquant_mode and value_antiquant_mode",
            (std::to_string(keyAntiquantMode) + " and " + std::to_string(valueAntiquantMode)).c_str(),
            "keyAntiquantMode and valueAntiquantMode must be equal "
            "when keyAntiquantMode!=0 and valueAntiquantMode!=1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerChannel(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType, const gert::Tensor *keyAntiquantScaleTensor)
{
    // per-tensor模式，仅当key/value的数据类型为INT8时支持
    gert::Shape expectedShape1 = gert::Shape({1});
    auto keyAntiquantScaleShape = keyAntiquantScaleTensor->GetStorageShape();
    if (inputKvType == ge::DT_INT4 || inputKvType == ge::DT_INT32) {
        OP_CHECK_IF((keyAntiquantScaleShape == expectedShape1),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
                ToString(inputKvType).c_str(),
                "The datatype of key and value must be INT8 "
                "when keyAntiquantMode and valueAntiquantMode are per-tensor mode"),
                return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((keyAntiquantScaleShape == expectedShape1) && inputKvType != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
                ToString(inputKvType).c_str(),
                "The datatype of key and value must be INT8 "
                "when keyAntiquantMode and valueAntiquantMode are per-tensor mode"),
                return ge::GRAPH_FAILED);
    }
    // per-channel模式，支持key/value的数据类型为INT8、INT4(INT32)、HIFLOAT8、FLOAT8_E4M3FN
    OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4 && inputKvType != ge::DT_HIFLOAT8 &&
        inputKvType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8, INT4(INT32), "
            "HIFLOAT8 or FLOAT8_E4M3FN when keyAntiquantMode and valueAntiquantMode are both per-channel mode"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantMixed(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // key支持per-channel叠加value支持per-token，支持key/value的数据类型为INT8、INT4(INT32)
    OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 or INT4 "
            "when keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.inputKvType == ge::DT_INT8 &&
        (fiaInfo.inputQType != ge::DT_FLOAT16 || fiaInfo.outputType != ge::DT_FLOAT16)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "query and attention_out",
            ToString(inputKvType).c_str(), "The datatype of query and attention_out must be FLOAT16 "
            "when keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode "
            "and the datatype of key is INT8"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerToken(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-token模式，支持key/value的数据类型为INT8、INT4(INT32)、FLOAT8_E4M3FN
    OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4 && inputKvType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8, INT4 or "
            "FLOAT8_E4M3FN when keyAntiquantMode and valueAntiquantMode are both per-token mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerTokenPA(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-token模式，使用page attention管理scale/offset, 支持key/value的数据类型为INT8、FLAOT8_E4M3FN
    OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 or FLOAT8_E4M3FN "
            "when keyAntiquantMode and valueAntiquantMode are per-token-pa mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerTensorHead(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-tensor-head模式，支持key/value的数据类型为INT8
    if (inputKvType == ge::DT_INT4) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 "
            "when keyAntiquantMode and valueAntiquantMode are per-tensor-head mode");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((inputKvType != ge::DT_INT8),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 "
            "when keyAntiquantMode and valueAntiquantMode are per-tensor-head mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerTokenHead(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-token-head模式，支持key/value的数据类型为INT8、INT4(INT32)
    OP_CHECK_IF((inputKvType != ge::DT_INT8 && inputKvType != ge::DT_INT4),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 or INT4 "
            "when keyAntiquantMode and valueAntiquantMode are per-token-head mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerTokenHeadPA(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-token-head模式使用page attention管理scale/offset，支持key/value的数据类型为INT8
    OP_CHECK_IF((inputKvType != ge::DT_INT8),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
                    ToString(inputKvType).c_str(), "The datatype of key and value must be INT8 "
                    "when keyAntiquantMode and valueAntiquantMode are per-token-head-PA mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquantPerTokenGroup(const FiaTilingInfo &fiaInfo,
    ge::DataType inputKvType)
{
    // per-token-group模式，支持key/value的数据类型为FLOAT4_E2M1
    OP_CHECK_IF((inputKvType != ge::DT_FLOAT4_E2M1),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            ToString(inputKvType).c_str(), "The datatype of key and value must be FLOAT4_E2M1 "
            "when keyAntiquantMode and valueAntiquantMode are per-token-group mode"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckInputKVTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 根据keyAntiquantMode和valueAntiquantMode，校验输入key、value的datatype的合法性
    auto inputKvType = fiaInfo.inputKvType;
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr) {
        // 若不存在keyAntiquantScaleTensor和valueAntiquantScaleTensor，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        return CheckInputKVTypeForAntiquantPerChannel(fiaInfo, inputKvType, keyAntiquantScaleTensor);
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        return CheckInputKVTypeForAntiquantPerToken(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        return CheckInputKVTypeForAntiquantPerTokenPA(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        return CheckInputKVTypeForAntiquantPerTensorHead(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        return CheckInputKVTypeForAntiquantPerTokenHead(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        return CheckInputKVTypeForAntiquantPerTokenHeadPA(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        return CheckInputKVTypeForAntiquantMixed(fiaInfo, inputKvType);
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        return CheckInputKVTypeForAntiquantPerTokenGroup(fiaInfo, inputKvType);
    }

    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus DequantChecker::CheckExistenceForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckScaleExistenceForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckDescExistenceForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetExistenceForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t antiquantMode = 0;
    if (fiaInfo.opParamInfo.antiquantMode != nullptr) {
        antiquantMode = *fiaInfo.opParamInfo.antiquantMode;
    }

    // 只支持KV分离场景，key和value的antiquantScaleTensor都应非空
    OP_CHECK_IF(keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_antiquant_scale and value_antiquant_scale",
                    "In antiquant scenario, keyAntiquantScale and valueAntiquantScale must exist"),
                return ge::GRAPH_FAILED);
    // 不支持KV不分离场景
    OP_CHECK_IF((antiquantMode != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "antiquant_mode", std::to_string(antiquantMode).c_str(),
            "Antiquant scenario only supports key/value split mode (antiquantMode 0)"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.opParamInfo.antiquantScale.tensor != nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "antiquant_scale",
                    "Antiquant scenario only supports key/value split mode; antiquantScale must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.opParamInfo.antiquantOffset.tensor != nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "antiquant_offset",
                    "Antiquant scenario only supports key/value split mode; antiquantOffset must be empty"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckDescExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // 所有输入若Tensor存在，则Desc必须存在
    // keyantiquantScale
    OP_CHECK_IF((fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr &&
                 fiaInfo.opParamInfo.keyAntiquantScale.desc == nullptr),
                OP_LOGE(fiaInfo.opName, "keyAntiquantScaleTensor exists, but keyAntiquantScaleDesc does not exist."),
                return ge::GRAPH_FAILED);
    // valueAntiquantScale
    OP_CHECK_IF(
        (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr &&
         fiaInfo.opParamInfo.valueAntiquantScale.desc == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                    "When valueAntiquantScaleTensor exists, valueAntiquantScaleDesc cannot be empty"),
        return ge::GRAPH_FAILED);
    // keyAntiquantOffset
    OP_CHECK_IF((fiaInfo.opParamInfo.keyAntiquantOffset.tensor != nullptr &&
                 fiaInfo.opParamInfo.keyAntiquantOffset.desc == nullptr),
                OP_LOGE(fiaInfo.opName, "keyAntiquantOffsetTensor exists, but keyAntiquantOffsetDesc does not exist."),
                return ge::GRAPH_FAILED);
    // valueAntiquantOffset
    OP_CHECK_IF(
        (fiaInfo.opParamInfo.valueAntiquantOffset.tensor != nullptr &&
         fiaInfo.opParamInfo.valueAntiquantOffset.desc == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "value_antiquant_offset",
                    "When valueAntiquantOffsetTensor exists, valueAntiquantOffsetDesc cannot be empty"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // 校验Offset的存在性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;
    auto inputKvType = fiaInfo.opParamInfo.key.desc->GetDataType();

    // keyAntiquantOffset和valueAntiquantOffset要么同时存在，要么同时不存在
    OP_CHECK_IF((keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_antiquant_offset and value_antiquant_offset",
            "When keyAntiquantOffsetTensor exists, valueAntiquantOffsetDesc cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((valueAntiquantOffsetTensor != nullptr && keyAntiquantOffsetTensor == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_antiquant_offset and value_antiquant_offset",
            "When keyAntiquantOffsetTensor is empty, valueAntiquantOffsetDesc must be empty"),
        return ge::GRAPH_FAILED);
    // key、value的datatype为FLOAT8_E4M3FN、HIFLOAT8和FLOAT4_E2M1时，不支持Offset
    if (inputKvType == ge::DT_FLOAT8_E4M3FN || inputKvType == ge::DT_HIFLOAT8 || inputKvType == ge::DT_FLOAT4_E2M1) {
        OP_CHECK_IF((keyAntiquantOffsetTensor != nullptr),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_antiquant_offset",
                "When the datatype is FLOAT8_E4M3FN/HIFLOAT8/FLOAT4_E2M1, key_antiquant_offset must be empty"),
            return ge::GRAPH_FAILED);
    }
    if (inputKvType == ge::DT_FLOAT8_E4M3FN || inputKvType == ge::DT_HIFLOAT8 || inputKvType == ge::DT_FLOAT4_E2M1) {
        OP_CHECK_IF((valueAntiquantOffsetTensor != nullptr),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "value_antiquant_offset",
                "When the datatype is FLOAT8_E4M3FN/HIFLOAT8/FLOAT4_E2M1, value_antiquant_offset must be empty"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureLayoutForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureQuerySForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureRopeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // 伪量化场景校验支持的inputLayout
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    const std::vector<std::string> supportedLayoutList = {
        "BSH", "BSND", "BNSD", "BNSD_BSND", "TND"};

    if ((std::find(supportedLayoutList.begin(), supportedLayoutList.end(), inputLayout) == supportedLayoutList.end())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            inputLayout.c_str(), "In antiquant scenario, inputLayout only supports BSH, BNSD, BSND, BNSD_BSND or TND");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (fiaInfo.inputKvType == ge::DT_INT8 && inputLayout == "TND"),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", inputLayout.c_str(),
            "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario, "
            "input_layout cannot be TND"),
        return ge::GRAPH_FAILED);

    if (fiaInfo.s1Size == 1) {
        OP_CHECK_IF(
            inputLayout == "BNSD_BSND",
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", "BNSD_BSND",
                "In antiquant scenario, input_layout cannot be BNSD_BSND when Q_S is 1"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.s1Size > 1) {
        int64_t keyAntiquantMode = 0;
        if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
            keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
        }
        if ((keyAntiquantMode == PER_TENSOR_HEAD_MODE || keyAntiquantMode == PER_TOKEN_HEAD_MODE ||
             keyAntiquantMode == PER_TOKEN_PA_MODE || keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE) &&
            fiaInfo.inputKvType == ge::DT_INT8) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "keyAntiquantMode",
                std::to_string(keyAntiquantMode).c_str(),
                "When the datatype of key/value is int8 and the S axis of query > 1, "
                "keyAntiquantMode 2, 3, 4, 5 are not supported");
            return ge::GRAPH_FAILED;
        }

        if ((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
            fiaInfo.inputKvType == ge::DT_INT8 &&
            (fiaInfo.inputQType != ge::DT_BF16 || fiaInfo.outputType != ge::DT_BF16)) {
            std::string dtypeMsg = ToString(fiaInfo.inputQType) + " and " + ToString(fiaInfo.outputType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "query and attention_out",
                dtypeMsg.c_str(),
                "When the datatype of key/value is int8 and keyAntiquantMode is 0 or 1, "
                "the datatype of query and attention_out must be BF16");
            return ge::GRAPH_FAILED;
        }

        if ((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
            fiaInfo.inputKvType == ge::DT_INT8 && fiaInfo.s1Size > NUM_16) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                std::to_string(fiaInfo.s1Size).c_str(),
                "When the datatype of key/value is int8 and keyAntiquantMode is 0 or 1, "
                "the axis S of query should not be greater than 16");
            return ge::GRAPH_FAILED;
        }

        if ((keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
            fiaInfo.inputKvType == ge::DT_INT8 && !fiaInfo.batchContinuousFlag) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key and value",
                "When the datatype of key/value is int8 and keyAntiquantMode is 0 or 1, tensorlist is not supported");
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.inputKvType == ge::DT_INT4 || fiaInfo.inputKvType == ge::DT_INT32) {
            std::string dtypeMsg = ToString(fiaInfo.inputKvType);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                dtypeMsg.c_str(),
                "In keyAntiquant/valueAntiquant split mode scenario, the datatypes of key "
                "and value cannot be int4 or int32");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // 校验交叉特性page attention约束
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;
    int32_t blockSize = fiaInfo.blockSize;
    uint32_t maxBlockNumPerBatch = fiaInfo.maxBlockNumPerBatch;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    // per-token模式，per-token叠加per-head模式，antiquantScale输入的最后一维需要大于等于maxBlockNumPerBatch * blockSize
    // per-token模式
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // shape 可能为(1, B, S)或(B, S)
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        if (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize) {
            std::string shapeStr = ToStringRaw(keyAntiquantScaleTensor->GetStorageShape());
            std::string reasonMsg =
                "The last dimension of key_antiquant_scale should be >= (maxBlockNumPerBatch * blockSize) when "
                "keyAntiquantMode=1 and valueAntiquantMode=1 and keyAntiquant/valueAntiquant is splited";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale", shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize) {
            std::string shapeStr = ToStringRaw(valueAntiquantScaleTensor->GetStorageShape());
            std::string reasonMsg =
                "The last dimension of value_antiquant_scale should be >= (maxBlockNumPerBatch * blockSize) when "
                "keyAntiquantMode=1 and valueAntiquantMode=1 and keyAntiquant/valueAntiquant is splited";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale", shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    // per-token叠加per-head模式
    // shape可能为(B,N,S)
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        if (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize) {
            std::string shapeStr = ToStringRaw(keyAntiquantScaleTensor->GetStorageShape());
            std::string reasonMsg = "The last dimension of key_antiquant_scale should be >= maxBlockNumPerBatch * "
                                    "blockSize when keyAntiquantMode and valueAntiquantMode are per-token-head mode "
                                    "and keyAntiquant/valueAntiquant is splited";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale", shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (valueAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 1) < maxBlockNumPerBatch * blockSize) {
            std::string shapeStr = ToStringRaw(valueAntiquantScaleTensor->GetStorageShape());
            std::string reasonMsg = "The last dimension of value_antiquant_scale should be >= maxBlockNumPerBatch * "
                                    "blockSize when keyAntiquantMode and valueAntiquantMode are per-token-head mode "
                                    "and keyAntiquant/valueAntiquant is splited";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale", shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    // per-token-group模式，antiquantScale输入倒数第二维需要大于等于maxBlockNumPerSeq * blockSize
    // shape为(1,B,N,S,D/32)
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        uint32_t dimNum = keyAntiquantScaleTensor->GetStorageShape().GetDimNum();
        if (keyAntiquantScaleTensor->GetStorageShape().GetDim(dimNum - 2) < maxBlockNumPerBatch * blockSize) {
            std::string shapeStr = ToStringRaw(keyAntiquantScaleTensor->GetStorageShape());
            std::string reasonMsg = "The second-to-last dimension of key_antiquant_scale should be >= "
                                    "maxBlockNumPerBatch * blockSize when keyAntiquantMode and valueAntiquantMode are "
                                    "per-token-group mode and keyAntiquant/valueAntiquant is splited";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale", shapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const
{
     OP_CHECK_IF((fiaInfo.ropeMode == RopeMode::ROPE_COMBINE),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "rope_mode", "ROPE_COMBINE",
                "Antiquant scenario does not support combined rope"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus DequantChecker::CheckMultiParaForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckScaleTypeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckScaleShapeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetTypeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckOffsetShapeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验Scale矩阵的datatype的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    auto &keyAntiquantScaleDesc = fiaInfo.opParamInfo.keyAntiquantScale.desc;
    auto &valueAntiquantScaleDesc = fiaInfo.opParamInfo.valueAntiquantScale.desc;
    auto &queryDesc = fiaInfo.opParamInfo.query.desc;

    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        // per-channel/per-tensor模式，keyAntiquantScale和valueAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be the same as query when keyAntiquantMode is per-channel or per-tensor mode"
                " and keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be the same as query when valueAntiquantMode is per-channel or per-tensor"
                " mode and keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }

    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // per-token模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be FLOAT32 when keyAntiquantMode is per-token mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT32 when valueAntiquantMode is per-token mode and "
                "valueAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        // per-tensor-head模式，keyAntiquantScale和valueAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be the same as query when keyAntiquantMode is per-tensor-head mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be the same as query when valueAntiquantMode is per-tensor-head mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        // per-token-head模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be FLOAT32 when keyAntiquantMode is per-token-head mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT32 when valueAntiquantMode is per-token-head mode and "
                "valueAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        // per-token-head-PA模式，数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be FLOAT32 when keyAntiquantMode is per-token-head-PA mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT32 when valueAntiquantMode is per-token-head-PA mode and "
                "valueAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        // per-token-PA模式，数据类型固定为FLOAT32
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be FLOAT32 when keyAntiquantMode is per-token-PA mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT32 when valueAntiquantMode is per-token-PA mode and "
                "valueAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // key支持per-channel叠加value支持per-token
        // keyAntiquantScale的数据类型与query相同
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != queryDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be the same as query when keyAntiquantMode is per-channel mode and "
                "valueAntiquantMode is per-token mode"),
            return ge::GRAPH_FAILED);
        // valueAntiquantScale的数据类型固定为FLOAT32
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT32 when keyAntiquantMode is per-channel mode and "
                "valueAntiquantMode is per-token mode"),
            return ge::GRAPH_FAILED);
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        // per-token-group模式，keyAntiquantScale和valueAntiquantScale的数据类型固定为FLOAT8_E8M0
        OP_CHECK_IF((keyAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                ToString(keyAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_scale must be FLOAT8_E8M0 when keyAntiquantMode is per-token-group mode and "
                "keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueAntiquantScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                ToString(valueAntiquantScaleDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_scale must be FLOAT8_E8M0 when valueAntiquantMode is per-token-group mode and "
                "valueAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckScaleShapeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验scale矩阵的shape的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }

    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t valueAntiquantScaleTensorDimNum = valueAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerChannelPerTensorMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTensorHeadMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenHeadMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenPAMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenHeadPAMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        // key支持per-channel叠加value支持per-token
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerChannelPerTensorMode(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckVScaleShapeForPerTokenMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantMode == PER_TOKEN_GROUP_MODE && valueAntiquantMode == PER_TOKEN_GROUP_MODE) {
        if (ge::GRAPH_SUCCESS != CheckKScaleShapeForPerTokenGroupMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!((keyAntiquantMode == PER_CHANNEL_MODE) && (valueAntiquantMode == PER_TOKEN_MODE))) {
        // 除key支持per-channel叠加value支持per-token模式以外
        // keyAntiquantScale的shape必须与valueAntiquantScale一致
        if (keyAntiquantScaleTensorShape != valueAntiquantScaleTensorShape) {
            std::string shapeMsg =
                ToString(keyAntiquantScaleTensorShape) + " and " + ToString(valueAntiquantScaleTensorShape);
            std::string reasonMsg =
                "The shape of key_antiquant_scale and value_antiquant_scale must be the same when keyAntiquantMode is "
                "not per-channel mode and valueAntiquantMode is not per-token mode";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_antiquant_scale and value_antiquant_scale",
                                                   shapeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetTypeForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // 校验offset矩阵的datatype的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;

    // keyAntiquantOffset的datatype与keyAntiquantScale一致
    auto &keyAntiquantOffsetDesc = fiaInfo.opParamInfo.keyAntiquantOffset.desc;
    auto &keyAntiquantScaleDesc = fiaInfo.opParamInfo.keyAntiquantScale.desc;
    if (keyAntiquantOffsetDesc != nullptr && keyAntiquantScaleDesc != nullptr) {
        OP_CHECK_IF((keyAntiquantOffsetDesc->GetDataType() != keyAntiquantScaleDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_antiquant_offset",
                ToString(keyAntiquantOffsetDesc->GetDataType()).c_str(), "The datatype of "
                "key_antiquant_offset must be the same as key_antiquant_scale "
                "when keyAntiquant/valueAntiquant is in split mode"),
            return ge::GRAPH_FAILED);
    }
    // valueAntiquantOffset的datatype与valueAntiquantScale一致
    auto &valueAntiquantOffsetDesc = fiaInfo.opParamInfo.valueAntiquantOffset.desc;
    auto &valueAntiquantScaleDesc = fiaInfo.opParamInfo.valueAntiquantScale.desc;
    if (valueAntiquantOffsetDesc != nullptr && valueAntiquantScaleDesc != nullptr) {
        OP_CHECK_IF((valueAntiquantOffsetDesc->GetDataType() != valueAntiquantScaleDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value_antiquant_offset",
                ToString(valueAntiquantOffsetDesc->GetDataType()).c_str(), "The datatype of "
                "value_antiquant_offset must be the same as value_antiquant_scale "
                "when keyAntiquant/valueAntiquant is in split mode"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckOffsetShapeForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    // 校验offset矩阵的shape的合法性
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    auto &keyAntiquantOffsetTensor = fiaInfo.opParamInfo.keyAntiquantOffset.tensor;
    auto &valueAntiquantOffsetTensor = fiaInfo.opParamInfo.valueAntiquantOffset.tensor;

    if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr ||
        keyAntiquantOffsetTensor == nullptr || valueAntiquantOffsetTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape keyAntiquantOffsetTensorShape = keyAntiquantOffsetTensor->GetStorageShape();
    gert::Shape valueAntiquantOffsetTensorShape = valueAntiquantOffsetTensor->GetStorageShape();
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();

    if (keyAntiquantOffsetTensorShape != keyAntiquantScaleTensorShape) {
        std::string shapeMsg =
            ToString(keyAntiquantOffsetTensorShape) + " and " + ToString(keyAntiquantScaleTensorShape);
        std::string reasonMsg = "The shape of key_antiquant_offset and key_antiquant_scale must be the same when "
                                "keyAntiquant/valueAntiquant is in split mode";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_antiquant_offset and key_antiquant_scale",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (valueAntiquantOffsetTensorShape != valueAntiquantScaleTensorShape) {
        std::string shapeMsg =
            ToString(valueAntiquantOffsetTensorShape) + " and " + ToString(valueAntiquantScaleTensorShape);
        std::string reasonMsg = "The shape of value_antiquant_offset and value_antiquant_scale must be the same when "
                                "keyAntiquant/valueAntiquant is in split mode";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "value_antiquant_offset and value_antiquant_scale",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo) const
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    // per-channel/per-tensor模式
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
        // 维度为1，shape可能为[H]，per-channel模式
        // 维度为1，shape可能为[1]，per-tensor模式
        gert::Shape expectedShapeH = gert::Shape({numKeyValueHeads * headDim});
        gert::Shape expectedShape1 = gert::Shape({1});
        if ((keyAntiquantScaleTensorShape != expectedShapeH) && (keyAntiquantScaleTensorShape != expectedShape1)) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [H(" +
                std::to_string(numKeyValueHeads * headDim) + ")] when keyAntiquantMode is per-channel mode "
                "or [1] when keyAntiquantMode is per-tensor mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // 维度为2，shape可能为[1, H]或者[N, D]
        gert::Shape expectedShape1H = gert::Shape({1, numKeyValueHeads * headDim});
        gert::Shape expectedShapeND = gert::Shape({numKeyValueHeads, headDim});
        if ((keyAntiquantScaleTensorShape != expectedShape1H) && (keyAntiquantScaleTensorShape != expectedShapeND)) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [1, H(" +
                std::to_string(numKeyValueHeads * headDim) + ")] or [N(" + std::to_string(numKeyValueHeads) +
                "), D(" + std::to_string(headDim) + ")] when keyAntiquantMode is per-channel mode and "
                "the shape of keyAntiquantScale is 2D";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // 维度为3，shape可能为[1, N, D]或者[N, 1, D]
        gert::Shape expectedShape1ND = gert::Shape({1, numKeyValueHeads, headDim});
        gert::Shape expectedShapeN1D = gert::Shape({numKeyValueHeads, 1, headDim});
        if ((keyAntiquantScaleTensorShape != expectedShape1ND) && (keyAntiquantScaleTensorShape != expectedShapeN1D)) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [1, N(" +
                std::to_string(numKeyValueHeads) + "), D(" + std::to_string(headDim) + ")] or [N(" +
                std::to_string(numKeyValueHeads) + "), 1, D(" + std::to_string(headDim) +
                ")] when keyAntiquantMode is per-channel mode and the shape of keyAntiquantScale is 3D";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_4) {
        // 维度为4，shape仅支持[1, N, 1, D] 或者 [1, 1, N, D]
        gert::Shape expectedShape1N1D = gert::Shape({1, numKeyValueHeads, 1, headDim});
        gert::Shape expectedShape11ND = gert::Shape({1, 1, numKeyValueHeads, headDim});
        if ((keyAntiquantScaleTensorShape != expectedShape1N1D) &&
            (keyAntiquantScaleTensorShape != expectedShape11ND)) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [1, N(" +
                std::to_string(numKeyValueHeads) + "), 1, D(" + std::to_string(headDim) + ")] or [1, 1, N(" +
                std::to_string(numKeyValueHeads) + "), D(" + std::to_string(headDim) +
                ")] when keyAntiquantMode is per-channel mode and the shape of keyAntiquantScale is 4D";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "1D, 2D, 3D or 4D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isMaxWorkspace && (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD)) {
        return ge::GRAPH_SUCCESS;
    }
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    // per-token模式
    // 支持shape为[1, B, S],[B, S]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // 维度为2，shape可能为[B, >=KV_S]
        if (keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) < seqLength) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of key_antiquant_scale must be [1, B(" +
                std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) + ")] or "
                "[B(" + std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) + ")] "
                "when keyAntiquantMode is per-token mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale", actualShape.c_str(),
                reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // 维度为3，shape可能为[1, B, >=KV_S]
        if (keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of key_antiquant_scale must be [1, B(" +
                std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) + ")] or "
                "[B(" + std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) + ")] "
                "when keyAntiquantMode is per-token mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale", actualShape.c_str(),
                reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "2D or 3D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTensorHeadMode(const FiaTilingInfo &fiaInfo) const
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    // per-tensor叠加per-head模式
    // shape仅支持[N]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
        gert::Shape expectedShapeN = gert::Shape({numKeyValueHeads});
        if (keyAntiquantScaleTensorShape != expectedShapeN) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [N(" +
                std::to_string(numKeyValueHeads) + ")] when keyAntiquantMode is per-tensor-head mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "1D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenHeadMode(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isMaxWorkspace && (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD)) {
        return ge::GRAPH_SUCCESS;
    }
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint64_t seqLength = fiaInfo.s2Size;
    // per-token叠加per-head模式
    // shape仅支持[B, N, >=S]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        if (keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != numKeyValueHeads ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [B(" + std::to_string(batchSize) +
                "), N(" + std::to_string(numKeyValueHeads) + "), >=S(" + std::to_string(seqLength) +
                ")] when keyAntiquantMode is per-token-head mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "3D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenPAMode(const FiaTilingInfo &fiaInfo) const
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    // per-token模式使用page attention管理scale/offset
    // shape仅支持[blockNum, blockSize]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_2) {
        uint32_t blockNum = fiaInfo.totalBlockNum;
        uint32_t blockSize = fiaInfo.blockSize;
        gert::Shape expectedShape = gert::Shape({blockNum, blockSize});
        if (keyAntiquantScaleTensorShape != expectedShape) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [blockNum(" +
                std::to_string(blockNum) + "), blockSize(" + std::to_string(blockSize) +
                ")] when keyAntiquantMode is per-token-pa mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "2D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenHeadPAMode(const FiaTilingInfo &fiaInfo) const
{
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    // per-token叠加per-head模式并使用page attention管理scale/offset
    // shape仅支持[blockNum, N, blockSize]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_3) {
        uint32_t blockNum = fiaInfo.totalBlockNum;
        uint32_t blockSize = fiaInfo.blockSize;
        gert::Shape expectedShape = gert::Shape({blockNum, numKeyValueHeads, blockSize});
        if (keyAntiquantScaleTensorShape != expectedShape) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [blockNum(" +
                std::to_string(blockNum) + "), N(" + std::to_string(numKeyValueHeads) + "), blockSize(" +
                std::to_string(blockSize) + ")] when keyAntiquantMode is per-token-head-pa mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "3D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckKScaleShapeForPerTokenGroupMode(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isMaxWorkspace && (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD)) {
        return ge::GRAPH_SUCCESS;
    }
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t numKeyValueHeads = fiaInfo.n2Size;
    uint64_t seqLength = fiaInfo.s2Size;
    uint32_t headDim = fiaInfo.qkHeadDim;
    // per-token-group模式
    // shape支持[1, B, N, >=KV_S, D/32]
    if (keyAntiquantScaleTensorDimNum == DIM_NUM_5) {
        if (keyAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_2) != numKeyValueHeads ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_3) < fiaInfo.s2Size ||
            keyAntiquantScaleTensorShape.GetDim(DIM_NUM_4) != headDim / 32) {
            std::string actualShape = ToStringRaw(keyAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of keyAntiquantScale must be [1, B(" +
                std::to_string(batchSize) + "), N(" + std::to_string(numKeyValueHeads) + "), >=S(" +
                std::to_string(seqLength) + "), D/32(" + std::to_string(headDim / 32) +
                ")] when keyAntiquantMode is per-token-group mode";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(keyAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "key_antiquant_scale", dimStr.c_str(), "5D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckVScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isMaxWorkspace && (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD)) {
        return ge::GRAPH_SUCCESS;
    }
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();
    uint32_t valueAntiquantScaleTensorDimNum = valueAntiquantScaleTensorShape.GetDimNum();
    uint32_t batchSize = fiaInfo.bSize;
    uint64_t seqLength = fiaInfo.s2Size;
    // 校验value支持per-token
    // shape支持[1, B, S]或[B, S]
    if (valueAntiquantScaleTensorDimNum == DIM_NUM_2) {
        // [B, >=KV_S]
        if (valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != batchSize ||
            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1) < seqLength) {
            std::string actualShape = ToStringRaw(valueAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of valueAntiquantScale must be [B(" +
                std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) +
                ")] when valueAntiquantMode is per-token mode, the shape of valueAntiquantScale is 2D";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (valueAntiquantScaleTensorDimNum == DIM_NUM_3) {
        // [1, B, >=KV_S]
        if (valueAntiquantScaleTensorShape.GetDim(DIM_NUM_0) != 1 ||
            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_1) != batchSize ||
            valueAntiquantScaleTensorShape.GetDim(DIM_NUM_2) < seqLength) {
            std::string actualShape = ToStringRaw(valueAntiquantScaleTensorShape);
            std::string reasonMsg = "The shape of valueAntiquantScale must be [1, B(" +
                std::to_string(batchSize) + "), >=S(" + std::to_string(seqLength) +
                ")] when valueAntiquantMode is per-token mode, the shape of valueAntiquantScale is 3D";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_antiquant_scale",
                                                  actualShape.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    std::string dimStr = std::to_string(valueAntiquantScaleTensorDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "value_antiquant_scale", dimStr.c_str(), "2D or 3D");
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        // 量化方式
        if (fiaInfo.fullQuantMode == FiaFullQuantMode::Q_PER_TOKEN_HEAD_KV_PER_TENSOR_FULL_QUANT) {
            enableQPerTokenHeadKVPerTensor_ = true;
        } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
            enableQKVPertensorQuant_ = true;
        } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
            enableQKVPerblockQuant_ = true;
            OP_LOGW(fiaInfo.opName, "Per-block full quantization scenario will be deprecated in future versions. "
                    "It is recommended to use mxfp8 full quantization instead.");
        } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
            enableQKVMxfp8FullQuant_ = true;
        } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD) {
            enableQKPerTokenHeadVPerHead_ = true;
        }

        if (ge::GRAPH_SUCCESS != CheckDequantScaleDtypeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantModeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantScaleShapeFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (CheckSingleParaForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            return CheckExistenceNoquant(fiaInfo);
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckExistencePertensorFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistenceMLAFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistencePerblockFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistenceMXFP8Fullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckExistenceFP8GQAFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckExistenceForAntiquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckDataTypeSupportFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckInputDTypeFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckInputLayoutFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckInputAxisFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckStrideFP8GQAFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckDequantScaleShapeCrossFullquant(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckFeatureSupportFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (CheckFeatureForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (enableAntiQuant_) {
        if (CheckMultiParaForAntiquant(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling