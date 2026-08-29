/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_v2_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_v2_tiling.h"

#include "../op_kernel/quant_lightning_indexer_v2_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

static const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"},
    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}};

std::string QLIV2DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("QLIV2DataTypeToSerialString ", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

// --------------------------QLIV2InfoParser类成员函数定义-------------------------------------
ge::graphStatus QLIV2InfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "The shape of q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "The desc of q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "k", "The shape of k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "k", "The desc of k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "w", "The shape of w is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "w", "The desc of w is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The shape of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The desc of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The shape of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The desc of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The shape of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The desc of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOutQuery == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_q", "Layout_q is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_k", "Layout_k is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseCount == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_count", "Sparse_count is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_mode", "Sparse_mode is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.quantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_quant_mode", "Query_quant_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("LightningIndexerV2", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if ((npuArch_ != NpuArch::DAV_2201) && (npuArch_ != NpuArch::DAV_3510)) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIV2InfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.cuSeqLensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensK.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_K_INDEX);
    opParamInfo_.cuSeqLensK.desc = context_->GetOptionalInputDesc(CU_SEQLENS_K_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedK.tensor = context_->GetOptionalInputTensor(SEQUSED_K_INDEX);
    opParamInfo_.sequsedK.desc = context_->GetOptionalInputDesc(SEQUSED_K_INDEX);
    opParamInfo_.cmpResidualK.tensor = context_->GetOptionalInputTensor(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.cmpResidualK.desc = context_->GetOptionalInputDesc(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.outputIdxOffset.tensor = context_->GetOptionalInputTensor(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.outputIdxOffset.desc = context_->GetOptionalInputDesc(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void QLIV2InfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    opParamInfo_.query_dequant_scale.desc = context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.query_dequant_scale.shape = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.desc = context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.shape = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    GetOptionalInputParaInfo();
}

void QLIV2InfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(SPARSE_INDICES_INDEX);
    opParamInfo_.sparseValues.desc = context_->GetOutputDesc(SPARSE_VALUES_INDEX);
    opParamInfo_.sparseValues.shape = context_->GetOutputShape(SPARSE_VALUES_INDEX);
}

ge::graphStatus QLIV2InfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");

    opParamInfo_.quantMode = attrs->GetAttrPointer<int32_t>(ATTR_QUANT_MODE_INDEX);
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<int32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);
    auto keyStrides = context_->GetDynamicInputStride(KEY_INDEX, 0);
    auto keyDequantScaleStrides = context_->GetDynamicInputStride(KEY_DEQUANT_SCALE_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyStrides->GetDimNum(); i++) {
            keyStridesVec_.push_back(keyStrides->GetStride(i));
        }
    }
    if (keyDequantScaleStrides != nullptr && keyDequantScaleStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyDequantScaleStrides->GetDimNum(); i++) {
            keyDequantScaleStridesVec_.push_back(keyDequantScaleStrides->GetStride(i));
        }
    }

    if (opParamInfo_.layOutQuery != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOutQuery);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.sparseCount != nullptr) {
        OP_LOGI(context_->GetNodeName(), "selscted count is:%d", *opParamInfo_.sparseCount);
    }
    if (opParamInfo_.sparseMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "sparse mode is:%d", *opParamInfo_.sparseMode);
    }
    if (opParamInfo_.cmpRatio != nullptr) {
        OP_LOGI(context_->GetNodeName(), "cmpRatio is:%d", *opParamInfo_.cmpRatio);
    }
    if (opParamInfo_.returnValue != nullptr) {
        OP_LOGI(context_->GetNodeName(), "returnValue is:%s", *opParamInfo_.returnValue ? "true" : "false");
    }
    if (opParamInfo_.maxSeqlenQ != nullptr) {
        OP_LOGI(context_->GetNodeName(), "maxSeqlenQ  is:%d", *opParamInfo_.maxSeqlenQ);
    }
    if (opParamInfo_.quantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "query_quant_mode mode is:%d", *opParamInfo_.quantMode);
    }
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckAttrParaInfo()
{
    std::string layout_key(opParamInfo_.layOutKey);
    std::string layout_query(opParamInfo_.layOutQuery);

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF((std::string(opParamInfo_.layOutKey) != "PA_BBND"),
                    OP_LOGE(opName_,
                            "input attr layout_key only supported PA_BBND,"
                            "but now layout_key is %s.",
                            layout_key.c_str()),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(
            ((std::string(opParamInfo_.layOutKey) != "PA_BBND") && (std::string(opParamInfo_.layOutKey) != "BSND") &&
             (std::string(opParamInfo_.layOutKey) != "TND")),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_k", std::string(opParamInfo_.layOutKey).c_str(),
                                                  "Layout_k only supports PA_BBND, BSND or TND"),
            return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
                    OP_LOGE(opName_, "input attr sparse_count must > 0 and <= %d, but now sparse_count is %d",
                            SPARSE_LIMIT, *opParamInfo_.sparseCount),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128) ||
                        ((*opParamInfo_.cmpRatio & (*opParamInfo_.cmpRatio - 1)) != 0),
                    OP_LOGE(opName_,
                            "input attr cmpRatio must > 0 and <= 128 and should be powers of 2,"
                            " but now cmpRatio is %ld.",
                            *opParamInfo_.cmpRatio),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(
            !((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT_8K)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk", std::to_string(*opParamInfo_.sparseCount),
                                                  "Sparse_count must > 0 and <= " + std::to_string(SPARSE_LIMIT_8K)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_ratio", std::to_string(*opParamInfo_.cmpRatio),
                                                          "Cmp_ratio must > 0 and <= 128"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutQuery) != "BSND") && (std::string(opParamInfo_.layOutQuery) != "TND")),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", std::string(opParamInfo_.layOutQuery).c_str(),
                                              "Layout_q only supports BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
         (std::string(opParamInfo_.layOutQuery)) != (std::string(opParamInfo_.layOutKey))),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "layout_q and layout_k", layout_query + " and " + layout_key,
                                               "Outside of PA, layout_q and layout_k must be the same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode", std::to_string(*opParamInfo_.sparseMode).c_str(),
                                              "Sparse_mode only supports 0 or 3"),
        return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(*opParamInfo_.quantMode != 2, OP_LOGE(opName_, "input attr quant_mode only supported 2."),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((*opParamInfo_.quantMode != QUANT_MODE_FP8) && (*opParamInfo_.quantMode != QUANT_MODE_INT8) &&
                        (*opParamInfo_.quantMode != QUANT_MODE_MXFP8) &&
                        (*opParamInfo_.quantMode != QUANT_MODE_HIFLOAT8) &&
                        (*opParamInfo_.quantMode != QUANT_MODE_MXFP4),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode",
                                                          std::to_string(*opParamInfo_.quantMode).c_str(),
                                                          "Quant_mode only supports 1, 2, 3, 4 and 5"),
                    return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(*opParamInfo_.returnValue, OP_LOGE(opName_, "input attr returnValue only supported False."),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((*opParamInfo_.returnValue != 0) && (*opParamInfo_.returnValue != 1),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "return_value",
                                                          std::to_string(*opParamInfo_.returnValue).c_str(),
                                                          "Return_value only supports 0 or 1"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (*opParamInfo_.maxSeqlenQ < -1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "max_seqlen_q", std::to_string(*opParamInfo_.maxSeqlenQ).c_str(),
                                              "Max_seqlen_q must >= -1"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    inputQueryScaleType_ = opParamInfo_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = opParamInfo_.key_dequant_scale.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.sparseValues.desc->GetDataType();

    OP_CHECK_IF(!(inputQType_ == inputKType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "q and k",
                    QLIV2DataTypeToSerialString(inputQType_) + " and " + QLIV2DataTypeToSerialString(inputKType_),
                    "The dtype of q and k must be same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!(inputQueryScaleType_ == inputKeyScaleType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "q_descale and k_descale",
                                                       QLIV2DataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                           QLIV2DataTypeToSerialString(inputKeyScaleType_),
                                                       "The dtype of q_descale and k_descale must be same"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(inputQType_ != ge::DT_INT8,
                    OP_LOGE(opName_, "The data types of the input query and key must be int8."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputQueryScaleType_ != ge::DT_FLOAT16,
            OP_LOGE(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be float16."),
            return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        ge::DataType expectQType = ge::DT_FLOAT8_E4M3FN;
        ge::DataType expectScaleType = ge::DT_FLOAT;
        if (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) {
            expectScaleType = ge::DT_FLOAT8_E8M0;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_HIFLOAT8) {
            expectQType = ge::DT_HIFLOAT8;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
            expectQType = ge::DT_FLOAT4_E2M1;
            expectScaleType = ge::DT_FLOAT8_E8M0;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_INT8) {
            expectQType = ge::DT_INT8;
            expectScaleType = ge::DT_FLOAT16;
        }
        OP_CHECK_IF(inputQType_ != expectQType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "q and k",
                        QLIV2DataTypeToSerialString(inputQType_) + " and " + QLIV2DataTypeToSerialString(inputKType_),
                        "The dtype of q and k must match quant_mode"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputQueryScaleType_ != expectScaleType,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "q_descale and k_descale",
                                                   QLIV2DataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                       QLIV2DataTypeToSerialString(inputKeyScaleType_),
                                                   "The dtype of q_descale and k_descale must match quant_mode"),
            return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(weightsType_ != ge::DT_FLOAT16,
                    OP_LOGE(opName_, "The data types of the input weights must be float16."), return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        if (inputQType_ == ge::DT_INT8) {
            OP_CHECK_IF(weightsType_ != ge::DT_FLOAT16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "w", QLIV2DataTypeToSerialString(weightsType_).c_str(),
                            "When the dtype of query is int8, the dtype of w must be float16"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(weightsType_ != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "w", QLIV2DataTypeToSerialString(weightsType_).c_str(),
                            "When the dtype of query is not int8, the dtype of w must be float"),
                        return ge::GRAPH_FAILED);
        }
    }

    OP_CHECK_IF(outputType_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices",
                                                      QLIV2DataTypeToSerialString(outputType_).c_str(),
                                                      "The dtype of sparse_indices must be int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valuesOutType_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_values",
                                                      QLIV2DataTypeToSerialString(valuesOutType_).c_str(),
                                                      "The dtype of sparse_values must be bfloat16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutQueryMap = {{"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}};

    std::string layout_query(opParamInfo_.layOutQuery);
    auto QLayout_ = layoutQueryMap.find(layout_query);
    if (QLayout_ != layoutQueryMap.end()) {
        qLayout_ = QLayout_->second;
    }

    const map<string, DataLayout> layoutKeyMap = {{"BSND", DataLayout::BSND},
                                                  {"TND", DataLayout::TND},
                                                  {"PA_BSND", DataLayout::PA_BBND},
                                                  {"PA_BBND", DataLayout::PA_BBND}};
    std::string layout_key(opParamInfo_.layOutKey);
    auto KLayout = layoutKeyMap.find(layout_key);
    if (KLayout != layoutKeyMap.end()) {
        kLayout_ = KLayout->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckOptionalInput()
{
    // =============== K 侧校验 ===============
    if (kLayout_ == DataLayout::PA_BBND) {
        // PA_BBND: block_table 必传, seqused_k 必传, cu_seqlens_k 不传
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "block_table",
                                                             "When layout_k is PA_BBND, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.sequsedK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "seqused_k",
                                                             "When layout_k is PA_BBND, seqused_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cu_seqlens_k", "When layout_k is PA_BBND, cu_seqlens_k must not be provided"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "block_table",
                        QLIV2DataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                        "The dtype of block_table only supports int32"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "seqused_k", QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                "The dtype of seqused_k only supports int32"),
            return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        // TND: cu_seqlens_k 必传, seqused_k 可选, cu_seqlens_k 不传
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_k",
                                                             "When layout_k is TND, cu_seqlens_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cu_seqlens_k",
                        QLIV2DataTypeToSerialString(opParamInfo_.cuSeqLensK.desc->GetDataType()).c_str(),
                        "The dtype of cu_seqlens_k only supports int32"),
                    return ge::GRAPH_FAILED);
        // seqused_k 可选 - 仅校验数据类型
        if (opParamInfo_.sequsedK.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_k",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                            "The dtype of seqused_k only supports int32"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // BSND: cu_seqlens_k 不传, seqused_k 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_k",
                                                             "When layout_k is BSND, cu_seqlens_k must not be null"),
                    return ge::GRAPH_FAILED);
        if (opParamInfo_.sequsedK.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_k",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                            "The dtype of seqused_k only supports int32"),
                        return ge::GRAPH_FAILED);
        }
    }

    // block_table 非 PA 场景必须为空
    if (kLayout_ != DataLayout::PA_BBND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "block_table", "When layout_k is not PA_BBND, block_table must not be null"),
                    return ge::GRAPH_FAILED);
    }

    // =============== cmpResidualK 校验 ===============
    // cmpRatio 不等于 1 且 sparseMode 不等于 0 时 cmpResidualK 必传
    if (opParamInfo_.cmpRatio != nullptr && *opParamInfo_.cmpRatio != 1 && opParamInfo_.sparseMode != nullptr &&
        *opParamInfo_.sparseMode != 0) {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cmp_residual_k",
                        "Cmp_ratio is not 1 and sparse_mode is not 0, cmp_residual_k must not be null"),
                    return ge::GRAPH_FAILED);
        // cmpResidualK 传入时校验维度 & 数据类型
        if (qLayout_ == DataLayout::BSND) {
            OP_CHECK_IF(
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) !=
                    opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize(),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_residual_k", ToStringRaw(opParamInfo_.cmpResidualK.tensor->GetStorageShape()).c_str(),
                    "When layout_q is BSND, the shape of cmp_residual_k must be (B,)"),
                return ge::GRAPH_FAILED);
        } else if (qLayout_ == DataLayout::TND) {
            OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() - 1,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            opName_, "cmp_residual_k",
                            std::to_string(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize()),
                            "When layout_q is TND, the shape size of cmp_residual_k "
                            "must equal the shape size - 1 of cu_seqlens_q"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cmp_residual_k", "Cmp_ratio is 1 or sparse_mode is 0, cmp_residual_k must be null"),
                    return ge::GRAPH_FAILED);
    }
    // cmpResidualK 传入时校验数据类型
    if (opParamInfo_.cmpResidualK.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cmp_residual_k",
                        QLIV2DataTypeToSerialString(opParamInfo_.cmpResidualK.desc->GetDataType()).c_str(),
                        "The dtype of cmp_residual_k supports int32"),
                    return ge::GRAPH_FAILED);
    }

    // =============== Q 侧校验 ===============
    if (qLayout_ == DataLayout::TND) {
        // TND: cu_seqlens_q 必传, seqused_q 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                             "When layout_q is TND, cu_seqlens_q must not be null"),
                    return ge::GRAPH_FAILED);
        if (kLayout_ == DataLayout::PA_BBND) {
            // k为PA_BBND必传sequsedK, 用sequsedK的维度校验
            OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.sequsedK.tensor->GetStorageShape().GetShapeSize() + 1,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            opName_, "cmp_residual_k",
                            std::to_string(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize()),
                            "When layout_q is TND and layout_k is PA_BBND, "
                            "the shape size of cu_seqlens_q must equal the shape size + 1 of seqused_k"),
                        return ge::GRAPH_FAILED);
        } else if (kLayout_ == DataLayout::TND) {
            // q、k都为TND, cuSeqlensQ与cuSeqlensK维度一致校验
            OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.cuSeqLensK.tensor->GetStorageShape().GetShapeSize(),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "cu_seqlens_q and cu_seqlens_k",
                            Ops::Base::ToString(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.cuSeqLensK.tensor->GetStorageShape()),
                            "When layout_q is TND and layout_k is TND, "
                            "the shape of cu_seqlens_q must equal the shape of cu_seqlens_k"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cu_seqlens_q",
                        QLIV2DataTypeToSerialString(opParamInfo_.cuSeqLensK.desc->GetDataType()).c_str(),
                        "The dtype of cu_seqlens_q only supports int32"),
                    return ge::GRAPH_FAILED);
        // seqused_q 可选 - 仅校验数据类型
        if (opParamInfo_.sequsedQ.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_q",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                            "The dtype of seqused_q only supports int32"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // BSND: cu_seqlens_q 不传, seqused_q 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cu_seqlens_q", "When layout_q is BSND, cu_seqlens_q must not be provided"),
                    return ge::GRAPH_FAILED);
        if (opParamInfo_.sequsedQ.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_q",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                            "The dtype of seqused_q only supports int32"),
                        return ge::GRAPH_FAILED);
        }
    }
    if (npuArch_ == NpuArch::DAV_3510) {
        if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.outputIdxOffset.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "output_idx_offset",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                            "The dtype of output_idx_offset only supports int32"),
                        return ge::GRAPH_FAILED);
        }
    }
    // metadata 必传
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "metadata", "Metadata must not be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                    (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "block_table",
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((kLayout_ == DataLayout::PA_BBND) || (kLayout_ == DataLayout::BSND)) &&
                    (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR),
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "k", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (kLayout_ == DataLayout::TND) && (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName_, "k", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "3"),
        return ge::GRAPH_FAILED);

    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(qShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "q", std::to_string(qShapeDim).c_str(),
                                             std::to_string(expectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                                             std::to_string(expectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
        uint32_t outputIdxOffsetShapeDim = opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            (outputIdxOffsetShapeDim != expectShapeDim - 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "output_idx_offset", std::to_string(outputIdxOffsetShapeDim).c_str(),
                                         std::to_string(expectShapeDim - 1).c_str()),
            return ge::GRAPH_FAILED);
    }
    if (npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.returnValue == 1) {
        uint32_t sparseValuesShapeDim = opParamInfo_.sparseValues.shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(sparseValuesShapeDim != expectShapeDim,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "sparse_values", std::to_string(sparseValuesShapeDim).c_str(),
                                                 std::to_string(expectShapeDim).c_str()),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!(weightsShapeDim == expectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "w", std::to_string(weightsShapeDim).c_str(),
                                             std::to_string(expectShapeDim - 1).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                     const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, actualSeqLenName.c_str(), std::to_string(size).c_str(),
            "The shape size of " + actualSeqLenName + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckN2Size()
{
    // PA_BBND
    if (kLayout_ == DataLayout::TND) {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    } else {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    }
    OP_LOGI(context_->GetNodeName(), "N2 is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "k",
                                                      ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                                                      "The head num of k only supports 1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "q and k",
                                               Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) +
                                                   " and " +
                                                   Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                               "The head num of q can not be a multiple of the head num of k");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(gSize_ > G_SIZE_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q and k",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                        "The value of (the head num of q divided by the head num of k) must <= 64"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(gSize_ != G_SIZE_LIMIT,
                    OP_LOGE(opName_, "N1 is %u, N2 is %u, N1 divided by N2 must equal 64.", n1Size_, n2Size_),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, 以cu_seqlens_q的shape[0]-1为B轴大小
    if (qLayout_ == DataLayout::BSND) {
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        OP_LOGI(context_->GetNodeName(), "b: %d, s: %d, n: %d,d :%d",
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        return ge::GRAPH_SUCCESS;
    } else { // TND
        // cu_seqlens_q shape is [B+1], batch_size = shape[0] - 1
        uint32_t cuSeqLensQSize = 0;
        if (GetActualSeqLenSize(cuSeqLensQSize, opParamInfo_.cuSeqLensQ.tensor, "input cu_seqlens_q") !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            cuSeqLensQSize <= 1,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "cu_seqlens_q", std::to_string(cuSeqLensQSize).c_str(),
                                                      "The shape size of cu_seqlens_q should be greater than  1 (B+1)"),
            return ge::GRAPH_FAILED);
        bSize_ = cuSeqLensQSize - 1;

        // Validate key side batch size consistency
        if (kLayout_ == DataLayout::TND) {
            uint32_t cuSeqLensKSize = 0;
            if (GetActualSeqLenSize(cuSeqLensKSize, opParamInfo_.cuSeqLensK.tensor, "cu_seqlens_k") !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF((cuSeqLensKSize - 1) != bSize_,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "cu_seqlens_q and cu_seqlens_k",
                            Ops::Base::ToString(opParamInfo_.cuSeqLensK.tensor->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.cuSeqLensK.tensor->GetStorageShape()),
                            "The batch sizes derived from cu_seqlens_q and cu_seqlens_k must be same"),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QLIV2InfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OP_CHECK_IF(
        headDim_ != HEAD_DIM_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "q", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()),
                                              "The head dim of q only supports 128"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(
        ((blockSize_ % BLOCK_SIZE_FACTOR != 0) || (blockSize_ == 0) || (blockSize_ > BLOCK_SIZE_LIMIT)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "k", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()),
                                              "The block_size of k must be a multiple of 16 and belong to (0, 1024]"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF(
        (blockCount_ == 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "k", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()),
                                              "The block_count of k cannot be 0"),
        return ge::GRAPH_FAILED);

    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2SizeForBatchContinuous()
{
    std::string layout_key(opParamInfo_.layOutKey);
    if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    }
    OP_CHECK_IF((kLayout_ != DataLayout::BSND) && (kLayout_ != DataLayout::TND),
                OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_k", layout_key.c_str(), "BSND or TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    }
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus QLIV2InfoParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;

    if (qLayout_ == DataLayout::TND) {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于cu_seqlens_q (shape=[B+1], bSize_=B)
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BBND) &&
                        ((opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                         (opParamInfo_.blockTable.tensor != nullptr &&
                          opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "cu_seqlens_q, seqused_k and block_table",
                        Ops::Base::ToString(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.sequsedK.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "TND case, the dim 0 of cu_seqlens_q, seqused_k and block_table must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((kLayout_ == DataLayout::TND) && (opParamInfo_.cuSeqLensK.tensor->GetShapeSize() != bSize_ + 1),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        opName_, "cu_seqlens_q and cu_seqlens_k",
                        std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize()) + " and " +
                            std::to_string(opParamInfo_.cuSeqLensK.tensor->GetStorageShape().GetShapeSize()),
                        "TND case, the shape size of cu_seqlens_q and cu_seqlens_k must be same"),
                    return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q, w and sparse_indices",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                        "TND case q, w, sparse_values dim 0 are " + std::to_string(qTsize) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        if (npuArch_ == NpuArch::DAV_3510) {
            if (*opParamInfo_.returnValue == 1 && opParamInfo_.sparseValues.shape != nullptr) {
                OP_CHECK_IF((opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(0) != qTsize),
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                opName_, "q and sparse_values",
                                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                                    Ops::Base::ToString(opParamInfo_.sparseValues.shape->GetStorageShape()),
                                "TND case q and sparse_values dim 0 are " + std::to_string(qTsize) + ", " +
                                    std::to_string(opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(0)) +
                                    " respectively, they must be same"),
                            return ge::GRAPH_FAILED);
            }
            if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
                OP_CHECK_IF((opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0) != qTsize),
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                opName_, "q and output_idx_offset",
                                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                                    Ops::Base::ToString(opParamInfo_.outputIdxOffset.tensor->GetStorageShape()),
                                "TND case q and output_idx_offset dim 0 are " + std::to_string(qTsize) + " and " +
                                    std::to_string(opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0)) +
                                    " respectively, they must be same"),
                            return ge::GRAPH_FAILED);
            }
        }
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BBND) &&
                        ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                         (opParamInfo_.blockTable.tensor != nullptr &&
                          opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                         (opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                         (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q, w, seqused_k, block_table and sparse_indices",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.sequsedK.tensor->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                        "BSND case q, w, seqused_k, block_table, sparse_indices dim 0 are " + std::to_string(bSize_) +
                            ", " + std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(opParamInfo_.sequsedK.tensor->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ != DataLayout::PA_BBND) &&
                ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                 (opParamInfo_.sequsedK.tensor != nullptr && opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                 (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "q, w, seqused_k and sparse_indices",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.sequsedK.tensor->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                "BSND case q, w, seqused_k, sparse_indices dim 0 are " + std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(opParamInfo_.sequsedK.tensor->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (opParamInfo_.sequsedQ.tensor != nullptr) && (opParamInfo_.sequsedQ.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "q and seqused_q",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.sequsedQ.tensor->GetStorageShape()),
                "BSND case q, seqused_q dim 0 are " + std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.sequsedQ.tensor->GetStorageShape().GetDim(0)) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q, w and sparse_indices",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                        "BSND case q, w and sparse_indices dim 1 are " + std::to_string(s1Size_) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(1)) + ", " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q and w",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()),
                    "BSND case the head num of q, w are " + std::to_string(n1Size_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF(
        ((kLayout_ != DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_) ||
         (kLayout_ == DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != headDim_)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "q and k",
                                               Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) +
                                                   " and " +
                                                   Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                               "BSND case q, k last dim are " + std::to_string(headDim_) + ", " +
                                                   std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDim(
                                                       (kLayout_ == DataLayout::TND) ? DIM_IDX_TWO : DIM_IDX_THREE)) +
                                                   " respectively, they must be same"),
        return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "k and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case the head num of k, sparse_indices are " + std::to_string(n2Size_) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "sparse_count and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case sparse_count, sparse_indices last dim are " + std::to_string(*opParamInfo_.sparseCount) +
                        ", " + std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check cmp_residual_k-------------------
    if (opParamInfo_.cmpResidualK.tensor != nullptr) {
        OP_CHECK_IF(
            (opParamInfo_.cmpResidualK.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_residual_k", std::to_string(opParamInfo_.cmpResidualK.tensor->GetShapeSize()),
                "The shape size of cmp_residual_k must be equal to batch_size (" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    // -----------------------check sparse_values------------------
    if (npuArch_ == NpuArch::DAV_3510) {
        if (*opParamInfo_.returnValue == 1) {
            OP_CHECK_IF((opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "k and sparse_values",
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.sparseValues.shape->GetStorageShape()),
                            "The head num of k and sparse_values must be same"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                (opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "topk and sparse_values",
                    std::to_string(*opParamInfo_.sparseCount) + " and " +
                        Ops::Base::ToString(opParamInfo_.sparseValues.shape->GetStorageShape()),
                    "The last dim of sparse_values must be same as topk"),
                return ge::GRAPH_FAILED);
        }
    }
    // -----------------------check metadata-------------------
    OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "metadata", std::to_string(opParamInfo_.metadata.tensor->GetShapeSize()).c_str(),
                    "The shape size of metadata must be " + std::to_string(METADATA_LIMIT)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckScaleShape()
{
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDimNum();
    bool isMxQuantMode = (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4);

    if (*opParamInfo_.quantMode == QUANT_MODE_HIFLOAT8) {
        OP_CHECK_IF(qDequantScaleShapeDim != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "q_descale", std::to_string(qDequantScaleShapeDim).c_str(),
                        "When quant_mode is 4, the dim num of q_descale should be 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDequantScaleShapeDim != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "k_descale", std::to_string(kDequantScaleShapeDim).c_str(),
                        "When quant_mode is 4, the dim num of k_descale should be 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(0) != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "q_descale", ToStringRaw(opParamInfo_.query_dequant_scale.shape->GetStorageShape()).c_str(),
                "When quant_mode is 4, q_descale's shape[0] should be 1, but now is " +
                    std::to_string(opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(0))),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(0) != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "k_descale", ToStringRaw(opParamInfo_.key_dequant_scale.shape->GetStorageShape()).c_str(),
                "When quant_mode is 4, k_descale's shape[0] should be 1, but now is " +
                    std::to_string(opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(0))),
            return ge::GRAPH_FAILED);
    } else if (isMxQuantMode) {
        OP_CHECK_IF(headDim_ % MX_SCALE_SHAPE_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "q", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                        "When quant_mode is " + std::to_string(*opParamInfo_.quantMode) +
                            ", head_dim should be a multiple of " + std::to_string(MX_SCALE_SHAPE_ALIGN) +
                            ", but now is " + std::to_string(headDim_)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim + 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "q_descale", std::to_string(qDequantScaleShapeDim).c_str(),
                        "When quant_mode is " + std::to_string(*opParamInfo_.quantMode) +
                            ", the dim num of q_descale should be " + std::to_string(qShapeDim + 1)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim + 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "k_descale", std::to_string(qDequantScaleShapeDim).c_str(),
                        "When quant_mode is " + std::to_string(*opParamInfo_.quantMode) +
                            ", the dim num of k_descale should be " + std::to_string(kShapeDim + 1)),
                    return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
            uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(
                dimValueQueryScale != dimValueQuery,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q and q_descale",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()),
                    "Q_descale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQueryScale) +
                        " and q's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQuery) + " are not same"),
                return ge::GRAPH_FAILED);
        }
        for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
            uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueKey = opParamInfo_.key.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueKeyScale != dimValueKey,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "k and k_descale",
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()),
                            "K_descale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKeyScale) +
                                " and k's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKey) +
                                " are not the same"),
                        return ge::GRAPH_FAILED);
        }
        uint32_t expectScaleD = headDim_ / MX_SCALE_SHAPE_ALIGN;
        OP_CHECK_IF(
            (opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(qShapeDim - 1) != expectScaleD) ||
                (opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(qShapeDim) != MX_E8M0_SCALE_PACK_NUM),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "q_descale", Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()),
                "When quant_mode is " + std::to_string(*opParamInfo_.quantMode) +
                    ", q_descale's last dims should be [" + std::to_string(expectScaleD) + ", " +
                    std::to_string(MX_E8M0_SCALE_PACK_NUM) + "]"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(kShapeDim - 1) != expectScaleD) ||
                (opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(kShapeDim) != MX_E8M0_SCALE_PACK_NUM),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "k_descale", Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()),
                "When quant_mode is " + std::to_string(*opParamInfo_.quantMode) +
                    ", k_descale's last dims should be [" + std::to_string(expectScaleD) + ", " +
                    std::to_string(MX_E8M0_SCALE_PACK_NUM) + "]"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim - 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "q_descale", std::to_string(qDequantScaleShapeDim).c_str(),
                                                 std::to_string(qShapeDim - 1).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim - 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "k_descale", std::to_string(kDequantScaleShapeDim).c_str(),
                                                 std::to_string(kShapeDim - 1).c_str()),
                    return ge::GRAPH_FAILED);
        // check q scale
        for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
            uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueQueryScale != dimValueQuery,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "q_descale and q",
                            Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()),
                            "q_descale's shape[" + std::to_string(i) + "] and q's shape[" + std::to_string(i) +
                                "] are not the same"),
                        return ge::GRAPH_FAILED);
        }
        // check k scale
        for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
            uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueKey = opParamInfo_.key.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueKeyScale != dimValueKey,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "k_descale and k",
                            Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                            "k_descale's shape[" + std::to_string(i) + "] and k's shape[" + std::to_string(i) +
                                "] are not the same"),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BBND时，只允许0轴非连续，其余轴必须连续
// 非PA_BBND时，所有轴都必须连续
ge::graphStatus QLIV2InfoParser::CheckKeyContiguous() const
{
    bool keyNonContiguous = false;
    bool scaleNonContiguous = false;
    // PA_BBND: axis 0 may be non-contiguous (paged cache stride0); remaining axes must be contiguous.
    // Non-PA_BBND: every axis must be contiguous.
    size_t checkStartIdx = (kLayout_ == DataLayout::PA_BBND) ? 1 : 0;
    if (!keyStridesVec_.empty() && opParamInfo_.key.shape != nullptr) {
        auto &shape = opParamInfo_.key.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3), shape.GetDim(2) * shape.GetDim(3),
                               shape.GetDim(3), 1};
        } else if (kLayout_ == DataLayout::TND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyStridesVec_.size() && keyStridesVec_[i] != expectedStrides[i]) {
                keyNonContiguous = true;
                break;
            }
        }
    }
    bool isMxQuantMode = (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4);
    if ((*opParamInfo_.quantMode != QUANT_MODE_HIFLOAT8) && !keyDequantScaleStridesVec_.empty() &&
        opParamInfo_.key_dequant_scale.shape != nullptr) {
        auto &shape = opParamInfo_.key_dequant_scale.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (isMxQuantMode) {
            if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3) * shape.GetDim(4),
                                   shape.GetDim(2) * shape.GetDim(3) * shape.GetDim(4),
                                   shape.GetDim(3) * shape.GetDim(4), shape.GetDim(4), 1};
            } else if (kLayout_ == DataLayout::TND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3),
                                   shape.GetDim(2) * shape.GetDim(3), shape.GetDim(3), 1};
            }
        } else {
            if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
            } else if (kLayout_ == DataLayout::TND) {
                expectedStrides = {shape.GetDim(1), 1};
            }
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyDequantScaleStridesVec_.size() && keyDequantScaleStridesVec_[i] != expectedStrides[i]) {
                scaleNonContiguous = true;
                break;
            }
        }
    }
    if (kLayout_ == DataLayout::PA_BBND) {
        if (!keyStridesVec_.empty()) {
            if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
                OP_CHECK_IF(
                    keyStridesVec_[0] <= 0 || keyStridesVec_[0] % MXFP4_PACK_NUM != 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "k",
                        "When quant_mode is 5 and layout_k is PA_BBND, key stride0 must be positive and satisfy "
                        "2-element FP4 packing alignment, but got " +
                            std::to_string(keyStridesVec_[0])),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(keyStridesVec_[0] <= 0,
                            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                                opName_, "k",
                                "When layout_k is PA_BBND, key stride0 must be positive, but got " +
                                    std::to_string(keyStridesVec_[0])),
                            return ge::GRAPH_FAILED);
            }
        }
        if (isMxQuantMode && !keyDequantScaleStridesVec_.empty()) {
            OP_CHECK_IF(
                keyDequantScaleStridesVec_[0] <= 0 || keyDequantScaleStridesVec_[0] % MX_E8M0_SCALE_PACK_NUM != 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "k_descale",
                    "When quant_mode is 3 or 5 and layout_k is PA_BBND, key_dequant_scale stride0 must be positive "
                    "and satisfy 2-element E8M0 packing alignment, but got " +
                        std::to_string(keyDequantScaleStridesVec_[0])),
                return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(keyNonContiguous || scaleNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "k and k_descale", "k and k_descale only support non-continuous keying on the 0-axis"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIV2InfoParser::GenerateInfo(QLIV2TilingInfo &QLIV2Info)
{
    QLIV2Info.opName = opName_;
    QLIV2Info.platformInfo = platformInfo_;
    QLIV2Info.opParamInfo = opParamInfo_;
    QLIV2Info.socVersion = socVersion_;
    QLIV2Info.npuArch = npuArch_;

    QLIV2Info.bSize = bSize_;
    QLIV2Info.n1Size = n1Size_;
    QLIV2Info.n2Size = n2Size_;
    QLIV2Info.s1Size = s1Size_;
    QLIV2Info.s2Size = s2Size_;
    QLIV2Info.gSize = gSize_;

    QLIV2Info.inputQType = inputQType_;
    QLIV2Info.inputKType = inputKType_;
    QLIV2Info.outputType = outputType_;

    QLIV2Info.blockSize = blockSize_;
    QLIV2Info.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    QLIV2Info.pageAttentionFlag = (kLayout_ == DataLayout::PA_BBND);
    QLIV2Info.sparseMode = *opParamInfo_.sparseMode;
    QLIV2Info.sparseCount = *opParamInfo_.sparseCount;
    QLIV2Info.cmpRatio = *opParamInfo_.cmpRatio;
    QLIV2Info.returnValue = *opParamInfo_.returnValue;
    QLIV2Info.maxSeqlenQ = (opParamInfo_.maxSeqlenQ != nullptr) ? *opParamInfo_.maxSeqlenQ : -1;

    QLIV2Info.keyStridesVec = keyStridesVec_;
    QLIV2Info.keyDequantScaleStridesVec = keyDequantScaleStridesVec_;
    if (!keyStridesVec_.empty()) {
        uint32_t keyStride0 = static_cast<uint32_t>(keyStridesVec_[0]);
        if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
            // FP4 shape stride以逻辑元素计数，kernel侧以打包后的uint8为寻址单位。
            keyStride0 /= MXFP4_PACK_NUM;
        }
        QLIV2Info.keyStride0 = keyStride0;
    } else {
        QLIV2Info.keyStride0 = 0; // 非PA无需使用stride
    }
    if (!keyDequantScaleStridesVec_.empty()) {
        QLIV2Info.keyDequantScaleStride0 = static_cast<uint32_t>(keyDequantScaleStridesVec_[0]);
    } else if ((*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4)) {
        QLIV2Info.keyDequantScaleStride0 = static_cast<uint32_t>(blockSize_) * (headDim_ / MX_SCALE_GROUP_SIZE);
    } else {
        QLIV2Info.keyDequantScaleStride0 = 0;
    }

    QLIV2Info.inputQLayout = qLayout_;
    QLIV2Info.inputKLayout = kLayout_;
}

ge::graphStatus QLIV2InfoParser::ParseAndCheck(QLIV2TilingInfo &QLIV2Info)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape() ||
        ge::GRAPH_SUCCESS != CheckKeyContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(QLIV2Info);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus QuantLightningIndexerV2Tiling::DoTiling(QLIV2TilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 256;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 2048;           // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 4;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    constexpr uint32_t TOPK_MAX_SIZE_950 = 8192;      // A5 TopK最大选取个数
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t S1_BASE_SIZE_950 = 4;
        constexpr uint32_t S2_BASE_SIZE_950 = 128;
        workspaceSize += S1_BASE_SIZE_950 * ((tilingInfo->s2Size + S2_BASE_SIZE_950 - 1) / S2_BASE_SIZE_950) *
                         S2_BASE_SIZE_950 * sizeof(uint16_t) * aicNum;
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE_950 * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    } else {
        uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.set_returnValue(tilingInfo->returnValue);
    tilingData_.set_maxSeqlenQ(tilingInfo->maxSeqlenQ);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_keyDequantScaleStride0(tilingInfo->keyDequantScaleStride0);
    tilingData_.set_quantMode(*tilingInfo->opParamInfo.quantMode);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantLightningIndexerV2(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantLightningIndexerV2", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    QLIV2TilingInfo QLIV2Info;
    QLIV2InfoParser QLIV2InfoParser(context);
    if (QLIV2InfoParser.ParseAndCheck(QLIV2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerV2Tiling QLIV2Tiling(context);
    return QLIV2Tiling.DoTiling(&QLIV2Info);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexerV2)
    .Tiling(TilingForQuantLightningIndexerV2)
    .TilingParse<QLIV2CompileInfo>(TilingPrepareForQuantLightningIndexerV2);

} // namespace optiling
