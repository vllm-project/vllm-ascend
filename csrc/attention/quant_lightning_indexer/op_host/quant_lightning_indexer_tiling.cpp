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
 * \file quant_lightning_indexer_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_tiling.h"

#include "../op_kernel/quant_lightning_indexer_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
// --------------------------QLIInfoParser类成员函数定义-------------------------------------
ge::graphStatus QLIInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE(opName_, "Shape of tensor weights is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE(opName_, "Desc of tensor weights is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.shape == nullptr,
               OP_LOGE(opName_, "Shape of tensor query_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.desc == nullptr,
               OP_LOGE(opName_, "Desc of tensor query_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.shape == nullptr,
               OP_LOGE(opName_, "Shape of tensor key_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.desc == nullptr,
               OP_LOGE(opName_, "Desc of tensor key_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOutQuery == nullptr, OP_LOGE(opName_, "attr layout_query is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr, OP_LOGE(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseCount == nullptr, OP_LOGE(opName_, "attr sparse_count is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE(opName_, "attr sparse_mode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.queryQuantMode == nullptr, OP_LOGE(opName_, "query_quant_mode is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.keyQuantMode == nullptr, OP_LOGE(opName_, "key_quant_mode is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("QuantLightningIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) && 
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_95)) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
               OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void QLIInfoParser::GetInputParaInfo()
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

void QLIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(quant_lightning_indexer);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(quant_lightning_indexer);
}

ge::graphStatus QLIInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);

    opParamInfo_.queryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_QUANT_MODE_INDEX);
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_COUNT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.returnValues = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUES_INDEX);

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
    if (opParamInfo_.preTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "preTokens is:%d", *opParamInfo_.preTokens);
    }
    if (opParamInfo_.nextTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "nextTokens is:%d", *opParamInfo_.nextTokens);
    }
    if (opParamInfo_.cmpRatio != nullptr) {
        OP_LOGI(context_->GetNodeName(), "cmpRatio is:%d", *opParamInfo_.cmpRatio);
    }
    if (opParamInfo_.returnValues != nullptr) {
        OP_LOGI(context_->GetNodeName(), "returnValues is:%s", *opParamInfo_.returnValues ? "true" : "false");
    }
    if (opParamInfo_.queryQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "query_quant_mode mode is:%d", *opParamInfo_.queryQuantMode);
    }
    if (opParamInfo_.keyQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "key_quant_mode mode is:%d", *opParamInfo_.keyQuantMode);
    }
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckAttrParaInfo()
{
    std::string layout_key(opParamInfo_.layOutKey);
    std::string layout_query(opParamInfo_.layOutQuery);
    if ((socVersion_ == platform_ascendc::SocVersion::ASCEND910B) ||
        (socVersion_ == platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_CHECK_IF(
            ((std::string(opParamInfo_.layOutKey) == "BNSD") || (std::string(opParamInfo_.layOutKey) == "PA_BBND")),
            OP_LOGE(opName_, "input attr layout_key only supported PA_BSND, PA_BBND, BSND or TND,"
                    "but now layout_key is %s.", layout_key.c_str()),
                    return ge::GRAPH_FAILED);
    } else if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        OP_CHECK_IF(
            ((std::string(opParamInfo_.layOutKey) != "PA_BSND")),
            OP_LOGE(opName_, "input attr layout_key only supported PA_BSND,"
                        "but now layout_key is %s.", layout_key.c_str()),
                        return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(((std::string(opParamInfo_.layOutQuery) != "BSND") && (std::string(opParamInfo_.layOutQuery) != "TND")),
               OP_LOGE(opName_, "input attr layout_query only supported BSND or TND."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND") &&
        (std::string(opParamInfo_.layOutQuery)) != (std::string(opParamInfo_.layOutKey))),
        OP_LOGE(opName_,  "outside of PA, input attr layout_query and input attr layout_key must be the same,"
                  "but now layout_key is %s, layout_query is %s.",
         layout_key.c_str(),  layout_query.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
                OP_LOGE(opName_, "input attr sparse_count must > 0 and <= %d, but now sparse_count is %d",
                       SPARSE_LIMIT, *opParamInfo_.sparseCount),return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
                OP_LOGE(opName_, "input attr sparse_mode only supported 0 or 3, but now sparseMode is %d.",
                       *opParamInfo_.sparseMode), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.preTokens != 9223372036854775807,
                OP_LOGE(opName_, "input attr preTokens only supported 9223372036854775807, but now preTokens is %ld.",
                *opParamInfo_.preTokens), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.nextTokens != 9223372036854775807,
                OP_LOGE(opName_, "input attr nextTokens only supported 9223372036854775807, but now nextTokens is %ld.",
                *opParamInfo_.nextTokens), return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128) 
                || ((*opParamInfo_.cmpRatio & (*opParamInfo_.cmpRatio - 1)) != 0),
                OP_LOGE(opName_, "input attr cmpRatio must > 0 and <= 128 and should be powers of 2, but now cmpRatio is %ld.",
                *opParamInfo_.cmpRatio), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.queryQuantMode != 0, OP_LOGE(opName_, "input attr query_quant_mode only supported 0."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.keyQuantMode != 0, OP_LOGE(opName_, "input attr key_quant_mode only supported 0."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.returnValues, OP_LOGE(opName_, "input attr returnValues only supported False."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetOpParaInfo()
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

ge::graphStatus QLIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    inputQueryScaleType_ = opParamInfo_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = opParamInfo_.key_dequant_scale.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    socVersion_ = ascendcPlatform.GetSocVersion();

    OP_CHECK_IF(!(inputQType_ == inputKType_),
               OP_LOGE(opName_, "The data types of the input query and key must be the same."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !(inputQueryScaleType_ == inputKeyScaleType_),
        OP_LOGE(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be the same."),
        return ge::GRAPH_FAILED);
    if ((socVersion_ == platform_ascendc::SocVersion::ASCEND910B) ||
        (socVersion_ == platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_CHECK_IF(inputQType_ != ge::DT_INT8,
               OP_LOGE(opName_, "The data types of the input query and key must be int8."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputQueryScaleType_ != ge::DT_FLOAT16,
            OP_LOGE(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be float16."),
            return ge::GRAPH_FAILED);
    } else if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        OP_CHECK_IF(inputQType_ != ge::DT_FLOAT8_E4M3FN,
               OP_LOGE(opName_, "The data types of the input query and key must be float8_e4m3."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputQueryScaleType_ != ge::DT_FLOAT,
            OP_LOGE(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be float."),
            return ge::GRAPH_FAILED);
    }

    if ((socVersion_ == platform_ascendc::SocVersion::ASCEND910B) ||
        (socVersion_ == platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_CHECK_IF(weightsType_ != ge::DT_FLOAT16,
                OP_LOGE(opName_, "The data types of the input weights must be float16."), return ge::GRAPH_FAILED);
    } else if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
            OP_CHECK_IF(weightsType_ != ge::DT_FLOAT,
                OP_LOGE(opName_, "The data types of the input weights must be float."), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(outputType_ != ge::DT_INT32,
               OP_LOGE(opName_, "The data types of the output sparse_indices must be int32."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutQueryMap = {{"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}};

    std::string layout_query(opParamInfo_.layOutQuery);
    auto QLayout_ = layoutQueryMap.find(layout_query);
    if (QLayout_ != layoutQueryMap.end()) {
        qLayout_ = QLayout_->second;
    }

    const map<string, DataLayout> layoutKeyMap = {
        {"BSND", DataLayout::BSND}, {"TND", DataLayout::TND},
        {"PA_BSND", DataLayout::PA_BSND}, {"PA_BBND", DataLayout::PA_BSND}};
    std::string layout_key(opParamInfo_.layOutKey);
    auto KLayout = layoutKeyMap.find(layout_key);
    if (KLayout != layoutKeyMap.end()) {
        kLayout_ = KLayout->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::PA_BSND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OP_LOGE(opName_, "key layout only supported PA_BSND, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.actualSeqLengthsK.tensor == nullptr,
            OP_LOGE(opName_, "key layout only supported PA_BSND, input actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OP_LOGE(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                   OP_LOGE(opName_, "key layout is not PA_BSND, input block_table must be null"),
                   return ge::GRAPH_FAILED);
    }

    if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsK.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_key is TND, input actual_seq_lengths_key must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsK.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsK.desc->GetDataType() != ge::DT_INT32,
                   OP_LOGE(opName_, "input actual_seq_lengths_key data type only support int32"),
                   return ge::GRAPH_FAILED);
    if (qLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                   opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input actual_seq_lengths_query data type only support int32"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
               OP_LOGE(opName_, "input metadata must not be null"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE(opName_, "the dim num of block_table's shape should be 2, but now is %u",
                opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((kLayout_ == DataLayout::PA_BSND)||(kLayout_ == DataLayout::BSND)) &&
        (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR),
        OP_LOGE(opName_, "the dim num of key's shape should be 4, but now is %u",
                opParamInfo_.key.shape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (kLayout_ == DataLayout::TND) && (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE),
        OP_LOGE(opName_, "the dim num of key's shape should be 3, but now is %u",
                opParamInfo_.key.shape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(
        qShapeDim != expectShapeDim,
        OP_LOGE(opName_, "the dim num of query's shape should be %u, but now is %u", expectShapeDim, qShapeDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != expectShapeDim,
               OP_LOGE(opName_, "the dim num of sparse_indices's shape should be %u, but now is %u", expectShapeDim,
                         outShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == expectShapeDim - 1),
               OP_LOGE(opName_, "the dim num of weights's shape should be %u, but now is %u", expectShapeDim - 1,
                         weightsShapeDim),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetN1Size()
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

ge::graphStatus QLIInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                   const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OP_LOGE(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckN2Size()
{
    // PA_BSND
    if (kLayout_ == DataLayout::TND) {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    } else {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    }
    OP_LOGI(context_->GetNodeName(), "N2 is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1, OP_LOGE(opName_, "key shape[2] is numhead, only support 1."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    OP_CHECK_IF(gSize_ != G_SIZE_LIMIT,
               OP_LOGE(opName_, "N1 is %u, N2 is %u, N1 divided by N2 must equal 64.", n1Size_, n2Size_),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、Q和K都为TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    // 3、Q为TND，K为PA_BSND时，以actual_seq_lens_k数组的长度为B轴大小
    if (qLayout_ == DataLayout::BSND) {
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        OP_LOGI(context_->GetNodeName(), "b: %d, s: %d, n: %d,d :%d",
            opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
            opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
            opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
            opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        return ge::GRAPH_SUCCESS;
    } else {  // TND
        uint32_t bSizeQuery;
        uint32_t bSizeKey;
        GetActualSeqLenSize(bSizeQuery, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
        GetActualSeqLenSize(bSizeKey, opParamInfo_.actualSeqLengthsK.tensor, "input actual_seq_lengths_key");
        if (kLayout_ == DataLayout::TND) {
            OP_CHECK_IF(bSizeQuery != bSizeKey,
                OP_LOGE(opName_, "the lengths of actual_seq_lengths_query is %u, %u respectively, they must be same.",
                        bSizeQuery, bSizeKey),
                return ge::GRAPH_FAILED);
            bSize_ = bSizeQuery;
        } else {
            bSize_ = bSizeKey; // Q为TND，batch从Key中获取
        }
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QLIInfoParser::GetHeadDim()
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
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT, OP_LOGE(opName_, "input query's last dim head_dim only support 128, but now is %u.", headDim_),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(
        ((blockSize_ % BLOCK_SIZE_FACTOR != 0) || (blockSize_ == 0) || (blockSize_ > BLOCK_SIZE_LIMIT)),
        OP_LOGE(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024], but now is %d.", blockSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF((blockCount_ == 0), OP_LOGE(opName_, "input key's block_count cannot be 0."), return ge::GRAPH_FAILED);

    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2SizeForBatchContinuous()
{
    std::string layout_key(opParamInfo_.layOutKey);
    if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    }
    OP_CHECK_IF((kLayout_ != DataLayout::BSND) && (kLayout_ != DataLayout::TND),
        OP_LOGE(opName_, "the layout of key is %s, it is unsupported.", layout_key.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BSND) {
        return GetS2SizeForPageAttention();
    }
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus QLIInfoParser::ValidateInputShapesMatch()
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
        // bSize_ 来源于act_seq_q
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BSND) &&
                ((opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                (opParamInfo_.blockTable.tensor != nullptr &&
                opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE(
                opName_,
                "TND case input actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are %u, %u, %u "
                "respectively, they must be same.",
                bSize_, opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize(),
                opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((kLayout_ != DataLayout::PA_BSND) &&
                   (opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_),
            OP_LOGE(
                opName_,
                "TND case input actual_seq_lengths_query, actual_seq_lengths_key, are %u, %u "
                "respectively, they must be same.",
                bSize_, opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize()),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                       (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                   OP_LOGE(opName_,
                             "TND case input query, weights, sparse_indices dim 0 are %u, %u, %u "
                             "respectively, they must be same.",
                             qTsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                             opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                   return ge::GRAPH_FAILED);
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BSND) &&
                    ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.blockTable.tensor != nullptr &&
                    opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
                   OP_LOGE(opName_,
                             "BSND case input query, weight, actual_seq_lengths_key, block_table, sparse_indices dim 0 are %u, %u, %u, %u, %u respectively, they must be same.",
                              bSize_, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                              opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize(),
                              opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0),
                              opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF((kLayout_ != DataLayout::PA_BSND) &&
                    ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.actualSeqLengthsK.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
                   OP_LOGE(opName_,
                             "BSND case input query, weight, actual_seq_lengths_key, sparse_indices dim 0 are %u, %u, %u, %u respectively, they must be same.",
                              bSize_, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                              opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize(),
                              opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
                (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
            OP_LOGE(
                opName_,
                "BSND case input query, actual_seq_lengths_query dim 0 are %u, %ld respectively, they must be same",
                bSize_, opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize()),
            return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OP_CHECK_IF(
            (opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
            OP_LOGE(opName_, "BSND case input query, weight, sparse_indices dim 1 are %u, %u, %u, they must be same.",
                      s1Size_, opParamInfo_.weights.shape->GetStorageShape().GetDim(1),
                      opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)),
            return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
               OP_LOGE(opName_, "input query, weight shape dim N1 must be same, but now are %u, %u respectively, they must be same.",
               opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim), n1Size_),
               return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF(
        ((kLayout_ != DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_)
        || (kLayout_ == DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != headDim_)),
                OP_LOGE(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE(opName_, "input query and output sparse_indices shape n2 dim must be same."),
                return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
               OP_LOGE(opName_, "output sparse_indices shape last dim must be same as attr sparse_count."),
               return ge::GRAPH_FAILED);
    // -----------------------check metadata-------------------
     OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
                OP_LOGE(opName_, "input metadata dim 0 must be %u.", METADATA_LIMIT),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckScaleShape()
{
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim - 1),
               OP_LOGE(opName_, "the dim num of query_dequant_scale's shape should be %u, but now is %u",
                         qShapeDim - 1, qDequantScaleShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim - 1),
               OP_LOGE(opName_, "the dim num of key_dequant_scale's shape should be %u, but now is %u", kShapeDim - 1,
                         kDequantScaleShapeDim),
               return ge::GRAPH_FAILED);
    // check q scale
    for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
        uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
        uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(dimValueQueryScale != dimValueQuery,
                   OP_LOGE(opName_, "query_dequant_scale's shape[%u] %u and query's shape[%u] %u is not same", i,
                             dimValueQueryScale, i, dimValueQuery),
                   return ge::GRAPH_FAILED);
    }
    // check k scale
    for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
        uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(i);
        uint32_t dimValueKey = opParamInfo_.key.shape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(dimValueKeyScale != dimValueKey,
                   OP_LOGE(opName_, "key_dequant_scale's shape[%u] %u and key's shape[%u] %u is not same", i,
                             dimValueKeyScale, i, dimValueKey),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void QLIInfoParser::GenerateInfo(QLITilingInfo &QLIInfo)
{
    QLIInfo.opName = opName_;
    QLIInfo.platformInfo = platformInfo_;
    QLIInfo.opParamInfo = opParamInfo_;
    QLIInfo.socVersion = socVersion_;

    QLIInfo.bSize = bSize_;
    QLIInfo.n1Size = n1Size_;
    QLIInfo.n2Size = n2Size_;
    QLIInfo.s1Size = s1Size_;
    QLIInfo.s2Size = s2Size_;
    QLIInfo.gSize = gSize_;

    QLIInfo.inputQType = inputQType_;
    QLIInfo.inputKType = inputKType_;
    QLIInfo.outputType = outputType_;

    QLIInfo.blockSize = blockSize_;
    QLIInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    QLIInfo.pageAttentionFlag = (kLayout_ == DataLayout::PA_BSND);
    QLIInfo.sparseMode = *opParamInfo_.sparseMode;
    QLIInfo.sparseCount = *opParamInfo_.sparseCount;
    QLIInfo.preTokens = *opParamInfo_.preTokens;
    QLIInfo.nextTokens = *opParamInfo_.nextTokens;
    QLIInfo.cmpRatio = *opParamInfo_.cmpRatio;
    QLIInfo.returnValues = *opParamInfo_.returnValues;

    QLIInfo.inputQLayout = qLayout_;
    QLIInfo.inputKLayout = kLayout_;
}

ge::graphStatus QLIInfoParser::ParseAndCheck(QLITilingInfo &QLIInfo)
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
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(QLIInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerTiling类成员函数定义-----------------------
ge::graphStatus QuantLightningIndexerTiling::DoTiling(QLITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;          // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;              // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;              // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;             // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;           // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;           // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8;  // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;       // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;         // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;               // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;           // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // Decode流程(LD)需要Workspace大小
    // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
    // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
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
    tilingData_.set_returnValues(tilingInfo->returnValues);
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
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantLightningIndexer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantLightningIndexer", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    QLITilingInfo QLIInfo;
    QLIInfoParser QLIInfoParser(context);
    if (QLIInfoParser.ParseAndCheck(QLIInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerTiling QLITiling(context);
    return QLITiling.DoTiling(&QLIInfo);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexer)
    .Tiling(TilingForQuantLightningIndexer)
    .TilingParse<QLICompileInfo>(TilingPrepareForQuantLightningIndexer);

}  // namespace optiling
