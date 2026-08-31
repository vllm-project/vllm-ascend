/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <string>
#include <utility>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "error/ops_error.h"
#include "fused_infer_attention_score_v2_sink_tiling_index.h"
#include "fused_infer_attention_score_v2_sink_tiling_info_parser.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaInfoParser::CheckRequiredInOutExistence() const
{
    OPS_CHECK(opParamInfo_.query.shape == nullptr,
              OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.query.desc == nullptr,
              OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(
        opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(
        opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.value.shape == nullptr,
              OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.value.desc == nullptr,
              OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.attenOut.shape == nullptr,
              OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.attenOut.desc == nullptr,
              OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::CheckRequiredAttrExistence() const
{
    OPS_CHECK(
        opParamInfo_.numHeads == nullptr, OPS_LOG_E(opName_, "attr numHeads is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(
        opParamInfo_.scaleValue == nullptr, OPS_LOG_E(opName_, "attr scaleValue is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(
        opParamInfo_.kvHeadNums == nullptr, OPS_LOG_E(opName_, "attr kvHeadNums is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.layOut == nullptr, OPS_LOG_E(opName_, "attr layout is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(
        opParamInfo_.blockSize == nullptr, OPS_LOG_E(opName_, "attr blockSize is nullptr"), return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.antiquantMode == nullptr,
              OPS_LOG_E(opName_, "attr antiquantMode is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.softmaxLseFlag == nullptr,
              OPS_LOG_E(opName_, "attr softmaxLseFlag is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.batchInvariant == nullptr,
              OPS_LOG_E(opName_, "attr batchInvariant is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.softmaxMaxSumFlag == nullptr,
              OPS_LOG_E(opName_, "attr softmaxMaxSumFlag is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.keyAntiquantMode == nullptr,
              OPS_LOG_E(opName_, "attr keyAntiquantMode is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.valueAntiquantMode == nullptr,
              OPS_LOG_E(opName_, "attr valueAntiquantMode is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.innerPrecise == nullptr,
              OPS_LOG_E(opName_, "attr innerPrecise is nullptr"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(opParamInfo_.queryQuantMode == nullptr,
              OPS_LOG_E(opName_, "attr queryQuantMode is nullptr"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetEmptyTensorFlag()
{
    if ((opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0 &&
         opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() != 0) ||
        (opParamInfo_.query.shape->GetStorageShape().GetShapeSize() != 0 &&
         opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() == 0)) {
        OPS_LOG_E(opName_,
                  "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty "
                  "while the other is not",
                  opParamInfo_.query.shape->GetStorageShape().GetShapeSize(),
                  opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0 &&
        opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize() == 0) {
        emptyTensorFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    if (*opParamInfo_.softmaxLseFlag) {
        if ((opParamInfo_.lseOut.shape == nullptr) ||
            (opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize() == 0)) {
            OPS_LOG_E(opName_, "lse Flag is %u, but lse shape size is 0 byte", *opParamInfo_.softmaxLseFlag);
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &kTensor : kCache_) {
        if (kTensor->GetShape().GetShapeSize() != 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    for (auto &vTensor : vCache_) {
        if (vTensor->GetShape().GetShapeSize() != 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    emptyTensorFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxWorkspaceFlag()
{
    isMaxWorkspace_ = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetLegacyIfaFlag()
{
    std::string layout(opParamInfo_.layOut);
    if ((layout == "BSH" || layout == "BSND" || layout == "BNSD") && s1Size_ == 1U && qkHeadDim_ == vHeadDim_ &&
        opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor == nullptr) {
        isLegacyIfa_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqLenSize(uint32_t &size,
                                                   const gert::Tensor *tensor,
                                                   const FiaLayout &layout,
                                                   const std::string &actualSeqLenName,
                                                   const std::string &attrName)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_,
                  "when %s's layout is %s, %s must be provided.",
                  attrName.c_str(),
                  LayoutToSerialString(layout).c_str(),
                  actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "%s's shape size is %ld, it should be greater than 0.", actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(
        size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME);
}

ge::graphStatus FiaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_CHECK(platformInfo_ == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
              return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_CHECK(aicNum == 0 || aivNum == 0,
              OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
              return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND310P) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND950) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_55)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.pseShift.tensor = context_->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    opParamInfo_.pseShift.desc = context_->GetOptionalInputDesc(PSE_SHIFT_INDEX);
    opParamInfo_.attenMask.tensor = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    opParamInfo_.attenMask.desc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.deqScale1.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    opParamInfo_.deqScale1.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.tensor = context_->GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.desc = context_->GetOptionalInputDesc(QUANT_SCALE1_INDEX);
    opParamInfo_.deqScale2.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    opParamInfo_.deqScale2.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE2_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    GetOptionalInputParaPostQuantInfo();
    opParamInfo_.antiquantScale.tensor = context_->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantScale.desc = context_->GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantOffset.tensor = context_->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.antiquantOffset.desc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    opParamInfo_.queryPaddingSize.desc = context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX);
    opParamInfo_.kvPaddingSize.tensor = context_->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    opParamInfo_.kvPaddingSize.desc = context_->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX);
    opParamInfo_.keyAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantOffset.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.keyAntiquantOffset.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantScale.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantScale.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantOffset.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantOffset.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    GetOptionalInputParaRopeInfo();
    opParamInfo_.dequantScaleQuery.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    opParamInfo_.dequantScaleQuery.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
    opParamInfo_.learnableSink.tensor = context_->GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    opParamInfo_.learnableSink.desc = context_->GetOptionalInputDesc(LEARNABLE_SINK_INDEX);
    opParamInfo_.keySink.tensor = context_->GetOptionalInputTensor(KEY_SINK_INDEX);
    opParamInfo_.keySink.desc = context_->GetOptionalInputDesc(KEY_SINK_INDEX);
    opParamInfo_.keyRopeSink.tensor = context_->GetOptionalInputTensor(KEY_ROPE_SINK_INDEX);
    opParamInfo_.keyRopeSink.desc = context_->GetOptionalInputDesc(KEY_ROPE_SINK_INDEX);
    opParamInfo_.valueSink.tensor = context_->GetOptionalInputTensor(VALUE_SINK_INDEX);
    opParamInfo_.valueSink.desc = context_->GetOptionalInputDesc(VALUE_SINK_INDEX);
}

void FiaInfoParser::GetOptionalInputParaPostQuantInfo()
{
    opParamInfo_.quantScale2.tensor = context_->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    opParamInfo_.quantScale2.desc = context_->GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    opParamInfo_.quantOffset2.tensor = context_->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    opParamInfo_.quantOffset2.desc = context_->GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
}

void FiaInfoParser::GetOptionalInputParaRopeInfo()
{
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX);
    opParamInfo_.keyRopeAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyRopeAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
}

void FiaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    GetOptionalInputParaInfo();
}

void FiaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(ATTENTION_OUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
    opParamInfo_.softmaxMaxOut.desc = context_->GetOutputDesc(SOFTMAX_MAX_INDEX);
    opParamInfo_.softmaxMaxOut.shape = context_->GetOutputShape(SOFTMAX_MAX_INDEX);
    opParamInfo_.softmaxSumOut.desc = context_->GetOutputDesc(SOFTMAX_SUM_INDEX);
    opParamInfo_.softmaxSumOut.shape = context_->GetOutputShape(SOFTMAX_SUM_INDEX);
}

ge::graphStatus FiaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
              return ge::GRAPH_FAILED);

    opParamInfo_.numHeads = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    opParamInfo_.kvHeadNums = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    opParamInfo_.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    opParamInfo_.antiquantMode = attrs->GetAttrPointer<int64_t>(ANTIQUANT_MODE_INDEX);
    opParamInfo_.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    opParamInfo_.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    opParamInfo_.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    opParamInfo_.innerPrecise = attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX);
    opParamInfo_.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX);
    opParamInfo_.sinkNumber = attrs->GetAttrPointer<int64_t>(ATTR_SINK_NUMBER_INDEX);
    opParamInfo_.batchInvariant = attrs->GetAttrPointer<bool>(ATTR_BATCH_INVARIANT_INDEX);
    opParamInfo_.softmaxMaxSumFlag = attrs->GetAttrPointer<bool>(SOFTMAX_MAX_SUM_FLAG_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetSparseMode()
{
    auto attrs = context_->GetAttrs();
    static int32_t SPARSE_ZERO = 0U;
    if (isLegacyIfa_) {
        opParamInfo_.sparseMode = &SPARSE_ZERO;
    } else {
        opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    }

    OPS_CHECK(
        opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparseMode is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetPreNextToken()
{
    auto attrs = context_->GetAttrs();
    static int64_t TOKEN_MAX = 2147483647;
    if (isLegacyIfa_) {
        opParamInfo_.preToken = &TOKEN_MAX;
        opParamInfo_.nextToken = &TOKEN_MAX;
    } else {
        opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
        opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    }

    int32_t sparseMode = (*opParamInfo_.sparseMode);
    if (sparseMode == SPARSE_MODE_ALL_MASK) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode == SPARSE_MODE_LEFT_UP || sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        nextToken_ = 0;
        preToken_ = SPARSE_MODE_INT_MAX;
    } else {
        preToken_ = opParamInfo_.preToken == nullptr ? 0 : *opParamInfo_.preToken;
        nextToken_ = opParamInfo_.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;
    }

    if (preToken_ > SPARSE_MODE_INT_MAX) {
        preToken_ = SPARSE_MODE_INT_MAX;
    } else if (preToken_ < -(SPARSE_MODE_INT_MAX)) {
        preToken_ = -(SPARSE_MODE_INT_MAX);
    }
    if (nextToken_ > SPARSE_MODE_INT_MAX) {
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (nextToken_ < -(SPARSE_MODE_INT_MAX)) {
        nextToken_ = -(SPARSE_MODE_INT_MAX);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvCache()
{
    // 处理Key和value的TensorList场景
    kCache_.clear();
    uint32_t keyBIdx = 0;
    while ((context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        kCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)));
        keyBIdx++;
    }
    OPS_CHECK(keyBIdx == 0,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tensor list of %s is empty.", KEY_NAME.c_str()),
              return ge::GRAPH_FAILED);
    kCache_.resize(keyBIdx);

    vCache_.clear();
    uint32_t valueBIdx = 0;
    while ((context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        vCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)));
        valueBIdx++;
    }
    vCache_.resize(valueBIdx);

    OPS_CHECK(kCache_.size() != vCache_.size(),
              OPS_REPORT_VECTOR_INNER_ERR(
                  context_->GetNodeName(),
                  "tensor list of %s has %zu tensor, but tensor list of %s has %zu tensor, they should be equal.",
                  KEY_NAME.c_str(),
                  kCache_.size(),
                  VALUE_NAME.c_str(),
                  vCache_.size()),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetB();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FiaInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (queryShape_->HasT()) ? static_cast<uint32_t>(queryShape_->GetT()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKTSize()
{
    kTSize_ = (keyShape_->HasT()) ? static_cast<uint32_t>(keyShape_->GetT()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND/NTD时, 以query的S维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        uint32_t b = 0;
        if (GetActualSeqLenQSize(b) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int64_t qActualSeqMax = 0;
        qActualSeqMax = (queryShape_->HasT()) ? static_cast<uint32_t>(queryShape_->GetT()) : 0;
        s1Size_ = static_cast<uint32_t>(qActualSeqMax);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetS());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvStorageMode()
{
    // kv存储模式基准值
    if (kCache_.size() > 1) {
        kvStorageMode_ = KvStorageMode::TENSOR_LIST;
    } else {
        if (opParamInfo_.blockTable.tensor != nullptr) {
            kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
        } else {
            kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        kvLayout_ = qLayout_;
    } else {
        uint32_t keyDimNum = kCache_[0]->GetShape().GetDimNum();
        if (keyDimNum == 3U) {
            kvLayout_ = FiaLayout::BnBsH;
        } else if (keyDimNum == 4U) {
            kvLayout_ = FiaLayout::BnNBsD;
        } else if (keyDimNum == 5U) {
            kvLayout_ = FiaLayout::NZ;
        } else {
            OPS_LOG_E(opName_,
                      "the first tensor of %s's tensor list is %u dim, only support 3/4/5.",
                      KEY_NAME.c_str(),
                      keyDimNum);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxActualSeq(
    const gert::Tensor *actualSeqLensTensor, FiaLayout layout, int64_t &maxActualSeqLen)
{
    maxActualSeqLen = (keyShape_->HasT()) ? static_cast<uint32_t>(keyShape_->GetT()) : 0;
    return GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeFromActualSeqLens()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.actualSeqLengths.tensor, kvLayout_, s2Size_);
}

ge::graphStatus FiaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetS();
        kvListSeqLens_.push_back(s2Size_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForTensorList()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        s2Size_ = 0;
        for (uint32_t i = 0; i < kCache_.size(); i++) {
            auto keyShape =
                std::make_shared<FiaTilingShape>(kCache_[i]->GetShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
            if (keyShape->GetS() > s2Size_) {
                s2Size_ = keyShape->GetS();
            }
            if (keyShape->GetS() != keyShape_->GetS()) {
                isSameSeqAllKVTensor_ = false;
            }
            kvListSeqLens_.push_back(keyShape->GetS());
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxBlockNumPerBatch()
{
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != 2U) {
        OPS_LOG_E(opName_, "the dim num of %s is %u, it should be 2.", BLOCK_TABLE_NAME.c_str(), dimNum);
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBlockSize()
{
    if (opParamInfo_.blockSize == nullptr) {
        OPS_LOG_E(opName_, "Attr %s not exist", BLOCK_SIZE_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    blockSize_ = *(opParamInfo_.blockSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、TENSOR_LIST时, 从kCache_的所有Tensor的S轴的最大值
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        return GetS2SizeForTensorList();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FiaInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    if (valueShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeMode()
{
    bool existSplitRopeTensor =
        ((opParamInfo_.queryRope.tensor != nullptr) && (opParamInfo_.queryRope.desc != nullptr));
    if (qkHeadDim_ < vHeadDim_) {
        OPS_LOG_E(opName_,
                  "the query's head dim(%u) should be greater than or equal to the value's head dim(%u)",
                  qkHeadDim_,
                  vHeadDim_);
        return ge::GRAPH_FAILED;
    } else if (qkHeadDim_ > vHeadDim_) {
        if (existSplitRopeTensor) {
            OPS_LOG_E(opName_,
                      "when %s exist, the query's head dim(%u) should be equal to the value's head dim(%u). ",
                      QUERY_ROPE_NAME.c_str(),
                      qkHeadDim_,
                      vHeadDim_);
            return ge::GRAPH_FAILED;
        } else {
            ropeMode_ = RopeMode::ROPE_COMBINE;
        }
    } else {
        if (existSplitRopeTensor) {
            ropeMode_ = RopeMode::ROPE_SPLIT;
        } else {
            ropeMode_ = RopeMode::NO_ROPE;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeHeadDim()
{
    if (ge::GRAPH_SUCCESS != GetRopeMode()) {
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::NO_ROPE) {
        ropeHeadDim_ = 0U;
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        ropeHeadDim_ = qkHeadDim_ - vHeadDim_;
    } else {
        queryRopeShape_ = std::make_shared<FiaTilingShape>(
            opParamInfo_.queryRope.tensor->GetStorageShape(), qLayout_, QUERY_ROPE_NAME, opName_, n1Size_);
        if (queryRopeShape_->CheckHasD(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        ropeHeadDim_ = static_cast<uint32_t>(queryRopeShape_->GetD());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKeySinkNumber()
{
    if (opParamInfo_.keySink.tensor != nullptr) {
        keySinkNumber_ = opParamInfo_.keySink.tensor->GetStorageShape().GetDim(0);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKVAndKRopeStride()
{
    auto strideKeyPtr = context_->GetInputStride(KEY_INDEX);
    auto strideValuePtr = context_->GetInputStride(VALUE_INDEX);
    auto strideKeyRopePtr = context_->GetOptionalInputStride(KEY_ROPE_INDEX);

    // 判断输入tensor是否连续
    bool isKeyContinguous = strideKeyPtr == nullptr ? true : strideKeyPtr->GetDimNum() == 0;
    bool isValueContinguous = strideValuePtr == nullptr ? true : strideValuePtr->GetDimNum() == 0;
    bool isKeyRopeContinguous = strideKeyRopePtr == nullptr ? true : strideKeyRopePtr->GetDimNum() == 0;

    const uint32_t MAX_STRIDE_NUM = 5;
    for (int32_t i = 0; i < MAX_STRIDE_NUM; ++i) {
        keyRopeStride_.push_back(0);
        keyStride_.push_back(0);
        valueStride_.push_back(0);
    }

    // 校验 stride 维度数不能超过 5
    OPS_CHECK(strideKeyPtr != nullptr && strideKeyPtr->GetDimNum() > MAX_STRIDE_NUM,
              OPS_LOG_E(opName_, "keyStride dimension %u exceeds max %u", strideKeyPtr->GetDimNum(), MAX_STRIDE_NUM),
              return ge::GRAPH_FAILED);
    OPS_CHECK(
        strideValuePtr != nullptr && strideKeyPtr->GetDimNum() > MAX_STRIDE_NUM,
        OPS_LOG_E(opName_, "valueStride dimension %u exceeds max %u", strideValuePtr->GetDimNum(), MAX_STRIDE_NUM),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        strideKeyRopePtr != nullptr && strideKeyRopePtr->GetDimNum() > MAX_STRIDE_NUM,
        OPS_LOG_E(opName_, "keyRopeStride dimension %u exceeds max %u", strideKeyRopePtr->GetDimNum(), MAX_STRIDE_NUM),
        return ge::GRAPH_FAILED);

    OPS_CHECK(isKeyContinguous != isValueContinguous,
              OPS_LOG_E(opName_, "Key and value should be either both contiguous or both non-contiguous."),
              return ge::GRAPH_FAILED);

    // 非PA 场景不支持传入非连续tensor
    if ((kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) &&
        !(isKeyContinguous && isValueContinguous && isKeyRopeContinguous)) {
        OPS_LOG_E(opName_, "Only supports non-contiguous tensor in Page Attention mode.");
        return ge::GRAPH_FAILED;
    }

    // 获取偏移值
    if (!isKeyContinguous) {
        for (size_t i = 0; i < strideKeyPtr->GetDimNum(); ++i) {
            keyStride_[i] = strideKeyPtr->GetStride(i);
            valueStride_[i] = strideValuePtr->GetStride(i);
        }
    }

    if (!isKeyRopeContinguous) {
        for (size_t i = 0; i < strideKeyRopePtr->GetDimNum(); ++i) {
            keyRopeStride_[i] = strideKeyRopePtr->GetStride(i);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // inputLayout: {qLayout, outLayout}
    const map<string, pair<FiaLayout, FiaLayout>> layoutMap = {{"BSH", {FiaLayout::BSH, FiaLayout::BSH}},
                                                               {"BSND", {FiaLayout::BSND, FiaLayout::BSND}},
                                                               {"BNSD", {FiaLayout::BNSD, FiaLayout::BNSD}},
                                                               {"TND", {FiaLayout::TND, FiaLayout::TND}},
                                                               {"BSH_NBSD", {FiaLayout::BSH, FiaLayout::NBSD}},
                                                               {"BSND_NBSD", {FiaLayout::BSND, FiaLayout::NBSD}},
                                                               {"BNSD_NBSD", {FiaLayout::BNSD, FiaLayout::NBSD}},
                                                               {"TND_NTD", {FiaLayout::TND, FiaLayout::NTD}},
                                                               {"NTD_TND", {FiaLayout::NTD, FiaLayout::TND}},
                                                               {"BNSD_BSND", {FiaLayout::BNSD, FiaLayout::BSND}},
                                                               {"BSND_BNSD", {FiaLayout::BSND, FiaLayout::BNSD}},
                                                               {"BSH_BNSD", {FiaLayout::BSH, FiaLayout::BNSD}},
                                                               {"NTD", {FiaLayout::NTD, FiaLayout::NTD}}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OPS_LOG_E(opName_, "input layout is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN1Size()
{
    // 获取N1基准值
    int32_t numHeads = *(opParamInfo_.numHeads);
    if (numHeads <= 0) {
        OPS_LOG_E(opName_, "%s is %d, it should be greater than 0.", QUERY_HEADS_NUM_NAME.c_str(), numHeads);
        return ge::GRAPH_FAILED;
    }
    n1Size_ = static_cast<uint32_t>(numHeads);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN2Size()
{
    // 获取N2基准值
    int32_t kvHeadNums = *(opParamInfo_.kvHeadNums);
    if (kvHeadNums < 0) {
        OPS_LOG_E(opName_, "%s is %d, it should be greater than 0.", KV_HEADS_NUM_NAME.c_str(), kvHeadNums);
        return ge::GRAPH_FAILED;
    }
    n2Size_ = (kvHeadNums == 0) ? n1Size_ : static_cast<uint32_t>(kvHeadNums);
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::SetFiaShape()
{
    queryShape_ = std::make_shared<FiaTilingShape>(
        opParamInfo_.query.shape->GetStorageShape(), qLayout_, QUERY_NAME, opName_, n1Size_);
    keyShape_ = std::make_shared<FiaTilingShape>(kCache_[0]->GetShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
    valueShape_ = std::make_shared<FiaTilingShape>(vCache_[0]->GetShape(), kvLayout_, VALUE_NAME, opName_, n2Size_);
}

ge::graphStatus FiaInfoParser::GetGSize()
{
    // 获取G基准值
    if (n1Size_ % n2Size_ != 0U) {
        OPS_LOG_E(opName_,
                  "%s(%u) should be a multiple of %s(%u).",
                  QUERY_HEADS_NUM_NAME.c_str(),
                  n1Size_,
                  KV_HEADS_NUM_NAME.c_str(),
                  n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetAttenMaskInfo()
{
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    attenMaskFlag_ = (maskTensor != nullptr) && (maskTensor->GetStorageShape().GetShapeSize() != 0);
    // only bss & b1ss & bs need to calc attenMaskSize_ , attenMaskSize_ is used to calc batch offset
    if (attenMaskFlag_) {
        uint32_t maskDimNum = maskTensor->GetStorageShape().GetDimNum();
        if (maskDimNum == 2U) {
            if (s1Size_ == 1U) { // qs=1 仅支持BS
                attenMaskSize_ = maskTensor->GetStorageShape().GetDim(1);
            } else { // qs > 1 仅支持SS
                attenMaskSize_ = 0;
            }
        } else if (maskDimNum == 3U || maskDimNum == 4U) {
            if (maskTensor->GetStorageShape().GetDim(0) == bSize_) { // BSS B1SS BatchStride = S1*S2
                /*
                 * maskDimNum-1表示BSS/B1SS的S2维度，GetDim(maskDimNum-1)表示获取S2维度大小
                 * maskDimNum-2表示BSS/B1SS的S1维度，GetDim(maskDimNum-1)表示获取S1维度大小
                */
                attenMaskSize_ = maskTensor->GetStorageShape().GetDim(maskDimNum - 1)
                                * maskTensor->GetStorageShape().GetDim(maskDimNum - 2);
            } else { // 1SS 11SS
                attenMaskSize_ = 0;
            }
        } else {
            OPS_LOG_E(opName_, "mask matrix dim only support 2/3/4.");
        }
        if (*opParamInfo_.sparseMode == 0U || *opParamInfo_.sparseMode == 1U) {
            attenMaskStride_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        } else {
            attenMaskStride_ = 2048U; // compress mask
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetPaddingSizeFlag()
{
    qPaddingSizeFlag_ = (opParamInfo_.queryPaddingSize.tensor != nullptr);
    kvPaddingSizeFlag_ = (opParamInfo_.kvPaddingSize.tensor != nullptr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqInfo()
{
    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FiaInfoParser::MapStringToLayout(FiaLayout &layoutString) const
{
    const std::map<FiaLayout, TilingKeyLayout> layoutMap = {
        {FiaLayout::BSH, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BNSD, TilingKeyLayout::BNSD},
        {FiaLayout::NZ, TilingKeyLayout::NZ},
        {FiaLayout::TND, TilingKeyLayout::TND},
        {FiaLayout::NBSD, TilingKeyLayout::NBSD},
        {FiaLayout::NTD, TilingKeyLayout::NTD},
        {FiaLayout::BnBsH, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BnNBsD, TilingKeyLayout::BNSD},
    };

    auto it = layoutMap.find(layoutString);
    if (it != layoutMap.end()) {
        return it->second;
    }
    return TilingKeyLayout::BSH_BSND;
}

void FiaInfoParser::GenerateFeatureInfo(FiaTilingInfo &fiaInfo)
{
    // empty tensor
    fiaInfo.emptyTensorFlag = emptyTensorFlag_;

    // pa
    fiaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    fiaInfo.blockSize = blockSize_;
    fiaInfo.blockTypeSize = sizeof(float);

    // inner precise
    fiaInfo.innerPrecise = *opParamInfo_.innerPrecise;

    // atten mask
    fiaInfo.attenMaskFlag = attenMaskFlag_;
    fiaInfo.attenMaskSize = attenMaskSize_;
    fiaInfo.attenMaskStride = attenMaskStride_;
    fiaInfo.sparseMode = *opParamInfo_.sparseMode;
    fiaInfo.sinkNumber = *opParamInfo_.sinkNumber;
    // 4: only mla noquant & band mode support slidingFlag
    fiaInfo.slidingFlag =
        (*opParamInfo_.sparseMode == 4) && (ropeMode_ == RopeMode::ROPE_SPLIT) && (qkHeadDim_ == 512U);
    fiaInfo.qPaddingSizeFlag = qPaddingSizeFlag_;
    fiaInfo.kvPaddingSizeFlag = kvPaddingSizeFlag_;
    fiaInfo.pseShiftFlag = pseShiftFlag_;
    fiaInfo.softmaxLseFlag = *opParamInfo_.softmaxLseFlag;
    fiaInfo.batchInvariant = *opParamInfo_.batchInvariant;
    fiaInfo.softmaxMaxSumFlag = *opParamInfo_.softmaxMaxSumFlag;
    fiaInfo.isMaxWorkspace = isMaxWorkspace_;
    fiaInfo.isLegacyIfa = isLegacyIfa_;
    fiaInfo.preToken = preToken_;
    fiaInfo.nextToken = nextToken_;
    fiaInfo.learnableSinkFlag = (opParamInfo_.learnableSink.tensor != nullptr);
}

void FiaInfoParser::GenerateLayoutInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.qLayout = qLayout_;
    fiaInfo.kvLayout = kvLayout_;
    fiaInfo.outLayout = outLayout_;
    fiaInfo.inputKvLayout = MapStringToLayout(kvLayout_);
    fiaInfo.inputLayout = MapStringToLayout(qLayout_);
    fiaInfo.outputLayout = MapStringToLayout(outLayout_);
}

void FiaInfoParser::GenerateInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.opName = opName_;
    fiaInfo.platformInfo = platformInfo_;
    fiaInfo.opParamInfo = opParamInfo_;
    fiaInfo.socVersion = socVersion_;
    GenerateAxisInfo(fiaInfo);
    GenerateDtypeInfo(fiaInfo);
    fiaInfo.kvStorageMode = kvStorageMode_;
    fiaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    fiaInfo.ropeMode = ropeMode_;
    fiaInfo.l2CacheSize = l2CacheSize_;

    fiaInfo.kCache = kCache_;
    fiaInfo.vCache = vCache_;

    fiaInfo.keyStride = keyStride_;
    fiaInfo.valueStride = valueStride_;
    fiaInfo.keyRopeStride = keyRopeStride_;

    fiaInfo.l2CacheOffFlag = false;
    fiaInfo.totalBlockNum = kCache_[0]->GetShape().GetDim(0);
    fiaInfo.scaleValue = *opParamInfo_.scaleValue;
    fiaInfo.needInit = needInit_;
    fiaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    fiaInfo.actualLenQDims = actualLenQDims_;
    fiaInfo.actualLenDims = actualLenDims_;
    fiaInfo.maxActualseq = maxActualseq_;
    fiaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    fiaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    fiaInfo.isSameActualseq = isSameActualseq_;
    fiaInfo.kvListSeqLens = kvListSeqLens_;

    fiaInfo.isAccumQSeq = isAccumQSeq_;
    fiaInfo.isAccumKVSeq = isAccumKVSeq_;

    GenerateFeatureInfo(fiaInfo);
    GenerateLayoutInfo(fiaInfo);
}

void FiaInfoParser::GenerateAxisInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.bSize = bSize_;
    fiaInfo.n1Size = n1Size_;
    fiaInfo.n2Size = n2Size_;
    fiaInfo.keySinkNumber = keySinkNumber_;
    fiaInfo.s1Size = s1Size_;
    if (fiaInfo.keySinkNumber) {
        fiaInfo.s2Size = s2Size_ + fiaInfo.keySinkNumber;
    } else {
        fiaInfo.s2Size = s2Size_;
    }
    fiaInfo.gSize = gSize_;
    fiaInfo.qkHeadDim = qkHeadDim_;
    fiaInfo.vHeadDim = vHeadDim_;
    fiaInfo.ropeHeadDim = ropeHeadDim_;
    fiaInfo.qTSize = qTSize_;
    fiaInfo.kTSize = kTSize_;
}

void FiaInfoParser::GenerateDtypeInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.inputQType = inputQType_;
    fiaInfo.inputKvType = inputKvType_;
    fiaInfo.inputQRopeType = inputQRopeType_;
    fiaInfo.inputKRopeType = inputKRopeType_;
    fiaInfo.outputType = outputType_;
}

ge::graphStatus FiaInfoParser::Parse(FiaTilingInfo &fiaInfo)
{
    if (context_ == nullptr) {
        OPS_LOG_E("FusedInferAttentionScoreV2Sink", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvCache() || ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() || ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode() || ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetEmptyTensorFlag()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(fiaInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseAxisInfo()
{
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }
    SetFiaShape();
    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKTSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim() || ge::GRAPH_SUCCESS != GetKeySinkNumber() ||
        ge::GRAPH_SUCCESS != GetKVAndKRopeStride()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseFeatureInfo()
{
    if (ge::GRAPH_SUCCESS != GetLegacyIfaFlag() || ge::GRAPH_SUCCESS != GetSparseMode() ||
        ge::GRAPH_SUCCESS != GetPreNextToken() || ge::GRAPH_SUCCESS != GetAttenMaskInfo() ||
        ge::GRAPH_SUCCESS != GetMaxWorkspaceFlag() || ge::GRAPH_SUCCESS != GetPaddingSizeFlag() ||
        ge::GRAPH_SUCCESS != GetActualSeqInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
