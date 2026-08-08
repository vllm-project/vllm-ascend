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
 * \file fused_infer_attention_score_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "fused_infer_attention_score_tiling_index.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "fused_infer_attention_score_tiling_utils.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "shape of query cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "desc of query cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "shape of key cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "desc of key cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "value", "shape of value cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "value", "desc of value cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "output", "shape of output cannot be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "output", "desc of output cannot be empty"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::CheckRequiredAttrExistence() const
{
    if (opParamInfo_.numHeads == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "num_heads");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.scaleValue == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "scale");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.kvHeadNums == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "num_key_value_heads");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.layOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "input_layout");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "block_size");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.antiquantMode == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "antiquant_mode");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.softmaxLseFlag == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "softmax_lse_flag");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.keyAntiquantMode == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "key_antiquant_mode");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.valueAntiquantMode == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "value_antiquant_mode");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.innerPrecise == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "inner_precise");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.queryQuantMode == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "query_quant_mode");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.pseType == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "pse_type");
        return ge::GRAPH_FAILED;
    }

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
        std::string sizesStr = std::to_string(opParamInfo_.query.shape->GetStorageShape().GetShapeSize()) + " and " +
            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize());
        std::string reason = "The shape sizes of query and attention_out must "
            "be both greater than 0 or both equal to 0";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "query and attention_out",
            sizesStr.c_str(), reason.c_str());
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
            std::string reason = "The shape size of softmax_lse should be greater than 0 when lse Flag is 1";
            OP_LOGE_FOR_INVALID_SHAPESIZE(opName_, "softmax_lse", std::to_string(0).c_str(),
                reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &kTensor : kCache_) {
        if (kTensor->GetStorageShape().GetShapeSize() != 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    for (auto &vTensor : vCache_) {
        if (vTensor->GetStorageShape().GetShapeSize() != 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    emptyTensorFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxWorkspaceFlag()
{
    if ((opParamInfo_.actualSeqLengths.tensor != nullptr &&
         opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr) ||
        (opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
         opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr)) {
        isMaxWorkspace_ = true;
        OP_LOGI(opName_, "FIA tiling sink");
    } else {
        isMaxWorkspace_ = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetLegacyIfaFlag()
{
    uint32_t querySize = 0;
    std::string layout(opParamInfo_.layOut);
    if (layout == "BSH" || layout == "BSND" || layout == "BNSD") {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        querySize = static_cast<uint32_t>(queryShape_->GetShapeS());
        if (querySize == 1U && qkHeadDim_ == vHeadDim_ && opParamInfo_.queryRope.tensor == nullptr &&
            opParamInfo_.keyRope.tensor == nullptr) {
            isLegacyIfa_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actualSeqLengths",
            "When inputLayout is " + LayoutToSerialString(qLayout_) + ", actualSeqLengths cannot be empty");
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName_, "actual_seq_lengths",
            std::to_string(shapeSize).c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetStrides()
{
    // 用opbase版本判断，老版本opbase为tensorv1
    isTensorV1_ = context_->InputIsView(KEY_INDEX) == false;
    keyStrides_ = context_->GetDynamicInputStride(KEY_INDEX, 0);
    valueStrides_ = context_->GetDynamicInputStride(VALUE_INDEX, 0);
    if (opParamInfo_.keyAntiquantScale.tensor != nullptr || opParamInfo_.valueAntiquantScale.tensor != nullptr) {
        kScaleStrides_ = context_->GetInputStride(KEY_ANTIQUANT_SCALE_INDEX);
        vScaleStrides_ = context_->GetInputStride(VALUE_ANTIQUANT_SCALE_INDEX);
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        kRopeStrides_ = context_->GetInputStride(KEY_ROPE_INDEX);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
                return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND310P) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) && (npuArch_ != NpuArch::DAV_3510)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d]/NpuArch[%d] is not support.",
                                    static_cast<int32_t>(socVersion_), static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetOptionalInputParaInfo()
{
    // actualSeqLengthsQ和queryPaddingSize在GetUpdateInfo()中获取
    GetOptionalInputParaMaskInfo();
    GetOptionalInputParaActualSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaPostQuantInfo();
    GetOptionalInputParaRopeInfo();
    GetOptionalInputParaPseInfo();
    GetOptionalInputParaLeftPaddingInfo();
    GetOptionalInputParaPrefixInfo();
    GetOptionalInputParaLearnableSinkInfo();
    GetOptionalInputParaQuantInfo();
}
void FiaInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attenMask.tensor = context_->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    opParamInfo_.attenMask.desc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX);
}

void FiaInfoParser::GetOptionalInputParaActualSeqLengthInfo()
{
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_KV_INDEX);
}

void FiaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
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

void FiaInfoParser::GetOptionalInputParaPseInfo()
{
    opParamInfo_.pseShift.tensor = context_->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    opParamInfo_.pseShift.desc = context_->GetOptionalInputDesc(PSE_SHIFT_INDEX);
    opParamInfo_.qStartIdx.tensor = context_->GetOptionalInputTensor(Q_START_IDX_INDEX);
    opParamInfo_.qStartIdx.desc = context_->GetOptionalInputDesc(Q_START_IDX_INDEX);
    opParamInfo_.kvStartIdx.tensor = context_->GetOptionalInputTensor(KV_START_IDX_INDEX);
    opParamInfo_.kvStartIdx.desc = context_->GetOptionalInputDesc(KV_START_IDX_INDEX);
}

void FiaInfoParser::GetOptionalInputParaLeftPaddingInfo()
{
    opParamInfo_.kvPaddingSize.tensor = context_->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    opParamInfo_.kvPaddingSize.desc = context_->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX);
}

void FiaInfoParser::GetOptionalInputParaPrefixInfo()
{
    opParamInfo_.keySharedPrefix.tensor = context_->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    opParamInfo_.keySharedPrefix.desc = context_->GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    opParamInfo_.valueSharedPrefix.tensor = context_->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    opParamInfo_.valueSharedPrefix.desc = context_->GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    opParamInfo_.actualSharedPrefixLen.tensor = context_->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    opParamInfo_.actualSharedPrefixLen.desc = context_->GetOptionalInputDesc(ACTUAL_SHARED_PREFIX_LEN_INDEX);
}

void FiaInfoParser::GetOptionalInputParaLearnableSinkInfo()
{
    opParamInfo_.learnableSink.tensor = context_->GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    opParamInfo_.learnableSink.desc = context_->GetOptionalInputDesc(LEARNABLE_SINK_INDEX);
}

void FiaInfoParser::GetOptionalInputParaQuantInfo()
{
    opParamInfo_.deqScale1.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    opParamInfo_.deqScale1.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.tensor = context_->GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    opParamInfo_.quantScale1.desc = context_->GetOptionalInputDesc(QUANT_SCALE1_INDEX);
    opParamInfo_.deqScale2.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    opParamInfo_.deqScale2.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE2_INDEX);
    opParamInfo_.dequantScaleQuery.tensor = context_->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    opParamInfo_.dequantScaleQuery.desc = context_->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
    opParamInfo_.antiquantScale.tensor = context_->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantScale.desc = context_->GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    opParamInfo_.antiquantOffset.tensor = context_->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.antiquantOffset.desc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.keyAntiquantScale.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantScale.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.keyAntiquantOffset.tensor = context_->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.keyAntiquantOffset.desc = context_->GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantScale.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantScale.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    opParamInfo_.valueAntiquantOffset.tensor = context_->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    opParamInfo_.valueAntiquantOffset.desc = context_->GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
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
}

ge::graphStatus FiaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    // sparseMode,preToken,nextToken在GetUpdateInfo()中获取
    opParamInfo_.numHeads = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    opParamInfo_.kvHeadNums = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    opParamInfo_.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    opParamInfo_.antiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_ANTIQUANT_MODE_INDEX);
    opParamInfo_.softmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    opParamInfo_.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
    opParamInfo_.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);
    opParamInfo_.innerPrecise = attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX);
    opParamInfo_.queryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    opParamInfo_.pseType = attrs->GetAttrPointer<int64_t>(ATTR_PSE_TYPE_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetUpdateInfo()
{
    auto attrs = context_->GetAttrs();
    static int32_t SPARSE_ZERO = 0U;
    static int64_t TOKEN_MAX = 2147483647;
    if (isLegacyIfa_ && (socVersion_ == platform_ascendc::SocVersion::ASCEND310P ||
                         socVersion_ == platform_ascendc::SocVersion::ASCEND910B)) {
        opParamInfo_.sparseMode = &SPARSE_ZERO;
        opParamInfo_.preToken = &TOKEN_MAX;
        opParamInfo_.nextToken = &TOKEN_MAX;
        opParamInfo_.actualSeqLengthsQ.tensor = nullptr;
        opParamInfo_.actualSeqLengthsQ.desc = nullptr;
        opParamInfo_.queryPaddingSize.tensor = nullptr;
        opParamInfo_.queryPaddingSize.desc = nullptr;
    } else {
        opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
        opParamInfo_.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
        opParamInfo_.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
        opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
        opParamInfo_.queryPaddingSize.tensor = context_->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
        opParamInfo_.queryPaddingSize.desc = context_->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX);
    }
}

void FiaInfoParser::UpdatePreNextTokenBySparseMode()
{
    if (sparseMode_ == SPARSE_MODE_NO_MASK) {
        if (attenMaskFlag_ && (qPaddingSizeFlag_ || kvPaddingSizeFlag_)) {
            preToken_ = SPARSE_MODE_INT_MAX;
            nextToken_ = SPARSE_MODE_INT_MAX;
        }
    } else if (sparseMode_ == SPARSE_MODE_ALL_MASK) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == SPARSE_MODE_LEFT_UP || sparseMode_ == SPARSE_MODE_RIGHT_DOWN ||
               sparseMode_ == SPARSE_MODE_TREE) {
        nextToken_ = 0;
        preToken_ = SPARSE_MODE_INT_MAX;
    }
}

void FiaInfoParser::ClampPreNextToken()
{
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
}

void FiaInfoParser::GetPreNextToken()
{
    preToken_ = opParamInfo_.preToken == nullptr ? 0 : *opParamInfo_.preToken;
    nextToken_ = opParamInfo_.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;

    sparseMode_ = opParamInfo_.sparseMode == nullptr ? 0 : *opParamInfo_.sparseMode;
    UpdatePreNextTokenBySparseMode();
    ClampPreNextToken();

    if (opParamInfo_.sparseMode != nullptr && sparseMode_ == SPARSE_MODE_NO_MASK) {
        if (!attenMaskFlag_) {
            preToken_ = SPARSE_MODE_INT_MAX;
            nextToken_ = SPARSE_MODE_INT_MAX;
        }
    }

    if (sparseMode_ == SPARSE_MODE_NO_MASK && qPaddingSizeFlag_) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    }

    if ((quantMode_ == FiaQuantMode::ANTI_QUANT) && (s1Size_ == 1)) {
        preToken_ = SPARSE_MODE_INT_MAX;
        nextToken_ = SPARSE_MODE_INT_MAX;
    }
}

ge::graphStatus FiaInfoParser::GetTensorListCache(uint32_t index, const std::string &name,
                                                  std::vector<gert::StorageShape *> &cache)
{
    cache.clear();
    uint32_t bIdx = 0;
    while ((context_->GetDynamicInputShape(index, bIdx)) != nullptr) {
        cache.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(index, bIdx)));
        bIdx++;
    }
    if (bIdx == 0) {
        OP_LOGE_FOR_INVALID_TENSORNUM(opName_, name.c_str(), bIdx, ">0");
        return ge::GRAPH_FAILED;
    }
    cache.resize(bIdx);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvCache()
{
    if (GetTensorListCache(KEY_INDEX, KEY_NAME, kCache_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetTensorListCache(VALUE_INDEX, VALUE_NAME, vCache_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (kCache_.size() != vCache_.size()) {
        std::string numsStr = std::to_string(kCache_.size()) + " and " + std::to_string(vCache_.size());
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(context_->GetNodeName(), "key and value",
            numsStr.c_str(), "The tensor nums in key and value must be the same");
        return ge::GRAPH_FAILED;
    }

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

void FiaInfoParser::GetInOutDataType()
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
}

ge::graphStatus FiaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        if (isMaxWorkspace_) {
            bSize_ = 1;
            return ge::GRAPH_SUCCESS;
        }
        return GetActualSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void FiaInfoParser::GetQueryTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void FiaInfoParser::GetKeyTSize()
{
    keyTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus FiaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND/NTD时, 以query的S维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
        if (isMaxWorkspace_) {
            if (qLayout_ == FiaLayout::TND) {
                s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
            } else {
                s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
            }
        }
        const int64_t *actualSeqQ = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        if (actualSeqQ == nullptr) {
            if (queryShape_->CheckHasShapeT(__func__) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeT());
            qSize_.push_back(s1Size_);
            return ge::GRAPH_SUCCESS;
        }

        uint32_t b = 0;
        if (GetActualSeqLenQSize(b) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int64_t qActualSeqMax = 0;
        for (uint32_t i = 0; i < b; i++) {
            int64_t tmpS1 = (i == 0U) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
            qSize_.push_back(tmpS1);
            if (tmpS1 > qActualSeqMax) {
                qActualSeqMax = tmpS1;
            }
        }
        s1Size_ = static_cast<uint32_t>(qActualSeqMax);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
        qSize_.push_back(s1Size_);
    }
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetKvStorageMode()
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
}

void FiaInfoParser::GetQuantMode()
{
    if (inputQType_ != inputKvType_) {
        quantMode_ = FiaQuantMode::ANTI_QUANT;
    } else if (inputQType_ == ge::DT_BF16 || inputQType_ == ge::DT_FLOAT16) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else {
        quantMode_ = FiaQuantMode::FULL_QUANT;
    }
}

ge::graphStatus FiaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        kvLayout_ = qLayout_;
    } else {
        OP_CHECK_IF(kCache_.empty() || kCache_[0] == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "kCache", "kCache cannot be empty"),
            return ge::GRAPH_FAILED);
        uint32_t keyDimNum = kCache_[0]->GetStorageShape().GetDimNum();
        if (keyDimNum == 3U) {
            kvLayout_ = FiaLayout::BnBsH;
        } else if (keyDimNum == 4U) {
            kvLayout_ = FiaLayout::BnNBsD;
        } else if (keyDimNum == 5U) {
            kvLayout_ = FiaLayout::NZ;
        } else {
            std::string reason = "The shape dim of key in the 1st tensor must be within the range "
                "{3, 4, 5} when page attention is enabled";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "key in the 1st tensor",
                std::to_string(keyDimNum).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FiaLayout layout,
                                               int64_t &maxActualSeqLen)
{
    maxActualSeqLen = 0;
    const int64_t *actualLenData = actualSeqLensTensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLensTensor->GetShapeSize()), bSize_);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (layout == FiaLayout::TND || layout == FiaLayout::NTD) {
                tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            } else {
                tmpS = actualLenData[i];
            }
            kvSize_.push_back(tmpS);
            maxActualSeqLen = std::max(maxActualSeqLen, tmpS);
        }
    }
    if (isMaxWorkspace_) {
        maxActualSeqLen = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeFromActualSeqLens()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize_.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }
    return GetMaxActualSeq(opParamInfo_.actualSeqLengths.tensor, kvLayout_, s2Size_);
}

ge::graphStatus FiaInfoParser::GetS2SizeForBatchContinuous()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
        kvListSeqLens_.push_back(s2Size_);
        kvSize_.push_back(s2Size_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForTensorList()
{
    if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
        return GetS2SizeFromActualSeqLens();
    } else { // BSH/BSND/BNSD
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        s2Size_ = 0;
        for (uint32_t i = 0; i < kCache_.size(); i++) {
            auto keyShape =
                std::make_shared<FiaTilingShape>(kCache_[i]->GetStorageShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
            if (keyShape->GetShapeS() > s2Size_) {
                s2Size_ = keyShape->GetShapeS();
            }
            if (keyShape->GetShapeS() != keyShape_->GetShapeS()) {
                isSameSeqAllKVTensor_ = false;
            }
            kvListSeqLens_.push_back(keyShape->GetShapeS());
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetMaxBlockNumPerBatch()
{
    if (antiQuantFlag_) {
        uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
        if (dimNum != 2U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "block_table", std::to_string(dimNum).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    } else {
        const gert::Tensor *actSeqLenKV = opParamInfo_.actualSeqLengths.tensor;
        int32_t actualSeqKVPerBatch = 0;
        int32_t blockNumPerBatch = 0;
        int64_t blockNumValid = 0;
        maxBlockNumPerBatch_ = 0;
        if (actSeqLenKV == nullptr || actSeqLenKV->GetData<int64_t>() == nullptr || blockSize_ == 0) {
            return ge::GRAPH_SUCCESS;
        }
        uint32_t loops = std::min(bSize_, static_cast<uint32_t>(actSeqLenKV->GetShapeSize()));
        for (uint32_t i = 0; i < loops; i++) {
            actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ?
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
            blockNumPerBatch = (actualSeqKVPerBatch + blockSize_ - 1) / blockSize_;
            blockNumValid += blockNumPerBatch;
            if (blockNumPerBatch > maxBlockNumPerBatch_) {
                maxBlockNumPerBatch_ = blockNumPerBatch;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBlockSize()
{
    if (opParamInfo_.blockSize == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "block_size", "block_size cannot be empty");
        return ge::GRAPH_FAILED;
    }
    blockSize_ = *(opParamInfo_.blockSize);
    if (blockSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "block_size", std::to_string(blockSize_).c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBlockTableDim2()
{
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != 2U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "block_table", std::to_string(dimNum).c_str(), "2");
        return ge::GRAPH_FAILED;
    }
    blockTableDim2_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS2SizeForPageAttention()
{
    if (GetBlockSize() != ge::GRAPH_SUCCESS || GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS ||
        GetBlockTableDim2() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (isMaxWorkspace_) {
        s2Size_ = static_cast<int64_t>(blockTableDim2_) * blockSize_;
        return ge::GRAPH_SUCCESS;
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
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeMode()
{
    bool existQuerySplitRopeTensor =
        ((opParamInfo_.queryRope.tensor != nullptr) && (opParamInfo_.queryRope.desc != nullptr));
    bool existKeySplitRopeTensor = ((opParamInfo_.keyRope.tensor != nullptr) && (opParamInfo_.keyRope.desc != nullptr));
    bool existSplitRopeTensor = existQuerySplitRopeTensor && existKeySplitRopeTensor;
    if (npuArch_ == NpuArch::DAV_3510) {
        if (existSplitRopeTensor) {
            ropeMode_ = RopeMode::ROPE_SPLIT;
        } else if (qkHeadDim_ == 192 && vHeadDim_ == 128) {
            ropeMode_ = RopeMode::ROPE_COMBINE;
        } else {
            ropeMode_ = RopeMode::NO_ROPE;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (qkHeadDim_ < vHeadDim_) {
        std::string shapeStr = ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
            ToString(opParamInfo_.value.shape->GetStorageShape());
        std::string reason = "D of query must be greater than or equal to axis D of value";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and value", shapeStr.c_str(),
            reason.c_str());
        return ge::GRAPH_FAILED;
    } else if (qkHeadDim_ > vHeadDim_) {
        if (existSplitRopeTensor) {
            std::string shapeStr = ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                ToString(opParamInfo_.value.shape->GetStorageShape());
            std::string reason = "D of query must be equal to the same axis of value when query_rope exists";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and value", shapeStr.c_str(),
                reason.c_str());
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

ge::graphStatus FiaInfoParser::GetMlaMode()
{
    if (ropeMode_ == RopeMode::NO_ROPE) {
        mlaMode_ = MlaMode::NO_MLA;
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        mlaMode_ = MlaMode::ROPE_COMBINE_D128;
    } else {
        if (qkHeadDim_ == 512) {
            mlaMode_ = MlaMode::ROPE_SPLIT_D512;
        } else if (qkHeadDim_ == 128) {
            mlaMode_ = MlaMode::ROPE_SPLIT_D128;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeHeadDim()
{
    if (ge::GRAPH_SUCCESS != GetRopeMode() || ge::GRAPH_SUCCESS != GetMlaMode()) {
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::NO_ROPE) {
        ropeHeadDim_ = 0U;
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        ropeHeadDim_ = qkHeadDim_ - vHeadDim_;
    } else {
        queryRopeShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.queryRope.tensor->GetStorageShape(), qLayout_,
                                                           QUERY_ROPE_NAME, opName_, n1Size_);
        if (queryRopeShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        keyRopeShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.keyRope.tensor->GetStorageShape(), kvLayout_,
                                                         KEY_ROPE_NAME, opName_, n2Size_);
        if (keyRopeShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        uint32_t queryRopeHeadDim = static_cast<uint32_t>(queryRopeShape_->GetShapeD());
        uint32_t keyRopeHeadDim = static_cast<uint32_t>(keyRopeShape_->GetShapeD());
        if (queryRopeHeadDim != keyRopeHeadDim) {
            std::string shapesStr = ToString(opParamInfo_.queryRope.tensor->GetStorageShape()) + " and " +
                ToString(opParamInfo_.keyRope.tensor->GetStorageShape());
            std::string reason = "D of query_rope must be equal to D of key_rope";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query_rope and key_rope",
                shapesStr.c_str(), reason.c_str());
            return GRAPH_FAILED;
        }
        ropeHeadDim_ = queryRopeHeadDim;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // inputLayout: {qLayout, outLayout}
    const map<string, pair<FiaLayout, FiaLayout>> layoutMap = {
        {"BSH", {FiaLayout::BSH, FiaLayout::BSH}},         {"BSND", {FiaLayout::BSND, FiaLayout::BSND}},
        {"BNSD", {FiaLayout::BNSD, FiaLayout::BNSD}},      {"TND", {FiaLayout::TND, FiaLayout::TND}},
        {"BSH_NBSD", {FiaLayout::BSH, FiaLayout::NBSD}},   {"BSND_NBSD", {FiaLayout::BSND, FiaLayout::NBSD}},
        {"BNSD_NBSD", {FiaLayout::BNSD, FiaLayout::NBSD}}, {"TND_NTD", {FiaLayout::TND, FiaLayout::NTD}},
        {"NTD_TND", {FiaLayout::NTD, FiaLayout::TND}},     {"BNSD_BSND", {FiaLayout::BNSD, FiaLayout::BSND}},
        {"BSND_BNSD", {FiaLayout::BSND, FiaLayout::BNSD}}, {"BSH_BNSD", {FiaLayout::BSH, FiaLayout::BNSD}},
        {"NTD", {FiaLayout::NTD, FiaLayout::NTD}}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input_layout", layout.c_str(),
            "Input layout must be within the range {BSH, BSND, BNSD, TND, BSH_NBSD, BSND_NBSD, "
            "BNSD_NBSD, TND_NTD, NTD_TND, BNSD_BSND, BSND_BNSD, BSH_BNSD, NTD}");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN1Size()
{
    // 获取N1基准值
    int32_t numHeads = *(opParamInfo_.numHeads);
    if (numHeads <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_heads", std::to_string(numHeads).c_str(),
            "The value of num_heads must be greater than 0");
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_key_value_heads", std::to_string(kvHeadNums).c_str(),
            "The value of num_key_value_heads must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    n2Size_ = (kvHeadNums == 0) ? n1Size_ : static_cast<uint32_t>(kvHeadNums);
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::SetFiaShape()
{
    queryShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), qLayout_, QUERY_NAME,
                                                   opName_, n1Size_);
    keyShape_ = std::make_shared<FiaTilingShape>(kCache_[0]->GetStorageShape(), kvLayout_, KEY_NAME, opName_, n1Size_);
    valueShape_ =
        std::make_shared<FiaTilingShape>(vCache_[0]->GetStorageShape(), kvLayout_, VALUE_NAME, opName_, n2Size_);
}

ge::graphStatus FiaInfoParser::GetGSize()
{
    // 获取G基准值
    if (n1Size_ % n2Size_ != 0U) {
        std::string reason = "The value of num_heads must be a multiple of num_key_value_heads(" +
            std::to_string(n2Size_) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_heads", std::to_string(n1Size_).c_str(),
            reason.c_str());
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetAttenMaskSparse9Info()
{
    // sparse9时，mask形状：TND时传入∑s1²，其余场景传入[B,S1,S1]
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    uint32_t maskDimNum = maskTensor->GetStorageShape().GetDimNum();

    // TND传入的mask
    if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
        if (maskDimNum == 1U) {
            attenMaskBatchStride_ = 1;
            attenMaskStride_ = 1;
        } else {
            std::string maskDimStr = std::to_string(maskDimNum) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "atten_mask", maskDimStr.c_str(),
                "The shape dim of atten_mask must be 1 when the layout is TND or NTD");
        }
    } else {
        if (maskDimNum == 3U) {
            attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(maskDimNum - 1) *
                                    maskTensor->GetStorageShape().GetDim(maskDimNum - 2);
            attenMaskStride_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        } else {
            std::string maskDimStr3 = std::to_string(maskDimNum) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "atten_mask", maskDimStr3.c_str(),
                "The shape dim of atten_mask must be 3 when the layout is not TND or NTD");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetAntiQuantInfo()
{
    antiQuantFlag_ = inputQType_ != inputKvType_;
    if (opParamInfo_.keyAntiquantScale.tensor == nullptr || opParamInfo_.valueAntiquantScale.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (!antiQuantFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    bool kPerChnVPerTokFlag_ = (inputKvType_ == ge::DT_INT4 || inputKvType_ == ge::DT_INT8) &&
                               (*(opParamInfo_.keyAntiquantMode) == 0 && *(opParamInfo_.valueAntiquantMode) == 1);

    auto tmpAntiquantMode = *(opParamInfo_.keyAntiquantMode);
    auto tmpAntiquant = opParamInfo_.keyAntiquantScale.tensor->GetStorageShape();
    if (opParamInfo_.keyAntiquantOffset.tensor != nullptr) {
        tmpAntiquant = opParamInfo_.keyAntiquantOffset.tensor->GetStorageShape();
    }

    if (kPerChnVPerTokFlag_) {
        tmpAntiquantMode = *(opParamInfo_.valueAntiquantMode);
        tmpAntiquant = opParamInfo_.valueAntiquantScale.tensor->GetStorageShape();
        if (opParamInfo_.valueAntiquantOffset.tensor != nullptr) {
            tmpAntiquant = opParamInfo_.valueAntiquantOffset.tensor->GetStorageShape();
        }
    }

    if (tmpAntiquantMode == 6) {
        if (systemPrefixFlag_) {
            if (tmpAntiquant.GetDimNum() != 5) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "antiquant",
                    std::to_string(tmpAntiquant.GetDimNum()).c_str(),
                    "The shape dim of antiquant must be 5 when per-token-group mode is enabled");
                return ge::GRAPH_FAILED;
            }
            antiquantParaSeqSize_ = tmpAntiquant.GetDim(3);
        }
    } else if (tmpAntiquantMode == 1) {
        if (tmpAntiquant.GetDimNum() != 2 && tmpAntiquant.GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "antiquant",
                std::to_string(tmpAntiquant.GetDimNum()).c_str(),
                "The shape dim of antiquant must be 2 or 3 when per-token mode is enabled");
            return ge::GRAPH_FAILED;
        }
        antiquantParaSeqSize_ = tmpAntiquant.GetDimNum() == 3U ? tmpAntiquant.GetDim(2) : tmpAntiquant.GetDim(1);
    } else if (tmpAntiquantMode == 3) {
        if (tmpAntiquant.GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "antiquant",
                std::to_string(tmpAntiquant.GetDimNum()).c_str(),
                "The shape dim of antiquant must be 3 when per-token-head mode is enabled");
            return ge::GRAPH_FAILED;
        }
        antiquantParaSeqSize_ = tmpAntiquant.GetDim(2);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetAttenMaskInfo()
{
    // only bss & b1ss & bs need to calc attenMaskSize_ , attenMaskSize_ is uesed to calc batch offset
    if (attenMaskFlag_) {
        if (*opParamInfo_.sparseMode == 9U && npuArch_ == NpuArch::DAV_3510) {
            OP_LOGE(opName_, "NpuArch[%d] currently does not support sparse9.", static_cast<int32_t>(npuArch_));
            return ge::GRAPH_FAILED;
        }
        if (*opParamInfo_.sparseMode == 9U) {
            return GetAttenMaskSparse9Info();
        }
        auto *maskTensor = opParamInfo_.attenMask.tensor;
        uint32_t maskDimNum = maskTensor->GetStorageShape().GetDimNum();
        if (maskDimNum == 2U) {
            if (s1Size_ == 1U) { // qs=1 仅支持BS
                attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(1);
            } else { // qs > 1 仅支持SS
                attenMaskBatchStride_ = 0;
            }
        } else if (maskDimNum == 3U || maskDimNum == 4U) {
            if (maskTensor->GetStorageShape().GetDim(0) == bSize_) { // BSS B1SS BatchStride = S1*S2
                attenMaskBatchStride_ = maskTensor->GetStorageShape().GetDim(maskDimNum - 1) *
                                        maskTensor->GetStorageShape().GetDim(maskDimNum - 2);
            } else { // 1SS 11SS
                attenMaskBatchStride_ = 0;
            }
        } else {
            std::string reason = "The shape dim of atten_mask must be within the range 2/3/4";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "atten_mask",
                std::to_string(maskDimNum).c_str(), reason.c_str());
        }
        if (*opParamInfo_.sparseMode == 0U || *opParamInfo_.sparseMode == 1U) {
            attenMaskStride_ = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        } else {
            attenMaskStride_ = 2048U; // compress mask
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetPaddingSizeFlag()
{
    qPaddingSizeFlag_ =
        ((!isLegacyIfa_ || npuArch_ == NpuArch::DAV_3510) && (opParamInfo_.queryPaddingSize.tensor != nullptr));

    if (isLegacyIfa_) {
        if ((opParamInfo_.kvPaddingSize.tensor != nullptr &&
             opParamInfo_.kvPaddingSize.tensor->GetStorageShape().GetShapeSize() > 0) &&
            kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS && opParamInfo_.actualSeqLengths.tensor != nullptr) {
            kvPaddingSizeFlag_ = true;
        }
    } else {
        if (opParamInfo_.kvPaddingSize.tensor != nullptr) {
            kvPaddingSizeFlag_ = true;
        }
    }

    if (antiQuantFlag_) {
        if (qPaddingSizeFlag_ || (kvPaddingSizeFlag_ && s1Size_ == 1)) {
            needInit_ = true;
        }
    } else {
        if (qPaddingSizeFlag_ || kvPaddingSizeFlag_) {
            needInit_ = true;
        }
    }
    if ((quantMode_ == FiaQuantMode::ANTI_QUANT) && (s1Size_ == 1)) {
        qPaddingSizeFlag_ = false;
    }
}

void FiaInfoParser::GetMaskFlag()
{
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    attenMaskFlag_ = (maskTensor != nullptr) && (maskTensor->GetStorageShape().GetShapeSize() != 0);
}

ge::graphStatus FiaInfoParser::GetPseShiftFlag()
{
    const gert::StorageShape *pseShiftShape = context_->GetOptionalInputShape(PSE_SHIFT_INDEX);
    if (opParamInfo_.pseType != nullptr) {
        pseType_ = *opParamInfo_.pseType;
        if (pseType_ != static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pse_type", std::to_string(pseType_).c_str(),
                "The value of pse_type must be 0");
            return ge::GRAPH_FAILED;
        }
    }
    if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
        pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        enableAlibiPse_ = true;
    }
    if ((pseShiftShape == nullptr) ||
        ((pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() == 0))) {
        pseShiftFlag_ = false;
    } else if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_OUTER_MUL_ADD_TYPE)) {
        pseShiftFlag_ = true;
        pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDim(0);
        if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B) {
            pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDim(0) != 1U ? 1 : 0;
        }
        pseShiftS1_ = pseShiftShape->GetStorageShape().GetDim(ATTR_PSE_SHIFT_S1_INDEX);
        pseShiftS2_ = pseShiftShape->GetStorageShape().GetDim(ATTR_PSE_SHIFT_S2_INDEX);
    } else if (pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
               pseType_ == static_cast<int64_t>(IfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        if (antiQuantFlag_) {
            pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDimNum() != 0 ? 1 : 0;
        } else {
            pseShiftByBatch_ = pseShiftShape->GetStorageShape().GetDimNum() == 1 ? 1 : 0;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetSystemPrefix()
{
    if (opParamInfo_.keySharedPrefix.tensor != nullptr && opParamInfo_.valueSharedPrefix.tensor != nullptr) {
        keyPrefixShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.keySharedPrefix.tensor->GetStorageShape(),
                                                           kvLayout_, KEY_SHARED_PREFIX_NAME, opName_, n2Size_);
        systemPrefixMaxLen_ = keyPrefixShape_->GetShapeS();
        systemPrefixFlag_ = true;
        systemPrefixLen_ = systemPrefixMaxLen_;
        if (opParamInfo_.actualSharedPrefixLen.tensor != nullptr &&
            opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0) {
            gert::Shape prefixShape{1};
            if (prefixShape != opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape()) {
                std::string reason = "The shape of actual_shared_prefix_len must be {1} when system prefix is enabled";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "actual_shared_prefix_len",
                    ToString(opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape()).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.actualSharedPrefixLen.tensor->GetData<int64_t>() != nullptr) {
                systemPrefixLen_ = opParamInfo_.actualSharedPrefixLen.tensor->GetData<int64_t>()[0];
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::ProcessKVActualSeqLoop(const int64_t *actualLenData, uint32_t loop)
{
    int64_t preActualLen = 0;
    for (uint32_t i = 0; i < loop; i++) {
        int64_t actLen = actualLenData[i];
        if (antiQuantFlag_) {
            if (qLayout_ == FiaLayout::TND && i > 0 && kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
                actLen -= actualLenData[i - 1];
            }
            if (s1Size_ == 1) {
                needInit_ = needInit_ || (actLen == 0);
            }
        } else {
            if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
                actLen -= preActualLen;
                preActualLen = actualLenData[i];
            }
            needInit_ = needInit_ || (actLen == 0);
            if (actLen != s2Size_ && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD) {
                needInit_ = 1;
            }
        }
        maxActualseq_ = maxActualseq_ < actLen ? actLen : maxActualseq_;
        if (actualLenData[i] != actualLenData[0]) {
            isSameActualseq_ = false;
        }
    }
}

void FiaInfoParser::ProcessQActualSeqLoop(const int64_t *actualLenQData, uint32_t loop)
{
    int64_t preActualLen = 0;
    for (uint32_t i = 0; i < loop; i++) {
        int64_t actLen = actualLenQData[i];
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            actLen -= preActualLen;
            preActualLen = actualLenQData[i];
        }
        if (actLen != static_cast<int64_t>(s1Size_)) {
            needInit_ = true;
            break;
        }
    }
}

ge::graphStatus FiaInfoParser::GetActualSeqInfo()
{
    maxActualseq_ = 0;
    if (isMaxWorkspace_) {
        maxActualseq_ = s2Size_;
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDims_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
        if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
            isAccumKVSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) {
            const int64_t *actualLenData = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
            uint32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : bSize_;
            loop = std::min(loop, actualLenDims_);
            ProcessKVActualSeqLoop(actualLenData, loop);
        }
    } else {
        maxActualseq_ = s2Size_;
    }

    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        const int64_t *actualLenQData = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualLenQDims_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
        if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
            isAccumQSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr) {
            uint32_t loop = std::min(bSize_, actualLenQDims_);
            ProcessQActualSeqLoop(actualLenQData, loop);
        }
    }

    if ((quantMode_ == FiaQuantMode::ANTI_QUANT) && (s1Size_ == 1)) {
        if (qLayout_ != FiaLayout::TND) {
            actualLenQDims_ = 0;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetPostQuantInfo()
{
    ge::DataType outputType = opParamInfo_.attenOut.desc->GetDataType();
    if (outputType != ge::DT_BF16 && outputType != ge::DT_FLOAT16) {
        isPostQuantEnable_ = true;
        if (opParamInfo_.quantScale2.tensor != nullptr && opParamInfo_.quantScale2.desc != nullptr) {
            isOutQuantPerChnOut_ = opParamInfo_.quantScale2.tensor->GetStorageShape().GetShapeSize() != 1;
            isOutQuantTypeBf16_ = opParamInfo_.quantScale2.desc->GetDataType() == DT_BF16;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetFullQuantMode()
{
    if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        if (*opParamInfo_.queryQuantMode == 7 && *opParamInfo_.keyAntiquantMode == 7 &&
            *opParamInfo_.valueAntiquantMode == 7) {
            fullQuantMode_ = FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT;
        } else if (*opParamInfo_.queryQuantMode == 6 && *opParamInfo_.keyAntiquantMode == 6 &&
                   *opParamInfo_.valueAntiquantMode == 8) {
            fullQuantMode_ = FiaFullQuantMode::QKV_MXFP8_FULL_QUANT;
        } else if (*opParamInfo_.queryQuantMode == 3 && *opParamInfo_.keyAntiquantMode == 3 &&
                   *opParamInfo_.valueAntiquantMode == 2) {
            fullQuantMode_ = FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD;
        } else if (ropeMode_ == RopeMode::ROPE_SPLIT &&
                   (*opParamInfo_.queryQuantMode == 3 && *opParamInfo_.keyAntiquantMode == 0 &&
                    *opParamInfo_.valueAntiquantMode == 0)) {
            fullQuantMode_ = FiaFullQuantMode::Q_PER_TOKEN_HEAD_KV_PER_TENSOR_FULL_QUANT;
        } else {
            fullQuantMode_ = FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FiaInfoParser::MapStringToLayout(FiaLayout &layoutString) const
{
    const std::map<FiaLayout, TilingKeyLayout> layoutMap = {
        {FiaLayout::BSH, TilingKeyLayout::BSH_BSND}, {FiaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BNSD, TilingKeyLayout::BNSD},    {FiaLayout::NZ, TilingKeyLayout::NZ},
        {FiaLayout::TND, TilingKeyLayout::TND},      {FiaLayout::NBSD, TilingKeyLayout::NBSD},
        {FiaLayout::NTD, TilingKeyLayout::NTD},      {FiaLayout::BnBsH, TilingKeyLayout::BSH_BSND},
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

    // pse shift
    fiaInfo.pseShiftFlag = pseShiftFlag_;
    fiaInfo.pseShiftByBatch = pseShiftByBatch_;
    fiaInfo.pseShiftS1 = pseShiftS1_;
    fiaInfo.pseShiftS2 = pseShiftS2_;
    fiaInfo.enableAlibiPse = enableAlibiPse_;

    // atten mask
    fiaInfo.attenMaskFlag = attenMaskFlag_;
    fiaInfo.attenMaskBatchStride = attenMaskBatchStride_;
    fiaInfo.attenMaskStride = attenMaskStride_;
    fiaInfo.sparseMode = sparseMode_;
    // 4: only mla noquant & band mode suppport slidingFlag
    fiaInfo.slidingFlag =
        (*opParamInfo_.sparseMode == 4) && (ropeMode_ == RopeMode::ROPE_SPLIT) && (qkHeadDim_ == 512U);
    fiaInfo.qPaddingSizeFlag = qPaddingSizeFlag_;
    fiaInfo.kvPaddingSizeFlag = kvPaddingSizeFlag_;
    fiaInfo.softmaxLseFlag = *opParamInfo_.softmaxLseFlag;
    fiaInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();
    fiaInfo.isMaxWorkspace = isMaxWorkspace_;
    fiaInfo.isLegacyIfa = isLegacyIfa_;
    fiaInfo.preToken = preToken_;
    fiaInfo.nextToken = nextToken_;
    fiaInfo.learnableSinkFlag = (opParamInfo_.learnableSink.tensor != nullptr);
    fiaInfo.antiQuantFlag = antiQuantFlag_;
    fiaInfo.antiqSeqSize = antiquantParaSeqSize_;
    fiaInfo.keyAntiquantMode = static_cast<uint32_t>(*opParamInfo_.keyAntiquantMode);
    fiaInfo.valueAntiquantMode = static_cast<uint32_t>(*opParamInfo_.valueAntiquantMode);

    fiaInfo.isQKVDDifferent = (qkHeadDim_ != vHeadDim_) && !(qkHeadDim_ == 192U && vHeadDim_ == 128U);

    fiaInfo.sysPrefixFlag = systemPrefixFlag_;
    fiaInfo.systemPrefixLen = systemPrefixLen_;
    fiaInfo.systemPrefixMaxLen = systemPrefixMaxLen_;
    // postquant
    fiaInfo.isOutQuantPerChnOut = isOutQuantPerChnOut_;
    fiaInfo.isOutQuantTypeBf16 = isOutQuantTypeBf16_;
    fiaInfo.isOutQuantEnable = isPostQuantEnable_;
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
    fiaInfo.npuArch = npuArch_;
    GenerateAxisInfo(fiaInfo);
    GenerateDtypeInfo(fiaInfo);
    fiaInfo.kvStorageMode = kvStorageMode_;
    fiaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    fiaInfo.ropeMode = ropeMode_;
    fiaInfo.mlaMode = mlaMode_;
    fiaInfo.quantMode = quantMode_;
    fiaInfo.fullQuantMode = fullQuantMode_;
    fiaInfo.l2CacheSize = l2CacheSize_;

    fiaInfo.kCache = kCache_;
    fiaInfo.vCache = vCache_;
    fiaInfo.qSize = qSize_;
    fiaInfo.kvSize = kvSize_;

    fiaInfo.keyStrides = keyStrides_;
    fiaInfo.valueStrides = valueStrides_;
    fiaInfo.kRopeStrides = kRopeStrides_;
    fiaInfo.kScaleStrides = kScaleStrides_;
    fiaInfo.vScaleStrides = vScaleStrides_;
    fiaInfo.isTensorV1 = isTensorV1_;

    fiaInfo.totalOutputSize = opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize();

    fiaInfo.l2CacheOffFlag = false;
    fiaInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    fiaInfo.scaleValue = *opParamInfo_.scaleValue;
    fiaInfo.needInit = needInit_;
    fiaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        fiaInfo.maxBlockNumPerBatch = blockTableDim2_;
    }

    fiaInfo.actualLenQDims = actualLenQDims_;
    fiaInfo.actualLenKvDims = actualLenDims_;
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
    fiaInfo.s1Size = s1Size_;
    fiaInfo.s2Size = s2Size_;
    fiaInfo.gSize = gSize_;
    fiaInfo.qkHeadDim = qkHeadDim_;
    fiaInfo.vHeadDim = vHeadDim_;
    fiaInfo.ropeHeadDim = ropeHeadDim_;
    fiaInfo.qTSize = queryTSize_;
    fiaInfo.kTSize = keyTSize_;
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
        OP_LOGE(fiaInfo.opName, "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvCache() || ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }
    GetInOutDataType();
    GetKvStorageMode();
    GetQuantMode();
    if (ge::GRAPH_SUCCESS != GetAntiQuantInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetQueryAndOutLayout() || ge::GRAPH_SUCCESS != GetKvLayout()) {
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
    GetQueryTSize();
    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetLegacyIfaFlag()) {
        return ge::GRAPH_FAILED;
    }
    GetUpdateInfo();
    if (ge::GRAPH_SUCCESS != GetMaxWorkspaceFlag()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }
    if (emptyTensorFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    GetKeyTSize();
    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetRopeHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetStrides()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseFeatureInfo()
{
    GetMaskFlag();
    GetPaddingSizeFlag();
    GetPreNextToken();
    if (ge::GRAPH_SUCCESS != GetAttenMaskInfo() || ge::GRAPH_SUCCESS != GetActualSeqInfo() ||
        ge::GRAPH_SUCCESS != GetSystemPrefix() || ge::GRAPH_SUCCESS != GetPseShiftFlag() ||
        ge::GRAPH_SUCCESS != GetPostQuantInfo() || ge::GRAPH_SUCCESS != GetFullQuantMode()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
