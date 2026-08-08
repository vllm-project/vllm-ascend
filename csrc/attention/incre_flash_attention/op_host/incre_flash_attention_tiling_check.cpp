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
 * \file incre_flash_attention_tiling_check.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "incre_flash_attention_tiling_impl.h"
#include "incre_flash_attention_tiling_base.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"


using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus IFATiling::CheckPABlockSize() const
{
    OP_CHECK_IF(
        blockSize_ == 0,
        OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockSize_ > MAX_BLOCK_SIZE,
        OP_LOGE(ifaContext_->opName,
            "When Page Attention is enabled, input attribute blocksize %u can not be larger than %u.",
            blockSize_, MAX_BLOCK_SIZE),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0)),
        OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, if kv cache dtype is int8, input attr "
            "blocksize[%u] should be 32 aligned.", blockSize_),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
        OP_LOGE(ifaContext_->opName,
            "When Page Attention is enabled, "
            "if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckBaseInputsNull() const
{
    // Check base input tensors
    OP_CHECK_IF(ifaContext_->query.shape == nullptr, OP_LOGE(ifaContext_->opName, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->query.shape->GetStorageShape().GetShapeSize() == 0,
               OP_LOGE(ifaContext_->opName, "Tensor query is empty cause shapesize is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->query.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.shape == nullptr, OP_LOGE(ifaContext_->opName, "Shape of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.shape == nullptr, OP_LOGE(ifaContext_->opName, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.shape == nullptr, OP_LOGE(ifaContext_->opName, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    // Check base input attrs
    OP_CHECK_IF(ifaContext_->numHeads == nullptr, OP_LOGE(ifaContext_->opName, "attr the query's heads num is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->scaleValue == nullptr, OP_LOGE(ifaContext_->opName, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvHeadNums == nullptr, OP_LOGE(ifaContext_->opName, "attr kvHeadNums is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->layOut == nullptr, OP_LOGE(ifaContext_->opName, "attr layOut is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->blockSize == nullptr, OP_LOGE(ifaContext_->opName, "attr blockSize is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::IsSupportFormat(const ge::Format format) const
{
    if (format == ge::FORMAT_ND) {
        return true;
    }
    if (format == ge::FORMAT_NCHW) {
        return true;
    }
    if (format == ge::FORMAT_NHWC) {
        return true;
    }
    if (format == ge::FORMAT_NCDHW) {
        return true;
    }
    return false;
}

bool IFATiling::IsZeroDimTensor(const gert::Shape shape) const
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

std::string IFATiling::GetTensorDimString(const gert::Shape shape) const
{
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << std::to_string(shape.GetDim(i));
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

ge::graphStatus IFATiling::CheckInputParameterFormat() const
{
    auto qFormat = ifaContext_->query.desc->GetOriginFormat();
    auto kFormat = ifaContext_->key.desc->GetOriginFormat();
    auto vFormat = ifaContext_->value.desc->GetOriginFormat();

    OP_CHECK_IF(
        (!IsSupportFormat(qFormat)), OP_LOGE(ifaContext_->opName, "Query format %d should be ND/NCHW/NHWC/NCDHW", qFormat), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!IsSupportFormat(kFormat)), OP_LOGE(ifaContext_->opName, "Key format %d should be ND/NCHW/NHWC/NCDHW", kFormat), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!IsSupportFormat(vFormat)), OP_LOGE(ifaContext_->opName, "Value format %d should be ND/NCHW/NHWC/NCDHW", vFormat), return ge::GRAPH_FAILED);
    if(ifaContext_->attenMask.desc != nullptr){
        auto mFormat = ifaContext_->attenMask.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(mFormat)),
                OP_LOGE(ifaContext_->opName, "atten_mask format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->kvPaddingSize.desc != nullptr){
        auto kvPaddingFormat = ifaContext_->kvPaddingSize.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kvPaddingFormat)),
                OP_LOGE(ifaContext_->opName, "padding_size format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->keySharedPrefix.desc != nullptr){
        auto kPrefixFormat = ifaContext_->keySharedPrefix.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kPrefixFormat)),
                OP_LOGE(ifaContext_->opName, "k_prefix format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->valueSharedPrefix.desc != nullptr){
        auto vPrefixFormat = ifaContext_->valueSharedPrefix.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vPrefixFormat)),
                OP_LOGE(ifaContext_->opName, "v_prefix format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputAntiquantFormat() const
{
    if(ifaContext_->antiquantScale.desc != nullptr){
        auto aScaleFormat = ifaContext_->antiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(aScaleFormat)),
                OP_LOGE(ifaContext_->opName, "antiquantScale format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->antiquantOffset.desc != nullptr){
        auto aOffsetFormat = ifaContext_->antiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(aOffsetFormat)),
                OP_LOGE(ifaContext_->opName, "antiquantOffset format should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->keyAntiquantScale.desc != nullptr){
        auto kScaleFormat = ifaContext_->keyAntiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kScaleFormat)),
                OP_LOGE(ifaContext_->opName, "The format of the key's dequant scale should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->keyAntiquantOffset.desc != nullptr){
        auto kOffsetFormat = ifaContext_->keyAntiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kOffsetFormat)),
                OP_LOGE(ifaContext_->opName, "The format of the key's dequant offset should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->valueAntiquantScale.desc != nullptr){
        auto vScaleFormat = ifaContext_->valueAntiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vScaleFormat)),
                OP_LOGE(ifaContext_->opName, "The format of the value's dequant scale should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    if(ifaContext_->valueAntiquantOffset.desc != nullptr){
        auto vOffsetFormat = ifaContext_->valueAntiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vOffsetFormat)),
                OP_LOGE(ifaContext_->opName, "The format of the value's dequant offset should be ND/NCHW/NHWC/NCDHW"),
                return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static const size_t KV_CACHE_DIM_NUM = 4;
static const size_t BLOCK_TABLE_DIM_NUM = 2;

ge::graphStatus IFATiling::ProcessCheckATBInputWithPage() const
{
    auto blockTablesShape = ifaContext_->blockTable.tensor->GetStorageShape();
    ge::DataType inputBlockTableType_ = ifaContext_->blockTable.desc->GetDataType();
    uint64_t taskNumI64 = static_cast<uint64_t>(numHeads_) * batchSize_;
    OP_CHECK_IF((inputBlockTableType_ != ge::DT_INT32),
        OP_LOGE(ifaContext_->opName, "blockTables dtype %d invalid, should be int32", inputKvType_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((blockTablesShape.GetDimNum() != BLOCK_TABLE_DIM_NUM),
        OP_LOGE(ifaContext_->opName, "blockTables dim num %lu, invalid, should be %lu",
            blockTablesShape.GetDimNum(), BLOCK_TABLE_DIM_NUM),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((taskNumI64 > UINT32_MAX), OP_LOGE(ifaContext_->opName, "param is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((blockSize_ <= 0 || blockSize_ % 16U != 0),
        OP_LOGE(ifaContext_->opName, "blockSize is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((static_cast<uint64_t>(headDim_) * blockSize_ > 128U * 128U),
        OP_LOGE(ifaContext_->opName, "headDim * blockSize should no greater than 128 * 128"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyShapeInput() const
{
    const gert::StorageShape *keyShape = ifaContext_->kCache[0];
    OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
            OP_LOGE(ifaContext_->opName,
            "key dim num %lu, invalid, should be %lu", keyShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF((keyShape->GetStorageShape().GetDim(3) != 16),
            OP_LOGE(ifaContext_->opName, "K_cache Shape should be in nz format"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckValueShapeInput() const
{
    const gert::StorageShape *valueShape = ifaContext_->vCache[0];
    OP_CHECK_IF((valueShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
            OP_LOGE(ifaContext_->opName,
            "value dim num %lu, invalid, should be %lu", valueShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF((valueShape->GetStorageShape().GetDim(3) != 16),
            OP_LOGE(ifaContext_->opName, "V_cache Shape should be in nz format"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessCheckATBInput()
{
    OP_CHECK_IF((numHeads_ <= 0),
        OP_LOGE(ifaContext_->opName, "headSize is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((numKvHeads_ <= 0 || numKvHeads_ > numHeads_ || numHeads_ % numKvHeads_ != 0),
            OP_LOGE(ifaContext_->opName, "kvHead is invalid."),
            return ge::GRAPH_FAILED);

    static const size_t Q_CACHE_DIM_NUM = 4;
    const gert::StorageShape *queryShape = ifaContext_->query.shape;

    OP_CHECK_IF((inputQType_ != ge::DT_FLOAT16),
        OP_LOGE(ifaContext_->opName, "query dtype %d invalid, should be float16", inputQType_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKvType_ != ge::DT_FLOAT16),
        OP_LOGE(ifaContext_->opName, "K and V dtype %d invalid, should be float16", inputKvType_),
        return ge::GRAPH_FAILED);
    if (pageAttentionFlag_) {
        if (ProcessCheckATBInputWithPage() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF((ifaContext_->actualSeqLengths.tensor == nullptr),
            OP_LOGE(ifaContext_->opName, "The query's actual sequence lengths cannot be nullptr in GQA mode"),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != Q_CACHE_DIM_NUM),
            OP_LOGE(ifaContext_->opName,
            "query dim num %lu, invalid, should be %lu", queryShape->GetStorageShape().GetDimNum(), Q_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);

    if (CheckKeyShapeInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckValueShapeInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((headDim_ > 256 || headDim_ <= 0),
        OP_LOGE(ifaContext_->opName, "headdim is invalid"), return ge::GRAPH_FAILED);

    int32_t numTokens = queryShape->GetStorageShape().GetDim(2);

    OP_CHECK_IF((numTokens <= 0 || batchSize_ <= 0 || numHeads_ <= 0),
            OP_LOGE(ifaContext_->opName, "param must > 0"),
            return ge::GRAPH_FAILED);

    if (ifaContext_->pseShift.tensor != nullptr) {
        attenMaskFlag_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputQKVTypeMatch() const
{
    OP_CHECK_IF(
        (!ropeFlag_) && (nNumOfQInOneGroup_ > 64),
        OP_LOGE(ifaContext_->opName, "the query's heads num divided by the key/value's heads num = %u, cannot be greater than 64", nNumOfQInOneGroup_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((!ropeFlag_) && inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8),
        OP_LOGE(ifaContext_->opName, "When QueryRope/KeyRope is null and Qs(%u) in [1, 32], not support qkv datatype all int8.", qSeqSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((inputQType_ == ge::DT_FLOAT16) && (inputKvType_ != ge::DT_FLOAT16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)), OP_LOGE(ifaContext_->opName, "when input Q type is fp16, KV type %d should be fp16 or int8 or int4", inputKvType_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((inputQType_ == ge::DT_BF16) && (inputKvType_ != ge::DT_BF16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)), OP_LOGE(ifaContext_->opName, "when input Q type is bf16, KV type %d should be bf16 or int8 or int4", inputKvType_),
        return ge::GRAPH_FAILED);

    if (pageAttentionFlag_) {
        OP_CHECK_IF(
            (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
            OP_LOGE(ifaContext_->opName, "blockSize=%u, it need align to 16 when kv dtype is fp16/bf16.", blockSize_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0),
            OP_LOGE(ifaContext_->opName, "blockSize=%u, it need align to 32 when kv dtype is int8.", blockSize_),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus IFATiling::CheckInputFormatAndLimits()
{
    if (CheckInputParameterFormat() != ge::GRAPH_SUCCESS ||
        CheckInputAntiquantFormat() != ge::GRAPH_SUCCESS ||
        CheckInputQKVTypeMatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
		if(atbRunFlag_ ){
            return ProcessCheckATBInput();
        }
        std::string layOutStr = ifaContext_->layOut;
        OP_CHECK_IF(
            (headDim_ > 128U || numHeads_ * qSeqSize_ / numKvHeads_ > 32) && numHeads_ * qSeqSize_ != numKvHeads_,  
            OP_LOGE(ifaContext_->opName, "On 300I DUO, when gqa > 1 or qlen > 1, it requires that gqa * qlen <= 32 and headDim <=128."),
            return ge::GRAPH_FAILED);
        
        OP_CHECK_IF(
            pageAttentionFlag_ && layOutStr != "BSH" && layOutStr != "BSND" && numHeads_ * qSeqSize_ != numKvHeads_,  
            OP_LOGE(ifaContext_->opName, "On 300I DUO, when gqa > 1 or qlen > 1, it requires that the input layout to be BSH / BSND with page attention enabled."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (!pageAttentionFlag_) && layOutStr != "BNSD" && numHeads_ * qSeqSize_ != numKvHeads_,  
            OP_LOGE(ifaContext_->opName, "On 300I DUO, when gqa > 1 or qlen > 1, it requires that the input layout to be BNSD without page attention enabled."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (batchSize_ > 256),
            OP_LOGE(ifaContext_->opName, "batch size:%u cannot be greater than 256 when 310P.", batchSize_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (sMax_ > 65536),
            OP_LOGE(ifaContext_->opName, "sMax:%u cannot be greater than 65536 when 310P.", sMax_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (headDim_ % 16 != 0),
            OP_LOGE(ifaContext_->opName, "in 310P, headDim:%u need align to 16.", headDim_),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (antiQuantFlag_ && (headDim_ % 32 != 0)),
            OP_LOGE(ifaContext_->opName, "in 310P, headDim:%u need align to 32 when kv dtype is int8.", headDim_),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (batchSize_ > 65536),
            OP_LOGE(ifaContext_->opName, "batch size:%u cannot be greater than 65536.", batchSize_),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        (!ropeFlag_) && (headDim_ > 512U),
        OP_LOGE(ifaContext_->opName, "headDim:%u cannot be greater than 512.", headDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!ropeFlag_) && (numKvHeads_ > 256U),
        OP_LOGE(ifaContext_->opName, "numHead of key and value:%u cannot be greater than 256.", numKvHeads_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVHeadNum(const gert::StorageShape *inputShape) const
{
    uint32_t tmpNumHeads = 0;
    std::string layOutStr = ifaContext_->layOut;
    constexpr uint32_t dimIndex2 = 2;
    if (layOutStr == "BSH") {
        auto H = inputShape->GetStorageShape().GetDim(dimIndex2);
        tmpNumHeads = H / headDim_;
    } else if (layOutStr == "TND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BNSD") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BSND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(dimIndex2);
    }
    OP_CHECK_IF(
        tmpNumHeads != numKvHeads_,
        OP_LOGE(ifaContext_->opName, "IFA check input param failed, tensor in list head num(%u) should be %u.",
            tmpNumHeads, numKvHeads_),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVShape(const size_t &size, const gert::StorageShape *keyTensorInList, const gert::StorageShape *valueTensorInList) const
{
    /* kv not continuous */
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH") {
        OP_CHECK_IF(
            (keyTensorInList->GetStorageShape().GetDimNum() != DIM_BSH) ||
            (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BSH),
            OP_LOGE(ifaContext_->opName,
                "IFA check input param failed, tensor in list dim num should be 3, k: %lu, v: %lu.",
                keyTensorInList->GetStorageShape().GetDimNum(),
                valueTensorInList->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    }
    if ((layOutStr == "BNSD") || (layOutStr == "BSND")) {
        OP_CHECK_IF(
            (keyTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND) ||
            (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND),
            OP_LOGE(ifaContext_->opName,
                "IFA check input param failed, tensor in list dim num should be 4, k: %lu, v: %lu.",
                keyTensorInList->GetStorageShape().GetDimNum(),
                valueTensorInList->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    }
    if (layOutStr == "TND") {
        OP_CHECK_IF(
            (keyTensorInList->GetStorageShape().GetDimNum() != DIM_TND) ||
            (valueTensorInList->GetStorageShape().GetDimNum() != DIM_TND),
            OP_LOGE(ifaContext_->opName,
                "IFA check input param failed, tensor in list dim num should be 4, k: %lu, v: %lu.",
                keyTensorInList->GetStorageShape().GetDimNum(),
                valueTensorInList->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        keyTensorInList->GetStorageShape().GetDim(0) != 1,
        OP_LOGE(
            ifaContext_->opName,
            "IFA check input param failed, b of tensor in tensor list should be 1, now b is: %ld, list index: %lu",
            keyTensorInList->GetStorageShape().GetDim(0), size),
        return ge::GRAPH_FAILED);
    if (CheckKVHeadNum(keyTensorInList) != ge::GRAPH_SUCCESS ||
        CheckKVHeadNum(valueTensorInList) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeBsh(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
    const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const
{
    const std::string inputLayoutStr = ifaContext_->layOut;
    auto outExpectDim = inputLayoutStr == "BSH" ? DIM_BSH : DIM_BNSD;
    OP_CHECK_IF((dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != outExpectDim),
        OP_LOGE(ifaContext_->opName, "When input layout is BSH/BSH_NBSD, the dimension should be 3, "
        "the dimOfOut should be %u, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu",
        outExpectDim, dimOfQ, dimOfK, dimOfOut), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(1) != 1 && !ropeFlag_ && !gqaMtpFlag_,
        OP_LOGE(ifaContext_->opName, "When input layout is BSH, the 2nd dimOfQ should be 1,the 2nd dimOfQ: %ld",
        queryShape->GetStorageShape().GetDim(1)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(2) / numHeads_ != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
        OP_LOGE(ifaContext_->opName, "When input layout is BSH,"
        "the 3rd dimOfQ/numHeads(%ld) should be equal to the 3rd dimOfK/numKvHeads(%ld)",
        queryShape->GetStorageShape().GetDim(2) / numHeads_, keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeTnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
    const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const
{
    OP_CHECK_IF((dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
        OP_LOGE(ifaContext_->opName, "When input layout is TND, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu",
        dimOfQ, dimOfK, dimOfOut), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(1) / numHeads_ != keyShape->GetStorageShape().GetDim(1) / numKvHeads_,
        OP_LOGE(ifaContext_->opName, "When input layout is TND, "
        "the 2nd dimOfQ/numHeads(%ld) should be equal to the 2nd dimOfK/numKvHeads(%ld)",
        queryShape->GetStorageShape().GetDim(1) / numHeads_, keyShape->GetStorageShape().GetDim(1) / numKvHeads_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeBnsdOrBsnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
    const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const
{
    OP_CHECK_IF((dimOfQ != DIM_BNSD_OR_BSND) || (dimOfK != DIM_BNSD_OR_BSND) || (dimOfOut != DIM_BNSD_OR_BSND),
        OP_LOGE(ifaContext_->opName, "When input layout is BNSD/BSND, the dim should be 4, "
        "4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu", dimOfQ, dimOfK, dimOfOut), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
        OP_LOGE(ifaContext_->opName,
        "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld",
        queryShape->GetStorageShape().GetDim(3), keyShape->GetStorageShape().GetDim(3)), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShape() const
{
    if (pageAttentionFlag_) { // page_attention don't check this place
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *queryShape = ifaContext_->query.shape;
    const gert::StorageShape *keyShape = ifaContext_->kCache[0];
    const std::string inputLayoutStr = ifaContext_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = ifaContext_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD") {
        return CheckQKOutShapeBsh(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        return CheckQKOutShapeTnd(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    } else {
        return CheckQKOutShapeBnsdOrBsnd(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyShapeTensor(const gert::Shape &aShape) const
{
    auto firstKeyShape = ifaContext_->kCache[0];
    std::string layOutStr = ifaContext_->layOut;
    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (((layOutStr == "BNSD") && (idx == 2)) || // BNSD s index is 2
            ((layOutStr == "BSND") && (idx == 1)) || // BSND s index is 1
            ((layOutStr == "BSH") && (idx == 1))) {  // BSH s index is 1
            continue;                                // s can be different
        }
        OP_CHECK_IF(
            firstKeyShape->GetStorageShape().GetDim(idx) != aShape.GetDim(idx),
            OP_LOGE(ifaContext_->opName,
                "IFA check input param failed, tensor in keyShape except S must be same, index:[%lu] is not same, k0: %ld, k: %ld",
                idx, firstKeyShape->GetStorageShape().GetDim(idx), aShape.GetDim(idx)),
                return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::CheckIfRollBack() const
{
    if (sMax_ == 0U) {
        return false; // 空tensor由新模板处理
    }

    if (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) {
        // 1 page attention
        if (ifaContext_->blockTable.tensor != nullptr) {
            return false;
        }
    }

    // 4 D>=1024
    if (headDim_ >= 1024U) {
        return true;
    }

    if (CanChangeToNew()) {
        return false;
    }

    return true;
}

bool IFATiling::ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape) const
{
    if (aShape.GetDimNum() != bShape.GetDimNum()) {
        return false;
    }

    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (aShape.GetDim(idx) != bShape.GetDim(idx)) {
            return false;
        }
    }

    return true;
}

bool IFATiling::CanChangeToNew() const
{
    if (inOutMode_ == TilingInOutMode::BF16_BF16) {
        return true;
    }
    if (inOutMode_ == TilingInOutMode::BF16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::INT8_BF16 || inOutMode_ == TilingInOutMode::INT8_FP16) {
        return true;
    }
    return false;
}

static bool IsQuant2ShapeValid(const gert::Shape &inputParaShape, uint32_t headnum, uint32_t headsize)
{
    gert::Shape expectParamShapeBNSD = gert::Shape({1, headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_2 = gert::Shape({headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBSND = gert::Shape({1, 1, headnum, headsize});
    gert::Shape expectParamShapeBSND_2 = gert::Shape({1, headnum, headsize});
    gert::Shape expectParamShapeBSND_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBH = gert::Shape({1, headnum * headsize});
    gert::Shape expectParamShapeBH_2 = gert::Shape({1, 1, headnum * headsize});
    gert::Shape expectParamShapeBH_3 = gert::Shape({headnum * headsize});

    bool validShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSND) ||
                      (inputParaShape == expectParamShapeBH) ||
                      (inputParaShape == expectParamShapeBNSD_2) || (inputParaShape == expectParamShapeBSND_2) ||
                      (inputParaShape == expectParamShapeBH_2) ||
                      (inputParaShape == expectParamShapeBNSD_3) || (inputParaShape == expectParamShapeBSND_3) ||
                      (inputParaShape == expectParamShapeBH_3);
    
    return validShape;
}

ge::graphStatus IFATiling::CheckQuant2Shape(const gert::Shape &inputParaShape) const
{
    auto headsize = headDim_; // D
    auto headnum = numHeads_; // Q's N
    bool validShape = IsQuant2ShapeValid(inputParaShape, headnum, headsize);
    if (!validShape && inputParaShape.GetDimNum() == DIM_BNSD) {
        OP_LOGE(ifaContext_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld, %ld] is not expected shape."
                  "Expect [1, %u, 1, %u] or [1, 1, %u, %u]",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX), headnum, headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 3) { // dim is 3
        OP_LOGE(ifaContext_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld] is not expected shape."
                  "Expect [%u, 1, %u], [1, %u, %u] or [1, 1, %u].",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), headnum, headsize, headnum, headsize, headnum * headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == DIM_BH) {
        OP_LOGE(ifaContext_->opName, "The shape of postquant parameter[%ld, %ld] is not expected[1, %u] or [%u, %u].",
                  inputParaShape.GetDim(BH_B_IDX), inputParaShape.GetDim(BH_H_IDX), headnum * headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 1) {
        OP_LOGE(ifaContext_->opName, "The shape of postquant parameter[%ld] is not expected[%u].",
                  inputParaShape.GetDim(BH_B_IDX), headnum * headsize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape) const
{
    if (antiquantMode_ == PER_TOKEN_MODE) { // per-token head
        OP_CHECK_IF((inputParaShape.GetDimNum() != 3), // 3: Dim of BGS is 3
                   OP_LOGE(ifaContext_->opName, "The dim of antiquant should be 3 instead of the current %lu",
                             inputParaShape.GetDimNum()),
                             return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(0) != batchSize_),
                   OP_LOGE(ifaContext_->opName, "The 1st dim of antiquant should be %u instead of the current %ld",
                             batchSize_, inputParaShape.GetDim(0)),
                             return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(1) != numKvHeads_),
                   OP_LOGE(ifaContext_->opName, "The 2nd dim of antiquant should be %u instead of the current %ld",
                             numKvHeads_, inputParaShape.GetDim(1)),
                             return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(2) < seqSize_),
                   OP_LOGE(ifaContext_->opName, "The 3rd dim of antiquant should bigger than or equal to %u instead of the current %ld",
                             seqSize_, inputParaShape.GetDim(2)), // 2 : BGS S index is 2
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else { // per-tensor head
        gert::Shape expectParamShape = gert::Shape({numKvHeads_});
        OP_CHECK_IF((inputParaShape != expectParamShape),
                   OP_LOGE(ifaContext_->opName,
                             "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor_head mode.",
                             inputParaShape.GetDim(0), numKvHeads_),
               return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerChannel(const gert::Shape& inputParaShape) const
{
    std::string layOutStr = ifaContext_->layOut;
    if (gqaKvNZFlag_ && kvAntiParamSplitFlag_) {
        return CheckGqaAntiQuantPerChannel(inputParaShape);
    } else {
        bool validOffsetShape = false;
        gert::Shape expectParamShapeBNSD = gert::Shape({antiquantNum_, numKvHeads_, 1, headDim_});
        gert::Shape expectParamShapeBSNDType1 = gert::Shape({antiquantNum_, 1, numKvHeads_, headDim_});
        gert::Shape expectParamShapeBSNDType2 = gert::Shape({antiquantNum_, numKvHeads_, headDim_});
        gert::Shape expectParamShapeBH = gert::Shape({antiquantNum_, numKvHeads_ * headDim_});
        validOffsetShape = (inputParaShape == expectParamShapeBNSD) ||
                           (inputParaShape == expectParamShapeBSNDType1) ||
                           (inputParaShape == expectParamShapeBSNDType2) ||
                           (inputParaShape == expectParamShapeBH);
        OP_CHECK_IF(
            (!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld, %ld] is not expected. "
                "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BNSD_B_IDX),
                inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX),
                antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
                "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BND_B_IDX),
                inputParaShape.GetDim(BND_N_IDX), inputParaShape.GetDim(BND_D_IDX), antiquantNum_, numKvHeads_, headDim_,
                antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
                "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BH_B_IDX),
                inputParaShape.GetDim(BH_H_IDX), antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_,
                antiquantNum_, numKvHeads_ * headDim_),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiQuantPerChannel(const gert::Shape& inputParaShape) const
{
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BNSD") {
        OP_CHECK_IF(
            (inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BNSD),
            OP_LOGE(ifaContext_->opName, "The shape dim[%zu] of antiquant parameter is not expected. "
                "Expect [%u] when per_channel mode in GQA KV NZ and layout is BNSD.", inputParaShape.GetDimNum(), DIM_PER_CHANNEL_KVNZ_BNSD),
            return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBNSD = gert::Shape({numKvHeads_, 1, headDim_});
        OP_CHECK_IF(
            (inputParaShape != expectParamShapeBNSD),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
                "Expect [%u, %u, %u] when per_channel mode in GQA KV NZ and layout is BNSD.", inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_N_IDX),
                inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_S_IDX), inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_D_IDX),
                numKvHeads_, 1U, headDim_),
            return ge::GRAPH_FAILED);
    } else if (layOutStr == "BSND" || layOutStr == "TND") {
        OP_CHECK_IF(
            (inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BSND),
            OP_LOGE(ifaContext_->opName, "The shape dim[%zu] of antiquant parameter is not expected. "
                "Expect [%u] when per_channel mode in GQA KV NZ and layout is BSND or TND.", inputParaShape.GetDimNum(), DIM_PER_CHANNEL_KVNZ_BSND),
            return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBSND = gert::Shape({numKvHeads_, headDim_});
        OP_CHECK_IF(
            (inputParaShape != expectParamShapeBSND),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
                "Expect [%u, %u] when per_channel mode in GQA KV NZ and layout is BSND.", inputParaShape.GetDim(BSND_NZ_ANTIQUANT_N_IDX),
                inputParaShape.GetDim(BSND_NZ_ANTIQUANT_D_IDX), numKvHeads_, headDim_),
            return ge::GRAPH_FAILED);
    } else if (layOutStr == "BSH") {
        OP_CHECK_IF(
            (inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BSH),
            OP_LOGE(ifaContext_->opName, "The shape dim[%zu] of antiquant parameter is not expected. "
                "Expect [%u] when per_channel mode in GQA KV NZ and layout is BSH.", inputParaShape.GetDimNum(), DIM_PER_CHANNEL_KVNZ_BSH),
            return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBH = gert::Shape({numKvHeads_ * headDim_});
        OP_CHECK_IF(
            (inputParaShape != expectParamShapeBH),
            OP_LOGE(ifaContext_->opName, "The shape of antiquant parameter[%ld] is not expected. "
                "Expect [%u] when per_channel mode in GQA KV NZ and layout is BSH.", inputParaShape.GetDim(BSH_NZ_ANTIQUANT_H_IDX),
                numKvHeads_ * headDim_),
            return ge::GRAPH_FAILED);
    } // 其他无效layout场景已校验过，不需要再校验
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiQuantParam(const gert::Tensor *antiquantScaleTensor,
                                               const gert::Tensor *antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                               const gert::CompileTimeTensorDesc *antiquantOffsetDesc)
{
    OP_CHECK_IF(
        (antiquantMode_ != 0) && (antiquantMode_ != 1) && (antiquantMode_ != 2), // unseparated antiquant need this
        OP_LOGE(ifaContext_->opName, "antiquantMode value:%u is invalid, it should be 0 or 1 or 2", antiquantMode_),
        return ge::GRAPH_FAILED);
    if (antiquantScaleTensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "KV antiquant is enabled, but the input antiquant scale is NULL");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr &&
        antiquantScaleTensor->GetStorageShape().GetDimNum() != antiquantOffsetTensor->GetStorageShape().GetDimNum()) {
        OP_LOGE(ifaContext_->opName,
                  "KV antiquant is enabled, but antiquant params have different layouts[scale: %lu, offset: %lu].",
                  antiquantScaleTensor->GetStorageShape().GetDimNum(),
                  antiquantOffsetTensor->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto tmpAntiquantScale = antiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantScale) == ge::GRAPH_FAILED) {
        OP_LOGE(ifaContext_->opName, "illegal shape of antiquant scale.");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr) {
        auto tmpAntiquantOffset = antiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantOffset) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();
    if (antiquantMode_ == DEQUANT_PER_CHANNEL_MODE) {
        if (antiquantScaleType != inputQType_) {
            OP_LOGE(ifaContext_->opName, "illegal datatype of antiquant scale, it should be same with input qtype");
            return ge::GRAPH_FAILED;
        }
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_MODE) {
        if (antiquantScaleType != ge::DT_FLOAT) {
            OP_LOGE(ifaContext_->opName, "per-token mode is enabled, datatype of antiquant scale should be float32 ");
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckAntiquantOffsetType(antiquantOffsetTensor, antiquantOffsetDesc, antiquantScaleType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiquantOffsetType(const gert::Tensor *antiquantOffsetTensor,
                                                    const gert::CompileTimeTensorDesc *antiquantOffsetDesc,
                                                    ge::DataType antiquantScaleType) const
{
    if (antiquantOffsetTensor != nullptr && antiquantOffsetDesc != nullptr) {
        ge::DataType antiquantOffsetType = antiquantOffsetDesc->GetDataType();
        if (antiquantScaleType != antiquantOffsetType) {
            OP_LOGE(ifaContext_->opName, "datatype of antiquant scale and antiquant offset should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckSupportKVLeftPadding()
{
    if (inputKvType_ == ge::DT_INT4) {
        OP_LOGE(ifaContext_->opName, "When input Kv Dtypes is INT4 or INT32, KvLeftPadding is not supported currently.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ropeFlag_,
        OP_LOGE(ifaContext_->opName, "kvPadding is not supported in MLA."),
        return ge::GRAPH_FAILED);
    if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition:  \
      pagedAttention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d.",
                  pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
        return ge::GRAPH_SUCCESS;
    }
    kvPaddingSizeFlag_ = true;
    OP_LOGD(ifaContext_->opName, "KVLeftPadding starts to be used.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckBasic()
{
    OP_CHECK_IF(ifaContext_->keySharedPrefix.tensor == nullptr,
               OP_LOGE(ifaContext_->opName, "tensor of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc == nullptr,
               OP_LOGE(ifaContext_->opName, "desc of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.tensor == nullptr,
               OP_LOGE(ifaContext_->opName, "tensor of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc == nullptr,
               OP_LOGE(ifaContext_->opName, "desc of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc->GetDataType() != inputKvType_,
               OP_LOGE(ifaContext_->opName, "type of key_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc->GetDataType() != inputKvType_,
               OP_LOGE(ifaContext_->opName, "type of value_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(pageAttentionFlag_, OP_LOGE(ifaContext_->opName, "shared prefix with page attention is not supported"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_, OP_LOGE(ifaContext_->opName, "shared prefix with kv padding is not supported"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(socVersion_ == IfaSocVersion::SOC_ASCEND_310P,
               OP_LOGE(ifaContext_->opName, "shared prefix is not supported on 310p"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckShapes(const gert::Shape &keyShape, const gert::Shape &valueShape) const
{
    OP_CHECK_IF(
        !ShapeEqual(keyShape, valueShape),
        OP_LOGE(ifaContext_->opName, "tensor shape of key_shared_prefix and value_shared_prefix not equal."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        keyShape.GetDimNum() != ifaContext_->query.shape->GetStorageShape().GetDimNum(),
        OP_LOGE(ifaContext_->opName, "tensor shape dim of key_shared_prefix[%lu] is not compatable with query",
            keyShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        keyShape.GetDim(0) != 1,
        OP_LOGE(ifaContext_->opName, "batch of key_shared_prefix[%ld] must be 1", keyShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        OP_CHECK_IF(
            keyShape.GetDimNum() == 3 && keyShape.GetDim(2) != numKvHeads_ * headDim_,
            OP_LOGE(ifaContext_->opName, "H of key_shared_prefix[%lu] is not equal to H of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(2) != numKvHeads_,
            OP_LOGE(ifaContext_->opName, "N of key_shared_prefix[%lu] is not equal to N of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(3) != headDim_,
            OP_LOGE(ifaContext_->opName, "D of key_shared_prefix[%lu] is not equal to D of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            keyShape.GetDim(1) != numKvHeads_,
            OP_LOGE(ifaContext_->opName, "N of key_shared_prefix[%ld] is not equal to N of key", keyShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            keyShape.GetDim(3) != headDim_,
            OP_LOGE(ifaContext_->opName, "D of key_shared_prefix[%ld] is not equal to D of key", keyShape.GetDim(3)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckUbSpace()
{
    if (!CalcUbBmm() || !CalcUbSoftMax() || !CalcUbAttenMask()) {
        return false;
    }
    return true;
}

bool IFATiling::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_ -> opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }
    if (balanceModeFlag_) {
        uint32_t tndFDCoreArrLen = tilingDataMla_.tndSplitCoreParams.get_tndFDCoreArrLen();
        return tndFDCoreArrLen > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}

bool IFATiling::CheckCoreOkFlag(uint32_t coreNum,IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMM() const
{
    if (!CheckConstraints()) {
        return false;
    }

    if (!CheckDataCopyNd2Nz()) {
        return false;
    }

    uint32_t s2Loop = (maxActualseq_ + 2048U - 1U) / 2048U;
    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ * s2Loop <= 128U) { // 128: 性能收益临界值
        return false;
    }

    if (antiQuantFlag_) {
        if (inputKvType_ == ge::DT_INT4) {
            return false;
        }
    }
    std::string layOutStr = ifaContext_->layOut;
    if (pageAttentionFlag_ && layOutStr == "BNSD") {
        bool quantFlag = antiQuantFlag_ && (antiquantMode_ == PER_CHANNEL_MODE) && (inputQType_ == ge::DT_FLOAT16 || inputQType_ == ge::DT_BF16) && (inputKvType_ == ge::DT_INT8);
        bool noQuantFlag = (inputQType_ == ge::DT_FLOAT16 && inputKvType_ == ge::DT_FLOAT16) || (inputQType_ == ge::DT_BF16 && inputKvType_ == ge::DT_BF16);
        if (quantFlag || noQuantFlag) {
            return true;
        }
    }

    return false;
}

bool IFATiling::CheckDataCopyNd2Nz() const
{
    if (innerPrecise_ == IFA_HIGH_PRECISION) {
        return false;
    }
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    return true;
}

bool IFATiling::CheckConstraints() const
{
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 16U) { // 16: group size
        return false;
    }
    if (headDim_ != 128U) { // 128: D dim
        return false;
    }
    return true;
}

bool IFATiling::EnableCubeViewMMDD() const
{
    if (!CheckConstraints()) {
        return false;
    }
    if (innerPrecise_ == IFA_HIGH_PRECISION){
        return false;
    }
    if(softmaxLseFlag_){
        return false;
    }

    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    if (layOutStr == "TND" || layOutStr == "TND_NTD") {
        return false;
    }
    if (ifaContext_->antiquantOffset.desc != nullptr || ifaContext_->keyAntiquantOffset.desc != nullptr || ifaContext_->valueAntiquantOffset.desc != nullptr) {
        return false;
    }
    if (gqaKvNZFlag_ && pageAttentionFlag_ && antiQuantFlag_ && (antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) &&
        (inputQType_ == ge::DT_BF16) && (inputKvType_ == ge::DT_INT8)) {
        return true;
    }
    return false;
}

bool IFATiling::DealSameSeqEachBatch() const
{
    if (!batchContinuousFlag_){
        if (actualSeqLenFlag_){
            return isSameActualseq_;
        } else {
            return isSameSeqAllKVTensor_;
        }
    } else {
        return isSameActualseq_;
    }
}

bool IFATiling::HasZeroSeqBatch() const
{
    if (!batchContinuousFlag_){
        if (actualSeqLenFlag_){
            return hasZeroActualseq_;
        } else {
            return hasZeroSeqKVTensor_;
        }
    } else {
        return hasZeroActualseq_;
    }
}

bool IFATiling::IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize, const std::vector<uint32_t> &s1OuterNum, const std::vector<uint32_t> &s2OuterNum) const
{
    if (needUpdate) {
        return false;
    }
    if ((lastValidBMoreIdx <= 0U) || (lastValidBMoreIdx >= bSize)) {
        return false;
    }
    if (s1OuterNum[lastValidBMoreIdx] == 0) {
        return false;
    }
    if (s2OuterNum[lastValidBMoreIdx] != 0) {
        return false;
    }
    return true;
}

bool IFATiling::EnableGQA310P()
{
    if (IsFlashDecode(aicNum_, IfaPerfMode::BMM_ALL_BY_VEC)){
        return false;
    }
    if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
        return true;
    }
    if (pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) && inputLayout_ == IfaLayout::BSH_BSND) {
        return true;
    }
    if (!pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) && inputLayout_ == IfaLayout::BNSD) {
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMMFullLoad()
{
    if (!CheckConstraints()) {
        return false;
    }
    if (IsFlashDecode(aicNum_, IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD)) {
        return false;
    }
    if (softmaxLseFlag_) {
        return false;
    }
    if (!CheckDataCopyNd2Nz()) {
        return false;
    }
    if (!DealSameSeqEachBatch()) {
        return false;
    }
    if (HasZeroSeqBatch()) {
        return false;
    }
    if (maxActualseq_ > 2048U || maxActualseq_ == 0U) {
        return false;
    }

    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ <= 128U) {  // 128: 性能收益临界值
        return false;
    }

    if (antiQuantFlag_) {
        if (inputKvType_ == ge::DT_INT4) {
            return false;
        }
        return false;
    } else {
        return false;
    }
}

bool IFATiling::EnableAllVec() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return true;
    }
    if (pageAttentionFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 1U) {
        return false;
    }
    if (headDim_ > 512U) { // 全VEC模板仅支持headDim_不大于512
        return false;
    }
    return (inputQType_ == ge::DT_FLOAT16) && (inputKvType_ == ge::DT_FLOAT16) && (outputType_ == ge::DT_FLOAT16);
}

bool IFATiling::EnableC1V1() const
{
    if (splitKVFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    // 2:核数不超过vector总核数一半，可以按1:1启动cube和vector
    return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * 2U <= aivNum_);
}

ge::graphStatus IFATiling::CheckPseShiftDataType() const
{
    auto pseShiftDataType = ifaContext_->pseShift.desc->GetDataType();
    if (pseShiftDataType != ge::DT_FLOAT16 && pseShiftDataType != DT_BF16) {
        OP_LOGE(ifaContext_->opName, "Data type of pse shift is %s, which is not supported.",
                  DataTypeToSerialString(pseShiftDataType).c_str());
        return ge::GRAPH_FAILED;
    }

    switch (pseShiftDataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            OP_CHECK_IF(
                (inputQType_ != ge::DT_INT8) && (inputQType_ != pseShiftDataType),
                OP_LOGE(ifaContext_->opName,
                    "Data type of pse is %s, which does not match data type of query: %s.",
                    DataTypeToSerialString(pseShiftDataType).c_str(),
                    DataTypeToSerialString(inputQType_).c_str()),
                return ge::GRAPH_FAILED);
            break;
        default:
            OP_LOGE(ifaContext_->opName, "Data type of pse %s is not currently supported.",
                      DataTypeToSerialString(pseShiftDataType).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessPseShift()
{
    // get pse shift data
    auto pseShiftInput = ifaContext_->pseShift.tensor;
    if (pseShiftInput == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(gqaKvNZFlag_, OP_LOGE(ifaContext_->opName, "PseShift is not support for IFA GQA with KV NZ!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->pseShift.desc == nullptr, OP_LOGE(ifaContext_->opName, "Desc of pse shift tensor is null."),
               return ge::GRAPH_FAILED);

    if (CheckPseShiftDataType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check pse shift shape (B/1, N, 1, Si)
    const gert::Shape pseShiftShape = pseShiftInput->GetStorageShape();
    uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
    OP_CHECK_IF(pseShiftDimNum != 4,
        OP_LOGE(ifaContext_->opName, "The input shape of pse shift must have 4 dims, current dim num is %u.",
            pseShiftDimNum),
        return GRAPH_FAILED);
    pseShiftBatch_ = pseShiftShape.GetDim(PSE_SHIFT_B);
    uint32_t pseShiftN = pseShiftShape.GetDim(PSE_SHIFT_N);
    uint32_t pseShiftS0 = pseShiftShape.GetDim(PSE_SHIFT_S0);
    pseShiftS1_ = pseShiftShape.GetDim(PSE_SHIFT_S1);

    OP_CHECK_IF(
        (pseShiftBatch_ != 1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0 != 1),
        OP_LOGE(ifaContext_->opName,
            "The shape of pse shift is (%u, %u, %u, %u), which does not match (B, N, 1, S) or (1, N, 1, S).",
            pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_),
        return ge::GRAPH_FAILED);

    if (pseShiftS1_ < seqSize_) {
        OP_LOGE(ifaContext_->opName,
                  "The shape of pse shift is (%u, %u, %u, %u), the 3rd dim S[%u] shouldn't be less than sMax[%u]."
                  "When Page Attention is enabled, sMax is maxBlockNumPerBatch * blockSize.",
                  pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_, pseShiftS1_, seqSize_);
        return GRAPH_FAILED;
    }

    pseShiftFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTreeSparseMask()
{
    // 仅支持MLA全量化，且要求s2 >= s1
    OP_CHECK_IF(socVersion_ != IfaSocVersion::SOC_ASCEND_910B,
        OP_LOGE(ifaContext_->opName, "Tree sparse9 is only supported in ASCEND_910B."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!(ropeFlag_ && quantFlag_),
        OP_LOGE(ifaContext_->opName, "Tree Sparse9 is only supported in MLA full quant situation."),
        return ge::GRAPH_FAILED);
        
    // tiling下沉场景，获取不到actualseqlen,不进行校验
    if (isWorkspace_) {
        return ge::GRAPH_SUCCESS;
    }

    // 补充s2 >= s1的拦截，在MLA全量化场景，必须开启PA(NZ)，所以actualSeqKV不进行累加
    if (inputLayout_ == IfaLayout::TND) {
        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        const int64_t *actualSeqKVTnd = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        int64_t qActSize = 0;
        int64_t kvActSize = 0;

        // s2=0 的 batch（入图padding或空tensor场景）不校验 s1<=s2
        for (int32_t b = 0; b < actualLenQDims_; b++) {
            kvActSize = actualSeqKVTnd[b];
            if (kvActSize == 0) {
                continue;
            }
            qActSize = (b == 0) ? actualSeqQTnd[0] : (actualSeqQTnd[b] - actualSeqQTnd[b - 1]);
            OP_CHECK_IF(qActSize > kvActSize,
                OP_LOGE(ifaContext_->opName,
                    "In MLA full quant situation, when sparse is %d, qSize(%ld) should less than or equal to kvSize(%ld).",
                    sparseMode_, qActSize, kvActSize),
            return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(qSeqSize_ > seqSize_,
            OP_LOGE(ifaContext_->opName,
                    "In MLA full quant situation, when sparse is %d, qSize(%ld) should less than or equal to kvSize(%ld).", 
                    sparseMode_, qSeqSize_, seqSize_),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTreeSparseMaskShape()
{
    // sparse9时需要传入Mask，在TND场景，传入∑s1²，一维矩阵
    // 非TND场景传入[B, S1, S1]
    auto maskShape = ifaContext_->attenMask.tensor;
    if (inputLayout_ == IfaLayout::TND) {
        // sparse9时需要传入Mask，在TND场景，传入∑s1²，一维矩阵
        OP_CHECK_IF(maskShape == nullptr,
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD need input attenMask when sparse = 9."), return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1U,
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD when sparse = 9, atten_mask tensor shape must be 1 dimension."),
            return ge::GRAPH_FAILED);
    
        if (!isWorkspace_) {
            OP_CHECK_IF(shape.GetDim(0) != qSeqSquareSum_,
                OP_LOGE(ifaContext_->opName, "TND/TND_NTD when sparse = 9, atten_mask tensor shape must be ∑s1²."),
                return ge::GRAPH_FAILED);
        }
        attenMaskFlag_ = true;
    } else {
        OP_CHECK_IF(maskShape == nullptr,
            OP_LOGE(ifaContext_->opName, "When sparse = 9, atten_mask should not be null."), return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 3U || shape.GetDim(0) != batchSize_ || shape.GetDim(1) != qSeqSize_ || shape.GetDim(2) != qSeqSize_ ,
            OP_LOGE(ifaContext_->opName, "BSH when sparse = 9, atten_mask tensor shape must be [B, S1, S1]."),
            return ge::GRAPH_FAILED);
        attenMaskFlag_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTndMaskShapeWithSparseMode()
{
    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    if (sparseMode_ == 0U) {  // sparse 0 模式只支持无mask
        OP_CHECK_IF(maskShape != nullptr,
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD only support noMask when sparse = 0."), return ge::GRAPH_FAILED);
        attenMaskFlag_ = false;
    } else if (sparseMode_ == 3U) {  // sparse 3 模式需要attenMask
        OP_CHECK_IF(maskShape == nullptr,
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD need input attenMask when sparse = 3."), return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 2U || shape.GetDim(0) != 2048 || shape.GetDim(1) != 2048,  // 2， 只支持2维shape
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD when sparse = 3, atten_mask tensor shape must be [2048, 2048]."),
            return ge::GRAPH_FAILED);
        attenMaskFlag_ = true;
    } else if (sparseMode_ == 9U) {
        if (ge::GRAPH_SUCCESS == CheckTreeSparseMask() &&
            ge::GRAPH_SUCCESS == CheckTreeSparseMaskShape()) {
            attenMaskFlag_ = true;
        }
    } else {
        OP_LOGE(ifaContext_->opName, "TND/TND_NTD only support sparse(%u) = 0/3/9.", sparseMode_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMaskShapeWithQSeq() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U && (static_cast<int32_t>(sparseMode_) != 3 && static_cast<int32_t>(sparseMode_) != 9)),
                OP_LOGE(ifaContext_->opName, "when queryS > 1, sparseMode(%d) only support 3/9 "
                    "in MLA when full quant situation.", static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && static_cast<int32_t>(sparseMode_) != 0),
                OP_LOGE(ifaContext_->opName, "when queryS = 1, sparseMode(%d) only support 0 "
                        "in MLA when full quant situation.", static_cast<int32_t>(sparseMode_)),
                    return ge::GRAPH_FAILED);
    } else if (antiQuantFlag_) {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U && static_cast<int32_t>(sparseMode_) != 3),
                OP_LOGE(ifaContext_->opName, "when queryS > 1, sparseMode(%d) only support 3 "
                    "in MLA when antiquant situation.", static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && static_cast<int32_t>(sparseMode_) != 0),
                OP_LOGE(ifaContext_->opName, "when queryS = 1, sparseMode(%d) only support 0 "
                        "in MLA when antiquant situation.", static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U &&
                (static_cast<int32_t>(sparseMode_) != 3 && static_cast<int32_t>(sparseMode_) != 4)), // 3 : rightDown, 4 : band
                OP_LOGE(ifaContext_->opName, "when queryS > 1, sparseMode(%d) support 3 or 4 "
                    "in MLA when no quant situation.", static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U &&
                (static_cast<int32_t>(sparseMode_) != 0 && static_cast<int32_t>(sparseMode_) != 4)),
                OP_LOGE(ifaContext_->opName, "when queryS = 1, sparseMode(%d) support 0 or 4 in MLA "
                    "when no quant situation.", static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((ropeFlag_ && (qSeqSize_ > 1U || static_cast<int32_t>(sparseMode_) == 4) && ifaContext_->attenMask.tensor == nullptr),
            OP_LOGE(ifaContext_->opName, "when queryS > 1 and sparse = 3, or sparse = 4, "
                "atten_mask tensor shape must be [2048, 2048] in MLA."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && ifaContext_->attenMask.tensor != nullptr && static_cast<int32_t>(sparseMode_) != 4),
            OP_LOGE(ifaContext_->opName, "when queryS = 1 and sparseMode is not 4, atten_mask tensor must be null in MLA."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAttenMaskShape()
{
    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    if (maskShape == nullptr) {
        attenMaskFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (maskShape->GetStorageShape().GetShapeSize() == 0) {
        attenMaskFlag_ = false;
        OP_LOGW(ifaContext_->opName, "atten_mask tensor exist, but atten_mask shape size is 0.");
        return ge::GRAPH_SUCCESS;
    }

    if (sparseMode_ == 9U && CheckTreeSparseMaskShape() == ge::GRAPH_SUCCESS) {
        attenMaskFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus IFATiling::ProcessAttenMask()
{
    // 与pfa保持一致，先判断sparsemode
    sparseMode_ = ifaContext_->sparseMode != nullptr ? *ifaContext_->sparseMode : 0;

    if (inputLayout_ == IfaLayout::TND) {
        return CheckTndMaskShapeWithSparseMode();
    }
    
    if (CheckMaskShapeWithQSeq() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    if (sparseMode_ == 9U && CheckTreeSparseMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckAttenMaskShape() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    ge::DataType attenMaskType = ifaContext_->attenMask.desc->GetDataType();
    if (attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_INT8 && attenMaskType != ge::DT_UINT8) {
        OP_LOGE(ifaContext_->opName, "not support atten_mask type %d, only support bool, int8 and uint8.",
                  attenMaskType);
        return ge::GRAPH_FAILED;
    }

    if ( (sparseMode_ == 3 || sparseMode_ == 4 || gqaMtpFlag_) && (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) ) { // 3 : rightDown, 4 : band
        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF((sparseMode_ != 0 && sparseMode_ != 1) && (shape.GetDimNum() != 2U || shape.GetDim(0) != 2048 || shape.GetDim(1) != 2048),  // 2， 只支持2维shape
            OP_LOGE(ifaContext_->opName, "when sparse = %d, atten_mask tensor shape must be [2048, 2048] in MLA.", static_cast<int32_t>(sparseMode_)),
            return ge::GRAPH_FAILED);

        attenMaskFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    
    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    uint32_t batchSizeOfMask = static_cast<uint32_t>(maskShape->GetStorageShape().GetDim(0));
    if (batchSizeOfMask != batchSize_) {
        OP_LOGE(ifaContext_->opName, "batchSize[%u] of atten_mask must be equal to batchSize[%u] of query.",
                  batchSizeOfMask, batchSize_);
        return ge::GRAPH_FAILED;
    }

    auto dimNumOfMask = maskShape->GetStorageShape().GetDimNum();
    attenMaskSize_ = static_cast<uint32_t>(maskShape->GetStorageShape().GetDim(dimNumOfMask - 1));
    uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
    if (attenMaskSize_ < minAttenMaskSize) {
        OP_LOGE(ifaContext_->opName, "s Size[%u] of atten_mask must be greater than or equal to sMax[%u].",
                  attenMaskSize_, minAttenMaskSize);
        return ge::GRAPH_FAILED;
    }

    attenMaskFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeDesc() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() !=  ge::DT_BF16),
            OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, queryRope [%d] must be bfloat16.",
                        ifaContext_->queryRope.desc->GetDataType()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() != ifaContext_->query.desc->GetDataType()),
            OP_LOGE(ifaContext_->opName, "queryRope [%d] and query [%d] must have same dType",
                        ifaContext_->queryRope.desc->GetDataType(), ifaContext_->query.desc->GetDataType()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBsndLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF(qRopeShape.GetDim(0) != qShape.GetDim(0),
        OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'B'", qRopeShape.GetDim(0),
            qShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qRopeShape.GetDim(1) != qShape.GetDim(1),
        OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'S'", qRopeShape.GetDim(1),
            qShape.GetDim(1)),
        return ge::GRAPH_FAILED);

    if (qRopeShape.GetDimNum() != 4U) {
        headDimRope_ = qRopeShape.GetDim(2) / numHeads_; // 2: H
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(qRopeShape.GetDim(2) != qShape.GetDim(2), // 2: N
        OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'N'", qRopeShape.GetDim(2),
            qShape.GetDim(2)),
        return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBnsdLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF((qRopeShape.GetDim(0) != qShape.GetDim(0)) ||
                (qRopeShape.GetDim(1) != qShape.GetDim(1)) ||
                (qRopeShape.GetDim(2) != qShape.GetDim(2)), // 2: S
        OP_LOGE(ifaContext_->opName,
            "queryRope [%ld, %ld, %ld, %ld] and query [%ld, %ld, %ld, %ld] must have same 'B', 'N' and 'S'",
                qRopeShape.GetDim(0), qRopeShape.GetDim(1), qRopeShape.GetDim(2), qRopeShape.GetDim(3),
                qShape.GetDim(0), qShape.GetDim(1), qShape.GetDim(2),qShape.GetDim(3)),
        return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeNzLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF ((qRopeShape.GetDim(0) != qShape.GetDim(0)) || (qRopeShape.GetDim(1) != qShape.GetDim(1)),
        OP_LOGE(ifaContext_->opName, "queryRope [%ld, %ld, %ld] and query [%ld, %ld, %ld] must have same 'T' and 'N'.",
            qRopeShape.GetDim(0), qRopeShape.GetDim(1), qRopeShape.GetDim(2),
            qShape.GetDim(0), qShape.GetDim(1), qShape.GetDim(2)),
        return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(2); // 2:D
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus IFATiling::CheckMlaQueryRope()
{
    if (CheckMlaQueryRopeDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto qRopeShape = ifaContext_->queryRope.tensor->GetStorageShape();

    auto qShape = ifaContext_->query.shape->GetStorageShape();
    OP_CHECK_IF(qRopeShape.GetDimNum() != qShape.GetDimNum(), OP_LOGE(ifaContext_->opName,
        "queryRope(%lu) and query(%lu) dimensions must be the same.", qRopeShape.GetDimNum(), qShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        return CheckMlaQueryRopeBsndLayout(qRopeShape, qShape);
    } else if (inputLayout_ == IfaLayout::BNSD) {
        return CheckMlaQueryRopeBnsdLayout(qRopeShape, qShape);
    } else if (inputLayout_ == IfaLayout::TND) {
        return CheckMlaQueryRopeNzLayout(qRopeShape, qShape);
    }

    return ge::GRAPH_SUCCESS; // never here
}

ge::graphStatus IFATiling::CheckMlaAttrs() const
{
    OP_CHECK_IF(numKvHeads_ != 1U,
        OP_LOGE(ifaContext_->opName, "the key/value's heads num(%u) only support 1 in MLA.", numKvHeads_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(nNumOfQInOneGroup_ !=1U && nNumOfQInOneGroup_ !=2U && nNumOfQInOneGroup_ !=4U &&
                nNumOfQInOneGroup_ !=8U && nNumOfQInOneGroup_ !=16U && nNumOfQInOneGroup_ !=32U &&
                nNumOfQInOneGroup_ != 64U && nNumOfQInOneGroup_ != 128U,
        OP_LOGE(ifaContext_->opName, "the query's heads num divided by the key/value's heads num = %u, MLA only support {1, 2, 4, 8, 16, 32, 64, 128}.", nNumOfQInOneGroup_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDim_ != 512U,
        OP_LOGE(ifaContext_->opName, "queryD(%u) only support 512 in MLA.", headDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDimRope_ != 64U,
        OP_LOGE(ifaContext_->opName, "headDimRope(%u) only support 64 in MLA.", headDimRope_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputKvLayout_ == IfaLayout::NZ && inputLayout_ != IfaLayout::BSH_BSND && inputLayout_ != IfaLayout::TND),
        OP_LOGE(ifaContext_->opName, "When kv is NZ, layout only support {BSH, BSND, TND, BSH_NBSD, BSND_NBSD, TND_NTD}."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeDesc() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ifaContext_->keyRope.desc->GetDataType() != ifaContext_->queryRope.desc->GetDataType()),
            OP_LOGE(ifaContext_->opName,
                "when the dtype of query is int8, keyRope [%d] and queryRope [%d] must have same dType.",
                ifaContext_->keyRope.desc->GetDataType(), ifaContext_->queryRope.desc->GetDataType()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ifaContext_->keyRope.desc->GetDataType() != ifaContext_->key.desc->GetDataType()),
            OP_LOGE(ifaContext_->opName, "keyRope [%d] and key [%d] must have same dType",
                ifaContext_->keyRope.desc->GetDataType(), ifaContext_->key.desc->GetDataType()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeShapeMatch(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyRopeShape.GetShapeSize() == 0,
        OP_LOGE(ifaContext_->opName, "empty keyRope Tensor is not supported in MLA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.GetShapeSize() == 0,
        OP_LOGE(ifaContext_->opName, "empty key Tensor is not supported in MLA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(0) != keyShape.GetDim(0),
        OP_LOGE(ifaContext_->opName, "KeyRope(%ld) and Key(%ld) must have same BlockNum", keyRopeShape.GetDim(0),
                  keyShape.GetDim(0)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyRopeShape.GetDimNum() != keyShape.GetDimNum(), OP_LOGE(ifaContext_->opName,
        "KeyRope(%lu) and Key(%lu) dimensions must be the same.", keyRopeShape.GetDimNum(), keyShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBsndLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyShape.GetDim(1) != blockSize_ || keyShape.GetDim(2) != (numKvHeads_ * headDim_),
        OP_LOGE(ifaContext_->opName, "The dim of KeyShape is 3, KeyShape [%ld, %ld, %ld] must be [%ld, %u, %u] in MLA.",
            keyShape.GetDim(0), keyShape.GetDim(1), keyShape.GetDim(2),
            keyShape.GetDim(0), blockSize_, (numKvHeads_ * headDim_)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(1) != blockSize_ || keyRopeShape.GetDim(2) != (numKvHeads_ * headDimRope_),
        OP_LOGE(ifaContext_->opName, "The dim of KeyShape is 3, KeyRopeShape [%ld, %ld, %ld] must be [%ld, %u, %u].",
            keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2),
            keyRopeShape.GetDim(0), blockSize_, (numKvHeads_ * headDimRope_)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBnsdLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(
        keyShape.GetDim(1) != numKvHeads_ || keyShape.GetDim(2) != blockSize_ || keyShape.GetDim(3) != headDim_,
        OP_LOGE(ifaContext_->opName,
            "The dim of KeyShape is 4, KeyShape [%ld, %ld, %ld, %ld] must be [%ld, %u, %u, %u] in MLA.",
            keyShape.GetDim(0), keyShape.GetDim(1), keyShape.GetDim(2), keyShape.GetDim(3),
            keyShape.GetDim(0), numKvHeads_, blockSize_, headDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        keyRopeShape.GetDim(1) != numKvHeads_ ||
        keyRopeShape.GetDim(2) != blockSize_ ||
        keyRopeShape.GetDim(3) != headDimRope_,
        OP_LOGE(ifaContext_->opName,
            "The dim of KeyShape is 4, KeyRopeShape [%ld, %ld, %ld, %ld] must be [%ld, %u, %u, %u].",
            keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2), keyRopeShape.GetDim(3),
            keyRopeShape.GetDim(0), numKvHeads_, blockSize_, headDimRope_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeNzLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    size_t kvTypeSize = static_cast<size_t>(GetTypeSize(ifaContext_->key.desc->GetDataType()));
    size_t kvRopeTypeSize = static_cast<size_t>(GetTypeSize(ifaContext_->keyRope.desc->GetDataType()));
    OP_CHECK_IF(
        static_cast<size_t>(keyShape.GetDim(1)) != numKvHeads_ ||
        static_cast<size_t>(keyShape.GetDim(2)) != headDim_ / (32 / kvTypeSize) ||
        static_cast<size_t>(keyShape.GetDim(3)) != blockSize_ ||
        static_cast<size_t>(keyShape.GetDim(4)) != 32 / kvTypeSize,
        OP_LOGE(ifaContext_->opName,
            "KvLayout is NZ and KvDtype is %s, keyShape [%ld, %ld, %ld, %ld, %ld] must be [%ld, %u, %lu, %u, %lu] in MLA.",
            DataTypeToSerialString(ifaContext_->key.desc->GetDataType()).c_str(), keyShape.GetDim(0), keyShape.GetDim(1),
            keyShape.GetDim(2), keyShape.GetDim(3), keyShape.GetDim(4), keyShape.GetDim(0), numKvHeads_,
            (headDim_ / (32 / kvTypeSize)), blockSize_, (32 / kvTypeSize)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        static_cast<size_t>(keyRopeShape.GetDim(1)) != numKvHeads_ ||
        static_cast<size_t>(keyRopeShape.GetDim(2)) != headDimRope_ / (32U / kvRopeTypeSize) ||
        static_cast<size_t>(keyRopeShape.GetDim(3)) != blockSize_ ||
        static_cast<size_t>(keyRopeShape.GetDim(4)) != 32U / kvRopeTypeSize,
        OP_LOGE(ifaContext_->opName,
            "KvLayout is NZ, keyRopeShape [%ld, %ld, %ld, %ld, %ld] must be [%ld, %u, %lu, %u, %lu].",
            keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2), keyRopeShape.GetDim(3),
            keyRopeShape.GetDim(4), keyRopeShape.GetDim(0), numKvHeads_, (headDimRope_ / (32U / kvRopeTypeSize)),
            blockSize_, (32U / kvRopeTypeSize)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRope() const
{
    if (!pageAttentionFlag_) {
        OP_LOGE(ifaContext_->opName, "only PageAttention KvCache is supported in MLA mode");
        return ge::GRAPH_FAILED;
    }
    if (CheckMlaKeyRopeDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape keyRopeShape = ifaContext_->keyRope.tensor->GetStorageShape();
    const gert::Shape keyShape = ifaContext_->key.shape->GetStorageShape();
    if (CheckMlaKeyRopeShapeMatch(keyRopeShape, keyShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    if (inputKvLayout_ == IfaLayout::BSH_BSND) { // PA Dims=3
        return CheckMlaKeyRopeBsndLayout(keyRopeShape, keyShape);
    } else if (inputKvLayout_ == IfaLayout::BNSD) { // Dims = 4
        return CheckMlaKeyRopeBnsdLayout(keyRopeShape, keyShape);
    } else if (inputKvLayout_ == IfaLayout::NZ) { // Dims = 5   [B， N， headDim / (32 / sizeof(KV)), block_size 32/sizeof(KV)]
        return CheckMlaKeyRopeNzLayout(keyRopeShape, keyShape);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaMisc() const
{
    OP_CHECK_IF(antiQuantFlag_,
        OP_LOGE(ifaContext_->opName, "antiquant is not supported in MLA."),
        return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("MLA") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (antiQuantFlag_) {
        OP_CHECK_IF(inputQType_ != ge::DT_BF16,
            OP_LOGE(ifaContext_->opName, "when antiquant is enabled, input Q dtype must be bf16 in MLA."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputLayout_ == IfaLayout::BNSD,
            OP_LOGE(ifaContext_->opName,
                "when antiquant is enabled, BNSD or BNSD_NBSD input layout are not supported in MLA."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(!pageAttentionFlag_,
            OP_LOGE(ifaContext_->opName, "when antiquant is enabled, Page Attention must be enabled in MLA."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputKvLayout_ != IfaLayout::NZ,
            OP_LOGE(ifaContext_->opName, "when antiquant is enabled, KvCache layout must be NZ in MLA."),
            return ge::GRAPH_FAILED);
    }

    // Kv NZ blocksize：128
    OP_CHECK_IF(inputKvLayout_ == IfaLayout::NZ && blockSize_ != 128U,
        OP_LOGE(ifaContext_->opName, "blockSize(%u), MLA only support {128} when KvCache layout is NZ.", blockSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ != 16U && blockSize_ != 128U,
        OP_LOGE(ifaContext_->opName, "blockSize(%u), MLA only support {16, 128} when KvCache layout is ND.", blockSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckDefaultMisc(std::string scene) const
{
    OP_CHECK_IF(pseShiftFlag_,
        OP_LOGE(ifaContext_->opName, "PseShift is not supported in %s.", scene.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!batchContinuousFlag_,
        OP_LOGE(ifaContext_->opName, "Kvcache must be continues in %s.", scene.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(sysPrefixFlag_,
        OP_LOGE(ifaContext_->opName, "SysPrefix is not supported in %s.", scene.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputType_ == ge::DT_INT8,
        OP_LOGE(ifaContext_->opName, "PostQuant is not supported in %s.", scene.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_,
        OP_LOGE(ifaContext_->opName, "kvPaddingSizeFlag_ is not supported in %s.", scene.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaTensor() const
{
    if (CheckGqaIOTensor() != ge::GRAPH_SUCCESS || CheckGqaAntiquantTensor() != ge::GRAPH_SUCCESS ||
        CheckGqaBlockTable() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // other tensors should be empty
    return CheckGqaTensorEmpty();
}

ge::graphStatus IFATiling::CheckGqaIOTensor() const
{
    if (!antiQuantFlag_) {
        OP_LOGE(ifaContext_->opName, "IFA GQA with KV NZ only support antiquant!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ifaContext_->query.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, query type should be BFLOAT16, but now it's %d!",
        ifaContext_->query.desc->GetDataType()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.desc->GetDataType() != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, key type should be INT8, but now it's %d!",
        ifaContext_->key.desc->GetDataType()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.desc->GetDataType() != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, value type should be INT8, but now it's %d!",
        ifaContext_->value.desc->GetDataType()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, attenOut type should be BFLOAT16, but now it's %d!",
        ifaContext_->attenOut.desc->GetDataType()), return ge::GRAPH_FAILED);
    constexpr uint32_t kvNzDimNum = 5U;
    if (ifaContext_->key.shape->GetStorageShape().GetDimNum() == kvNzDimNum) {  // NZ
        std::string kShapeStr = GetTensorDimString(ifaContext_->key.shape->GetStorageShape());
        std::string vShapeStr = GetTensorDimString(ifaContext_->value.shape->GetStorageShape());
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->key.shape->GetStorageShape()),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "The shape of K%s should not have 0 dim in GQA with KV NZ!", kShapeStr.c_str()), 
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->value.shape->GetStorageShape()),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "The shape of V%s should not have 0 dim in GQA with KV NZ!", vShapeStr.c_str()), 
            return ge::GRAPH_FAILED);
        uint32_t expected3rdDim = headDim_ / 32U;
        uint32_t expected5thDim = 32U;  // 32U / sizeof(int8)
        bool isK3rdDimValid = expected3rdDim == ifaContext_->key.shape->GetStorageShape()[2];
        bool isK5thDimValid =  expected5thDim == ifaContext_->key.shape->GetStorageShape()[4];
        OP_CHECK_IF(!isK3rdDimValid || !isK5thDimValid,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "The shape of K%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                kShapeStr.c_str(), ifaContext_->key.shape->GetStorageShape()[0],
                ifaContext_->key.shape->GetStorageShape()[1], expected3rdDim, ifaContext_->key.shape->GetStorageShape()[3], expected5thDim),
            return ge::GRAPH_FAILED);
        bool isV3rdDimValid = expected3rdDim == ifaContext_->value.shape->GetStorageShape()[2];
        bool isV5thDimValid =  expected5thDim == ifaContext_->value.shape->GetStorageShape()[4];
        OP_CHECK_IF(!isV3rdDimValid || !isV5thDimValid,
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "The shape of V%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                kShapeStr.c_str(), ifaContext_->value.shape->GetStorageShape()[0],
                ifaContext_->value.shape->GetStorageShape()[1], expected3rdDim, ifaContext_->value.shape->GetStorageShape()[3], expected5thDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiquantTensor() const
{
    OP_CHECK_IF((ifaContext_->keyAntiquantScale.desc == nullptr || ifaContext_->valueAntiquantScale.desc == nullptr),
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, the key/value's dequant scale desc should not be null!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((ifaContext_->keyAntiquantScale.tensor == nullptr || ifaContext_->valueAntiquantScale.tensor == nullptr),
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, the key/value's dequant scale tensor should not be null!"),
        return ge::GRAPH_FAILED);

    if (CheckGqaAntiquantType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiquantType() const
{
    // check type
    if (antiquantMode_ == PER_CHANNEL_MODE) {
        OP_CHECK_IF(ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ perchannel antiquant, the key's dequant scale type should be BFLOAT16, but now it's %s!",
        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ perchannel antiquant, the value's dequant scale type should be BFLOAT16, but now it's %s!",
        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    } else if (antiquantMode_ == PER_TOKEN_MODE) {
        OP_CHECK_IF(ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ pertoken antiquant, the key's dequant scale type should be DT_FLOAT, but now it's %s!",
        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName, "In IFA GQA with KV NZ pertoken antiquant, the value's dequant scale type should be DT_FLOAT, but now it's %s!",
        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaBlockTable() const
{
    OP_CHECK_IF(!pageAttentionFlag_,
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, blocktable should not be null!"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaTensorEmpty() const
{
    OP_CHECK_IF(ifaContext_->pseShift.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "PseShift not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSeqLengthsQ.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "The query's actual sequence lengths not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "deqScale1 not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "quantScale1 not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "deqScale2 not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "the output's dequant scale not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "the output's dequant offset not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "antiquantScale not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantOffset.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "antiquantOffset not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryPaddingSize.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "queryPaddingSize not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "kvPaddingSize not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyAntiquantOffset.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "the key's dequant offset not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueAntiquantOffset.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "the value's dequant offset not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "keySharedPrefix not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "valueSharedPrefix not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSharedPrefixLen.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryRope.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "queryRope not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRope.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "keyRope not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRopeAntiquantScale.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "the key_rope's dequant scale not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->dequantScaleQuery.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "The query's dequant scale not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->lseOut.desc != nullptr,
        OP_LOGE(ifaContext_->opName, "lseOut not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaSeqSize() const
{
    std::string layout(ifaContext_->layOut);
    if (layout == "TND" && isWorkspace_) { // tiling下沉场景
        return ge::GRAPH_SUCCESS;
    }
    if (qSeqSize_ == 1U) {
        if (layout == "TND") { // TND MTP场景qSeqSize有可能为1，也需要支持sparseMode3
            OP_CHECK_IF((sparseMode_ != 0 && sparseMode_ != 3),
                OP_LOGE(ifaContext_->opName,"SparseMode[%d] only support 0 or 3 in IFA GQA with KV NZ when query is 1 and layout is TND",
                static_cast<int32_t>(sparseMode_)), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((sparseMode_ != 0),
                OP_LOGE(ifaContext_->opName, "SparseMode[%d] only support 0 in IFA GQA with KV NZ when query is 1",
                static_cast<int32_t>(sparseMode_)), return ge::GRAPH_FAILED);
        }
    } else if (qSeqSize_ > 1U) {
        OP_CHECK_IF((sparseMode_ != 3),  // when qs bigger than 1, sparse mode only support 3
            OP_LOGE(ifaContext_->opName, "SparseMode[%d] only support 3 in IFA GQA with KV NZ when query S is bigger than 1",
            static_cast<int32_t>(sparseMode_)), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(ifaContext_->opName, "Invalid query S %u", qSeqSize_);
        return ge::GRAPH_FAILED;
    }
    if (sparseMode_ == 0) {
        OP_CHECK_IF(ifaContext_->attenMask.desc != nullptr || ifaContext_->attenMask.tensor != nullptr,
            OP_LOGE(ifaContext_->opName, "attenMask should be null for IFA GQA with KV NZ when sparseMode 0!"), return ge::GRAPH_FAILED);
    } else {
        auto attenMaskShape = ifaContext_->attenMask.desc;
        auto attenMaskTensor = ifaContext_->attenMask.tensor;
        OP_CHECK_IF(attenMaskShape == nullptr || attenMaskTensor == nullptr,  // mask shape: 2048*2048
            OP_LOGE(ifaContext_->opName, "When sparseMode 3, attenMask shape for IFA GQA with KV NZ should not be null!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDimNum() != 2U,  // mask shape: 2048*2048
            OP_LOGE(ifaContext_->opName, "The dim of attenMask shape[%lu] is not expected. "
                "Expect 2 when sparseMode 3 in GQA KV NZ.", attenMaskTensor->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDim(0) != 2048U || attenMaskTensor->GetStorageShape().GetDim(1) != 2048U,  // mask shape: 2048*2048
            OP_LOGE(ifaContext_->opName, "The shape of attenMask shape[%ld, %ld] is not expected. "
                "Expect [2048, 2048] when sparseMode 3 in GQA KV NZ.",
                attenMaskTensor->GetStorageShape().GetDim(0), attenMaskTensor->GetStorageShape().GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAttribute() const
{
    OP_CHECK_IF(headDim_ != 128U,        // 128: D dim
        OP_LOGE(ifaContext_->opName, "headDim = %u, IFA GQA with KV NZ only support 128.", headDim_), return ge::GRAPH_FAILED);

    std::string layout(ifaContext_->layOut);
    OP_CHECK_IF(layout != "BSH" && layout != "BSND" && layout != "BNSD" && layout != "TND",
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, only BSH, BSND, BNSD and TND layout are supported, but now it's %s", layout.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ifaContext_->keyAntiquantMode == nullptr || ifaContext_->valueAntiquantMode == nullptr,
        OP_LOGE(ifaContext_->opName, "the key's quant mode or the value's quant mode is null!"), return ge::GRAPH_FAILED);
    int64_t keyAntiquantMode = ifaContext_->keyAntiquantMode != nullptr ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueAntiquantMode = ifaContext_->valueAntiquantMode != nullptr ? *ifaContext_->valueAntiquantMode : 0;
    OP_CHECK_IF(antiquantMode_ == PER_CHANNEL_MODE && (keyAntiquantMode != 0 || valueAntiquantMode != 0),
        OP_LOGE(ifaContext_->opName, "the key's quant mode(%ld) and the value's quant mode(%ld) should be 0 in GQA perchannel antiquant.",
        keyAntiquantMode, valueAntiquantMode), return ge::GRAPH_FAILED);

    OP_CHECK_IF(antiquantMode_ == PER_TOKEN_MODE && (keyAntiquantMode != 1 || valueAntiquantMode != 1),
        OP_LOGE(ifaContext_->opName, "the key's quant mode(%ld) and the value's quant mode(%ld) should be 1 in GQA pertoken antiquant.",
        keyAntiquantMode, valueAntiquantMode), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ != 128U && blockSize_ != 512U,  // block size only support 128 and 512 in GQA
        OP_LOGE(ifaContext_->opName, "blockSize(%u) only support 128 and 512 in GQA.", blockSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF((innerPrecise_ != IFA_HIGH_PERFORMANCE),
        OP_LOGE(ifaContext_->opName, "precision mode[%u] only support 1(hign performance) in GQA antiquant with KV NZ", innerPrecise_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckGqaSeqSize() != ge::GRAPH_SUCCESS,
        OP_LOGE(ifaContext_->opName, "Invalid query S %u with sparseMode %d and attenMask", qSeqSize_,
        static_cast<int32_t>(sparseMode_)), return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("IFA GQA with KV NZ") != ge::GRAPH_SUCCESS|| CheckGqaDefault() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaDefault() const
{
    int64_t queryQuantMode = ifaContext_->queryQuantMode != nullptr ? *ifaContext_->queryQuantMode : 0;
    OP_CHECK_IF(queryQuantMode != DEQUANT_PER_CHANNEL_MODE,
        OP_LOGE(ifaContext_->opName, "The query's quant mode(%ld) for IFA GQA with KV NZ should be default value %u!", queryQuantMode, DEQUANT_PER_CHANNEL_MODE),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling