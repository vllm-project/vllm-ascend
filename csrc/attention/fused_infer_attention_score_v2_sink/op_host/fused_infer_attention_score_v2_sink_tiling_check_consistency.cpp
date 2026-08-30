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
 * \file fused_infer_attention_score_v2_sink_tiling_check_consistency.cpp
 * \brief
 */

#include <map>
#include <string>
#include <utility>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_v2_sink_tiling_check.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
namespace optiling {
void FiaTilingCheck::SetFiaShapeCompare()
{
    queryShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(
        opParamInfo_.query.shape->GetStorageShape(), qLayout_, QUERY_NAME, opName_);
    keyShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(kCache_[0]->GetShape(), kvLayout_, KEY_NAME, opName_);
    valueShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(vCache_[0]->GetShape(), kvLayout_, VALUE_NAME, opName_);
    attenOutShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(
        opParamInfo_.attenOut.shape->GetStorageShape(), outLayout_, ATTEN_OUT_NAME, opName_);
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        queryRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(
            opParamInfo_.queryRope.tensor->GetStorageShape(), qLayout_, QUERY_ROPE_NAME, opName_);
        keyRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(
            opParamInfo_.keyRope.tensor->GetOriginShape(), kvLayout_, KEY_ROPE_NAME, opName_);
    }
}

ge::graphStatus FiaTilingCheck::CheckQAndQRopeDType() const
{
    if (opParamInfo_.query.desc->GetDataType() != inputQType_) {
        OPS_LOG_E(opName_,
                  "%s's dtype is %s, it should be %s.",
                  QUERY_NAME.c_str(),
                  FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
                  FusedDataTypeToSerialString(inputQType_).c_str());
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.queryRope.desc->GetDataType() != inputQRopeType_) {
            OPS_LOG_E(opName_,
                      "%s's dtype is %s, it should be %s.",
                      QUERY_NAME.c_str(),
                      FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
                      FusedDataTypeToSerialString(inputQRopeType_).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckQShape() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckQRopeShape() const
{
    // rope分离模式时queryRope Tensor才存在
    if (ropeMode_ != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }

    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(ropeHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    return queryRopeShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckQAndQRopeShape() const
{
    if (ge::GRAPH_SUCCESS != CheckQShape() || ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckQAndQRope() const
{
    if (ge::GRAPH_SUCCESS != CheckQAndQRopeDType() || ge::GRAPH_SUCCESS != CheckQAndQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVDType() const
{
    if (opParamInfo_.key.desc->GetDataType() != inputKvType_) {
        OPS_LOG_E(opName_,
                  "%s's dtype is %s, it should be %s.",
                  KEY_NAME.c_str(),
                  FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str(),
                  FusedDataTypeToSerialString(inputKvType_).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.value.desc->GetDataType() != inputKvType_) {
        OPS_LOG_E(opName_,
                  "%s's dtype is %s, it should be %s.",
                  VALUE_NAME.c_str(),
                  FusedDataTypeToSerialString(opParamInfo_.value.desc->GetDataType()).c_str(),
                  FusedDataTypeToSerialString(inputKvType_).c_str());
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.keyRope.desc->GetDataType() != inputKRopeType_) {
            OPS_LOG_E(opName_,
                      "%s's dtype is %s, it should be %s.",
                      KEY_ROPE_NAME.c_str(),
                      FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
                      FusedDataTypeToSerialString(inputKRopeType_).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShapeForBatchContinuous() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n2Size_);
    shapeParams.S = s2Size_;
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    shapeParams.T = static_cast<int64_t>(kTSize_);
    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(vHeadDim_);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        shapeParams.D = static_cast<int64_t>(ropeHeadDim_);
        if (keyRopeShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShapeForTensorList() const
{
    FiaTilingShapeCompareParam keyShapeParams;
    keyShapeParams.B = 1;
    keyShapeParams.N = static_cast<int64_t>(n2Size_);
    keyShapeParams.S = s2Size_;
    keyShapeParams.D = static_cast<int64_t>(qkHeadDim_);
    keyShapeParams.T = static_cast<int64_t>(qTSize_);
    keyShapeParams.compareTypeMap = {{FiaAxis::S, FiaCompareType::LESS_EQUAL}};

    FiaTilingShapeCompareParam valueShapeParams = keyShapeParams;
    valueShapeParams.D = static_cast<int64_t>(vHeadDim_);

    for (uint32_t i = 0; i < bSize_; i++) {
        auto keyShapeCmp =
            std::make_shared<FiaTilingShapeCompare>(kCache_[i]->GetStorageShape(), kvLayout_, KEY_NAME, opName_);
        if (keyShapeCmp->CompareShape(keyShapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto valueShapeCmp =
            std::make_shared<FiaTilingShapeCompare>(vCache_[i]->GetStorageShape(), kvLayout_, VALUE_NAME, opName_);
        if (valueShapeCmp->CompareShape(valueShapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t FiaTilingCheck::GetTypeSize(ge::DataType dtype) const
{
    constexpr uint32_t NUM_BYTES_FLOAT = 4;
    constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
    constexpr uint32_t NUM_BYTES_BF16 = 2;
    constexpr uint32_t NUM_BYTES_BOOL = 1;
    constexpr uint32_t NUM_BYTES_INT8 = 1;

    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT:
            typeSize = NUM_BYTES_FLOAT;
            break;
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        case ge::DT_BOOL:
            typeSize = NUM_BYTES_BOOL;
            break;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT4:
            typeSize = NUM_BYTES_INT8;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
            break;
    }
    return typeSize;
}

ge::graphStatus FiaTilingCheck::CheckBlockTable() const
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_CHECK(opParamInfo_.blockTable.tensor->GetStorageShape().GetShapeSize() == 0,
              OPS_LOG_E(opName_, "%s shape size is zero.", BLOCK_TABLE_NAME.c_str()),
              return ge::GRAPH_FAILED);

    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OPS_CHECK(qLayout_ == FiaLayout::TND && blockTableBatch != bSize_,
              OPS_LOG_E(opName_,
                        "when %s's layout is TND, %s's first dimension(%u) should be equal to batch size(%u)",
                        QUERY_NAME.c_str(),
                        BLOCK_TABLE_NAME.c_str(),
                        blockTableBatch,
                        bSize_),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShapeForPageAttention() const
{
    uint32_t kvBlockElemNum = 32 / GetTypeSize(inputKvType_);
    if (blockSize_ % static_cast<int32_t>(kvBlockElemNum) != 0) {
        OPS_LOG_E(
            opName_,
            "when kv_dtype is %s, 32 / sizeof(kv_dtype) is %u, block_size %% (32 / sizeof(kv_dtype)) should be 0.",
            FusedDataTypeToSerialString(inputKvType_).c_str(),
            kvBlockElemNum);
        return ge::GRAPH_FAILED;
    }

    // key
    int64_t blockNum = keyShapeCmp_->shape_.GetDim(0);
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(blockNum);
    shapeParams.N = static_cast<int64_t>(n2Size_);
    shapeParams.Bs = static_cast<int64_t>(blockSize_);
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    shapeParams.D0 = static_cast<int64_t>(kvBlockElemNum);
    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // value
    shapeParams.D = static_cast<int64_t>(vHeadDim_);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // key rope
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        uint32_t kRopeBlockElemNum = 32 / GetTypeSize(inputKRopeType_);
        if (blockSize_ % static_cast<int32_t>(kRopeBlockElemNum) != 0) {
            OPS_LOG_E(opName_,
                      "when key_rope_dtype is %s, 32 / sizeof(key_rope_dtype) is %u, block_size %% (32 / "
                      "sizeof(key_rope_dtype)) should be 0.",
                      FusedDataTypeToSerialString(inputKRopeType_).c_str(),
                      kRopeBlockElemNum);
            return ge::GRAPH_FAILED;
        }
        shapeParams.D = static_cast<int64_t>(ropeHeadDim_);
        shapeParams.D0 = static_cast<int64_t>(kRopeBlockElemNum);
        if (keyRopeShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShape() const
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        return CheckKVShapeForTensorList();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OPS_LOG_E(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus FiaTilingCheck::CheckKV() const
{
    if (ge::GRAPH_SUCCESS != CheckKVDType() || ge::GRAPH_SUCCESS != CheckKVShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckAttenOut() const
{
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n1Size_);
    shapeParams.S = static_cast<int64_t>(s1Size_);
    shapeParams.D = static_cast<int64_t>(vHeadDim_);
    shapeParams.T = static_cast<int64_t>(qTSize_);
    if (attenOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckActualSeqLensQ() const
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (qLayout_ == FiaLayout::TND) {
        if (actualSeqLengthsQSize_ != bSize_ && actualSeqLengthsQSize_ != 1U) {
            OPS_LOG_E(opName_,
                      "%s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                      ACTUAL_SEQ_Q_LEN_NAME.c_str(),
                      actualSeqLengthsQSize_,
                      bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsQSize_ < bSize_ && actualSeqLengthsQSize_ != 1U) {
            OPS_LOG_E(opName_,
                      "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                      ACTUAL_SEQ_Q_LEN_NAME.c_str(),
                      actualSeqLengthsQSize_,
                      bSize_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckActualSeqLensKv() const
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (kvLayout_ == FiaLayout::TND) {
        if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr && actualSeqLengthsKvSize_ != actualSeqLengthsQSize_) {
            OPS_LOG_E(opName_,
                      "%s shape size is %u, it should be equal to %s shape size(%u).",
                      ACTUAL_SEQ_KV_LEN_NAME.c_str(),
                      actualSeqLengthsKvSize_,
                      ACTUAL_SEQ_Q_LEN_NAME.c_str(),
                      actualSeqLengthsQSize_);
            return ge::GRAPH_FAILED;
        }
        if (actualSeqLengthsKvSize_ != bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OPS_LOG_E(opName_,
                      "%s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                      ACTUAL_SEQ_KV_LEN_NAME.c_str(),
                      actualSeqLengthsKvSize_,
                      bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsKvSize_ < bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OPS_LOG_E(opName_,
                      "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                      ACTUAL_SEQ_KV_LEN_NAME.c_str(),
                      actualSeqLengthsKvSize_,
                      bSize_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckPseShift()
{
    if (opParamInfo_.pseShift.tensor == nullptr || opParamInfo_.pseShift.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    } else {
        OPS_LOG_E(opName_, "FiaSink not support Pse.");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMlaSinkNumber()
{
    int64_t sinkNumber = *opParamInfo_.sinkNumber;
    int32_t sparseMode = *opParamInfo_.sparseMode;
    // D=128时，走GQA模板
    if (ropeMode_ != RopeMode::ROPE_SPLIT || vHeadDim_ != 512U) {
        return ge::GRAPH_SUCCESS;
    }
    if (sparseMode != SPARSE_MODE_BAND) {
        OPS_LOG_E(opName_, "sparseMode only support 4 when sinkNumber(%ld) exist and D is 512", sinkNumber);
        return ge::GRAPH_FAILED;
    }
    // MLA sink 支持 TND、TND_NTD layout，继承GQA的校验
    // tiling下沉场景
    if (fiaInfo_.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

/*
1、sinkNumber范围为 0~512
2、sinkNumber只支持偶数，不支持奇数
3、sink功能只支持TND场景
4、sparsemode=1 || (sparsemode=0 && attentionMask != nullptr)，不允许携带sinknumber;
5、TND场景下，sink_num <= actual_seq_len_kv
6、sink_num功能和sink功能互斥，不支持两种sink功能共存，在tiling进行校验报错
*/
ge::graphStatus FiaTilingCheck::CheckSinkNumber()
{
    int64_t sinkNumber = *opParamInfo_.sinkNumber;
    int32_t sparseMode = *opParamInfo_.sparseMode;
    int64_t maxSinkNumber = 512;

    if (sinkNumber == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (sinkNumber < 0 || sinkNumber > maxSinkNumber || sinkNumber % 2 == 1) {
        OPS_LOG_E(opName_, "invalid Parameter sinkNumber");
        return ge::GRAPH_FAILED;
    }

    if (sinkNumber > 0 && qLayout_ != FiaLayout::TND) {
        OPS_LOG_E(opName_,
                  "when sinkNumber > 0, qLayout should TND, but input qlayout is %s",
                  LayoutToSerialString(qLayout_).c_str());
        return ge::GRAPH_FAILED;
    }

    bool isLearnableSinkEnable = opParamInfo_.learnableSink.tensor != nullptr ? true : false;
    if (isLearnableSinkEnable) {
        OPS_LOG_E(opName_, "Conflict parameter between learnableSink and sinkNumber");
        return ge::GRAPH_FAILED;
    }

    if ((sparseMode == SPARSE_MODE_NO_MASK &&
         (opParamInfo_.attenMask.tensor != nullptr && opParamInfo_.attenMask.desc != nullptr)) ||
        sparseMode == SPARSE_MODE_ALL_MASK) {
        OPS_LOG_E(opName_, "sparseMode cannot be NO_MASK or ALL_MASK when sinkNumber(%ld) exists", sinkNumber);
        return ge::GRAPH_FAILED;
    }

    return CheckMlaSinkNumber();
}

/*
  1、paramSink与 sinkNumber 属性互斥，不同时使用
  2、paramSink与 learnableSink 互斥，不同时使用
  3、paramSink token数只支持128
  4、paramSink只支持 sparseMode=4 (SWA)
  5、paramSink只支持TND场景
  */
ge::graphStatus FiaTilingCheck::CheckParamSinkNumber()
{
    bool hasParamSink = (opParamInfo_.keySink.tensor != nullptr || opParamInfo_.keyRopeSink.tensor != nullptr ||
                         opParamInfo_.valueSink.tensor != nullptr);

    OPS_LOG_I(opName_, "[DEBUG] CheckParamSinkNumber: hasParamSink=%d", hasParamSink);

    if (!hasParamSink) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t attrSinkNumber = *opParamInfo_.sinkNumber;
    OPS_LOG_I(opName_, "[DEBUG] CheckParamSinkNumber: sinkNumber=%ld", attrSinkNumber);
    if (attrSinkNumber != 0) {
        OPS_LOG_E(opName_,
                  "Conflict: paramSink (key_sink/key_rope_sink/value_sink) and sinkNumber attribute cannot coexist");
        return ge::GRAPH_FAILED;
    }

    bool isLearnableSinkEnable = opParamInfo_.learnableSink.tensor != nullptr ? true : false;
    if (isLearnableSinkEnable) {
        OPS_LOG_E(opName_,
                  "Conflict paramSink (key_sink/key_rope_sink/value_sink) and learnableSink attribute cannot coexist");
        return ge::GRAPH_FAILED;
    }

    int32_t sparseMode = *opParamInfo_.sparseMode;
    if (sparseMode != SPARSE_MODE_BAND) {
        OPS_LOG_E(opName_, "paramSink only supports sparseMode=4 (SWA), but got %d", sparseMode);
        return ge::GRAPH_FAILED;
    }

    if (qLayout_ != FiaLayout::TND) {
        OPS_LOG_E(opName_, "qLayout should TND, but input qlayout is %s", LayoutToSerialString(qLayout_).c_str());
        return ge::GRAPH_FAILED;
    }

    // prefill rope合并传入  keyRopeSink为空
    if (ropeMode_ != RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.keySink.tensor == nullptr || opParamInfo_.keyRopeSink.tensor != nullptr ||
            opParamInfo_.valueSink.tensor == nullptr) {
            OPS_LOG_E(opName_,
                      "Prefill merged rope mode requires: key_sink and value_sink non-null, key_rope_sink null");
            return ge::GRAPH_FAILED;
        }

        const auto &keySinkShape = opParamInfo_.keySink.tensor->GetStorageShape();
        const auto &valueSinkShape = opParamInfo_.valueSink.tensor->GetStorageShape();

        constexpr int64_t PARAM_SINK_TOKEN_NUM = 128;
        int64_t keySinkNum = keySinkShape.GetDim(0);
        int64_t valueSinkNum = valueSinkShape.GetDim(0);
        if (keySinkNum != PARAM_SINK_TOKEN_NUM || valueSinkNum != PARAM_SINK_TOKEN_NUM) {
            OPS_LOG_E(opName_,
                      "Param sink number must be 128, but got key_sink=%ld, value_sink=%ld",
                      keySinkNum,
                      valueSinkNum);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    //  decode场景：rope分离传入，三个参数都必须存在
    if (opParamInfo_.keySink.tensor == nullptr || opParamInfo_.keyRopeSink.tensor == nullptr ||
        opParamInfo_.valueSink.tensor == nullptr) {
        OPS_LOG_E(opName_, "Decode split rope mode requires all three: key_sink, key_rope_sink, value_sink non-null");
        return ge::GRAPH_FAILED;
    }

    const auto &keySinkShape = opParamInfo_.keySink.tensor->GetStorageShape();
    const auto &keyRopeSinkShape = opParamInfo_.keyRopeSink.tensor->GetStorageShape();
    const auto &valueSinkShape = opParamInfo_.valueSink.tensor->GetStorageShape();

    constexpr int64_t PARAM_SINK_TOKEN_NUM = 128;
    int64_t keySinkNum = keySinkShape.GetDim(0);
    int64_t keyRopeSinkNum = keyRopeSinkShape.GetDim(0);
    int64_t valueSinkNum = valueSinkShape.GetDim(0);
    if (keySinkNum != PARAM_SINK_TOKEN_NUM || keyRopeSinkNum != PARAM_SINK_TOKEN_NUM ||
        valueSinkNum != PARAM_SINK_TOKEN_NUM) {
        OPS_LOG_E(opName_,
                  "Param sink number must be 128, but got key_sink=%ld, key_rope_sink=%ld, value_sink=%ld",
                  keySinkNum,
                  keyRopeSinkNum,
                  valueSinkNum);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMask()
{
    if (CheckParamSinkNumber() != ge::GRAPH_SUCCESS || CheckSinkNumber() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (opParamInfo_.attenMask.tensor == nullptr || opParamInfo_.attenMask.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckAttentionMask() != ge::GRAPH_SUCCESS || CheckTokens() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FiaTilingCheck::SetAttenMaskCompare()
{
    size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
    int64_t maskDim0 = opParamInfo_.attenMask.tensor->GetStorageShape().GetDim(0);
    int32_t sparseMode = *opParamInfo_.sparseMode;
    FiaLayout maskLayout;

    if (sparseMode == SPARSE_MODE_NO_MASK || sparseMode == SPARSE_MODE_ALL_MASK) {
        if (maskDimNum == DIM_NUM_TWO) {
            if (s1Size_ == 1U && maskDim0 == static_cast<int64_t>(bSize_)) {
                maskLayout = FiaLayout::BS2;
            } else {
                maskLayout = FiaLayout::S1S2;
            }
        } else if (maskDimNum == DIM_NUM_THREE) {
            maskLayout = maskDim0 == 1 ? FiaLayout::IS1S2 : FiaLayout::BS1S2;
        } else if (maskDimNum == DIM_NUM_FOUR) {
            maskLayout = maskDim0 == 1 ? FiaLayout::I1S1S2 : FiaLayout::B1S1S2;
        } else {
            OPS_LOG_E(opName_,
                      "%s dim num only support %zu, %zu, %zu, but got %zu",
                      ATTEN_MASK_NAME.c_str(),
                      DIM_NUM_TWO,
                      DIM_NUM_THREE,
                      DIM_NUM_FOUR,
                      maskDimNum);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (maskDimNum == DIM_NUM_TWO) {
            maskLayout = FiaLayout::S1S2;
        } else if (maskDimNum == DIM_NUM_THREE) {
            maskLayout = FiaLayout::IS1S2;
        } else if (maskDimNum == DIM_NUM_FOUR) {
            maskLayout = FiaLayout::I1S1S2;
        } else {
            OPS_LOG_E(
                opName_, "%s dim num only support %zu, but got %zu", ATTEN_MASK_NAME.c_str(), DIM_NUM_TWO, maskDimNum);
            return ge::GRAPH_FAILED;
        }
    }

    attenMaskShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(
        opParamInfo_.attenMask.tensor->GetStorageShape(), maskLayout, ATTEN_MASK_NAME, opName_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckAttentionMask()
{
    int32_t sparseMode = *opParamInfo_.sparseMode;
    if (opParamInfo_.attenMask.tensor == nullptr || opParamInfo_.attenMask.desc == nullptr) {
        if (sparseMode != SPARSE_MODE_NO_MASK) {
            OPS_LOG_E(opName_,
                      "When %s(%d) not equals to %d, %s must exists",
                      SPARSE_MODE_NAME.c_str(),
                      sparseMode,
                      SPARSE_MODE_NO_MASK,
                      ATTEN_MASK_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (SetAttenMaskCompare() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    constexpr int64_t OPT_ATTEN_MASK_LEN = 2048; // 2048: ATTEN_MASK_LEN
    FiaTilingShapeCompareParam shapeParams;
    if (sparseMode == SPARSE_MODE_NO_MASK || sparseMode == SPARSE_MODE_ALL_MASK) {
        if (fiaInfo_.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        shapeParams.B = static_cast<int64_t>(bSize_);
        shapeParams.S1 = static_cast<int64_t>(s1Size_);
        shapeParams.S2 = s2Size_;
        shapeParams.compareTypeMap = {
            {FiaAxis::S1, FiaCompareType::GREATER_EQUAL},
            {FiaAxis::S2, FiaCompareType::GREATER_EQUAL},
        };
    } else if (sparseMode == SPARSE_MODE_LEFT_UP || sparseMode == SPARSE_MODE_RIGHT_DOWN ||
               sparseMode == SPARSE_MODE_BAND) {
        shapeParams.S1 = OPT_ATTEN_MASK_LEN;
        shapeParams.S2 = OPT_ATTEN_MASK_LEN;
    }

    return attenMaskShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckTokens()
{
    preTokens_ = fiaInfo_.preToken;
    nextTokens_ = fiaInfo_.nextToken;
    OPS_CHECK(
        preTokens_ < 0 && nextTokens_ < 0,
        OPS_LOG_E(
            opName_, "preTokens(%ld) and nextTokens(%ld) cannot neither be negative number.", preTokens_, nextTokens_),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        nextTokens_ * (-1) > preTokens_,
        OPS_LOG_E(opName_, "nextToken line(%ld) should be higher than preToken line(%ld).", nextTokens_, preTokens_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxLse()
{
    if (!fiaInfo_.softmaxLseFlag && opParamInfo_.lseOut.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo_.softmaxLseFlag && opParamInfo_.lseOut.desc == nullptr) {
        OPS_LOG_E(opName_, "when %s is enabled, softmaxlse should not be NULL.", SOFTMAX_LSE_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    if (!fiaInfo_.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckSoftmaxLseTypeDTypeAndShape(opParamInfo_.lseOut, SOFTMAX_LSE_NAME);
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxLseTypeDTypeAndShape(const FIARequiredParaInfo &outInfo,
                                                                const std::string &name)
{
    if (outInfo.desc == nullptr || outInfo.shape == nullptr) {
        OPS_LOG_E(opName_, "when softmaxLseFlag is enabled, %s should not be NULL.", name.c_str());
        return ge::GRAPH_FAILED;
    }
    if (outInfo.desc->GetDataType() != ge::DT_FLOAT) {
        OPS_LOG_E(opName_,
                  "%s only support dtype FP32, but got %s",
                  name.c_str(),
                  FusedDataTypeToSerialString(outInfo.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }

    FiaLayout lseLayout = FiaLayout::BNS11;
    if (outLayout_ == FiaLayout::TND || outLayout_ == FiaLayout::NTD) {
        lseLayout = FiaLayout::TN1;
    }
    auto shapeCmp = std::make_shared<FiaTilingShapeCompare>(outInfo.shape->GetStorageShape(), lseLayout, name, opName_);

    FiaTilingShapeCompareParam shapeParams;
    if (lseLayout == FiaLayout::TN1) {
        shapeParams.T = static_cast<int64_t>(qTSize_);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.CONST = 1;
    } else {
        shapeParams.B = static_cast<int64_t>(bSize_);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.S1 = static_cast<int64_t>(s1Size_);
        shapeParams.CONST = 1;
    }
    return shapeCmp->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxMaxSum()
{
    if (!fiaInfo_.softmaxMaxSumFlag) {
        return ge::GRAPH_SUCCESS;
    }

    // softmaxMaxSumFlag 与 softmaxLseFlag互斥
    if (fiaInfo_.softmaxLseFlag) {
        OPS_LOG_E(opName_,
                  "softmaxMaxSumFlag and softmaxLseFlag are mutually exclusive, "
                  "but both are set to true.");
        return ge::GRAPH_FAILED;
    }

    // D3: softmaxMax/softmaxSum 仅支持 MLA 吸收方案(rope_split + vHeadDim=512) + batchInvariant
    const bool isMlaAbsorb = (ropeMode_ == RopeMode::ROPE_SPLIT && vHeadDim_ == 512U && ropeHeadDim_ == 64U);
    if (!isMlaAbsorb) {
        OPS_LOG_E(opName_,
                  "softmaxMax/softmaxSum only support MLA absorb(rope_split + vHeadDim=512 + ropeHeadDim=64), "
                  "got ropeMode=%s vHeadDim=%u ropeHeadDim=%u",
                  RopeModeToSerialString(ropeMode_).c_str(), vHeadDim_, ropeHeadDim_);
        return ge::GRAPH_FAILED;
    }
    if (!fiaInfo_.batchInvariant) {
        OPS_LOG_E(opName_,
                  "softmaxMax/softmaxSum requires batchInvariant=true, got false");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckSoftmaxLseTypeDTypeAndShape(opParamInfo_.softmaxMaxOut, SOFTMAX_MAX_NAME) ||
        ge::GRAPH_SUCCESS != CheckSoftmaxLseTypeDTypeAndShape(opParamInfo_.softmaxSumOut, SOFTMAX_SUM_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParamSinkShape()
{
    if (opParamInfo_.keySink.tensor == nullptr && opParamInfo_.keyRopeSink.tensor == nullptr &&
        opParamInfo_.valueSink.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // prefill rope合并传入
    if (ropeMode_ != RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.keySink.tensor == nullptr || opParamInfo_.keyRopeSink.tensor != nullptr ||
            opParamInfo_.valueSink.tensor == nullptr) {
            OPS_LOG_E(opName_,
                      "Prefill merged rope mode requires: key_sink and value_sink non-null, key_rope_sink null");
            return ge::GRAPH_FAILED;
        }

        const auto &keySinkShape = opParamInfo_.keySink.tensor->GetStorageShape();
        const auto &valueSinkShape = opParamInfo_.valueSink.tensor->GetStorageShape();

        // 检查shape: (sink_number, N2, D)
        // keySinkShape、valueSinkShape形状为(sink_number, N2, D)，其维度固定为3维
        if (keySinkShape.GetDimNum() != 3 || valueSinkShape.GetDimNum() != 3) {
            OPS_LOG_E(opName_, "key_sink and value_sink shape must be 3D (sink_num, N2, D)");
            return ge::GRAPH_FAILED;
        }

        // N2（第二维）必须与KV的N2一致
        int64_t keySinkN2 = keySinkShape.GetDim(1);
        int64_t valueSinkN2 = valueSinkShape.GetDim(1);
        if (keySinkN2 != static_cast<int64_t>(n2Size_) || valueSinkN2 != static_cast<int64_t>(n2Size_)) {
            OPS_LOG_E(opName_,
                      "Param sink N2(%ld) and value_sink N2(%ld) must be equal to KV N2(%u)",
                      keySinkN2,
                      valueSinkN2,
                      n2Size_);
            return ge::GRAPH_FAILED;
        }

        // D_qk支持192
        int64_t keySinkD = keySinkShape.GetDim(2);
        if (keySinkD != 192) {
            OPS_LOG_E(opName_, "key_sink D_qk(%ld) only supports 192 in prefill merged mode", keySinkD);
            return ge::GRAPH_FAILED;
        }

        // key_sink的D_qk必须与K的qkHeadDim一致，且为192
        if (keySinkD != static_cast<int64_t>(qkHeadDim_)) {
            OPS_LOG_E(opName_, "key_sink D_qk(%ld) must be equal to KV qkHeadDim(%u)", keySinkD, qkHeadDim_);
            return ge::GRAPH_FAILED;
        }

        // D_v支持128
        int64_t valueSinkD = valueSinkShape.GetDim(2);
        if (valueSinkD != 128) {
            OPS_LOG_E(opName_, "value_sink D_v(%ld) only supports 128", valueSinkD);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 前置空指针检查
    if (opParamInfo_.keySink.tensor == nullptr || opParamInfo_.keyRopeSink.tensor == nullptr ||
        opParamInfo_.valueSink.tensor == nullptr) {
        OPS_LOG_E(opName_, "key_sink, key_rope_sink, and value_sink must all be valid");
        return ge::GRAPH_FAILED;
    }

    // prefill分离传入
    const auto &keySinkShape = opParamInfo_.keySink.tensor->GetStorageShape();
    const auto &keyRopeSinkShape = opParamInfo_.keyRopeSink.tensor->GetStorageShape();
    const auto &valueSinkShape = opParamInfo_.valueSink.tensor->GetStorageShape();

    // keySinkShape、keyRopeSinkShape、valueSinkShape形状为(sink_number, N2, D)，其维度固定为3维
    if (keySinkShape.GetDimNum() != 3 || keyRopeSinkShape.GetDimNum() != 3 || valueSinkShape.GetDimNum() != 3) {
        OPS_LOG_E(opName_, "key_sink, key_rope_sink, value_sink shape must be 3D (sink_num, N2, D)");
        return ge::GRAPH_FAILED;
    }

    // N2（第二维）必须与KV的N2一致
    int64_t keySinkN2 = keySinkShape.GetDim(1);
    int64_t keyRopeSinkN2 = keyRopeSinkShape.GetDim(1);
    int64_t valueSinkN2 = valueSinkShape.GetDim(1);
    if (keySinkN2 != static_cast<int64_t>(n2Size_) || keyRopeSinkN2 != static_cast<int64_t>(n2Size_) ||
        valueSinkN2 != static_cast<int64_t>(n2Size_)) {
        OPS_LOG_E(opName_, "Param sink N2(%ld) must be equal to KV N2(%u)", keySinkN2, n2Size_);
        return ge::GRAPH_FAILED;
    }

    // D_qk支持128/512
    int64_t keySinkD = keySinkShape.GetDim(2);
    const std::set<int64_t> supportedDQk = {128, 512};
    if (supportedDQk.find(keySinkD) == supportedDQk.end()) {
        OPS_LOG_E(opName_, "key_sink D_qk(%ld) only supports 128/512", keySinkD);
        return ge::GRAPH_FAILED;
    }

    // key_rope_sink的rope dim必须为64
    int64_t keyRopeSinkD = keyRopeSinkShape.GetDim(2);
    constexpr int64_t ROPE_DIM = 64;
    if (keyRopeSinkD != ROPE_DIM) {
        OPS_LOG_E(opName_, "key_rope_sink rope dim(%ld) must be 64", keyRopeSinkD);
        return ge::GRAPH_FAILED;
    }

    // D_v支持128
    int64_t valueSinkD = valueSinkShape.GetDim(2);
    const std::set<int64_t> supportedDV = {128, 512};
    if (supportedDV.find(valueSinkD) == supportedDV.end()) {
        OPS_LOG_E(opName_, "value_sink D_v(%ld) only supports 128/512", valueSinkD);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMultiParaConsistency()
{
    SetFiaShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLensKv() ||
        ge::GRAPH_SUCCESS != CheckBlockTable() || ge::GRAPH_SUCCESS != CheckQAndQRope() ||
        ge::GRAPH_SUCCESS != CheckKV() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckPseShift() || ge::GRAPH_SUCCESS != CheckParamSinkShape() ||
        ge::GRAPH_SUCCESS != CheckMask() || ge::GRAPH_SUCCESS != CheckSoftmaxLse() ||
        ge::GRAPH_SUCCESS != CheckSoftmaxMaxSum()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
