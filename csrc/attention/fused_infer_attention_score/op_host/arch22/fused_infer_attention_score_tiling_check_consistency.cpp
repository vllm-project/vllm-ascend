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
 * \file fused_infer_attention_score_tiling_check_consistency.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
void FiaTilingCheck::SetFiaShapeCompare()
{
    queryShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.query.shape->GetStorageShape(),
        qLayout_, QUERY_NAME, opName_);
    keyShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(kCache_[0]->GetStorageShape(),
        kvLayout_, KEY_NAME, opName_);
    valueShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(vCache_[0]->GetStorageShape(),
        kvLayout_, VALUE_NAME, opName_);
    attenOutShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.attenOut.shape->GetStorageShape(),
        outLayout_, ATTEN_OUT_NAME, opName_);
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        queryRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.queryRope.tensor->GetStorageShape(),
            qLayout_, QUERY_ROPE_NAME, opName_);
        keyRopeShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.keyRope.tensor->GetStorageShape(),
            kvLayout_, KEY_ROPE_NAME, opName_);
    }
}

ge::graphStatus FiaTilingCheck::CheckQAndQRopeDType() const
{
    if (opParamInfo_.query.desc->GetDataType() != inputQType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            QUERY_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.queryRope.desc->GetDataType() != inputQRopeType_) {
            OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
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
    if (ge::GRAPH_SUCCESS != CheckQShape() ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckQAndQRope() const
{
    if (ge::GRAPH_SUCCESS != CheckQAndQRopeDType() ||
        ge::GRAPH_SUCCESS != CheckQAndQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVDType() const
{
    if (opParamInfo_.key.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            KEY_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.value.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
            VALUE_NAME.c_str(),
            FusedDataTypeToSerialString(opParamInfo_.value.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str());
            return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        if (opParamInfo_.keyRope.desc->GetDataType() != inputKRopeType_) {
            OP_LOGE(opName_, "%s's dtype is %s, it should be %s.",
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
        auto keyShapeCmp = std::make_shared<FiaTilingShapeCompare>(kCache_[i]->GetStorageShape(),
            kvLayout_, KEY_NAME, opName_);
        if (keyShapeCmp->CompareShape(keyShapeParams, __func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto valueShapeCmp = std::make_shared<FiaTilingShapeCompare>(vCache_[i]->GetStorageShape(),
            kvLayout_, VALUE_NAME, opName_);
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
    }
    return typeSize;
}

ge::graphStatus FiaTilingCheck::CheckBlockTable() const
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        opParamInfo_.blockTable.tensor->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "%s shape size is zero.",
            BLOCK_TABLE_NAME.c_str()),
        return ge::GRAPH_FAILED);
    
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(qLayout_ == FiaLayout::TND && blockTableBatch != bSize_,
        OP_LOGE(opName_, "when %s's layout is TND, %s's first dimension(%u) should be equal to batch size(%u)",
            QUERY_NAME.c_str(), BLOCK_TABLE_NAME.c_str(),
            blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckKVShapeForPageAttention() const
{
    uint32_t kvBlockElemNum = 32 / GetTypeSize(inputKvType_);
    if (blockSize_ % static_cast<int32_t>(kvBlockElemNum) != 0) {
        OP_LOGE(opName_, "when kv_dtype is %s, 32 / sizeof(kv_dtype) is %u, block_size %% (32 / sizeof(kv_dtype)) should be 0.",
            FusedDataTypeToSerialString(inputKvType_).c_str(), kvBlockElemNum);
        return ge::GRAPH_FAILED;
    }

    // key
    FiaTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(fiaInfo_.totalBlockNum);
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
            OP_LOGE(opName_, "when key_rope_dtype is %s, 32 / sizeof(key_rope_dtype) is %u, block_size %% (32 / sizeof(key_rope_dtype)) should be 0.",
                FusedDataTypeToSerialString(inputKRopeType_).c_str(), kRopeBlockElemNum);
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

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus FiaTilingCheck::CheckKV() const
{
    if (ge::GRAPH_SUCCESS != CheckKVDType() ||
        ge::GRAPH_SUCCESS != CheckKVShape()) {
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
    if ((opParamInfo_.actualSeqLengthsQ.tensor == nullptr) ||
        (opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr)) {
        return ge::GRAPH_SUCCESS;
    }

    if (qLayout_ == FiaLayout::TND) {
        if (actualSeqLengthsQSize_ != bSize_ && actualSeqLengthsQSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsQSize_ < bSize_ && actualSeqLengthsQSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckActualSeqLensKv() const
{
    if ((opParamInfo_.actualSeqLengths.tensor == nullptr) ||
        (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr)) {
        return ge::GRAPH_SUCCESS;
    }

    if (kvLayout_ == FiaLayout::TND) {
        if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
            opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr &&
            actualSeqLengthsKvSize_ != actualSeqLengthsQSize_) {
            OP_LOGE(opName_, "when key/value's layout is TND, %s shape size is %u, it should be equal to %s shape size(%u).",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_,
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), actualSeqLengthsQSize_);
            return ge::GRAPH_FAILED;
        }
        if (actualSeqLengthsKvSize_ != bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OP_LOGE(opName_, "when key/value's layout is TND, %s shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (actualSeqLengthsKvSize_ < bSize_ && actualSeqLengthsKvSize_ != 1U) {
            OP_LOGE(opName_, "%s shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), actualSeqLengthsKvSize_, bSize_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckPseShift()
{
    if (opParamInfo_.pseShift.tensor == nullptr || opParamInfo_.pseShift.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckPseShiftDType() ||
        ge::GRAPH_SUCCESS != CheckPseShiftShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckPseShiftDType()
{
    if ((inputQType_ == ge::DT_FLOAT16 || inputQType_ == ge::DT_BF16) && opParamInfo_.pseShift.desc->GetDataType() != inputQType_) {
        OP_LOGE(opName_, "when query's dtype is %s, pseShift dtype should be %s, but got %s.",
            FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(opParamInfo_.pseShift.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }

    if (inputQType_ == ge::DT_INT8 && (s1Size_ <= 1 || opParamInfo_.pseShift.desc->GetDataType() != ge::DT_FLOAT16)) {
        OP_LOGE(opName_, "when query's dtype is %s, pseShift dtype should be %s, but got %s.",
            FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(opParamInfo_.pseShift.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckPseShiftShape()
{
    if (fiaInfo_.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    size_t pseShiftDim0 = opParamInfo_.pseShift.tensor->GetStorageShape().GetDim(0);
    if (pseShiftDim0 == 1U) {
        pseShiftLayout_ = FiaLayout::INS1S2;
    } else {
        pseShiftLayout_ = FiaLayout::BNS1S2;
    }

    pseShiftShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.pseShift.tensor->GetStorageShape(),
        pseShiftLayout_, PSE_SHIFT_NAME, opName_);

    FiaTilingShapeCompareParam shapeParams;
    if (pseShiftLayout_ == FiaLayout::BNS1S2) {
        shapeParams.B = static_cast<int64_t>(bSize_);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.S1 = static_cast<int64_t>(s1Size_);
        shapeParams.S2 = s2Size_;
    } else if (pseShiftLayout_ == FiaLayout::INS1S2) {
        shapeParams.CONST = static_cast<int64_t>(1);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.S1 = static_cast<int64_t>(s1Size_);
        shapeParams.S2 = s2Size_;
    }
    shapeParams.compareTypeMap = {{FiaAxis::S1, FiaCompareType::GREATER_EQUAL}, {FiaAxis::S2, FiaCompareType::GREATER_EQUAL}};
    
    return pseShiftShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckSystemPrefix()
{
    if (fiaInfo_.sysPrefixFlag) {
        if (CheckSystemPrefixDtype() != ge::GRAPH_SUCCESS || CheckSystemPrefixShape() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSystemPrefixDtype()
{
    if (opParamInfo_.keySharedPrefix.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "when key's dtype is %s, keySharedPrefix dtype should be %s, but got %s.",
            FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str(),
            FusedDataTypeToSerialString(opParamInfo_.keySharedPrefix.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.valueSharedPrefix.desc->GetDataType() != inputKvType_) {
        OP_LOGE(opName_, "when value's dtype is %s, valueSharedPrefix dtype should be %s, but got %s.",
            FusedDataTypeToSerialString(opParamInfo_.value.desc->GetDataType()).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str(),
            FusedDataTypeToSerialString(opParamInfo_.valueSharedPrefix.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSystemPrefixShape()
{
    if (!fiaInfo_.sysPrefixFlag) {
        return ge::GRAPH_SUCCESS;
    }
    auto prefixKShape = opParamInfo_.keySharedPrefix.tensor->GetStorageShape();
    auto prefixVShape = opParamInfo_.valueSharedPrefix.tensor->GetStorageShape();
    prefixKeyShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(prefixKShape, kvLayout_, KEY_SHARED_PREFIX_NAME, opName_);

    if (prefixKShape != prefixVShape) {
        OP_LOGE(opName_, "Prefix shapes mismatch: prefix key shape and prefix value shape");
        return ge::GRAPH_FAILED;
    }
    if (prefixKShape.GetDim(0) != 1) {
        OP_LOGE(opName_, "System prefix is enabled, Prefix batch dimension must be 1, got %ld", prefixKShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    FiaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(bSize_);
    shapeParams.N = static_cast<int64_t>(n2Size_);
    shapeParams.S = s2Size_;
    shapeParams.D = static_cast<int64_t>(qkHeadDim_);
    // 前缀的B和S2和正常的没关系
    shapeParams.compareTypeMap = {{FiaAxis::S, FiaCompareType::IGNORE_INPUT},
                                  {FiaAxis::B, FiaCompareType::IGNORE_INPUT}};
    
    if (ge::GRAPH_SUCCESS != prefixKeyShapeCmp_->CompareShape(shapeParams, __func__)) {
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo_.systemPrefixLen > fiaInfo_.systemPrefixMaxLen) {
        OP_LOGE(opName_, "actual prefix len should be less than or equal to prefixlen");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMask()
{
    if (opParamInfo_.attenMask.tensor == nullptr || opParamInfo_.attenMask.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckAttentionMask() ||
        ge::GRAPH_SUCCESS != CheckTokens()
        ) {
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
            if (s1Size_ == 1U) {
                maskLayout = FiaLayout::BS2;
            } else {
                maskLayout = FiaLayout::S1S2;
            }
        } else if (maskDimNum == DIM_NUM_THREE) {
            maskLayout = maskDim0 == 1 ? FiaLayout::IS1S2 : FiaLayout::BS1S2;
        } else if (maskDimNum == DIM_NUM_FOUR) {
            maskLayout = maskDim0 == 1 ? FiaLayout::I1S1S2 : FiaLayout::B1S1S2;
        } else {
            OP_LOGE(opName_, "%s dim num only support %zu, %zu, %zu, but got %zu",
                ATTEN_MASK_NAME.c_str(), DIM_NUM_TWO, DIM_NUM_THREE, DIM_NUM_FOUR, maskDimNum);
            return ge::GRAPH_FAILED;
        }
    } else if (sparseMode == SPARSE_MODE_TREE) {
        if (maskDimNum == DIM_NUM_ONE) {
            maskLayout = FiaLayout::S1S1;
        } else if (maskDimNum == DIM_NUM_THREE) {
            maskLayout = FiaLayout::BS1S2;
        } else {
            OP_LOGE(opName_, "%s dim num only support %zu, %zu, but got %zu",
                ATTEN_MASK_NAME.c_str(), DIM_NUM_ONE, DIM_NUM_THREE, maskDimNum);
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
            OP_LOGE(opName_, "%s dim num only support %zu, but got %zu",
                ATTEN_MASK_NAME.c_str(), DIM_NUM_TWO, maskDimNum);
            return ge::GRAPH_FAILED;
        }
    }

    attenMaskShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.attenMask.tensor->GetStorageShape(),
        maskLayout, ATTEN_MASK_NAME, opName_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckAttentionMask()
{
    int32_t sparseMode = *opParamInfo_.sparseMode;
    if (opParamInfo_.attenMask.tensor == nullptr || opParamInfo_.attenMask.desc == nullptr) {
        if (sparseMode != SPARSE_MODE_NO_MASK) {
            OP_LOGE(opName_, "When %s(%d) not equals to %d, %s must exists",
            SPARSE_MODE_NAME.c_str(), sparseMode, SPARSE_MODE_NO_MASK, ATTEN_MASK_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (SetAttenMaskCompare() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    constexpr int64_t OPT_ATTEN_MASK_LEN = 2048;  // 2048: ATTEN_MASK_LEN
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
    } else if (sparseMode == SPARSE_MODE_LEFT_UP || sparseMode == SPARSE_MODE_RIGHT_DOWN || sparseMode == SPARSE_MODE_BAND){
        shapeParams.S1 = OPT_ATTEN_MASK_LEN;
        shapeParams.S2 = OPT_ATTEN_MASK_LEN;
    } else if (sparseMode == SPARSE_MODE_TREE) {
        uint64_t sSize = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            // tiling下沉场景 由于actualSeqlen得不到，所以不进行校验
            if (fiaInfo_.isMaxWorkspace) {
                return ge::GRAPH_SUCCESS;
            }

            for (uint32_t i = 0; i < qSize.size(); i++) {
                sSize += qSize[i] * qSize[i];
            }
        } else {
            sSize = s1Size_;
        }
        shapeParams.B = static_cast<int64_t>(bSize_);
        shapeParams.S1 = static_cast<int64_t>(sSize);
        shapeParams.S2 = static_cast<int64_t>(sSize);
        shapeParams.compareTypeMap = {
            {FiaAxis::S1, FiaCompareType::GREATER_EQUAL},
            {FiaAxis::S2, FiaCompareType::GREATER_EQUAL},
        };
    }
    return attenMaskShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckTokens()
{
    preTokens_ = fiaInfo_.preToken;
    nextTokens_ = fiaInfo_.nextToken;
    OP_CHECK_IF(preTokens_ < 0 && nextTokens_ < 0, 
        OP_LOGE(opName_, "preTokens(%ld) and nextTokens(%ld) cannot neither be negative number.",
            preTokens_, nextTokens_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(nextTokens_ * (-1) > preTokens_, 
        OP_LOGE(opName_, "nextToken line(%ld) should be higher than preToken line(%ld).",
            nextTokens_, preTokens_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxLse()
{
    if (!fiaInfo_.softmaxLseFlag && opParamInfo_.lseOut.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo_.softmaxLseFlag && opParamInfo_.lseOut.desc == nullptr) {
        OP_LOGE(opName_, "when %s is enabled, softmaxlse should not be NULL.",
            SOFTMAX_LSE_NAME.c_str()); 
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckSoftmaxLseDType() ||
        ge::GRAPH_SUCCESS != CheckSoftmaxLseShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxLseDType() 
{
    if (opParamInfo_.lseOut.desc->GetDataType() != ge::DT_FLOAT) { 
        OP_LOGE(opName_, "only support dtype FP32, but got %s",
            FusedDataTypeToSerialString(opParamInfo_.lseOut.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSoftmaxLseShape()
{
    if (outLayout_ == FiaLayout::TND || outLayout_ == FiaLayout::NTD) {
        softmaxLseLayout_ = FiaLayout::TN1;
    } else {
        softmaxLseLayout_ = FiaLayout::BNS11;
    }
    softmaxLseShapeCmp_ = std::make_shared<FiaTilingShapeCompare>(opParamInfo_.lseOut.shape->GetStorageShape(),
         softmaxLseLayout_, SOFTMAX_LSE_NAME, opName_);

    FiaTilingShapeCompareParam shapeParams;
    if (fiaInfo_.softmaxLseFlag && softmaxLseLayout_ == FiaLayout::TN1) {
        shapeParams.T = static_cast<int64_t>(qTSize_);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.CONST = 1;
    } else if (fiaInfo_.softmaxLseFlag && softmaxLseLayout_ == FiaLayout::BNS11) {
        shapeParams.B = static_cast<int64_t>(bSize_);
        shapeParams.N = static_cast<int64_t>(n1Size_);
        shapeParams.S1 = static_cast<int64_t>(s1Size_);
        shapeParams.CONST = 1;
    } else if (!fiaInfo_.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }
    return softmaxLseShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus FiaTilingCheck::CheckPostQuant()
{
    if (fiaInfo_.isOutQuantEnable) {
        OP_CHECK_IF(opParamInfo_.attenOut.desc->GetDataType() != ge::DT_INT8, 
            OP_LOGE(opName_, "the quantScale2 exist, output data type should be INT8!"), 
            return ge::GRAPH_FAILED);
        const ge::DataType queryDataType_ = opParamInfo_.query.desc->GetDataType();
        const ge::DataType quantScale2Datatype_ = opParamInfo_.quantScale2.desc->GetDataType();
        auto quantScale2Shape_ = opParamInfo_.quantScale2.tensor->GetStorageShape();

        //scale2 and offset2 should have same dtype and shape
        if (opParamInfo_.quantOffset2.tensor != nullptr && opParamInfo_.quantOffset2.desc != nullptr) {
            const ge::DataType quantOffset2Datatype_ = opParamInfo_.quantOffset2.desc->GetDataType();
            OP_CHECK_IF(quantOffset2Datatype_ != quantScale2Datatype_,
                OP_LOGE(opName_, "quantScale2 and quantOffset2 should have same dtype!"),
                return ge::GRAPH_FAILED);
            auto quantOffset2Shape_ = opParamInfo_.quantOffset2.tensor->GetStorageShape();
            OP_CHECK_IF(quantOffset2Shape_ != quantScale2Shape_,
                OP_LOGE(opName_, "quantScale2 and quantOffset2 should have same shape!"),
                return ge::GRAPH_FAILED);
        }

        // scale2 dtype verification
        OP_CHECK_IF((queryDataType_ != ge::DT_FLOAT16 && queryDataType_ != ge::DT_BF16),
            OP_LOGE(opName_, "in postquant situation, query type should be FP16 or BF16"),
            return ge::GRAPH_FAILED);
 
        OP_CHECK_IF((queryDataType_ == ge::DT_FLOAT16 && quantScale2Datatype_ != ge::DT_FLOAT),
            OP_LOGE(opName_, "inputQ dtype is FP16, quantScale2 dtype should be FP32"),
            return ge::GRAPH_FAILED);

        //scale2 shape verfication
        std::string layout(opParamInfo_.layOut);
        if (quantScale2Shape_.GetShapeSize() != 1) {
            if (layout == "BSH" || layout == "BSND" || layout == "BNSD" || layout == "BNSD_BSND") {
                int64_t currentShapeSize = quantScale2Shape_.GetShapeSize();
                int64_t expectShapeSize = gert::Shape({n1Size_, vHeadDim_}).GetShapeSize();
                OP_CHECK_IF(currentShapeSize != expectShapeSize,
                    OP_LOGE(opName_, "when input_layout is %s, the shape size of quantScale2 should be %lld(Q_N * V_D), but current is %u, shape is %s",
                        layout.c_str(), expectShapeSize, currentShapeSize, GetShapeStr(quantScale2Shape_).c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(quantScale2Shape_ != gert::Shape({n1Size_, vHeadDim_}),
                    OP_LOGE(opName_, "when input_layout is %s, quantScale2 expectshape is [Q_N, V_D], it is [%u, %u], but current shape is %s",
                        layout.c_str(), n1Size_, vHeadDim_, GetShapeStr(quantScale2Shape_).c_str()),
                    return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckMultiParaConsistency()
{
    SetFiaShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensKv() ||
        ge::GRAPH_SUCCESS != CheckBlockTable() ||
        ge::GRAPH_SUCCESS != CheckQAndQRope() ||
        ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckPseShift() ||
        ge::GRAPH_SUCCESS != CheckMask() ||
        ge::GRAPH_SUCCESS != CheckSoftmaxLse()||
        ge::GRAPH_SUCCESS != CheckSystemPrefix() ||
        ge::GRAPH_SUCCESS != CheckPostQuant()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
