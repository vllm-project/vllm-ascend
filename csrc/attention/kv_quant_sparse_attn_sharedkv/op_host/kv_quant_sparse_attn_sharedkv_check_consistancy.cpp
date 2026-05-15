/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_check_consistancy.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

ge::graphStatus KvQuantSASTilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            SASDataTypeToSerialString(expectDtype).c_str(),
            SASDataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const SASLayout &layout, const std::string &name) const
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.",
            KvQuantSASLayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const KvQuantSASTilingShapeCompareParam &param, const SASLayout &layout) const
{
    if (layout == SASLayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == SASLayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == SASLayout::PA_ND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", KvQuantSASLayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CompareShape(KvQuantSASTilingShapeCompareParam &param,
    const gert::Shape &shape, const SASLayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OP_LOGE(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OP_LOGE(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), KvQuantSASLayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void KvQuantSASTilingCheck::SetSASShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.q.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.cmpSparseIndices.tensor->GetShape().GetStorageShape();
    keyShapeCmp_ = opParamInfo_.oriKv.tensor->GetShape().GetStorageShape();
    valueShapeCmp_ = opParamInfo_.cmpKv.tensor->GetShape().GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attnOut.shape->GetStorageShape();
}

ge::graphStatus KvQuantSASTilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor != nullptr,
            OP_LOGE(opName_, "when the layout_kv is %s, %s should be null",
                KvQuantSASLayoutToSerialString(kvLayout_).c_str(), ORI_BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor != nullptr,
            OP_LOGE(opName_, "when the layout_kv is %s, %s should be null",
                KvQuantSASLayoutToSerialString(kvLayout_).c_str(), CMP_BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    
    uint32_t oriBlockTableBatch = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(oriBlockTableBatch != bSize_,
        OP_LOGE(opName_, "oriBlockTableBatch's first dimension(%u) should be equal to batch size(%u)",
            oriBlockTableBatch, bSize_),
        return ge::GRAPH_FAILED);
    
    uint32_t cmpBlockTableBatch = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(cmpBlockTableBatch != bSize_,
        OP_LOGE(opName_, "cmpBlockTableBatch's first dimension(%u) should be equal to batch size(%u)",
            cmpBlockTableBatch, bSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckTopkShape()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckAttenOutShape()
{
    KvQuantSASTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = 512; // 512:输出的head_dim
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attnOut.desc->GetDataType(),
        qType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckKVShapeForBatchContinuous()
{
    KvQuantSASTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t KvQuantSASTilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus KvQuantSASTilingCheck::CheckKVShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    KvQuantSASTilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = bSize_;
    shapeParams.T = kvTSize_;
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckKVShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus KvQuantSASTilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_,
        oriKvType_, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.cuSeqLensQ.desc == nullptr) {
        OP_LOGE(opName_, "cuSeqLensQ is not empty,"
            "but cuSeqLensQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "cuSeqLensQ's dtype is %s, it should be DT_INT32.",
            SASDataTypeToSerialString(opParamInfo_.cuSeqLensQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_ + 1) {
        OP_LOGE(opName_, "cuSeqLensQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.sequsedKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.sequsedKv.desc == nullptr) {
        OP_LOGE(opName_, "sequsedKv is not empty,"
            "but sequsedKv's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sequsedKv.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "sequsedKv's dtype is %s, it should be DT_INT32.",
            SASDataTypeToSerialString(opParamInfo_.sequsedKv.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.sequsedKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.sequsedKv.tensor, kvLayout_, "sequsedKv") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE(opName_, "sequsedKv shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckMultiParaConsistency()
{
    SetSASShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

}