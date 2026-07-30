/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_tiling_check.cc
 * \brief
 */

#include "custom_fused_infer_attention_v310_tiling.h"
#include "custom_fused_infer_attention_v310_tiling_base.h"
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"


using namespace ge;
namespace optiling {

ge::graphStatus CustomFIATiling::CheckBaseInputsNull()
{
    // Check base input tensors
    OPS_ERR_IF(context_->query.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.shape->GetStorageShape().GetShapeSize() == 0,
               OPS_LOG_E(context_->opName, "Tensor q is empty cause shapesize is 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->actualSeqLengths.tensor == nullptr,
               OPS_LOG_E(context_->opName, "actualSeqLengths tensor is nullptr"),
               return ge::GRAPH_FAILED);

    // Check base input attrs
    OPS_ERR_IF(context_->innerPrecise == nullptr, OPS_LOG_E(context_->opName, "attr innerPrecise is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->numHeads == nullptr, OPS_LOG_E(context_->opName, "attr numHeads is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->scaleValue == nullptr, OPS_LOG_E(context_->opName, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->kvHeadNums == nullptr, OPS_LOG_E(context_->opName, "attr kvHeadNums is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->layOut == nullptr, OPS_LOG_E(context_->opName, "attr layOut is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->blockSize == nullptr, OPS_LOG_E(context_->opName, "attr blockSize is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckInputParameterFormat()
{
    // No format whitelist: the kernel only cares about storage shape dimensions,
    // not the origin format tag.  Accepting all formats allows callers to pass
    // NZ-layout tensors (with either ND or NZ metadata) without being rejected.
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckCustomFIABaseParams()
{
    OPS_ERR_IF((numHeads_ == 0),
        OPS_LOG_E(context_->opName, "headSize is invalid."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((numKvHeads_ == 0 || numHeads_ % numKvHeads_ != 0),
        OPS_LOG_E(context_->opName, "kvHead is invalid."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((context_->kCache.empty()),
        OPS_LOG_E(context_->opName, "kCache is null."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((context_->kCache[0] == nullptr),
        OPS_LOG_E(context_->opName, "kCache[0] shape is null."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((context_->vCache.empty()),
        OPS_LOG_E(context_->opName, "vCache is null."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((context_->vCache[0] == nullptr),
        OPS_LOG_E(context_->opName, "vCache[0] shape is null."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckCustomFIAInputDtype()
{
    OPS_ERR_IF((inputQType_ != ge::DT_FLOAT16),
        OPS_LOG_E(context_->opName, "query dtype %u invalid, should be float16", inputQType_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((inputKvType_ != ge::DT_FLOAT16),
        OPS_LOG_E(context_->opName, "key and value dtype %u invalid, should be float16", inputKvType_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckCustomFIAPageAttention()
{
    static const size_t BLOCK_TABLE_DIM_NUM = 2;

    OPS_ERR_IF(context_->blockTable.desc == nullptr,
        OPS_LOG_E(context_->opName, "blockTable desc is nullptr"),
        return ge::GRAPH_FAILED);

    auto blockTablesShape = context_->blockTable.tensor->GetStorageShape();
    ge::DataType inputBlockTableType_ = context_->blockTable.desc->GetDataType();
    int64_t taskNumI64 = static_cast<int64_t>(numHeads_) * batchSize_;

    OPS_ERR_IF((inputBlockTableType_ != ge::DT_INT32),
        OPS_LOG_E(context_->opName, "block_table dtype %u invalid, should be int32", inputBlockTableType_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((blockTablesShape.GetDimNum() != BLOCK_TABLE_DIM_NUM),
        OPS_LOG_E(context_->opName, "blockTables dim num %lu, invalid, should be %lu",
            blockTablesShape.GetDimNum(), BLOCK_TABLE_DIM_NUM),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((taskNumI64 > UINT32_MAX),
        OPS_LOG_E(context_->opName, "numHeads * batchSize overflow"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((blockSize_ == 0 || blockSize_ % 16 != 0),
        OPS_LOG_E(context_->opName, "blockSize is invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((headDim_ * blockSize_ > 128 * 128),
        OPS_LOG_E(context_->opName, "headDim * blockSize should no greater than 128 * 128"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckCustomFIAQueryShape(const gert::StorageShape *queryShape)
{
    static const size_t Q_CACHE_DIM_NUM_BSND = 4;
    static const size_t Q_CACHE_DIM_NUM_TND = 3;

    const size_t queryDimNum = queryShape->GetStorageShape().GetDimNum();

    if (inputLayout_ == IfaLayout::TND) {
        OPS_ERR_IF((queryDimNum != Q_CACHE_DIM_NUM_TND),
            OPS_LOG_E(context_->opName,
                "query dim num %lu, invalid, should be %lu for TND",
                queryDimNum, Q_CACHE_DIM_NUM_TND),
            return ge::GRAPH_FAILED);
    } else if (inputLayout_ == IfaLayout::BSND){
        OPS_ERR_IF((queryDimNum != Q_CACHE_DIM_NUM_BSND),
            OPS_LOG_E(context_->opName,
                "query dim num %lu, invalid, should be %lu",
                queryDimNum, Q_CACHE_DIM_NUM_BSND),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckCustomFIAKvShapeAndToken(const gert::StorageShape *queryShape,
                                                   const gert::StorageShape *keyShape,
                                                   const gert::StorageShape *valueShape)
{
    static const size_t KV_CACHE_DIM_NUM = 4;

    OPS_ERR_IF((keyShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
        OPS_LOG_E(context_->opName,
            "key dim num %lu, invalid, should be %lu",
            keyShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((valueShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
        OPS_LOG_E(context_->opName,
            "value dim num %lu, invalid, should be %lu",
            valueShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((keyShape->GetStorageShape().GetDim(3) != 16),
        OPS_LOG_E(context_->opName, "K_cache Shape should be in nz format"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((valueShape->GetStorageShape().GetDim(3) != 16),
        OPS_LOG_E(context_->opName, "V_cache Shape should be in nz format"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((headDim_ == 0 || headDim_ > 256),
        OPS_LOG_E(context_->opName, "headdim is invalid"),
        return ge::GRAPH_FAILED);

    int64_t numTokensI64 = (inputLayout_ == IfaLayout::TND) ?
        queryShape->GetStorageShape().GetDim(0) :
        queryShape->GetStorageShape().GetDim(2);

    OPS_ERR_IF((numTokensI64 <= 0 || numTokensI64 > INT32_MAX),
        OPS_LOG_E(context_->opName, "numTokens must be in (0, INT32_MAX]"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::ProcessCheckCustomFIAInput()
{
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->kCache[0];
    const gert::StorageShape *valueShape = context_->vCache[0];

    if (CheckCustomFIABaseParams() != ge::GRAPH_SUCCESS ||
        CheckCustomFIAInputDtype() != ge::GRAPH_SUCCESS ||
        CheckCustomFIAPageAttention() != ge::GRAPH_SUCCESS ||
        CheckCustomFIAQueryShape(queryShape) != ge::GRAPH_SUCCESS ||
        CheckCustomFIAKvShapeAndToken(queryShape, keyShape, valueShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CheckInputFormatAndLimits()
{
    if (CheckInputParameterFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ProcessCheckCustomFIAInput();
}

} // namespace optiling
