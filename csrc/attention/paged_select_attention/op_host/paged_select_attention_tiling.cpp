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
 * \file paged_select_attention_tiling.cpp
 * \brief Host tiling entry for specialized sparse paged decode attention.
 */

#include <algorithm>
#include "paged_select_attention_tiling.h"
#include "paged_select_attention_infer_tiling.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"

using namespace ge;

namespace optiling {
REGISTER_TILING_DATA_CLASS(PagedSelectAttention, PagedSelectAttentionTilingData)

namespace {
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;

graphStatus ValidateShapesAndAttrs(gert::TilingContext *context, PagedSelectAttentionContext &faInfo)
{
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GetAttrs() returned nullptr."),
        return ge::GRAPH_FAILED);

    auto queryShape = context->GetInputShape(QUERY_INDEX);
    auto keyShape = context->GetInputShape(KEY_INDEX);
    auto valueShape = context->GetInputShape(VALUE_INDEX);
    auto blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    auto selectedShape = context->GetInputShape(SELECTED_KV_INDICES_INDEX);
    auto outputShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    auto valueDesc = context->GetInputDesc(VALUE_INDEX);
    auto actualQSeqTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualKvSeqTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);

    OPS_ERR_IF(queryShape == nullptr || keyShape == nullptr || valueShape == nullptr || outputShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query/key/value/output shapes must be present."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(queryDesc == nullptr || keyDesc == nullptr || valueDesc == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query/key/value descs must be present."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockTableShape == nullptr || selectedShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "PagedSelectAttention requires block_table and selected_kv_indices."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(actualQSeqTensor == nullptr || actualKvSeqTensor == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "PagedSelectAttention requires actual_seq_lengths and actual_seq_lengths_kv."),
        return ge::GRAPH_FAILED);

    const int32_t *numHeadsPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_HEADS_INDEX);
    const int32_t *numKvHeadsPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    const int32_t *blockSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    const float *scalePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_VALUE_INDEX);
    OPS_ERR_IF(numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || blockSizePtr == nullptr || scalePtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Required attrs are missing."),
        return ge::GRAPH_FAILED);

    const int32_t numHeads = *numHeadsPtr;
    const int32_t numKvHeads = *numKvHeadsPtr;
    const int32_t blockSize = *blockSizePtr;
    OPS_ERR_IF(numHeads <= 0 || numKvHeads <= 0 || (numHeads % numKvHeads) != 0 || blockSize <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Invalid num_heads/num_key_value_heads/block_size attrs."),
        return ge::GRAPH_FAILED);

    const auto &qStorage = queryShape->GetStorageShape();
    const auto &kStorage = keyShape->GetStorageShape();
    const auto &vStorage = valueShape->GetStorageShape();
    const auto &blockStorage = blockTableShape->GetStorageShape();
    const auto &selectedStorage = selectedShape->GetStorageShape();

    OPS_ERR_IF(qStorage.GetDimNum() != DIM_NUM_3 || kStorage.GetDimNum() != DIM_NUM_3 ||
            vStorage.GetDimNum() != DIM_NUM_3,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "query/key/value must be rank-3 TND and paged-cache tensors."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockStorage.GetDimNum() != DIM_NUM_2 || selectedStorage.GetDimNum() != DIM_NUM_3,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "block_table must be rank-2 and selected_kv_indices must be rank-3."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(selectedStorage.GetDim(DIM_2) <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "selected_kv_indices.shape[2] must be > 0."),
        return ge::GRAPH_FAILED);

    int64_t totalQ = qStorage.GetDim(DIM_0);
    int64_t queryHeads = qStorage.GetDim(DIM_1);
    int64_t headDim = qStorage.GetDim(DIM_2);
    int64_t numBlocks = kStorage.GetDim(DIM_0);
    int64_t keyBlockSize = kStorage.GetDim(DIM_1);
    int64_t keyWidth = kStorage.GetDim(DIM_2);
    int64_t valueWidth = vStorage.GetDim(DIM_2);
    int64_t batch = blockStorage.GetDim(DIM_0);
    int64_t maxBlocksPerBatch = blockStorage.GetDim(DIM_1);
    int64_t selectedBatch = selectedStorage.GetDim(DIM_0);
    int64_t selectedHeads = selectedStorage.GetDim(DIM_1);

    OPS_ERR_IF(queryHeads != numHeads || selectedHeads != numHeads,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "query/select head dimension must equal num_heads."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyBlockSize != blockSize || vStorage.GetDim(DIM_1) != blockSize,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "key/value block size must equal block_size attr."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyWidth != static_cast<int64_t>(numKvHeads) * headDim ||
            valueWidth != static_cast<int64_t>(numKvHeads) * headDim,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "key/value width must equal num_key_value_heads * head_dim."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch <= 0 || selectedBatch != batch || numBlocks <= 0 || maxBlocksPerBatch <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Invalid paged metadata shapes."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(outputShape->GetStorageShape() != qStorage,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "attention_out shape must equal query shape."),
        return ge::GRAPH_FAILED);

    const ge::DataType queryDtype = queryDesc->GetDataType();
    const ge::DataType keyDtype = keyDesc->GetDataType();
    const ge::DataType valueDtype = valueDesc->GetDataType();
    OPS_ERR_IF(queryDtype != keyDtype || queryDtype != valueDtype,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "query, key, and value must share the same dtype."),
        return ge::GRAPH_FAILED);

    faInfo.numTokens = static_cast<int32_t>(totalQ);
    faInfo.numHeads = numHeads;
    faInfo.embeddingSize = static_cast<int32_t>(headDim);
    faInfo.embeddingSizeV = static_cast<int32_t>(headDim);
    faInfo.numBlocks = static_cast<int32_t>(numBlocks);
    faInfo.blockSize = blockSize;
    faInfo.kvHeads = numKvHeads;
    faInfo.batch = static_cast<int32_t>(batch);
    faInfo.maxNumBlocksPerBatch = static_cast<uint32_t>(maxBlocksPerBatch);
    faInfo.scaleValue = *scalePtr;
    faInfo.maskType = MaskType::NO_MASK;
    faInfo.dataType = (queryDtype == ge::DT_BF16) ?
        DataType::BF16 : DataType::FP16;
    faInfo.sparseLaunchEnabled = true;
    faInfo.sparseKMax = static_cast<uint32_t>(selectedStorage.GetDim(DIM_2));
    return ge::GRAPH_SUCCESS;
}

graphStatus ValidateData(gert::TilingContext *context, PagedSelectAttentionContext &faInfo)
{
    auto actualQSeqTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualKvSeqTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto selectedTensor = context->GetInputTensor(SELECTED_KV_INDICES_INDEX);
    auto blockTableTensor = context->GetInputTensor(BLOCK_TABLE_INDEX);
    OPS_ERR_IF(actualQSeqTensor == nullptr || actualKvSeqTensor == nullptr || selectedTensor == nullptr ||
            blockTableTensor == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Required tensors are missing."),
        return ge::GRAPH_FAILED);

    const int64_t *actualQSeq = actualQSeqTensor->GetData<int64_t>();
    const int64_t *actualKvSeq = actualKvSeqTensor->GetData<int64_t>();
    if (actualQSeq == nullptr || actualKvSeq == nullptr) {
        faInfo.isTilingSink = true;
        faInfo.maxQSeqlen = 1;
        faInfo.maxKvSeqlen = static_cast<int64_t>(faInfo.maxNumBlocksPerBatch) * faInfo.blockSize;
        return ge::GRAPH_SUCCESS;
    }

    faInfo.qSeqlenList = actualQSeq;
    faInfo.kvSeqlenList = actualKvSeq;
    faInfo.isTilingSink = false;
    faInfo.maxQSeqlen = 1;
    faInfo.maxKvSeqlen = 0;

    int64_t prevQ = 0;
    for (int32_t batchIdx = 0; batchIdx < faInfo.batch; ++batchIdx) {
        int64_t qSeq = actualQSeq[batchIdx];
        int64_t kvSeq = actualKvSeq[batchIdx];
        OPS_ERR_IF(qSeq <= prevQ || (qSeq - prevQ) != 1,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "PagedSelectAttention requires pure decode TND cumulative q lens with q_len=1 per batch."),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(kvSeq < 0 || kvSeq > static_cast<int64_t>(faInfo.maxNumBlocksPerBatch) * faInfo.blockSize,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Invalid actual_seq_lengths_kv value for paged decode."),
            return ge::GRAPH_FAILED);
        prevQ = qSeq;
        faInfo.maxKvSeqlen = std::max(faInfo.maxKvSeqlen, kvSeq);
    }

    OPS_ERR_IF(prevQ != faInfo.numTokens,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "query T must equal the last cumulative actual_seq_lengths entry."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

graphStatus PopulateContext(gert::TilingContext *context, PagedSelectAttentionContext &faInfo)
{
    OPS_ERR_IF(ValidateShapesAndAttrs(context, faInfo) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "PagedSelectAttention shape/attr validation failed."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(ValidateData(context, faInfo) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "PagedSelectAttention sparse contract validation failed."),
        return ge::GRAPH_FAILED);
    faInfo.workspaces = context->GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

graphStatus SaveTilingResult(
    gert::TilingContext *context,
    PagedSelectAttentionTiling &tiling,
    PagedSelectAttentionTilingData &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = 16U * 1024U * 1024U +
            static_cast<uint64_t>(tiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U;
    }
    context->SetBlockDim(tiling.GetCoreNum());
    context->SetTilingKey(tiling.GetTilingKey());
    return ge::GRAPH_SUCCESS;
}
} // namespace

graphStatus TilingPagedSelectAttention(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("PagedSelectAttention", "context is nullptr."),
        return ge::GRAPH_FAILED);

    PagedSelectAttentionContext faInfo;
    OPS_ERR_IF(PopulateContext(context, faInfo) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Failed to build paged select attention context."),
        return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "PlatformInfoPtr is null."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    PagedSelectAttentionTilingData tilingData;
    PagedSelectAttentionTiling tiling(faInfo);
    tiling.SetCoreNum(ascendcPlatform.GetCoreNumAic());
    OPS_ERR_IF(tiling.DoTiling(tilingData) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "DoTiling failed."),
        return ge::GRAPH_FAILED);
    return SaveTilingResult(context, tiling, tilingData);
}

FIA_EXTERN_C graphStatus DoOpTilingPagedSelectAttention(gert::TilingContext *context)
{
    return TilingPagedSelectAttention(context);
}

extern "C" graphStatus DeviceDoOpTilingPagedSelectAttention(gert::TilingContext *context)
{
    return TilingPagedSelectAttention(context);
}
} // namespace optiling
