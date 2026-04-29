/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quest_block_select_paged_tiling.h"
#include <algorithm>
#include "../op_kernel/quest_block_select_paged_tilingkey.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t MAXBLOCKS_INDEX = 1;
constexpr uint32_t MINBLOCKS_INDEX = 2;
constexpr uint32_t METADATA_BLOCK_TABLES_INDEX = 3;
constexpr uint32_t SEQ_LENS_INDEX = 4;
constexpr uint32_t ATTR_TOKENS_SINCE_METADATA_UPDATE_INDEX = 0;
constexpr uint32_t QUERY_DIM_NUM = 3;
constexpr uint32_t BLOCKS_DIM_NUM = 4;
constexpr uint32_t TABLE_DIM_NUM = 2;
constexpr uint32_t SEQ_LEN_DIM_NUM = 1;
constexpr uint32_t OUTPUT_DIM_NUM = 3;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
} // namespace

static ge::graphStatus QuestBlockSelectPagedTilingFunc(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("QuestBlockSelectPaged", "Tiling context is null."),
               return ge::GRAPH_FAILED);

    auto platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0,
               OPS_LOG_E(context->GetNodeName(), "GetCoreNumAiv returned 0."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape *query_shape = context->GetInputShape(QUERY_INDEX);
    const gert::StorageShape *maxblocks_shape = context->GetInputShape(MAXBLOCKS_INDEX);
    const gert::StorageShape *minblocks_shape = context->GetInputShape(MINBLOCKS_INDEX);
    const gert::StorageShape *metadata_block_tables_shape =
        context->GetInputShape(METADATA_BLOCK_TABLES_INDEX);
    const gert::StorageShape *seq_lens_shape = context->GetInputShape(SEQ_LENS_INDEX);
    const gert::StorageShape *selected_indices_shape = context->GetOutputShape(0);
    OPS_ERR_IF(query_shape == nullptr || maxblocks_shape == nullptr || minblocks_shape == nullptr ||
                   metadata_block_tables_shape == nullptr || seq_lens_shape == nullptr ||
                   selected_indices_shape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Required tensor shape is null."),
               return ge::GRAPH_FAILED);

    const auto &query_storage = query_shape->GetStorageShape();
    const auto &maxblocks_storage = maxblocks_shape->GetStorageShape();
    const auto &minblocks_storage = minblocks_shape->GetStorageShape();
    const auto &metadata_block_tables_storage =
        metadata_block_tables_shape->GetStorageShape();
    const auto &seq_lens_storage = seq_lens_shape->GetStorageShape();
    const auto &selected_indices_storage = selected_indices_shape->GetStorageShape();

    OPS_ERR_IF(query_storage.GetDimNum() != QUERY_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "query must be 3D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDimNum() != BLOCKS_DIM_NUM ||
                   minblocks_storage.GetDimNum() != BLOCKS_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "metadata tensors must be 4D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(metadata_block_tables_storage.GetDimNum() != TABLE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "metadata_block_tables must be 2D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seq_lens_storage.GetDimNum() != SEQ_LEN_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "seq_lens must be 1D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(selected_indices_storage.GetDimNum() != OUTPUT_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "selected_indices must be 3D."),
               return ge::GRAPH_FAILED);

    const auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t *tokens_since_metadata_update =
        attrs->GetInt(ATTR_TOKENS_SINCE_METADATA_UPDATE_INDEX);
    OPS_LOG_E_IF_NULL(context, tokens_since_metadata_update, return ge::GRAPH_FAILED);

    QuestBlockSelectPagedTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(query_storage.GetDim(DIM_0)));
    tiling.set_numKvHeads(static_cast<uint32_t>(maxblocks_storage.GetDim(DIM_2)));
    tiling.set_numHeads(static_cast<uint32_t>(query_storage.GetDim(DIM_1)));
    tiling.set_blockSize(static_cast<uint32_t>(maxblocks_storage.GetDim(DIM_1)));
    tiling.set_headDim(static_cast<uint32_t>(query_storage.GetDim(DIM_2)));
    tiling.set_maxMetadataBlocksPerRequest(
        static_cast<uint32_t>(metadata_block_tables_storage.GetDim(DIM_1)));
    tiling.set_k(static_cast<uint32_t>(selected_indices_storage.GetDim(DIM_2)));
    tiling.set_tokensSinceMetadataUpdate(
        static_cast<int32_t>(*tokens_since_metadata_update));

    const uint32_t batch_heads = tiling.get_batchSize() * tiling.get_numHeads();
    context->SetBlockDim(batch_heads == 0 ? 1 : std::min(batch_heads, aiv_num));

    const auto query_desc = context->GetInputDesc(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, query_desc, return ge::GRAPH_FAILED);
    context->SetTilingKey(
        query_desc->GetDataType() == ge::DT_BF16 ? QUEST_BLOCK_SELECT_PAGED_TILING_BF16
                                                 : QUEST_BLOCK_SELECT_PAGED_TILING_FP16);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct QuestBlockSelectPagedCompileInfo {};

static ge::graphStatus QuestBlockSelectPagedTilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuestBlockSelectPaged)
    .Tiling(QuestBlockSelectPagedTilingFunc)
    .TilingParse<QuestBlockSelectPagedCompileInfo>(QuestBlockSelectPagedTilingParse);
} // namespace optiling
