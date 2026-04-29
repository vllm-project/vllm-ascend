/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quest_prefill_metadata_tiling.h"
#include <algorithm>
#include "../op_kernel/quest_prefill_metadata_tilingkey.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t K_CACHE_INDEX = 0;
constexpr uint32_t BLOCK_TABLES_INDEX = 1;
constexpr uint32_t SEQ_LENS_INDEX = 2;
constexpr uint32_t METADATA_BLOCK_TABLES_INDEX = 3;
constexpr uint32_t K_CACHE_DIM_NUM = 4;
constexpr uint32_t TABLE_DIM_NUM = 2;
constexpr uint32_t SEQ_LEN_DIM_NUM = 1;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
} // namespace

static ge::graphStatus QuestPrefillMetadataTilingFunc(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("QuestPrefillMetadata", "Tiling context is null."),
               return ge::GRAPH_FAILED);

    auto platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0,
               OPS_LOG_E(context->GetNodeName(), "GetCoreNumAiv returned 0."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape *k_cache_shape = context->GetInputShape(K_CACHE_INDEX);
    const gert::StorageShape *block_tables_shape = context->GetInputShape(BLOCK_TABLES_INDEX);
    const gert::StorageShape *seq_lens_shape = context->GetInputShape(SEQ_LENS_INDEX);
    const gert::StorageShape *metadata_block_tables_shape = context->GetInputShape(METADATA_BLOCK_TABLES_INDEX);
    OPS_ERR_IF(k_cache_shape == nullptr || block_tables_shape == nullptr || seq_lens_shape == nullptr ||
                   metadata_block_tables_shape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Required input shape is null."),
               return ge::GRAPH_FAILED);

    const auto &k_cache_storage = k_cache_shape->GetStorageShape();
    const auto &block_tables_storage = block_tables_shape->GetStorageShape();
    const auto &seq_lens_storage = seq_lens_shape->GetStorageShape();
    const auto &metadata_block_tables_storage =
        metadata_block_tables_shape->GetStorageShape();

    OPS_ERR_IF(k_cache_storage.GetDimNum() != K_CACHE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "k_cache must be 4D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(block_tables_storage.GetDimNum() != TABLE_DIM_NUM ||
                   metadata_block_tables_storage.GetDimNum() != TABLE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "block tables must be 2D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seq_lens_storage.GetDimNum() != SEQ_LEN_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "seq_lens must be 1D."),
               return ge::GRAPH_FAILED);

    const uint32_t batch_size = static_cast<uint32_t>(seq_lens_storage.GetDim(DIM_0));
    const uint32_t num_kv_heads = static_cast<uint32_t>(k_cache_storage.GetDim(DIM_2));
    const uint32_t block_size = static_cast<uint32_t>(k_cache_storage.GetDim(DIM_1));
    const uint32_t head_dim = static_cast<uint32_t>(k_cache_storage.GetDim(DIM_3));
    const uint32_t max_kv_blocks_per_request =
        static_cast<uint32_t>(block_tables_storage.GetDim(DIM_1));
    const uint32_t max_metadata_blocks_per_request =
        static_cast<uint32_t>(metadata_block_tables_storage.GetDim(DIM_1));

    QuestPrefillMetadataTilingData tiling;
    tiling.set_batchSize(batch_size);
    tiling.set_numKvHeads(num_kv_heads);
    tiling.set_blockSize(block_size);
    tiling.set_headDim(head_dim);
    tiling.set_maxKvBlocksPerRequest(max_kv_blocks_per_request);
    tiling.set_maxMetadataBlocksPerRequest(max_metadata_blocks_per_request);

    const uint32_t batch_heads = batch_size * num_kv_heads;
    context->SetBlockDim(batch_heads == 0 ? 1 : std::min(batch_heads, aiv_num));

    const auto k_cache_desc = context->GetInputDesc(K_CACHE_INDEX);
    OPS_LOG_E_IF_NULL(context, k_cache_desc, return ge::GRAPH_FAILED);
    context->SetTilingKey(
        k_cache_desc->GetDataType() == ge::DT_BF16 ? QUEST_PREFILL_METADATA_TILING_BF16
                                                   : QUEST_PREFILL_METADATA_TILING_FP16);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct QuestPrefillMetadataCompileInfo {};

static ge::graphStatus QuestPrefillMetadataTilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuestPrefillMetadata)
    .Tiling(QuestPrefillMetadataTilingFunc)
    .TilingParse<QuestPrefillMetadataCompileInfo>(QuestPrefillMetadataTilingParse);
} // namespace optiling
