/*
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

#include <cstdint>
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "../op_kernel/attn_metadata_update_tiling.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

namespace {
// Input indices
constexpr uint32_t INPUT_REQ_IDS_IDX           = 0;
constexpr uint32_t INPUT_DISCARD_LABEL_IDX      = 1;
constexpr uint32_t INPUT_POSITIONS_IDX          = 2;
constexpr uint32_t INPUT_SLOT_MAPPING_IDX       = 3;
constexpr uint32_t INPUT_BLOCK_TABLE_IDX        = 4;
constexpr uint32_t INPUT_SEQ_LENS_IDX           = 5;
constexpr uint32_t INPUT_DISCARD_POSITIONS_IDX  = 6;
constexpr uint32_t INPUT_DISCARD_SLOT_MAP_IDX   = 7;
constexpr uint32_t INPUT_DISCARD_CTX_LENS_IDX   = 8;
constexpr uint32_t INPUT_DISCARD_BLOCK_TBL_IDX  = 9;
constexpr uint32_t INPUT_DISCARD_CNT_IDX        = 10;

// Attr indices
constexpr uint32_t ATTR_MAX_BS_IDX      = 0;
constexpr uint32_t ATTR_LAYER_IDX_IDX   = 1;

constexpr uint32_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;
}  // namespace

namespace optiling {

static ge::graphStatus AttnMetadataUpdateTilingFunc(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    AttnMetadataUpdateTilingData *tilingData =
        context->GetTilingData<AttnMetadataUpdateTilingData>();
    OPS_CHECK(tilingData == nullptr,
              OPS_LOG_E(nodeName, "tilingData is nullptr."),
              return ge::GRAPH_FAILED);

    // Read attrs
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OPS_LOG_E(nodeName, "attrs is nullptr."),
              return ge::GRAPH_FAILED);

    auto maxBsPtr      = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_IDX);
    auto layerIdxPtr   = attrs->GetAttrPointer<int64_t>(ATTR_LAYER_IDX_IDX);

    OPS_CHECK(maxBsPtr == nullptr,
              OPS_LOG_E(nodeName, "max_bs attr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(layerIdxPtr == nullptr,
              OPS_LOG_E(nodeName, "layer_idx attr is null."), return ge::GRAPH_FAILED);

    tilingData->info.maxBs      = static_cast<uint32_t>(*maxBsPtr);
    tilingData->info.layerIdx   = static_cast<uint32_t>(*layerIdxPtr);

    // Derive table_size from block_table shape (dim 1)
    const gert::StorageShape *blockTableShape =
        context->GetInputShape(INPUT_BLOCK_TABLE_IDX);
    OPS_CHECK(blockTableShape == nullptr,
              OPS_LOG_E(nodeName, "blockTable shape is null."), return ge::GRAPH_FAILED);
    tilingData->info.tableSize =
        static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));

    // Derive maxDiscardBs from discard_positions shape dim 1.
    // discardAttnCnt tensor shape is (layer_num,); actual count is read at runtime.
    // For tiling we use max_discard_bs from discard_positions shape dim 1.
    const gert::StorageShape *discardPosShape =
        context->GetInputShape(INPUT_DISCARD_POSITIONS_IDX);
    OPS_CHECK(discardPosShape == nullptr,
              OPS_LOG_E(nodeName, "discardPositions shape is null."), return ge::GRAPH_FAILED);
    tilingData->info.maxDiscardBs =
        static_cast<uint32_t>(discardPosShape->GetStorageShape().GetDim(1));

    // Platform info
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData->info.totalUbSize = ubSize;

    // Multi-core parallel kernel: distribute work across all AIV cores
    constexpr uint32_t AIV_NUM = 48U;
    context->SetBlockDim(AIV_NUM);

    // Workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_CHECK(workSpaces == nullptr,
              OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_WORKSPACE_SIZE;

    OPS_LOG_D(nodeName,
              "AttnMetadataUpdate tiling: maxBs=%u  tableSize=%u layerIdx=%u maxDiscardBs=%u",
              tilingData->info.maxBs,
              tilingData->info.tableSize, tilingData->info.layerIdx,
              tilingData->info.maxDiscardBs);

    return ge::GRAPH_SUCCESS;
}

struct AttnMetadataUpdateCompileInfo {};

ge::graphStatus TilingParseForAttnMetadataUpdate(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttnMetadataUpdate)
    .Tiling(AttnMetadataUpdateTilingFunc)
    .TilingParse<AttnMetadataUpdateCompileInfo>(TilingParseForAttnMetadataUpdate);

}  // namespace optiling
