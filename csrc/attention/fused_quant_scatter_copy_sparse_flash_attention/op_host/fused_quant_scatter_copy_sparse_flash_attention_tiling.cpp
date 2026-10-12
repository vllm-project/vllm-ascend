/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_scatter_copy_sparse_flash_attention_tiling.cpp
 * \brief
 */

#include <climits>
#include <cmath>
#include <cstdint>
#include "fused_quant_scatter_copy_sparse_flash_attention_tiling.h"
#include "../op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#include "../op_kernel/fused_quant_scatter_copy_sparse_flash_attention_template_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t INPUT_COUNT = 14;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t HBM_KV_INDEX = 1;
constexpr uint32_t DRAM_KV_INDEX = 2;
constexpr uint32_t HBM_BLOCK_TABLE_INDEX = 3;
constexpr uint32_t DRAM_BLOCK_TABLE_INDEX = 4;
constexpr uint32_t TOPK_DST_SLOTS_INDEX = 5;
constexpr uint32_t TOPK_SRC_IDS_INDEX = 6;
constexpr uint32_t TOPK_MISS_COUNTS_INDEX = 7;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_QUERY_INDEX = 8;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KV_INDEX = 9;
constexpr uint32_t NUM_CACHE_TOKENS_INDEX = 10;
constexpr uint32_t MISS_SRC_IDS_INDEX = 11;
constexpr uint32_t MISS_DST_SLOTS_INDEX = 12;
constexpr uint32_t MISS_COUNTS_INDEX = 13;

bool IsSupportedHeadCount(int64_t headCount)
{
    return headCount >= fused_quant_scatter_copy_sfa::MIN_QUERY_HEAD_COUNT &&
        headCount <= fused_quant_scatter_copy_sfa::MAX_QUERY_HEAD_COUNT &&
        fused_quant_scatter_copy_sfa::IsSupportedHeadCount(static_cast<uint32_t>(headCount));
}

bool IsVector(const gert::Shape &shape, int64_t rows)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == rows;
}

bool IsMatrix(const gert::Shape &shape, int64_t rows, int64_t width)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) == rows && shape.GetDim(1) == 1 && shape.GetDim(2) == width;
}

ge::graphStatus CheckParameters(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr || context->GetAttrs() == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index = 0; index < INPUT_COUNT; ++index) {
        if (context->GetInputShape(index) == nullptr || context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType queryType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    if ((queryType != ge::DT_BF16 && queryType != ge::DT_FLOAT16) ||
        context->GetInputDesc(HBM_KV_INDEX)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_KV_INDEX)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index = HBM_BLOCK_TABLE_INDEX; index < INPUT_COUNT; ++index) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const auto &query = context->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &numCacheTokens = context->GetInputShape(NUM_CACHE_TOKENS_INDEX)->GetStorageShape();
    if (query.GetDimNum() != 3 || numCacheTokens.GetDimNum() != 1 || numCacheTokens.GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batchSize = numCacheTokens.GetDim(0);
    const int64_t queryCount = query.GetDim(0);
    if (queryCount <= 0 || query.GetDim(2) != fused_quant_scatter_copy_sfa::QUERY_DIM ||
        !IsSupportedHeadCount(query.GetDim(1)) ||
        batchSize > INT32_MAX / fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT || queryCount < batchSize ||
        queryCount > static_cast<int64_t>(fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT) * batchSize) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t index : {HBM_KV_INDEX, DRAM_KV_INDEX}) {
        const auto &shape = context->GetInputShape(index)->GetStorageShape();
        if (shape.GetDimNum() != 4 || shape.GetDim(0) <= 0 || shape.GetDim(0) > UINT32_MAX ||
            shape.GetDim(1) != fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE ||
            shape.GetDim(2) != fused_quant_scatter_copy_sfa::KV_HEAD_COUNT ||
            shape.GetDim(3) != fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES) {
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t index : {HBM_BLOCK_TABLE_INDEX, DRAM_BLOCK_TABLE_INDEX}) {
        const auto &shape = context->GetInputShape(index)->GetStorageShape();
        if (shape.GetDimNum() != 2 || shape.GetDim(0) != batchSize || shape.GetDim(1) <= 0 ||
            shape.GetDim(1) > fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!IsMatrix(context->GetInputShape(TOPK_DST_SLOTS_INDEX)->GetStorageShape(), queryCount,
            fused_quant_scatter_copy_sfa::TOPK_CAPACITY) ||
        !IsMatrix(context->GetInputShape(TOPK_SRC_IDS_INDEX)->GetStorageShape(), queryCount,
            fused_quant_scatter_copy_sfa::TOPK_CAPACITY) ||
        !IsVector(context->GetInputShape(TOPK_MISS_COUNTS_INDEX)->GetStorageShape(), queryCount) ||
        !IsVector(context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY_INDEX)->GetStorageShape(), batchSize) ||
        !IsVector(context->GetInputShape(ACTUAL_SEQ_LENGTHS_KV_INDEX)->GetStorageShape(), batchSize) ||
        !IsVector(context->GetInputShape(NUM_CACHE_TOKENS_INDEX)->GetStorageShape(), batchSize) ||
        !IsMatrix(context->GetInputShape(MISS_SRC_IDS_INDEX)->GetStorageShape(), batchSize,
            fused_quant_scatter_copy_sfa::COPY_CAPACITY) ||
        !IsMatrix(context->GetInputShape(MISS_DST_SLOTS_INDEX)->GetStorageShape(), batchSize,
            fused_quant_scatter_copy_sfa::COPY_CAPACITY) ||
        !IsVector(context->GetInputShape(MISS_COUNTS_INDEX)->GetStorageShape(), batchSize)) {
        return ge::GRAPH_FAILED;
    }

    const float *scaleValue = context->GetAttrs()->GetAttrPointer<float>(0);
    return scaleValue != nullptr && std::isfinite(*scaleValue) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TilingFusedQuantScatterCopySparseFlashAttention(gert::TilingContext *context)
{
    if (CheckParameters(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto &query = context->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &hbmKv = context->GetInputShape(HBM_KV_INDEX)->GetStorageShape();
    const auto &dramKv = context->GetInputShape(DRAM_KV_INDEX)->GetStorageShape();
    const auto &hbmBlockTable = context->GetInputShape(HBM_BLOCK_TABLE_INDEX)->GetStorageShape();
    const auto &dramBlockTable = context->GetInputShape(DRAM_BLOCK_TABLE_INDEX)->GetStorageShape();
    const auto &numCacheTokens = context->GetInputShape(NUM_CACHE_TOKENS_INDEX)->GetStorageShape();
    const float scaleValue = *context->GetAttrs()->GetAttrPointer<float>(0);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCoreNum = platform.GetCoreNumAic();
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aicCoreNum == 0 || aivCoreNum < 2 * aicCoreNum || (aicCoreNum & 1U) != 0) {
        return ge::GRAPH_FAILED;
    }

    FusedQuantScatterCopySparseFlashAttentionTilingData tilingData;
    tilingData.baseParams.set_batchSize(static_cast<uint32_t>(numCacheTokens.GetDim(0)));
    tilingData.baseParams.set_seqSize(static_cast<uint32_t>(hbmBlockTable.GetDim(1) *
        fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE));
    tilingData.baseParams.set_qSeqSize(static_cast<uint32_t>(query.GetDim(0)));
    tilingData.baseParams.set_nNumOfQInOneGroup(static_cast<uint32_t>(query.GetDim(1)));
    tilingData.baseParams.set_sparseBlockCount(fused_quant_scatter_copy_sfa::TOPK_CAPACITY);
    tilingData.baseParams.set_sparseMode(3);
    tilingData.baseParams.set_outputLayout(1);
    tilingData.baseParams.set_blockSize(fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE);
    tilingData.baseParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(hbmBlockTable.GetDim(1)));
    tilingData.baseParams.set_isActualLenDimsNull(0);
    tilingData.baseParams.set_isActualLenDimsKVNull(0);
    tilingData.baseParams.set_dSizeVInput(fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES);
    tilingData.baseParams.set_scaleValue(scaleValue);
    tilingData.singleCoreParams.set_usedCoreNum(aicCoreNum);
    tilingData.set_dramMaxBlocks(static_cast<uint32_t>(dramBlockTable.GetDim(1)));
    tilingData.set_hbmPhysicalBlocks(static_cast<uint32_t>(hbmKv.GetDim(0)));
    tilingData.set_dramPhysicalBlocks(static_cast<uint32_t>(dramKv.GetDim(0)));

    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() +
        fused_quant_scatter_copy_sfa::UserWorkspaceBytes(
            aicCoreNum, static_cast<uint32_t>(query.GetDim(0)));
    context->SetBlockDim(aicCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(
        query.GetDim(1) == fused_quant_scatter_copy_sfa::MAX_QUERY_HEAD_COUNT));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct FusedQuantScatterCopySparseFlashAttentionCompileInfo {};

ge::graphStatus TilingPrepareFusedQuantScatterCopySparseFlashAttention(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(FusedQuantScatterCopySparseFlashAttention)
    .Tiling(TilingFusedQuantScatterCopySparseFlashAttention)
    .TilingParse<FusedQuantScatterCopySparseFlashAttentionCompileInfo>(
        TilingPrepareFusedQuantScatterCopySparseFlashAttention);
} // namespace optiling
