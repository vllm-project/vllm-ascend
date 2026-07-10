/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_attention_score_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/error_log.h"

using namespace ge;
using namespace std;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SELECT_IDX_INDEX = 3;
constexpr int BLOCK_TABLE_INDEX = 4;
constexpr int SELECT_NUM_IDX_INDEX = 5;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

constexpr int SELECT_IDX_DIM_KV_HEAD = 0;
constexpr int SELECT_IDX_DIM_SEQ = 1;
constexpr int SELECT_IDX_DIM_TOPK = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;

constexpr uint32_t SOC_VER_950_CODE = 4;

namespace optiling {

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

ge::graphStatus SASATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "GetAttrs returned nullptr."), return ge::GRAPH_FAILED);

    const int64_t *numKvHeadsPtr = attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX);
    if (numKvHeadsPtr != nullptr) {
        kvHeads_ = static_cast<uint32_t>(*numKvHeadsPtr);
    }

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    const int64_t *blockSizePtr = attrs->GetInt(ATTR_BLOCK_SIZE_INDEX);
    if (blockSizePtr != nullptr) {
        blockSize_ = static_cast<uint32_t>(*blockSizePtr);
    }

    const int64_t *topKPtr = attrs->GetInt(ATTR_TOP_K_INDEX);
    if (topKPtr != nullptr) {
        topK_ = static_cast<uint32_t>(*topKPtr);
    }

    const int64_t *innerPrecPtr = attrs->GetInt(ATTR_INNER_PRECISE_INDEX);
    if (innerPrecPtr != nullptr) {
        innerPrecise_ = static_cast<uint32_t>(*innerPrecPtr);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseInputTensors(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Query shape is nullptr."), return ge::GRAPH_FAILED);

    totalQTokens_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Key shape is nullptr."), return ge::GRAPH_FAILED);

    if (kvHeads_ == 0) {
        kvHeads_ = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
    }

    const gert::StorageShape *blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_IF(blockTableShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "BlockTable shape is nullptr."), return ge::GRAPH_FAILED);

    batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));

    const gert::StorageShape *selectIdxShape = context->GetInputShape(SELECT_IDX_INDEX);
    OP_CHECK_IF(selectIdxShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "SelectIdx shape is nullptr."), return ge::GRAPH_FAILED);

    kvHeads_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_KV_HEAD));
    maxQSeqlen_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_SEQ));
    topK_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_TOPK));

    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }

    if (scaleValue_ < 1e-9f && scaleValue_ > -1e-9f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseSeqlens(gert::TilingContext *context)
{
    const gert::Tensor *seqLensTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    if (seqLensTensor != nullptr) {
        qSeqLenList_ = reinterpret_cast<const int32_t *>(seqLensTensor->GetAddr());
    }

    const gert::Tensor *seqLensKvTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (seqLensKvTensor != nullptr) {
        kvSeqLenList_ = reinterpret_cast<const int32_t *>(seqLensKvTensor->GetAddr());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::CalculateTaskSplit(gert::TilingContext *context)
{
    totalTaskNum_ = totalQTokens_ * kvHeads_;
    blockDim_ = std::min(totalTaskNum_, aicNum_);
    if (blockDim_ == 0) {
        blockDim_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (socVer_ != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        workSpaceSize_ = libapiSize_ + identityIdxSize + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_;
    } else {
        uint32_t dtypeSize = (dataType_ == ge::DT_FLOAT8_E4M3FN) ? 1 : 2;
        uint64_t perTaskWorkspace = static_cast<uint64_t>(topK_) * blockSize_ * embeddingSize_ * dtypeSize * 2;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        workSpaceSize_ = libapiSize_ + identityIdxSize + static_cast<uint64_t>(blockDim_) * perTaskWorkspace;
    }

    context->SetBlockDim(blockDim_);
    size_t *workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t SASATiling::GenerateTilingKey()
{
    if (socVer_ != SOC_VER_950_CODE) {
        if (dataType_ == ge::DT_BF16 && embeddingSize_ == 128 && blockSize_ == 128) {
            return SASA_BF16_D128_ARCH22_TILING;
        }
        if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
            return SASA_FP16_D128_ARCH22_TILING;
        }
        return SASA_FP16_D128_ARCH22_TILING;
    }
    if (dataType_ == ge::DT_FLOAT8_E4M3FN && embeddingSize_ == 128 && blockSize_ == 128) {
        return SASA_FP8_D128_TILING;
    }
    if (dataType_ == ge::DT_BF16 && embeddingSize_ == 128 && blockSize_ == 128) {
        return SASA_BF16_D128_TILING;
    }
    if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
        return SASA_FP16_D128_TILING;
    }
    return SASA_FP16_D128_TILING;
}

ge::graphStatus SASATiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_firstBatchTaskNum(kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    uint32_t groupSize = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    tilingData_->set_groupSize(groupSize);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    // BaseTileInfo
    uint32_t qBaseTile = (embeddingSize_ <= 128) ? 128 : 64;
    uint32_t kvBaseTile = blockSize_;
    tilingData_->set_qBaseTile(qBaseTile);
    tilingData_->set_kvBaseTile(kvBaseTile);

    // MmPhaseL1TileInfo: QK matmul L1 tile = [qBaseTile, kvBaseTile, embed]
    tilingData_->set_mm1L1TileM(qBaseTile);
    tilingData_->set_mm1L1TileN(kvBaseTile);
    tilingData_->set_mm1L1TileKLeft(embeddingSize_);
    tilingData_->set_mm1L1TileKRight(embeddingSize_);
    // PV matmul L1 tile = [qBaseTile, embed, kvBaseTile]
    tilingData_->set_mm2L1TileM(qBaseTile);
    tilingData_->set_mm2L1TileN(embeddingSize_);
    tilingData_->set_mm2L1TileKLeft(kvBaseTile);
    tilingData_->set_mm2L1TileKRight(kvBaseTile);
    // Buffer counts
    tilingData_->set_qL1BufNum(1);
    tilingData_->set_kL1BufNum(1);
    tilingData_->set_vL1BufNum(1);
    tilingData_->set_pL1BufNum(3);  // PRE_LAUNCH + 1

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::GetTiling(gert::TilingContext *context,
    SparseAttentionScoreTilingData &tilingData)
{
    tilingData_ = &tilingData;

    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::SetTilingData(gert::TilingContext *context,
    SparseAttentionScoreTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingSparseAttentionScore(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Context is nullptr."), return ge::GRAPH_FAILED);
    SparseAttentionScoreTilingData tilingData;
    SASATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSparseAttentionScore(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseAttentionScore)
    .Tiling(TilingSparseAttentionScore)
    .TilingInputsDataDependency({5, 6, 7},
        {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<SparseAttentionScoreCompileInfo>(TilingPrepareForSparseAttentionScore);

}  // namespace optiling
