/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kvcache_scatter_copy_tiling.h"

namespace optiling {

constexpr int32_t HBM_K_ROPE_IDX = 0;
constexpr int32_t HBM_KV_CACHE_IDX = 1;
constexpr int32_t DRAM_K_ROPE_IDX = 2;
constexpr int32_t DRAM_KV_CACHE_IDX = 3;
constexpr int32_t HBM_BLOCK_TABLE_IDX = 4;
constexpr int32_t DRAM_BLOCK_TABLE_IDX = 5;
constexpr int32_t SRC_TOKEN_IDS_IDX = 6;
constexpr int32_t DST_SLOTS_IDX = 7;
constexpr int32_t COPY_COUNTS_IDX = 8;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;
constexpr int64_t COPY_CAP = 16384;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

ge::graphStatus KvcacheScatterCopyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheScatterCopyTiling::GetShapeInfo()
{
    auto hbmRope = context_->GetInputShape(HBM_K_ROPE_IDX);
    auto hbmKv = context_->GetInputShape(HBM_KV_CACHE_IDX);
    auto dramRope = context_->GetInputShape(DRAM_K_ROPE_IDX);
    auto dramKv = context_->GetInputShape(DRAM_KV_CACHE_IDX);
    auto hbmTable = context_->GetInputShape(HBM_BLOCK_TABLE_IDX);
    auto dramTable = context_->GetInputShape(DRAM_BLOCK_TABLE_IDX);
    auto srcIds = context_->GetInputShape(SRC_TOKEN_IDS_IDX);
    auto dstSlots = context_->GetInputShape(DST_SLOTS_IDX);
    auto copyCounts = context_->GetInputShape(COPY_COUNTS_IDX);
    OPS_ERR_IF(hbmRope == nullptr || hbmKv == nullptr || dramRope == nullptr || dramKv == nullptr ||
                   hbmTable == nullptr || dramTable == nullptr || srcIds == nullptr || dstSlots == nullptr ||
                   copyCounts == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get input shape nullptr."), return ge::GRAPH_FAILED);

    gert::Shape hbmRopeShape = hbmRope->GetStorageShape();
    gert::Shape hbmKvShape = hbmKv->GetStorageShape();
    gert::Shape dramRopeShape = dramRope->GetStorageShape();
    gert::Shape dramKvShape = dramKv->GetStorageShape();
    gert::Shape hbmTableShape = hbmTable->GetStorageShape();
    gert::Shape dramTableShape = dramTable->GetStorageShape();
    gert::Shape srcIdsShape = srcIds->GetStorageShape();
    gert::Shape dstSlotsShape = dstSlots->GetStorageShape();
    gert::Shape copyCountsShape = copyCounts->GetStorageShape();

    OPS_ERR_IF(hbmRopeShape.GetDimNum() != 3 || hbmKvShape.GetDimNum() != 3 ||
                   dramRopeShape.GetDimNum() != 3 || dramKvShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "KV tensors must be rank 3."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(hbmRopeShape.GetDim(DIM_1) != BLOCK_SIZE || hbmRopeShape.GetDim(DIM_2) != K_ROPE_DIM ||
                   dramRopeShape.GetDim(DIM_1) != BLOCK_SIZE || dramRopeShape.GetDim(DIM_2) != K_ROPE_DIM,
        OPS_LOG_E(context_->GetNodeName(), "K-RoPE tensors must have shape [blocks, 128, 64]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(hbmKvShape.GetDim(DIM_1) != BLOCK_SIZE || hbmKvShape.GetDim(DIM_2) != KV_CACHE_DIM ||
                   dramKvShape.GetDim(DIM_1) != BLOCK_SIZE || dramKvShape.GetDim(DIM_2) != KV_CACHE_DIM,
        OPS_LOG_E(context_->GetNodeName(), "KV tensors must have shape [blocks, 128, 512]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(hbmRopeShape.GetDim(DIM_0) != hbmKvShape.GetDim(DIM_0) ||
                   dramRopeShape.GetDim(DIM_0) != dramKvShape.GetDim(DIM_0),
        OPS_LOG_E(context_->GetNodeName(), "K-RoPE and KV tensors must have matching block counts."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(hbmTableShape.GetDimNum() != 2 || dramTableShape.GetDimNum() != 2 ||
                   srcIdsShape.GetDimNum() != 3 || dstSlotsShape.GetDimNum() != 3 ||
                   copyCountsShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "block tables/index tensors/copy_counts have invalid ranks."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(srcIdsShape.GetDim(DIM_1) != 1 || srcIdsShape.GetDim(DIM_2) != COPY_CAP ||
                   dstSlotsShape.GetDim(DIM_1) != 1 || dstSlotsShape.GetDim(DIM_2) != COPY_CAP,
        OPS_LOG_E(context_->GetNodeName(), "src_token_ids and dst_slots must have shape [B, 1, 16384]."),
        return ge::GRAPH_FAILED);

    batchSize_ = copyCountsShape.GetDim(DIM_0);
    OPS_ERR_IF(srcIdsShape.GetDim(DIM_0) != batchSize_ || dstSlotsShape.GetDim(DIM_0) != batchSize_ ||
                   hbmTableShape.GetDim(DIM_0) != batchSize_ || dramTableShape.GetDim(DIM_0) != batchSize_,
        OPS_LOG_E(context_->GetNodeName(), "all batch dimensions must match copy_counts."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(hbmTableShape.GetDim(DIM_1) <= 0 || dramTableShape.GetDim(DIM_1) <= 0,
        OPS_LOG_E(context_->GetNodeName(), "block tables must contain at least one block column."),
        return ge::GRAPH_FAILED);

    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_copyCap(COPY_CAP);
    tilingData_.set_hbmMaxBlockNum(hbmTableShape.GetDim(DIM_1));
    tilingData_.set_dramMaxBlockNum(dramTableShape.GetDim(DIM_1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheScatterCopyTiling::GetDtypeInfo()
{
    auto dataDesc = context_->GetInputDesc(HBM_K_ROPE_IDX);
    OPS_ERR_IF(dataDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get data desc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType dataType = dataDesc->GetDataType();
    OPS_ERR_IF(dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT16,
        OPS_LOG_E(context_->GetNodeName(), "KV tensors only support bf16/fp16."), return ge::GRAPH_FAILED);
    for (int32_t i = HBM_KV_CACHE_IDX; i <= DRAM_KV_CACHE_IDX; ++i) {
        auto desc = context_->GetInputDesc(i);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != dataType,
            OPS_LOG_E(context_->GetNodeName(), "all KV tensors must have the same dtype."),
            return ge::GRAPH_FAILED);
    }
    for (int32_t i = HBM_BLOCK_TABLE_IDX; i <= COPY_COUNTS_IDX; ++i) {
        auto desc = context_->GetInputDesc(i);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT32,
            OPS_LOG_E(context_->GetNodeName(), "block tables, indices and copy_counts must be int32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheScatterCopyTiling::DoOpTiling()
{
    if (batchSize_ == 0) {
        tilingData_.set_usedCoreNum(0);
        tilingData_.set_totalPairSlots(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t totalPairSlots = batchSize_ * COPY_CAP;
    int64_t usedCoreNum = totalPairSlots < coreNum_ ? totalPairSlots : coreNum_;
    tilingData_.set_totalPairSlots(totalPairSlots);
    tilingData_.set_usedCoreNum(usedCoreNum);
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheScatterCopyTiling::PostTiling()
{
    context_->SetTilingKey(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr, OPS_LOG_E(context_->GetNodeName(), "get tilingdata nullptr."),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvcacheScatterCopyTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetDtypeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4KvcacheScatterCopy(gert::TilingContext* context)
{
    KvcacheScatterCopyTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4KvcacheScatterCopy(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvcacheScatterCopy)
    .Tiling(Tiling4KvcacheScatterCopy)
    .TilingParse<KvcacheScatterCopyCompileInfo>(TilingPrepare4KvcacheScatterCopy);

} // namespace optiling
