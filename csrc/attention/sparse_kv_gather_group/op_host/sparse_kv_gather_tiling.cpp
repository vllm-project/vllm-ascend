/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"
#include "err/ops_err.h"

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <string>

#include "sparse_kv_gather_tiling.h"

namespace optiling {
namespace {

const std::string OP_NAME_STR = "SparseKvGatherGroup";

bool IsCacheType(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

bool IsIndexType(const ge::DataType dtype)
{
    return dtype == ge::DT_INT32 || dtype == ge::DT_INT64;
}

SKGIndexType ToIndexType(const ge::DataType dtype)
{
    return dtype == ge::DT_INT64 ? SKGIndexType::INT64 : SKGIndexType::INT32;
}

ge::graphStatus CheckRank(const gert::StorageShape *shape, const uint32_t expectedRank,
                          const char *tensorName)
{
    if (shape == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "%s shape is null.", tensorName);
        return ge::GRAPH_FAILED;
    }
    const auto rank = shape->GetStorageShape().GetDimNum();
    if (rank != expectedRank) {
        OP_LOGE(OP_NAME_STR.c_str(), "%s rank must be %u, got %u.",
                tensorName, expectedRank, static_cast<uint32_t>(rank));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool IsPositiveUint32Dim(const int64_t dim)
{
    return dim > 0 && static_cast<uint64_t>(dim) <= std::numeric_limits<uint32_t>::max();
}

ge::graphStatus CheckExact3DShape(const gert::StorageShape *shape,
                                  const uint32_t dim0,
                                  const uint32_t dim1,
                                  const uint32_t dim2,
                                  const char *tensorName)
{
    if (CheckRank(shape, 3, tensorName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto &storageShape = shape->GetStorageShape();
    if (storageShape.GetDim(0) != static_cast<int64_t>(dim0) ||
        storageShape.GetDim(1) != static_cast<int64_t>(dim1) ||
        storageShape.GetDim(2) != static_cast<int64_t>(dim2)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "%s shape must be [%u, %u, %u], got [%ld, %ld, %ld].",
                tensorName, dim0, dim1, dim2,
                storageShape.GetDim(0), storageShape.GetDim(1), storageShape.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus SKGGroupInfoParser::GetTensorInfo(SKGGroupParamInfo &params) const
{
    params.pagedCtkvDesc = context_->GetInputDesc(SKG_PAGED_CTKV_IDX);
    params.pagedCtkvShape = context_->GetInputShape(SKG_PAGED_CTKV_IDX);
    params.pagedKpeDesc = context_->GetInputDesc(SKG_PAGED_KPE_IDX);
    params.pagedKpeShape = context_->GetInputShape(SKG_PAGED_KPE_IDX);
    params.blockTableDesc = context_->GetInputDesc(SKG_BLOCK_TABLE_IDX);
    params.blockTableShape = context_->GetInputShape(SKG_BLOCK_TABLE_IDX);
    params.topkIndicesDesc = context_->GetInputDesc(SKG_TOPK_INDICES_IDX);
    params.topkIndicesShape = context_->GetInputShape(SKG_TOPK_INDICES_IDX);
    params.curPosDesc = context_->GetInputDesc(SKG_CUR_POS_IDX);
    params.curPosShape = context_->GetInputShape(SKG_CUR_POS_IDX);

    params.outCtkvDesc = context_->GetOutputDesc(SKG_OUT_CTKV_IDX);
    params.outCtkvShape = context_->GetOutputShape(SKG_OUT_CTKV_IDX);
    params.outKpeDesc = context_->GetOutputDesc(SKG_OUT_KPE_IDX);
    params.outKpeShape = context_->GetOutputShape(SKG_OUT_KPE_IDX);

    if (params.pagedCtkvDesc == nullptr || params.pagedKpeDesc == nullptr ||
        params.blockTableDesc == nullptr || params.topkIndicesDesc == nullptr ||
        params.curPosDesc == nullptr || params.outCtkvDesc == nullptr ||
        params.outKpeDesc == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Required input or output descriptor is null.");
        return ge::GRAPH_FAILED;
    }
    if (params.pagedCtkvShape == nullptr || params.pagedKpeShape == nullptr ||
        params.blockTableShape == nullptr || params.topkIndicesShape == nullptr ||
        params.curPosShape == nullptr || params.outCtkvShape == nullptr ||
        params.outKpeShape == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Required input or output shape is null.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGGroupInfoParser::GetAttrs(SKGGroupParamInfo &params) const
{
    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Attributes are null.");
        return ge::GRAPH_FAILED;
    }
    params.blockSize = attrs->GetAttrPointer<int64_t>(SKG_ATTR_BLOCK_SIZE);
    params.numCacheLayers = attrs->GetAttrPointer<int64_t>(SKG_ATTR_NUM_CACHE_LAYERS);
    if (params.blockSize == nullptr || params.numCacheLayers == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "block_size or num_cache_layers attribute is null.");
        return ge::GRAPH_FAILED;
    }
    if (*params.blockSize != static_cast<int64_t>(SKG_BLOCK_SIZE) ||
        *params.numCacheLayers < 1 || *params.numCacheLayers > 3) {
        OP_LOGE(OP_NAME_STR.c_str(), "invalid block_size=%ld or num_cache_layers=%ld.",
                *params.blockSize, *params.numCacheLayers);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGGroupInfoParser::CheckDtypes(SKGGroupTilingInfo &info) const
{
    const auto ctkvType = info.params.pagedCtkvDesc->GetDataType();
    const auto kpeType = info.params.pagedKpeDesc->GetDataType();
    const auto outCtkvType = info.params.outCtkvDesc->GetDataType();
    const auto outKpeType = info.params.outKpeDesc->GetDataType();

    if (!IsCacheType(ctkvType) || !IsCacheType(kpeType)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "paged_ctkv and paged_kpe must be FLOAT16 or BF16, got %d and %d.",
                static_cast<int32_t>(ctkvType), static_cast<int32_t>(kpeType));
        return ge::GRAPH_FAILED;
    }
    if (ctkvType != kpeType) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "paged_ctkv and paged_kpe must have the same dtype, got %d and %d.",
                static_cast<int32_t>(ctkvType), static_cast<int32_t>(kpeType));
        return ge::GRAPH_FAILED;
    }
    if (outCtkvType != ctkvType || outKpeType != kpeType) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "Output dtype must match its paged cache input: ctkv %d/%d, kpe %d/%d.",
                static_cast<int32_t>(ctkvType), static_cast<int32_t>(outCtkvType),
                static_cast<int32_t>(kpeType), static_cast<int32_t>(outKpeType));
        return ge::GRAPH_FAILED;
    }

    const auto blockTableType = info.params.blockTableDesc->GetDataType();
    const auto topkIndicesType = info.params.topkIndicesDesc->GetDataType();
    const auto curPosType = info.params.curPosDesc->GetDataType();
    if (!IsIndexType(blockTableType) || !IsIndexType(topkIndicesType) ||
        !IsIndexType(curPosType)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "block_table, topk_indices and cur_pos must be INT32 or INT64, got %d, %d, %d.",
                static_cast<int32_t>(blockTableType),
                static_cast<int32_t>(topkIndicesType),
                static_cast<int32_t>(curPosType));
        return ge::GRAPH_FAILED;
    }

    if (blockTableType != topkIndicesType || blockTableType != curPosType) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "block_table, topk_indices and cur_pos must use the same dtype, got %d, %d, %d.",
                static_cast<int32_t>(blockTableType),
                static_cast<int32_t>(topkIndicesType),
                static_cast<int32_t>(curPosType));
        return ge::GRAPH_FAILED;
    }
    if (ctkvType == ge::DT_FLOAT16 && blockTableType != ge::DT_INT32) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "FLOAT16 cache currently requires INT32 indices.");
        return ge::GRAPH_FAILED;
    }

    info.blockTableType = ToIndexType(blockTableType);
    info.topkIndicesType = ToIndexType(topkIndicesType);
    info.curPosType = ToIndexType(curPosType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGGroupInfoParser::CheckShapes(SKGGroupTilingInfo &info) const
{
    if (CheckRank(info.params.pagedCtkvShape, 4, "paged_ctkv") != ge::GRAPH_SUCCESS ||
        CheckRank(info.params.pagedKpeShape, 4, "paged_kpe") != ge::GRAPH_SUCCESS ||
        CheckRank(info.params.blockTableShape, 2, "block_table") != ge::GRAPH_SUCCESS ||
        CheckRank(info.params.topkIndicesShape, 2, "topk_indices") != ge::GRAPH_SUCCESS ||
        CheckRank(info.params.curPosShape, 1, "cur_pos") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto &ctkv = info.params.pagedCtkvShape->GetStorageShape();
    const auto &kpe = info.params.pagedKpeShape->GetStorageShape();
    const auto &blockTable = info.params.blockTableShape->GetStorageShape();
    const auto &topk = info.params.topkIndicesShape->GetStorageShape();
    const auto &curPos = info.params.curPosShape->GetStorageShape();

    if (!IsPositiveUint32Dim(ctkv.GetDim(0)) ||
        !IsPositiveUint32Dim(blockTable.GetDim(0)) ||
        !IsPositiveUint32Dim(blockTable.GetDim(1)) ||
        !IsPositiveUint32Dim(topk.GetDim(1))) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "num_blocks, num_actual, max_blocks and topk_n must be positive uint32 dimensions.");
        return ge::GRAPH_FAILED;
    }

    if (ctkv.GetDim(1) != static_cast<int64_t>(SKG_BLOCK_SIZE) ||
        ctkv.GetDim(2) != static_cast<int64_t>(SKG_HEAD_NUM) ||
        ctkv.GetDim(3) != static_cast<int64_t>(SKG_CTKV_DIM)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "paged_ctkv shape must be [num_blocks, %u, %u, %u], got [%ld, %ld, %ld, %ld].",
                SKG_BLOCK_SIZE, SKG_HEAD_NUM, SKG_CTKV_DIM,
                ctkv.GetDim(0), ctkv.GetDim(1), ctkv.GetDim(2), ctkv.GetDim(3));
        return ge::GRAPH_FAILED;
    }

    if (kpe.GetDim(0) != ctkv.GetDim(0) ||
        kpe.GetDim(1) != static_cast<int64_t>(SKG_BLOCK_SIZE) ||
        kpe.GetDim(2) != static_cast<int64_t>(SKG_HEAD_NUM) ||
        kpe.GetDim(3) != static_cast<int64_t>(SKG_KPE_DIM)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "paged_kpe shape must be [num_blocks, %u, %u, %u] and share num_blocks with paged_ctkv; "
                "got [%ld, %ld, %ld, %ld].",
                SKG_BLOCK_SIZE, SKG_HEAD_NUM, SKG_KPE_DIM,
                kpe.GetDim(0), kpe.GetDim(1), kpe.GetDim(2), kpe.GetDim(3));
        return ge::GRAPH_FAILED;
    }

    const auto numActual = static_cast<uint32_t>(topk.GetDim(0));
    if (blockTable.GetDim(0) != topk.GetDim(0) || curPos.GetDim(0) != topk.GetDim(0)) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "block_table, topk_indices and cur_pos must share num_actual; got %ld, %ld, %ld.",
                blockTable.GetDim(0), topk.GetDim(0), curPos.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    info.numBlocks = static_cast<uint32_t>(ctkv.GetDim(0));
    info.numActual = numActual;
    info.maxBlocks = static_cast<uint32_t>(blockTable.GetDim(1));
    info.topkN = static_cast<uint32_t>(topk.GetDim(1));
    info.numCacheLayers = static_cast<uint32_t>(*info.params.numCacheLayers);
    info.totalSlots = static_cast<uint64_t>(info.numActual) * info.topkN;

    if (CheckExact3DShape(info.params.outCtkvShape, info.numActual, info.topkN,
                          SKG_CTKV_DIM, "out_ctkv") != ge::GRAPH_SUCCESS ||
        CheckExact3DShape(info.params.outKpeShape, info.numActual, info.topkN,
                          SKG_KPE_DIM, "out_kpe") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGGroupInfoParser::Parse(SKGGroupTilingInfo &info)
{
    SKGGroupParamInfo params;
    if (GetTensorInfo(params) != ge::GRAPH_SUCCESS ||
        GetAttrs(params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    info.params = params;
    if (CheckDtypes(info) != ge::GRAPH_SUCCESS ||
        CheckShapes(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherGroupTiling::GetPlatformInfo(SKGGroupTilingInfo *info) const
{
    if (info->platformInfo == nullptr) {
        OP_LOGE(info->opName, "GetPlatformInfo returned nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(info->platformInfo);
    info->aivNum = platform.GetCoreNumAiv();
    if (info->aivNum == 0) {
        OP_LOGE(info->opName, "AIV core count is 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherGroupTiling::SplitWork(SKGGroupTilingInfo *info) const
{
    if (info->totalSlots == 0) {
        OP_LOGE(info->opName, "totalSlots is 0.");
        return ge::GRAPH_FAILED;
    }
    info->usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(info->aivNum, info->totalSlots));
    info->slotsPerCore =
        (info->totalSlots + info->usedCoreNum - 1) / info->usedCoreNum;
    return ge::GRAPH_SUCCESS;
}

void SparseKvGatherGroupTiling::FillTilingData(const SKGGroupTilingInfo *info)
{
    tilingData_.set_numBlocks(info->numBlocks);
    tilingData_.set_numActual(info->numActual);
    tilingData_.set_maxBlocks(info->maxBlocks);
    tilingData_.set_topkN(info->topkN);
    tilingData_.set_numCacheLayers(info->numCacheLayers);
    tilingData_.set_totalSlots(info->totalSlots);
    tilingData_.set_slotsPerCore(info->slotsPerCore);
    tilingData_.set_usedCoreNum(info->usedCoreNum);
    tilingData_.set_blockTableType(static_cast<uint32_t>(info->blockTableType));
    tilingData_.set_topkIndicesType(static_cast<uint32_t>(info->topkIndicesType));
    tilingData_.set_curPosType(static_cast<uint32_t>(info->curPosType));
}

ge::graphStatus SparseKvGatherGroupTiling::SetBlockDim(const uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherGroupTiling::SetWorkspaceSize(const uint64_t workspaceSize) const
{
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Workspace size pointer is null.");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherGroupTiling::SetTilingData(TilingDef &tilingData) const
{
    auto *rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "RawTilingData is null.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherGroupTiling::DoOpTiling(SKGGroupTilingInfo *info)
{
    if (GetPlatformInfo(info) != ge::GRAPH_SUCCESS ||
        SplitWork(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData(info);

    if (SetBlockDim(info->usedCoreNum) != ge::GRAPH_SUCCESS ||
        SetWorkspaceSize(0) != ge::GRAPH_SUCCESS ||
        SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SparseKvGatherGroupTilingFunc(
    gert::TilingContext *context)
{
    SKGGroupTilingInfo info;
    info.opName       = context->GetNodeName();
    info.platformInfo = context->GetPlatformInfo();

    SKGGroupInfoParser parser(context);
    if (parser.Parse(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SparseKvGatherGroupTiling tiling(context);
    return tiling.DoOpTiling(&info);
}

static ge::graphStatus TilingPrepareForSparseKvGatherGroup(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseKvGatherGroup)
    .Tiling(SparseKvGatherGroupTilingFunc)
    .TilingParse<SparseKvGatherGroupCompileInfo>(TilingPrepareForSparseKvGatherGroup);

}  // namespace optiling
