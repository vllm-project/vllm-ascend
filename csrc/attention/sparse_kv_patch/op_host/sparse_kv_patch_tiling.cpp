/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "sparse_kv_patch_tiling.h"

#include "err/ops_err.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <limits>
#include <string>

namespace optiling {
namespace {

const std::string OP_NAME_STR = "SparseKvPatch";

bool IsCacheType(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

bool IsIndexType(const ge::DataType dtype)
{
    return dtype == ge::DT_INT32 || dtype == ge::DT_INT64;
}

ge::graphStatus CheckRank(const gert::StorageShape *shape,
                          const uint32_t expectedRank,
                          const char *name)
{
    if (shape == nullptr ||
        shape->GetStorageShape().GetDimNum() != expectedRank) {
        OP_LOGE(OP_NAME_STR.c_str(), "%s rank must be %u.", name, expectedRank);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus SparseKvPatchTiling::Parse()
{
    pagedCtkvDesc_ = context_->GetInputDesc(SKP_PAGED_CTKV_IDX);
    pagedKpeDesc_ = context_->GetInputDesc(SKP_PAGED_KPE_IDX);
    slotMappingDesc_ = context_->GetInputDesc(SKP_SLOT_MAPPING_IDX);
    currentTopkSlotsDesc_ = context_->GetInputDesc(SKP_CURRENT_TOPK_SLOTS_IDX);
    prefetchedCtkvDesc_ = context_->GetOutputDesc(SKP_PREFETCHED_CTKV_IDX);
    prefetchedKpeDesc_ = context_->GetOutputDesc(SKP_PREFETCHED_KPE_IDX);

    pagedCtkvShape_ = context_->GetInputShape(SKP_PAGED_CTKV_IDX);
    pagedKpeShape_ = context_->GetInputShape(SKP_PAGED_KPE_IDX);
    slotMappingShape_ = context_->GetInputShape(SKP_SLOT_MAPPING_IDX);
    currentTopkSlotsShape_ = context_->GetInputShape(SKP_CURRENT_TOPK_SLOTS_IDX);
    prefetchedCtkvShape_ = context_->GetOutputShape(SKP_PREFETCHED_CTKV_IDX);
    prefetchedKpeShape_ = context_->GetOutputShape(SKP_PREFETCHED_KPE_IDX);

    if (pagedCtkvDesc_ == nullptr || pagedKpeDesc_ == nullptr ||
        slotMappingDesc_ == nullptr || currentTopkSlotsDesc_ == nullptr ||
        prefetchedCtkvDesc_ == nullptr || prefetchedKpeDesc_ == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Required descriptor is null.");
        return ge::GRAPH_FAILED;
    }

    if (CheckRank(pagedCtkvShape_, 4, "paged_ctkv") != ge::GRAPH_SUCCESS ||
        CheckRank(pagedKpeShape_, 4, "paged_kpe") != ge::GRAPH_SUCCESS ||
        CheckRank(slotMappingShape_, 1, "slot_mapping") != ge::GRAPH_SUCCESS ||
        CheckRank(currentTopkSlotsShape_, 2, "current_topk_slots") !=
            ge::GRAPH_SUCCESS ||
        CheckRank(prefetchedCtkvShape_, 3, "prefetched_ctkv") !=
            ge::GRAPH_SUCCESS ||
        CheckRank(prefetchedKpeShape_, 3, "prefetched_kpe") !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto cacheType = pagedCtkvDesc_->GetDataType();
    if (!IsCacheType(cacheType) ||
        pagedKpeDesc_->GetDataType() != cacheType ||
        prefetchedCtkvDesc_->GetDataType() != cacheType ||
        prefetchedKpeDesc_->GetDataType() != cacheType) {
        OP_LOGE(OP_NAME_STR.c_str(), "Cache and prefetch dtypes must match.");
        return ge::GRAPH_FAILED;
    }
    if (!IsIndexType(slotMappingDesc_->GetDataType()) ||
        currentTopkSlotsDesc_->GetDataType() != ge::DT_INT32) {
        OP_LOGE(OP_NAME_STR.c_str(),
                "slot_mapping must be INT32/INT64 and slots must be INT32.");
        return ge::GRAPH_FAILED;
    }

    const auto &ctkv = pagedCtkvShape_->GetStorageShape();
    const auto &kpe = pagedKpeShape_->GetStorageShape();
    const auto &slots = slotMappingShape_->GetStorageShape();
    const auto &topkSlots = currentTopkSlotsShape_->GetStorageShape();
    const auto &outCtkv = prefetchedCtkvShape_->GetStorageShape();
    const auto &outKpe = prefetchedKpeShape_->GetStorageShape();

    if (ctkv.GetDim(0) <= 0 || ctkv.GetDim(1) != SKP_BLOCK_SIZE ||
        ctkv.GetDim(2) != 1 || ctkv.GetDim(3) != SKP_CTKV_DIM ||
        kpe.GetDim(0) != ctkv.GetDim(0) ||
        kpe.GetDim(1) != SKP_BLOCK_SIZE || kpe.GetDim(2) != 1 ||
        kpe.GetDim(3) != SKP_KPE_DIM) {
        OP_LOGE(OP_NAME_STR.c_str(), "Invalid paged KV cache shape.");
        return ge::GRAPH_FAILED;
    }

    if (slots.GetDim(0) <= 0 || slots.GetDim(0) != topkSlots.GetDim(0) ||
        topkSlots.GetDim(1) != 8 ||
        slots.GetDim(0) != outCtkv.GetDim(0) ||
        slots.GetDim(0) != outKpe.GetDim(0) ||
        outCtkv.GetDim(1) <= 0 ||
        outCtkv.GetDim(1) != outKpe.GetDim(1) ||
        outCtkv.GetDim(2) != SKP_CTKV_DIM ||
        outKpe.GetDim(2) != SKP_KPE_DIM) {
        OP_LOGE(OP_NAME_STR.c_str(), "Invalid slot or prefetch output shape.");
        return ge::GRAPH_FAILED;
    }

    if (static_cast<uint64_t>(slots.GetDim(0)) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(outCtkv.GetDim(1)) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(ctkv.GetDim(0)) * SKP_BLOCK_SIZE >
            std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(OP_NAME_STR.c_str(), "Shape exceeds uint32 range.");
        return ge::GRAPH_FAILED;
    }

    numActual_ = static_cast<uint32_t>(slots.GetDim(0));
    topkN_ = static_cast<uint32_t>(outCtkv.GetDim(1));
    numPhysicalSlots_ =
        static_cast<uint32_t>(ctkv.GetDim(0) * SKP_BLOCK_SIZE);
    slotMappingType_ = slotMappingDesc_->GetDataType() == ge::DT_INT64
        ? SKP_INDEX_TYPE_INT64
        : SKP_INDEX_TYPE_INT32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvPatchTiling::DoTiling()
{
    const auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0 || Parse() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = std::min(aivNum, numActual_);
    tilingData_.set_numActual(numActual_);
    tilingData_.set_topkN(topkN_);
    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_slotMappingType(slotMappingType_);
    tilingData_.set_numPhysicalSlots(numPhysicalSlots_);

    context_->SetBlockDim(usedCoreNum);
    size_t *workspace = context_->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    auto *raw = context_->GetRawTilingData();
    if (raw == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SparseKvPatchTilingFunc(gert::TilingContext *context)
{
    SparseKvPatchTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForSparseKvPatch(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseKvPatch)
    .Tiling(SparseKvPatchTilingFunc)
    .TilingParse<SparseKvPatchCompileInfo>(TilingPrepareForSparseKvPatch);

}  // namespace optiling
