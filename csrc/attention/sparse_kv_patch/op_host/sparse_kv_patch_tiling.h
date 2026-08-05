/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SPARSE_KV_PATCH_TILING_H
#define SPARSE_KV_PATCH_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>

#include "../op_kernel/sparse_kv_patch_tiling_data.h"

namespace optiling {

constexpr uint32_t SKP_PAGED_CTKV_IDX = 0;
constexpr uint32_t SKP_PAGED_KPE_IDX = 1;
constexpr uint32_t SKP_SLOT_MAPPING_IDX = 2;
constexpr uint32_t SKP_CURRENT_TOPK_SLOTS_IDX = 3;
constexpr uint32_t SKP_PREFETCHED_CTKV_IDX = 0;
constexpr uint32_t SKP_PREFETCHED_KPE_IDX = 1;

struct SparseKvPatchCompileInfo {
    int64_t core_num;
};

class SparseKvPatchTiling {
public:
    explicit SparseKvPatchTiling(gert::TilingContext *context)
        : context_(context)
    {
    }

    ge::graphStatus DoTiling();

private:
    ge::graphStatus Parse();

    gert::TilingContext *context_ = nullptr;
    const gert::CompileTimeTensorDesc *pagedCtkvDesc_ = nullptr;
    const gert::CompileTimeTensorDesc *pagedKpeDesc_ = nullptr;
    const gert::CompileTimeTensorDesc *slotMappingDesc_ = nullptr;
    const gert::CompileTimeTensorDesc *currentTopkSlotsDesc_ = nullptr;
    const gert::CompileTimeTensorDesc *prefetchedCtkvDesc_ = nullptr;
    const gert::CompileTimeTensorDesc *prefetchedKpeDesc_ = nullptr;
    const gert::StorageShape *pagedCtkvShape_ = nullptr;
    const gert::StorageShape *pagedKpeShape_ = nullptr;
    const gert::StorageShape *slotMappingShape_ = nullptr;
    const gert::StorageShape *currentTopkSlotsShape_ = nullptr;
    const gert::StorageShape *prefetchedCtkvShape_ = nullptr;
    const gert::StorageShape *prefetchedKpeShape_ = nullptr;

    uint32_t numActual_ = 0;
    uint32_t topkN_ = 0;
    uint32_t numPhysicalSlots_ = 0;
    uint32_t slotMappingType_ = 0;
    SparseKvPatchTilingData tilingData_;
};

}  // namespace optiling

#endif  // SPARSE_KV_PATCH_TILING_H
