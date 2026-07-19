/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "dgemma_apply_router_scale_tiling.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/dgemma_apply_router_scale_tiling_data.h"
using namespace DgemmaApplyRouterScale;
namespace optiling {
namespace {
constexpr uint64_t TILING_KEY_FP32 = 0;
constexpr size_t IN_WEIGHTS = 0;
}
ge::graphStatus DgemmaApplyRouterScaleTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) { return ge::GRAPH_FAILED; }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) { aivNum = 1; }

    auto *shapeW = context->GetInputShape(IN_WEIGHTS);
    if (shapeW == nullptr) { return ge::GRAPH_FAILED; }
    const auto &s = shapeW->GetStorageShape();
    int64_t elems = 1;
    for (size_t d = 0; d < s.GetDimNum(); ++d) { elems *= s.GetDim(d); }

    DgemmaApplyRouterScaleTilingData td{};
    td.numElems = static_cast<uint32_t>(elems);

    auto *raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < sizeof(td)) { return ge::GRAPH_FAILED; }
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &td, sizeof(td)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(td));

    context->SetBlockDim(1);
    context->SetTilingKey(TILING_KEY_FP32);
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) { return ge::GRAPH_FAILED; }
    ws[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingPrepareForDgemmaApplyRouterScale(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(DgemmaApplyRouterScale)
    .Tiling(DgemmaApplyRouterScaleTilingFunc)
    .TilingParse<DgemmaApplyRouterScaleCompileInfo>(TilingPrepareForDgemmaApplyRouterScale);
}
