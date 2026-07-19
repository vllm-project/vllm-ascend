/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_tiling.cpp */
#include "dgemma_fused_router_front_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/dgemma_fused_router_front_tiling_data.h"
#include "securec.h"
#include <cmath>

using namespace DgemmaFusedRouterFront;

namespace optiling {

static constexpr uint32_t L1_TILE_M = 128;
static constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 4;
static constexpr uint32_t DEFAULT_SYNC_READY_FLAG = 6;
static constexpr uint32_t DEFAULT_SYNC_DONE_FLAG = 7;

ge::graphStatus DgemmaFusedRouterFrontTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }

    auto xShape = context->GetInputShape(0)->GetStorageShape();     // [m, k]
    auto wShape = context->GetInputShape(2)->GetStorageShape();     // proj_weight [n, k]
    uint32_t m = (uint32_t)xShape.GetDim(0);
    uint32_t k = (uint32_t)xShape.GetDim(1);
    uint32_t n = (uint32_t)wShape.GetDim(0);

    auto attrs = context->GetAttrs();
    const float *epsPtr = attrs->GetAttrPointer<float>(0);
    float epsilon = (epsPtr != nullptr) ? *epsPtr : 1e-6f;
    const int64_t *topKPtr = attrs->GetAttrPointer<int64_t>(3);
    uint32_t topK = (topKPtr != nullptr) ? static_cast<uint32_t>(*topKPtr) : 8U;
    const int64_t *syncBasePtr = attrs->GetAttrPointer<int64_t>(4);
    uint32_t syncBase = (syncBasePtr != nullptr) ? (uint32_t)(*syncBasePtr) : 1U;
    uint32_t syncReadyFlag = DEFAULT_SYNC_READY_FLAG;
    uint32_t syncDoneFlag = DEFAULT_SYNC_DONE_FLAG;
    if (syncBase > 0U && syncBase < 14U) {
        syncReadyFlag = syncBase;
        syncDoneFlag = syncBase + 1U;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();

    DgemmaFusedRouterFrontTilingData td;
    td.m = m; td.k = k; td.n = n; td.topK = topK;
    td.m0 = L1_TILE_M;
    td.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    td.coreNum = aicNum;
    td.syncReadyFlag = syncReadyFlag;
    td.syncDoneFlag = syncDoneFlag;
    td.epsilon = epsilon;
    td.invHidden = 1.0f / (float)k;
    td.rootSize = 1.0f / std::sqrt((float)k);

    auto *raw = context->GetRawTilingData();
    if (raw == nullptr) { return ge::GRAPH_FAILED; }
    size_t tsz = sizeof(DgemmaFusedRouterFrontTilingData);
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &td, tsz) != EOK) { return ge::GRAPH_FAILED; }
    raw->SetDataSize(tsz);

    context->SetBlockDim(aicNum);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) { return ge::GRAPH_FAILED; }
    size_t sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // normed intermediate is a caller-provided input tensor (graph-safe), not op workspace.
    workSpaces[0] = sysWs;

    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDgemmaFusedRouterFront(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DgemmaFusedRouterFront)
    .Tiling(DgemmaFusedRouterFrontTilingFunc)
    .TilingParse<DgemmaFusedRouterFrontCompileInfo>(TilingPrepareForDgemmaFusedRouterFront);

} // namespace optiling
