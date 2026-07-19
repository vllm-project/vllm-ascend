/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_tiling.cpp */
#include "dgemma_fused_qkv_proj_norm_rope_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/dgemma_fused_qkv_proj_norm_rope_tiling_data.h"
#include "securec.h"

using namespace DgemmaFusedQkvProjNormRope;

namespace optiling {

static constexpr uint32_t L1_TILE_M = 128;   // matches kernel L1TileShape M
static constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 4;
static constexpr uint32_t DEFAULT_SYNC_DONE_FLAG = 0;
static constexpr uint32_t DEFAULT_SYNC_READY_FLAG = 6;
static constexpr uint32_t MODE_SKIP_EPILOGUE = 0x10000U;
static constexpr uint32_t MODE_PUBLISH_OUTPUTS = 0x20000U;

ge::graphStatus DgemmaFusedQkvProjNormRopeTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }

    // shapes
    auto hiddenShape = context->GetInputShape(0)->GetStorageShape();  // [m, k]
    auto wqkvShape   = context->GetInputShape(1)->GetStorageShape();  // [n, k]
    uint32_t m = (uint32_t)hiddenShape.GetDim(0);
    uint32_t k = (uint32_t)hiddenShape.GetDim(1);
    uint32_t n = (uint32_t)wqkvShape.GetDim(0);

    // attrs
    auto attrs = context->GetAttrs();
    const float *epsPtr = attrs->GetAttrPointer<float>(0);
    const int64_t *numQPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *numKvPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *headDimPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *syncBasePtr = attrs->GetAttrPointer<int64_t>(5);
    float epsilon = (epsPtr != nullptr) ? *epsPtr : 1e-6f;
    uint32_t numQHeads = (uint32_t)(*numQPtr);
    uint32_t numKvHeads = (uint32_t)(*numKvPtr);
    uint32_t headDim = (uint32_t)(*headDimPtr);
    uint32_t syncBaseAttr = (syncBasePtr != nullptr) ? (uint32_t)(*syncBasePtr) : 0U;
    // Keep the public ABI stable: low byte is the softsync flag base (1..13),
    // higher bits select optional output modes.
    uint32_t syncBase = syncBaseAttr & 0xFFU;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();

    DgemmaFusedQkvProjNormRopeTilingData td;
    td.m = m; td.k = k; td.n = n;
    td.m0 = L1_TILE_M;
    td.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    td.coreNum = aicNum
        | ((syncBaseAttr & 0x100U) ? MODE_SKIP_EPILOGUE : 0U)
        | ((syncBaseAttr & 0x200U) ? MODE_PUBLISH_OUTPUTS : 0U);
    td.numQHeads = numQHeads;
    td.numKvHeads = numKvHeads;
    td.headDim = headDim;
    td.rotaryDim = headDim / 2;
    if (syncBase > 0U && syncBase < 14U) {
        td.syncReadyFlag = syncBase;
        td.syncDoneFlag = syncBase + 1U;
    } else {
        td.syncReadyFlag = DEFAULT_SYNC_READY_FLAG;
        td.syncDoneFlag = DEFAULT_SYNC_DONE_FLAG;
    }
    td.epsilon = epsilon;
    td.invHeadDim = 1.0f / (float)headDim;

    auto *raw = context->GetRawTilingData();
    if (raw == nullptr) { return ge::GRAPH_FAILED; }
    size_t tsz = sizeof(DgemmaFusedQkvProjNormRopeTilingData);
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &td, tsz) != EOK) { return ge::GRAPH_FAILED; }
    raw->SetDataSize(tsz);

    // block dim: MIX uses aic core count
    context->SetBlockDim(aicNum);

    // workspace: system prefix + intermediate qkv[m,n] scratch (bf16 = 2 bytes)
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) { return ge::GRAPH_FAILED; }
    size_t sysWs = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // qkv intermediate is now a caller-provided input tensor (graph-safe), not op workspace.
    workSpaces[0] = sysWs;

    context->SetTilingKey(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDgemmaFusedQkvProjNormRope(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DgemmaFusedQkvProjNormRope)
    .Tiling(DgemmaFusedQkvProjNormRopeTilingFunc)
    .TilingParse<DgemmaFusedQkvProjNormRopeCompileInfo>(TilingPrepareForDgemmaFusedQkvProjNormRope);

} // namespace optiling
