/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope_tiling.cpp */
#include "dgemma_fused_norm_rope_tiling.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/dgemma_fused_norm_rope_tiling_data.h"
using namespace DgemmaFusedNormRope;
namespace optiling {
namespace {
constexpr uint64_t TILING_KEY_BF16 = 1;
constexpr uint64_t TILING_KEY_FP16 = 2;
constexpr size_t IN_Q = 0;
}
ge::graphStatus DgemmaFusedNormRopeTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) { return ge::GRAPH_FAILED; }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) { aivNum = 1; }

    auto *shapeQ = context->GetInputShape(IN_Q);
    if (shapeQ == nullptr) { return ge::GRAPH_FAILED; }
    const auto &s = shapeQ->GetStorageShape();
    // q is [num_tokens, num_q_heads, head_dim] (3-D) or [num_tokens, num_q_heads*head_dim] (2-D)
    if (s.GetDimNum() < 2) { return ge::GRAPH_FAILED; }
    int64_t numTokens = s.GetDim(0);

    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    const float *epsAttr = attrs->GetAttrPointer<float>(0);
    const int64_t *nqAttr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *nkvAttr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *hdAttr = attrs->GetAttrPointer<int64_t>(3);
    if (nqAttr == nullptr || nkvAttr == nullptr || hdAttr == nullptr) { return ge::GRAPH_FAILED; }

    DgemmaFusedNormRopeTilingData td{};
    td.numTokens  = static_cast<uint32_t>(numTokens);
    td.numQHeads  = static_cast<uint32_t>(*nqAttr);
    td.numKvHeads = static_cast<uint32_t>(*nkvAttr);
    td.headDim    = static_cast<uint32_t>(*hdAttr);
    td.rotaryDim  = td.headDim / 2u;
    td.epsilon    = (epsAttr != nullptr) ? *epsAttr : 1e-6f;
    td.invHeadDim = 1.0f / static_cast<float>(td.headDim);

    // one AIV core handles a contiguous chunk of tokens
    uint32_t blockDim = static_cast<uint32_t>(numTokens);
    if (blockDim > aivNum) { blockDim = aivNum; }
    if (blockDim == 0) { blockDim = 1; }

    auto *aDesc = context->GetInputDesc(IN_Q);
    if (aDesc == nullptr) { return ge::GRAPH_FAILED; }
    uint64_t tilingKey = (aDesc->GetDataType() == ge::DT_FLOAT16) ? TILING_KEY_FP16 : TILING_KEY_BF16;

    const size_t tsz = sizeof(DgemmaFusedNormRopeTilingData);
    auto *raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < tsz) { return ge::GRAPH_FAILED; }
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &td, tsz) != EOK) { return ge::GRAPH_FAILED; }
    raw->SetDataSize(tsz);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) { ws[0] = 0; }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingPrepareForDgemmaFusedNormRope(gert::TilingParseContext *context)
{ (void)context; return ge::GRAPH_SUCCESS; }
} // namespace optiling
IMPL_OP_OPTILING(DgemmaFusedNormRope)
    .Tiling(optiling::DgemmaFusedNormRopeTilingFunc)
    .TilingParse<optiling::DgemmaFusedNormRopeCompileInfo>(optiling::TilingPrepareForDgemmaFusedNormRope);
