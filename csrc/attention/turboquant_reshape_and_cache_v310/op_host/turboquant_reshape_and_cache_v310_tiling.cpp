/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "turboquant_reshape_and_cache_v310_tiling.h"

#include <cmath>

#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"
#include "tiling_base/tiling_util.h"
#include "platform/platform_info.h"

namespace optiling {

namespace {
constexpr uint32_t kNzC0 = 16;          // fp16 NZ tile on 310P (elements per tile)
constexpr uint32_t kBitsPerHalf = 16;   // bits in an fp16 -- SAME VALUE as kNzC0 but a
                                        // different quantity; keep them distinct so a
                                        // future change to either does not silently
                                        // corrupt the other's arithmetic.
constexpr uint32_t kSysWorkspace = 16777216;
constexpr uint32_t kMinBits = 2;
constexpr uint32_t kMaxBits = 4;
}  // namespace

ge::graphStatus TurboquantReshapeAndCacheV310Tiling::ParseInputs()
{
    // key: [num_tokens, num_kv_heads, head_dim]
    auto keyShape = context_->GetInputShape(0);
    OP_CHECK_IF(keyShape == nullptr, OP_LOGE(context_->GetNodeName(), "key shape is null"),
                return ge::GRAPH_FAILED);
    const auto &ks = keyShape->GetStorageShape();
    OP_CHECK_IF(ks.GetDimNum() != 3, OP_LOGE(context_->GetNodeName(), "key must be 3D [tok, kvh, d]"),
                return ge::GRAPH_FAILED);
    tilingData_.numTokens = static_cast<uint32_t>(ks.GetDim(0));
    tilingData_.numKvHeads = static_cast<uint32_t>(ks.GetDim(1));
    tilingData_.headDim = static_cast<uint32_t>(ks.GetDim(2));

    // key_cache: (num_blocks, C1, block_size, 16)
    auto cacheShape = context_->GetInputShape(2);
    OP_CHECK_IF(cacheShape == nullptr, OP_LOGE(context_->GetNodeName(), "key_cache shape is null"),
                return ge::GRAPH_FAILED);
    const auto &cs = cacheShape->GetStorageShape();
    OP_CHECK_IF(cs.GetDimNum() != 4, OP_LOGE(context_->GetNodeName(), "key_cache must be 4D NZ"),
                return ge::GRAPH_FAILED);
    tilingData_.numBlocks = static_cast<uint32_t>(cs.GetDim(0));
    tilingData_.c1 = static_cast<uint32_t>(cs.GetDim(1));
    tilingData_.blockSize = static_cast<uint32_t>(cs.GetDim(2));
    OP_CHECK_IF(static_cast<uint32_t>(cs.GetDim(3)) != kNzC0,
                OP_LOGE(context_->GetNodeName(), "key_cache last dim must be %u", kNzC0),
                return ge::GRAPH_FAILED);

    auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs are null"),
                return ge::GRAPH_FAILED);
    const int64_t *bitsAttr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *variantAttr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *cbAttr = attrs->GetAttrPointer<int64_t>(2);
    bits_ = (bitsAttr != nullptr) ? static_cast<uint32_t>(*bitsAttr) : 3U;
    tilingData_.variant = (variantAttr != nullptr) ? static_cast<uint32_t>(*variantAttr) : 0U;
    tilingData_.codebookMode = (cbAttr != nullptr) ? static_cast<uint32_t>(*cbAttr) : 0U;

    OP_CHECK_IF(bits_ < kMinBits || bits_ > kMaxBits,
                OP_LOGE(context_->GetNodeName(), "bits must be in [2,4], got %u", bits_),
                return ge::GRAPH_FAILED);

    // head_dim * bits must be byte-aligned AND the packed run must be an even
    // number of halves so it can ride through the fp16-typed cache.
    const uint32_t packedBits = tilingData_.headDim * bits_;
    OP_CHECK_IF(packedBits % kBitsPerHalf != 0,
                OP_LOGE(context_->GetNodeName(), "head_dim*bits must be a whole number of fp16 slots"),
                return ge::GRAPH_FAILED);
    tilingData_.packedHalves = packedBits / kBitsPerHalf;
    // Kernel-side constants: AscendC Sqrt has no scalar overload, so compute here.
    tilingData_.sqrtHeadDim = std::sqrt(static_cast<float>(tilingData_.headDim));
    tilingData_.invSqrtHeadDim = 1.0f / tilingData_.sqrtHeadDim;

    // The per-head packed run must tile the NZ C0 exactly, else a head would
    // straddle a tile and the scatter would need a read-modify-write.
    OP_CHECK_IF(tilingData_.packedHalves % kNzC0 != 0,
                OP_LOGE(context_->GetNodeName(),
                        "packed halves per head (%u) must be a multiple of %u; "
                        "head_dim=%u bits=%u is not NZ-tileable",
                        tilingData_.packedHalves, kNzC0, tilingData_.headDim, bits_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.numKvHeads * tilingData_.packedHalves != tilingData_.c1 * kNzC0,
                OP_LOGE(context_->GetNodeName(),
                        "cache C1 (%u) inconsistent with kv_heads*packed_halves (%u)",
                        tilingData_.c1, tilingData_.numKvHeads * tilingData_.packedHalves),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantReshapeAndCacheV310Tiling::ComputeSplit()
{
    auto *platformInfo = context_->GetPlatformInfo();
    uint32_t coreNum = 8;
    if (platformInfo != nullptr) {
        auto ascendPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
    }
    if (coreNum == 0) {
        coreNum = 1;
    }
    /*
     * CACHE-LINE OWNERSHIP (multi-core correctness, not tuning).
     *
     * The norm planes are written with scalar SetValue to GM, and GM scalar
     * writes commit at CACHE-LINE granularity, not element granularity. One
     * token's norms occupy numKvHeads*2 bytes (8 B at 4 kv heads), so with a
     * small tokensPerCore several cores read-modify-write the SAME line and
     * clobber each other: measured [0.0, 16.59, 0.0, 0.0] at numTokens=4, i.e.
     * three of four norms silently lost, nondeterministically.
     *
     * Give every core a whole number of cache lines. At 4 kv heads that is 8
     * tokens per 64 B line; the last core simply gets a short tail, which is
     * safe because no other core touches its lines.
     */
    constexpr uint32_t kLineBytes = 64;
    const uint32_t bytesPerToken = tilingData_.numKvHeads * static_cast<uint32_t>(sizeof(uint16_t));
    uint32_t tokensPerLine = (bytesPerToken == 0) ? 1U : (kLineBytes + bytesPerToken - 1) / bytesPerToken;
    if (tokensPerLine == 0) {
        tokensPerLine = 1;
    }

    uint32_t perCore = (tilingData_.numTokens + coreNum - 1) / coreNum;
    if (perCore == 0) {
        perCore = 1;
    }
    perCore = ((perCore + tokensPerLine - 1) / tokensPerLine) * tokensPerLine;  // round up to a line
    tilingData_.tokensPerCore = perCore;
    tilingData_.vectorCoreNum = (tilingData_.numTokens + perCore - 1) / perCore;
    if (tilingData_.vectorCoreNum == 0) {
        tilingData_.vectorCoreNum = 1;
    }

    // Only the bit-width is compile-time; variant + codebook stay runtime.
    tilingKey_ = 200 + bits_;
    workspaceSize_ = kSysWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantReshapeAndCacheV310Tiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    context_->SetTilingKey(tilingKey_);
    auto rawTiling = context_->GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTiling->GetCapacity() < sizeof(tilingData_),
                OP_LOGE(context_->GetNodeName(), "tiling buffer too small"), return ge::GRAPH_FAILED);
    (void)memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &tilingData_, sizeof(tilingData_));
    rawTiling->SetDataSize(sizeof(tilingData_));

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspace ptr is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantReshapeAndCacheV310Tiling::Run()
{
    if (ParseInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ComputeSplit() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return PostTiling();
}

ge::graphStatus TilingForTurboquantReshapeAndCacheV310(gert::TilingContext *context)
{
    TurboquantReshapeAndCacheV310Tiling tiling(context);
    return tiling.Run();
}

static ge::graphStatus TilingPrepareForTurboquantReshapeAndCacheV310(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<TurboquantReshapeAndCacheV310CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto p = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->coreNum = static_cast<uint32_t>(p.GetCoreNumAiv());
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantReshapeAndCacheV310)
    .Tiling(TilingForTurboquantReshapeAndCacheV310)
    .TilingParse<TurboquantReshapeAndCacheV310CompileInfo>(TilingPrepareForTurboquantReshapeAndCacheV310);
}  // namespace optiling
