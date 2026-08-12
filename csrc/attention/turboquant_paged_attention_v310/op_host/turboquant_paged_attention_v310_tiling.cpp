/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "turboquant_paged_attention_v310_tiling.h"

#include <cmath>

#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"
#include "tiling_base/tiling_util.h"
#include "platform/platform_info.h"

namespace optiling {

namespace {
constexpr uint32_t kNzC0 = 16;         // fp16 NZ tile (elements)
constexpr uint32_t kBitsPerHalf = 16;  // bits in an fp16 -- same value, different quantity
constexpr uint32_t kSysWorkspace = 16777216;
}  // namespace

ge::graphStatus TurboquantPagedAttentionV310Tiling::ParseInputs()
{
    auto qShape = context_->GetInputShape(0);      // [batch, num_heads, head_dim]
    auto cacheShape = context_->GetInputShape(1);  // (num_blocks, C1, block_size, 16)
    auto btShape = context_->GetInputShape(5);     // [batch, max_blocks_per_seq]
    auto normShape = context_->GetInputShape(3);   // [num_slots, num_kv_heads]
    OP_CHECK_IF(qShape == nullptr || cacheShape == nullptr || btShape == nullptr || normShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is null"), return ge::GRAPH_FAILED);

    const auto &qs = qShape->GetStorageShape();
    OP_CHECK_IF(qs.GetDimNum() != 3, OP_LOGE(context_->GetNodeName(), "query must be [b, n, d]"),
                return ge::GRAPH_FAILED);
    tilingData_.batch = static_cast<uint32_t>(qs.GetDim(0));
    tilingData_.numHeads = static_cast<uint32_t>(qs.GetDim(1));
    tilingData_.headDim = static_cast<uint32_t>(qs.GetDim(2));

    const auto &cs = cacheShape->GetStorageShape();
    OP_CHECK_IF(cs.GetDimNum() != 4, OP_LOGE(context_->GetNodeName(), "key_cache must be 4D NZ"),
                return ge::GRAPH_FAILED);
    tilingData_.c1 = static_cast<uint32_t>(cs.GetDim(1));
    tilingData_.blockSize = static_cast<uint32_t>(cs.GetDim(2));
    tilingData_.maxBlocksPerSeq = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
    /*
     * numKvHeads used to be read from the norm plane's dim 1. That plane is now
     * [num_slots, kNzC0] -- one whole 32B block per slot, only the first
     * numKvHeads lanes carrying data -- so dim 1 is the FIXED padding, not the
     * head count. Reading it here silently produced numKvHeads = 16 and the
     * tiling failed with ret -1. It is derived from the cache below instead,
     * once packedHalves is known, via the NZ invariant
     *     numKvHeads * packedHalves == c1 * kNzC0
     */
    OP_CHECK_IF(normShape->GetStorageShape().GetDim(1) != kNzC0,
                OP_LOGE(context_->GetNodeName(),
                        "norm plane must be [num_slots, %d] (one 32B block per slot), got dim1=%ld",
                        kNzC0, static_cast<long>(normShape->GetStorageShape().GetDim(1))),
                return ge::GRAPH_FAILED);

    auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs are null"),
                return ge::GRAPH_FAILED);
    const int64_t *bitsAttr = attrs->GetAttrPointer<int64_t>(0);
    const float *scaleAttr = attrs->GetAttrPointer<float>(1);
    const int64_t *variantAttr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *cbAttr = attrs->GetAttrPointer<int64_t>(3);
    bits_ = (bitsAttr != nullptr) ? static_cast<uint32_t>(*bitsAttr) : 3U;
    /*
     * SCALE PLUMBING (measured bug).
     * The float attr does not survive the attr path on this op: passing
     * scale = 0.0625 / 0.25 / 1.0 / 0.0 from Python all produced BIT-IDENTICAL
     * output, with cos(out, mean(V)) = 0.9945 -- i.e. the kernel saw a scale of
     * zero, every score collapsed to 0, softmax went uniform and the output was
     * just mean(V). That alone accounts for the full-suite failure (0.690 vs
     * 0.991 achievable) and for why identical-key and injected-score probes all
     * passed. The int attrs (bits, variant, codebook_mode) plumb correctly, so
     * this is specific to the float attr.
     *
     * Attention scale is 1/sqrt(head_dim) by definition, so derive it. The attr
     * is still honoured when it reads back sane, which keeps a caller-supplied
     * scale working if the float path is fixed later.
     */
    const float derivedScale = 1.0f / std::sqrt(static_cast<float>(tilingData_.headDim));
    float attrScale = (scaleAttr != nullptr) ? *scaleAttr : 0.0f;
    if (!(attrScale > 0.0f) || !std::isfinite(attrScale)) {
        attrScale = derivedScale;
    }
    tilingData_.scale = attrScale;
    tilingData_.variant = (variantAttr != nullptr) ? static_cast<uint32_t>(*variantAttr) : 0U;
    tilingData_.codebookMode = (cbAttr != nullptr) ? static_cast<uint32_t>(*cbAttr) : 0U;

    OP_CHECK_IF(bits_ < 2 || bits_ > 4, OP_LOGE(context_->GetNodeName(), "bits must be in [2,4]"),
                return ge::GRAPH_FAILED);

    const uint32_t packedBits = tilingData_.headDim * bits_;
    OP_CHECK_IF(packedBits % kBitsPerHalf != 0,
                OP_LOGE(context_->GetNodeName(), "head_dim*bits must be a whole number of fp16 slots"),
                return ge::GRAPH_FAILED);
    tilingData_.packedHalves = packedBits / kBitsPerHalf;
    OP_CHECK_IF(tilingData_.packedHalves % kNzC0 != 0,
                OP_LOGE(context_->GetNodeName(),
                        "packed halves per head (%u) must be a multiple of %u; head_dim=%u bits=%u "
                        "is not NZ-tileable",
                        tilingData_.packedHalves, kNzC0, tilingData_.headDim, bits_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.packedHalves == 0 ||
                    (tilingData_.c1 * kNzC0) % tilingData_.packedHalves != 0,
                OP_LOGE(context_->GetNodeName(),
                        "c1*%d=%u not divisible by packedHalves=%u; cannot derive numKvHeads",
                        kNzC0, tilingData_.c1 * kNzC0, tilingData_.packedHalves),
                return ge::GRAPH_FAILED);
    tilingData_.numKvHeads = tilingData_.c1 * kNzC0 / tilingData_.packedHalves;
    OP_CHECK_IF(tilingData_.numKvHeads == 0 || tilingData_.numHeads % tilingData_.numKvHeads != 0,
                OP_LOGE(context_->GetNodeName(), "num_heads (%u) must be a multiple of num_kv_heads (%u)",
                        tilingData_.numHeads, tilingData_.numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.numKvHeads * tilingData_.packedHalves != tilingData_.c1 * kNzC0,
                OP_LOGE(context_->GetNodeName(), "cache C1 (%u) inconsistent with kv_heads*packed (%u)",
                        tilingData_.c1, tilingData_.numKvHeads * tilingData_.packedHalves),
                return ge::GRAPH_FAILED);

    // AscendC Sqrt is tensor-only; the kernel needs these as scalars.
    tilingData_.sqrtHeadDim = std::sqrt(static_cast<float>(tilingData_.headDim));
    tilingData_.invSqrtHeadDim = 1.0f / tilingData_.sqrtHeadDim;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantPagedAttentionV310Tiling::ComputeSplit()
{
    auto *platformInfo = context_->GetPlatformInfo();
    uint32_t coreNum = 8;
    if (platformInfo != nullptr) {
        auto p = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = static_cast<uint32_t>(p.GetCoreNumAiv());
    }
    if (coreNum == 0) {
        coreNum = 1;
    }
    // One (batch, kv_head) pair per task: all query heads in a group share the
    // unpacked block, which is what removes Tier 0's repeat_interleave.
    const uint32_t nTasks = tilingData_.batch * tilingData_.numKvHeads;
    const uint32_t used = (nTasks < coreNum) ? nTasks : coreNum;
    tilingData_.vectorCoreNum = (used == 0) ? 1 : used;
    tilingData_.tasksPerCore = (nTasks + tilingData_.vectorCoreNum - 1) / tilingData_.vectorCoreNum;

    tilingKey_ = 210 + bits_;
    workspaceSize_ = kSysWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantPagedAttentionV310Tiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    context_->SetTilingKey(tilingKey_);
    auto raw = context_->GetRawTilingData();
    OP_CHECK_IF(raw == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(raw->GetCapacity() < sizeof(tilingData_),
                OP_LOGE(context_->GetNodeName(), "tiling buffer too small"), return ge::GRAPH_FAILED);
    (void)memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData_, sizeof(tilingData_));
    raw->SetDataSize(sizeof(tilingData_));

    size_t *ws = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(ws == nullptr, OP_LOGE(context_->GetNodeName(), "workspace ptr is null"),
                return ge::GRAPH_FAILED);
    ws[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurboquantPagedAttentionV310Tiling::Run()
{
    if (ParseInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ComputeSplit() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return PostTiling();
}

ge::graphStatus TilingForTurboquantPagedAttentionV310(gert::TilingContext *context)
{
    TurboquantPagedAttentionV310Tiling tiling(context);
    return tiling.Run();
}

static ge::graphStatus TilingPrepareForTurboquantPagedAttentionV310(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<TurboquantPagedAttentionV310CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto p = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->coreNum = static_cast<uint32_t>(p.GetCoreNumAiv());
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantPagedAttentionV310)
    .Tiling(TilingForTurboquantPagedAttentionV310)
    .TilingParse<TurboquantPagedAttentionV310CompileInfo>(TilingPrepareForTurboquantPagedAttentionV310);
}  // namespace optiling
