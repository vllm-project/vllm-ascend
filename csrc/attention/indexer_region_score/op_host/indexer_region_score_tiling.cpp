// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "indexer_region_score_shard_plan.h"
#include "indexer_region_score_bounded_workspace.h"
#include "../op_kernel/indexer_region_score_paged_tiling_data.h"
namespace optiling {
namespace {
constexpr uint32_t MIN_HEADS = 4, MAX_HEADS = 128, DIM = 256, MAX_REGIONS = 16777216, MAX_TOP_K = 512;
constexpr uint32_t MASK_MODE = 3, TILING_KEY = 13;
using IndexerRegionScorePaged::PagedTilingData;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *regions = attrs->GetAttrPointer<int64_t>(0);
    const auto *topK = attrs->GetAttrPointer<int64_t>(1);
    const auto *regionSize = attrs->GetAttrPointer<int64_t>(2);
    const auto *headScale = attrs->GetAttrPointer<float>(3);
    const auto *cacheMode = attrs->GetAttrPointer<int64_t>(4);
    const auto *maskMode = attrs->GetAttrPointer<int64_t>(5);
    const auto *sortMode = attrs->GetAttrPointer<int64_t>(6);
    const auto *returnScore = attrs->GetAttrPointer<bool>(7);
    const bool fullScore = returnScore == nullptr || *returnScore;
    OP_CHECK_NULL_WITH_CONTEXT(context, regions);
    OP_CHECK_NULL_WITH_CONTEXT(context, topK);
    OP_CHECK_NULL_WITH_CONTEXT(context, regionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, headScale);
    OP_CHECK_IF(!cacheMode || !maskMode || !sortMode || *cacheMode != 1 || *maskMode != 3 || *sortMode != 1,
        OP_LOGE(context, "only cache_mode=1, mask_mode=3 and sort_mode=1 are supported"), return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < 6; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(i));
    }
    for (uint32_t i = 0; i < 2; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(i));
    }
    const auto &q = context->GetInputShape(0)->GetOriginShape();
    const auto &w = context->GetInputShape(1)->GetOriginShape();
    const auto &k = context->GetInputShape(2)->GetOriginShape();
    const auto &lengths = context->GetInputShape(3)->GetOriginShape();
    const auto &table = context->GetInputShape(4)->GetOriginShape();
    const auto &cu = context->GetInputShape(5)->GetOriginShape();
    const auto &out = context->GetOutputShape(0)->GetOriginShape();
    const auto &ids = context->GetOutputShape(1)->GetOriginShape();
    OP_CHECK_IF(q.GetDimNum() != 4 || w.GetDimNum() != 3 || k.GetDimNum() != 4 ||
        lengths.GetDimNum() != 1 || table.GetDimNum() != 2 || cu.GetDimNum() != 1 ||
        (fullScore ? out.GetDimNum() != 3 : out.GetDimNum() != 1 || out.GetDim(0) != 0) || ids.GetDimNum() != 3,
        OP_LOGE(context, "invalid tensor ranks"), return ge::GRAPH_FAILED);
    const int64_t groups = q.GetDim(1), heads = q.GetDim(2);
    OP_CHECK_IF(q.GetDim(0) <= 0 || q.GetDim(0) > INT32_MAX || groups <= 0 ||
        (groups != 1 && groups % MIN_HEADS != 0) || heads < MIN_HEADS || heads > MAX_HEADS ||
        heads % MIN_HEADS != 0 || groups > UINT32_MAX / q.GetDim(0) || groups > UINT32_MAX / (heads * DIM) ||
        q.GetDim(3) != DIM || w.GetDim(0) != q.GetDim(0) || w.GetDim(1) != groups || w.GetDim(2) != heads ||
        k.GetDim(0) <= 0 || k.GetDim(0) > UINT32_MAX || k.GetDim(1) <= 0 || k.GetDim(1) > 1024 ||
        k.GetDim(2) != 1 || k.GetDim(3) != DIM || table.GetDim(0) <= 0 || table.GetDim(0) > UINT32_MAX ||
        table.GetDim(1) <= 0 || table.GetDim(1) > UINT32_MAX / k.GetDim(1) ||
        lengths.GetDim(0) != table.GetDim(0) || cu.GetDim(0) != table.GetDim(0) + 1,
        OP_LOGE(context, "expected Q[T,G,H,256], W[T,G,H], K[P,S,1,256]; G=1 or a positive multiple of 4, H=4..128 in steps of 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*regions < 0 || *regions > MAX_REGIONS || *regions > table.GetDim(1) * k.GetDim(1) ||
        *topK <= 0 || *topK > MAX_TOP_K || *regionSize <= 0 || *regionSize > INT32_MAX ||
        !std::isfinite(*headScale) || (fullScore && (out.GetDim(0) != groups || out.GetDim(1) != q.GetDim(0) ||
        out.GetDim(2) < *regions || out.GetDim(2) > UINT32_MAX)) || ids.GetDim(0) != groups ||
        ids.GetDim(1) != q.GetDim(0) || ids.GetDim(2) != *topK,
        OP_LOGE(context, "invalid scalar attributes or output shapes"), return ge::GRAPH_FAILED);
    const auto qtype = context->GetInputDesc(0)->GetDataType();
    const auto wtype = context->GetInputDesc(1)->GetDataType();
    OP_CHECK_IF((qtype != ge::DT_FLOAT8_E4M3FN && qtype != ge::DT_FLOAT16 && qtype != ge::DT_BF16) ||
        context->GetInputDesc(2)->GetDataType() != qtype ||
        (wtype != ge::DT_FLOAT && wtype != ge::DT_FLOAT16 && wtype != ge::DT_BF16) ||
        context->GetOutputDesc(0)->GetDataType() != ge::DT_FLOAT ||
        context->GetOutputDesc(1)->GetDataType() != ge::DT_INT32,
        OP_LOGE(context, "invalid input/output dtypes"), return ge::GRAPH_FAILED);
    for (uint32_t i = 3; i < 6; ++i) {
        OP_CHECK_IF(context->GetInputDesc(i)->GetDataType() != ge::DT_INT32,
            OP_LOGE(context, "request metadata must be int32"), return ge::GRAPH_FAILED);
    }
    PagedTilingData data{};
    data.cacheStridePage = k.GetDim(1) * DIM;
    const auto *strides = context->GetRequiredInputStride(2);
    if (strides == nullptr) strides = context->GetInputStride(2);
    // TensorV2 can omit canonical dense strides for a workspace tensor.
    const gert::Stride denseStrides({k.GetDim(1) * DIM, DIM, DIM, 1});
    if (strides != nullptr && strides->GetDimNum() == 0) strides = &denseStrides;
    if (strides != nullptr || context->InputIsView(2)) {
        const int64_t elementBytes = qtype == ge::DT_FLOAT8_E4M3FN ? 1 : 2;
        int64_t offset = context->GetRequiredInputOffset(2);
        if (offset < 0) offset = context->GetInputOffset(2);
        if (offset < 0 && !context->InputIsView(2)) offset = 0;
        const bool validStrides = strides != nullptr && strides->GetDimNum() == 4;
        OP_CHECK_IF(!validStrides || strides->GetStride(3) != 1 ||
            (k.GetDim(1) > 1 && strides->GetStride(1) != DIM) ||
            strides->GetStride(0) < k.GetDim(1) * DIM ||
            strides->GetStride(0) % (32 / elementBytes) != 0 || offset < 0 ||
            offset % (32 / elementBytes) != 0 || offset > INT64_MAX / elementBytes - k.GetDim(1) * DIM ||
            k.GetDim(0) - 1 > (INT64_MAX / elementBytes - k.GetDim(1) * DIM - offset) / strides->GetStride(0),
            OP_LOGE(context, "paged cache view must have aligned, contiguous page contents: "
                "stride_dims=%ld, strides[0,1,3]=[%ld,%ld,%ld], offset=%ld, required_offset=%ld, input_offset=%ld, is_view=%d",
                strides ? static_cast<int64_t>(strides->GetDimNum()) : -1, validStrides ? strides->GetStride(0) : -1,
                validStrides ? strides->GetStride(1) : -1, validStrides ? strides->GetStride(3) : -1, offset,
                context->GetRequiredInputOffset(2), context->GetInputOffset(2), static_cast<int>(context->InputIsView(2))),
            return ge::GRAPH_FAILED);
        data.cacheStridePage = strides->GetStride(0);
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic(), aiv = platform.GetCoreNumAiv();
    OP_CHECK_IF(aic == 0 || aiv < 2, OP_LOGE(context, "invalid core counts"), return ge::GRAPH_FAILED);
    const uint32_t keyTile = qtype == ge::DT_FLOAT8_E4M3FN ? 512 : 256;
    const uint32_t keyRanges = (*regions + keyTile - 1) / keyTile;
    auto plan = IndexerRegionScoreHost::SelectPagedRequestShardPlan(
        q.GetDim(0), table.GetDim(0), keyRanges, std::min(aic, aiv / 2), groups, k.GetDim(1), heads, keyTile);
    if (qtype != ge::DT_FLOAT8_E4M3FN) {
        plan.shards = IndexerRegionScoreHost::FillPagedQueryShards(
            plan.shards, (q.GetDim(0) + plan.qMicroTile - 1) / plan.qMicroTile,
            keyRanges, std::min(aic, aiv / 2));
    }
    data.totalTokens = q.GetDim(0);
    data.groups = groups;
    data.headsPerGroup = heads;
    data.proxyDim = DIM;
    data.numRequests = table.GetDim(0);
    data.pageCols = table.GetDim(1);
    data.summariesPerPage = k.GetDim(1);
    data.seqK = *regions;
    data.qTile = plan.qMicroTile;
    data.kTile = keyTile;
    data.qShardCount = plan.shards.qShards;
    data.kShardCount = plan.shards.kShards;
    data.usedCubeCores = data.qShardCount * data.kShardCount;
    data.maskMode = MASK_MODE;
    data.headScale = *headScale;
    data.keyPagesPerMacro = keyTile % data.summariesPerPage == 0 ? keyTile / data.summariesPerPage :
        (keyTile + data.summariesPerPage - 1) / data.summariesPerPage + 1;
    data.pageLoadMode = keyTile % data.summariesPerPage == 0 ? 0 : 1;
    data.numPhysicalPages = k.GetDim(0);
    data.weightDtype = wtype == ge::DT_FLOAT ? 0 : wtype == ge::DT_BF16 ? 1 : 2;
    data.qStrideToken = groups * heads * DIM;
    data.qStrideGroup = heads * DIM;
    data.qStrideHead = DIM;
    data.weightStrideToken = groups * heads;
    data.weightStrideGroup = heads;
    data.cacheStrideRegion = DIM;
    data.blockTableStrideRequest = data.pageCols;
    data.outputStrideToken = fullScore ? out.GetDim(2) : 0U;
    data.outputStrideGroup = data.totalTokens * data.outputStrideToken;
    data.regionSize = *regionSize;
    data.selectionConfig = static_cast<uint32_t>(*topK) | (1U << 16);
    auto *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    if (!fullScore) {
        IndexerRegionScoreHost::BoundedWorkspacePlan bounded;
        OP_CHECK_IF(!IndexerRegionScoreHost::PlanBoundedWorkspace(workspace[0], groups, data.totalTokens,
            data.seqK, data.qTile, bounded), OP_LOGE(context, "TopK-only needs more than the 256 MiB workspace budget"),
            return ge::GRAPH_FAILED);
        data.batchTokens = bounded.batchTokens;
        data.scorePoolOffset = bounded.scorePoolOffset;
        data.outputStrideToken = bounded.regionStride;
        data.outputStrideGroup = data.batchTokens * bounded.regionStride;
        workspace[0] = bounded.totalBytes;
    }
    context->SetBlockDim(platform.CalcTschBlockDim(data.usedCubeCores * 2, aic, aiv));
    context->SetScheduleMode(1);
    context->SetTilingKey(TILING_KEY);
    auto *raw = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_CHECK_IF(raw->GetCapacity() < sizeof(data), OP_LOGE(context, "tiling buffer too small"), return ge::GRAPH_FAILED);
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
ge::graphStatus Parse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(IndexerRegionScore).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
} // namespace optiling
