// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/flash_attn_c8_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t QUERY_TILE = 128;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t MASK_SIZE = 2048;

bool ShapeIs(gert::TilingContext *ctx, size_t index, std::initializer_list<int64_t> expected)
{
    const auto *shape = ctx->GetInputShape(index);
    if (!shape || shape->GetStorageShape().GetDimNum() != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (auto value : expected) {
        if (shape->GetStorageShape().GetDim(dim++) != value) {
            return false;
        }
    }
    return true;
}

ge::graphStatus TilingFlashAttnC8(gert::TilingContext *ctx)
{
    const auto *q = ctx->GetInputShape(0);
    const auto *k = ctx->GetInputShape(1);
    const auto *cuq = ctx->GetInputShape(8);
    const auto *attrs = ctx->GetAttrs();
    if (!q || !k || !cuq || !attrs || !ctx->GetPlatformInfo() ||
        q->GetStorageShape().GetDimNum() != 3 || k->GetStorageShape().GetDimNum() != 3 ||
        cuq->GetStorageShape().GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tq = q->GetStorageShape().GetDim(0);
    const int64_t tk = k->GetStorageShape().GetDim(0);
    const int64_t heads = q->GetStorageShape().GetDim(1);
    const int64_t batch = cuq->GetStorageShape().GetDim(0) - 1;
    const int64_t maskMode = *attrs->GetAttrPointer<int64_t>(1);
    const int64_t maxQ = *attrs->GetAttrPointer<int64_t>(2);
    const int64_t maxK = *attrs->GetAttrPointer<int64_t>(3);
    const float scale = *attrs->GetAttrPointer<float>(0);
    if (tq <= 0 || tk <= 0 || heads <= 0 || batch <= 0 ||
        tq > INT32_MAX || tk > INT32_MAX || heads > UINT32_MAX ||
        maxQ <= 0 || maxQ > tq || maxK <= 0 || maxK > tk ||
        !std::isfinite(scale) || scale <= 0 || (maskMode != 0 && maskMode != 3) ||
        !ShapeIs(ctx, 0, {tq, heads, HEAD_DIM}) ||
        !ShapeIs(ctx, 1, {tk, heads, HEAD_DIM}) || !ShapeIs(ctx, 2, {tk, heads, HEAD_DIM}) ||
        !ShapeIs(ctx, 3, {tq, heads, ROPE_DIM}) || !ShapeIs(ctx, 4, {tk, heads, ROPE_DIM}) ||
        !ShapeIs(ctx, 5, {tq, heads}) || !ShapeIs(ctx, 6, {heads}) || !ShapeIs(ctx, 7, {heads}) ||
        !ShapeIs(ctx, 9, {batch + 1})) {
        OP_LOGE(ctx->GetNodeName(), "Invalid FlashAttnC8 TND dimensions, scales or sequence bounds");
        return ge::GRAPH_FAILED;
    }
    if (ctx->GetOptionalInputShape(10) && !ShapeIs(ctx, 10, {batch})) {
        return ge::GRAPH_FAILED;
    }
    if (maskMode == 3 && !ShapeIs(ctx, 11, {MASK_SIZE, MASK_SIZE})) {
        OP_LOGE(ctx->GetNodeName(), "Causal FlashAttnC8 requires the shared 2048x2048 mask");
        return ge::GRAPH_FAILED;
    }
    const auto *metadata = ctx->GetInputShape(12);
    if (!metadata || metadata->GetStorageShape().GetDimNum() != 1 ||
        metadata->GetStorageShape().GetShapeSize() < 4096) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    uint64_t ub = 0, l1 = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    if (aic == 0 || aic > FA_AIC_CORE_NUM || aiv != 2 * aic || ub < 248 * 1024 || l1 < 512 * 1024) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t blocks = platform.CalcTschBlockDim(aiv, aic, aiv);
    auto *data = ctx->GetTilingData<FlashAttnC8TilingData>();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    *data = FlashAttnC8TilingData{};
    auto &base = data->baseTiling.flashMlaWithKvcacheBaseParams;
    base.bSize = batch;
    base.t1Size = tq;
    base.t2Size = tk;
    base.n2Size = heads;
    base.gSize = 1;
    base.s1Size = maxQ;
    base.s2Size = maxK;
    base.dSize = HEAD_DIM + ROPE_DIM;
    base.dSizeV = HEAD_DIM;
    base.dSizeRope = ROPE_DIM;
    base.actualSeqLengthsQSize = batch + 1;
    // The kernel passes cu_seqlens_kv + 1 to the cumulative-end parser.
    base.actualSeqLengthsKVSize = batch;
    base.scaleValue = scale;
    base.isKvContinuous = 1;
    base.isSoftMaxLseEnable = *attrs->GetAttrPointer<bool>(4);
    base.coreNum = blocks;
    base.outputLayout = 3; // AttentionCommon::FIA_LAYOUT::TND
    base.l2CacheOffFlag = 0;
    auto &mask = data->baseTiling.flashMlaWithKvcacheAttenMaskParams;
    mask.sparseMode = maskMode;
    mask.preTokens = INT32_MAX;
    mask.nextTokens = 0;
    if (maskMode == 3) {
        mask.attenMaskBatch = 1;
        mask.attenMaskS1Size = MASK_SIZE;
        mask.attenMaskS2Size = MASK_SIZE;
    }
    data->baseTiling.flashMlaWithKvcacheSystemPrefixParams.isActualSharedPrefixLenNull = 1;
    auto &ws = data->baseTiling.flashMlaWithKvcacheWorkspaceParams;
    ws.accumOutSize = blocks * 2 * QUERY_TILE * HEAD_DIM;
    ws.logSumExpSize = blocks * 2 * QUERY_TILE * 8;
    auto &empty = data->baseTiling.flashMlaWithKvcacheEmptyTensorParams;
    empty.totalOutputSize = static_cast<uint64_t>(tq) * heads * HEAD_DIM;
    empty.totalSoftMaxLseOutputSize = base.isSoftMaxLseEnable ? static_cast<uint64_t>(tq) * heads : 0;
    empty.singleCoreSize = (empty.totalOutputSize + aiv - 1) / aiv;
    empty.needInit = 1;
    ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        static_cast<uint64_t>(ws.accumOutSize + 2 * ws.logSumExpSize) * sizeof(float);
    ctx->SetBlockDim(blocks);
    ctx->SetTilingKey(maskMode == 3 ? 3 : 2);
    ctx->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_OPTILING(FlashAttnC8).Tiling(TilingFlashAttnC8);
}
