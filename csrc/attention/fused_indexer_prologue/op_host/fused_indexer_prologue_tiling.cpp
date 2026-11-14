/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../mla_preprocess/op_host/mla_preprocess_tiling.h"
#include "../op_kernel/fused_indexer_prologue_tiling_data.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr uint32_t LATENT_INPUT = 0;
constexpr uint32_t WEIGHT_INPUT = 1;
constexpr uint32_t NORM_WEIGHT_INPUT = 2;
constexpr uint32_t COS_INPUT = 3;
constexpr uint32_t SIN_INPUT = 4;
constexpr uint32_t SLOT_MAPPING_INPUT = 5;
constexpr uint32_t K_CACHE_INPUT = 6;
constexpr uint32_t SCALE_CACHE_INPUT = 7;

constexpr uint32_t EPSILON_ATTR = 0;
constexpr uint32_t CACHE_BLOCK_SIZE_ATTR = 1;
constexpr uint32_t K_CACHE_STRIDE0_ATTR = 2;
constexpr uint32_t K_CACHE_STRIDE1_ATTR = 3;
constexpr uint32_t SCALE_CACHE_STRIDE0_ATTR = 4;
constexpr uint32_t SCALE_CACHE_STRIDE1_ATTR = 5;

constexpr int64_t HEAD_DIM = 128;
constexpr int64_t ROPE_DIM = 64;
constexpr int64_t INPUT_DIM = 512;
constexpr uint64_t TILING_KEY = 0;

struct FusedIndexerPrologueCompileInfo {};

ge::graphStatus FillPlatformInfo(
    gert::TilingContext *context, mlapo::PlatformInfo &platformInfo,
    uint64_t &systemWorkspaceSize)
{
    fe::PlatFormInfos *platform = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform);
    const platform_ascendc::PlatformAscendC ascendcPlatform(platform);
    platformInfo.coreNum = ascendcPlatform.GetCoreNum();
    platformInfo.coreNumAic = ascendcPlatform.GetCoreNumAic();
    platformInfo.coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);
    systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(
        platformInfo.coreNumAic == 0 || platformInfo.coreNumAiv == 0,
        OP_LOGE(context, "A2/A3 mixed AIC/AIV cores are required."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFusedIndexerPrologue(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FusedIndexerPrologue", "TilingContext is null."),
                return ge::GRAPH_FAILED);

    const auto *latentShape = context->GetInputShape(LATENT_INPUT);
    const auto *weightShape = context->GetInputShape(WEIGHT_INPUT);
    const auto *normWeightShape = context->GetInputShape(NORM_WEIGHT_INPUT);
    const auto *cosShape = context->GetInputShape(COS_INPUT);
    const auto *sinShape = context->GetInputShape(SIN_INPUT);
    const auto *slotShape = context->GetInputShape(SLOT_MAPPING_INPUT);
    const auto *kCacheShape = context->GetInputShape(K_CACHE_INPUT);
    const auto *scaleCacheShape = context->GetInputShape(SCALE_CACHE_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleCacheShape);

    const gert::Shape &latent = latentShape->GetOriginShape();
    const gert::Shape &weight = weightShape->GetOriginShape();
    const gert::Shape &normWeight = normWeightShape->GetOriginShape();
    const gert::Shape &cos = cosShape->GetOriginShape();
    const gert::Shape &sin = sinShape->GetOriginShape();
    const gert::Shape &slots = slotShape->GetOriginShape();
    const gert::Shape &kCache = kCacheShape->GetOriginShape();
    const gert::Shape &scaleCache = scaleCacheShape->GetOriginShape();

    OP_CHECK_IF(
        latent.GetDimNum() != 2 || weight.GetDimNum() != 2 ||
            normWeight.GetDimNum() != 1,
        OP_LOGE(context, "latent/weight/norm_weight ranks must be 2/2/1."),
        return ge::GRAPH_FAILED);
    const int64_t numTokens = latent.GetDim(0);
    const int64_t inputDim = latent.GetDim(1);
    OP_CHECK_IF(
        numTokens <= 0 || inputDim != INPUT_DIM || weight.GetDim(0) != HEAD_DIM ||
            weight.GetDim(1) != inputDim || normWeight.GetDim(0) != HEAD_DIM,
        OP_LOGE(context, "invalid 512-to-128 indexer projection geometry."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        cos.GetDimNum() == 0 || sin.GetDimNum() == 0 ||
            cos.GetDim(cos.GetDimNum() - 1) != ROPE_DIM ||
            sin.GetDim(sin.GetDimNum() - 1) != ROPE_DIM ||
            cos.GetShapeSize() != numTokens * ROPE_DIM ||
            sin.GetShapeSize() != numTokens * ROPE_DIM,
        OP_LOGE(context, "cos/sin must provide [T, ..., 64] with singleton middle dimensions."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        slots.GetDimNum() != 2 || slots.GetDim(0) != numTokens ||
            slots.GetDim(1) != 2,
        OP_LOGE(context, "slot_mapping must provide [T, 2]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kCache.GetDimNum() != 4 || kCache.GetDim(2) != 1 ||
            kCache.GetDim(3) != HEAD_DIM || scaleCache.GetDimNum() != 4 ||
            scaleCache.GetDim(2) != 1 || scaleCache.GetDim(3) != 1 ||
            kCache.GetDim(0) != scaleCache.GetDim(0),
        OP_LOGE(context, "invalid index K/scale cache shapes."),
        return ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float *epsilon = attrs->GetFloat(EPSILON_ATTR);
    const int64_t *cacheBlockSize = attrs->GetInt(CACHE_BLOCK_SIZE_ATTR);
    const int64_t *kStride0 = attrs->GetInt(K_CACHE_STRIDE0_ATTR);
    const int64_t *kStride1 = attrs->GetInt(K_CACHE_STRIDE1_ATTR);
    const int64_t *scaleStride0 = attrs->GetInt(SCALE_CACHE_STRIDE0_ATTR);
    const int64_t *scaleStride1 = attrs->GetInt(SCALE_CACHE_STRIDE1_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheBlockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, kStride0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kStride1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleStride0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleStride1);
    OP_CHECK_IF(
        *epsilon <= 0.0F || *cacheBlockSize <= 0 || *kStride0 <= 0 ||
            *kStride1 <= 0 || *scaleStride0 <= 0 || *scaleStride1 <= 0,
        OP_LOGE(context, "epsilon and cache geometry attributes must be positive."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *cacheBlockSize != kCache.GetDim(1) ||
            *cacheBlockSize != scaleCache.GetDim(1),
        OP_LOGE(context, "cache_block_size must match both cache tensors."),
        return ge::GRAPH_FAILED);

    mlapo::PlatformInfo platformInfo{};
    uint64_t systemWorkspaceSize = 0;
    OP_CHECK_IF(
        FillPlatformInfo(context, platformInfo, systemWorkspaceSize) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to query A2/A3 platform information."),
        return ge::GRAPH_FAILED);

    auto *tilingData = context->GetTilingData<FusedIndexerPrologueTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = FusedIndexerPrologueTilingData{};
    mlapo::PpMatmulTilingApi matmulTiling(
        platformInfo, 1, static_cast<uint32_t>(numTokens),
        static_cast<uint32_t>(inputDim), HEAD_DIM, false, true, false, false);
    matmulTiling.GetTilingData(tilingData->matmul);
    tilingData->numTokens = static_cast<uint32_t>(numTokens);
    tilingData->inputDim = static_cast<uint32_t>(inputDim);
    tilingData->headDim = HEAD_DIM;
    tilingData->ropeDim = ROPE_DIM;
    tilingData->vectorCoreNum = std::min(
        platformInfo.coreNumAiv, platformInfo.coreNumAic * 2U);
    tilingData->cacheBlocks = static_cast<uint32_t>(kCache.GetDim(0));
    tilingData->cacheBlockSize = static_cast<uint32_t>(*cacheBlockSize);
    tilingData->epsilon = *epsilon;
    tilingData->kCacheStride0 = static_cast<uint64_t>(*kStride0);
    tilingData->kCacheStride1 = static_cast<uint64_t>(*kStride1);
    tilingData->scaleCacheStride0 = static_cast<uint64_t>(*scaleStride0);
    tilingData->scaleCacheStride1 = static_cast<uint64_t>(*scaleStride1);

    const uint64_t projectionBytes =
        static_cast<uint64_t>(numTokens) * HEAD_DIM * sizeof(uint16_t);
    OP_CHECK_IF(
        projectionBytes > std::numeric_limits<uint64_t>::max() - systemWorkspaceSize,
        OP_LOGE(context, "workspace size overflow."),
        return ge::GRAPH_FAILED);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(systemWorkspaceSize + projectionBytes);
    context->SetBlockDim(platformInfo.coreNumAic);
    context->SetTilingKey(TILING_KEY);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseFusedIndexerPrologue(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedIndexerPrologue)
    .Tiling(TilingFusedIndexerPrologue)
    .TilingParse<FusedIndexerPrologueCompileInfo>(
        TilingParseFusedIndexerPrologue);
}  // namespace optiling
