/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "build_dspark_swa_indices_tiling.h"

#include <algorithm>
#include <cstdio>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

// Self-contained error-handling macros — avoids the dependency on
// tiling_base/error_log.h (in csrc/common/include/) so that opbuild can
// compile this file in the standalone CANN-template build project without
// the extra include path. The macros mirror error_log.h semantics.
#ifndef TILING_OPS_LOGE
#define TILING_OPS_LOGE(opname, ...)  \
    do {                              \
        (void)(opname);               \
        std::printf("[ERROR] ");      \
        std::printf(__VA_ARGS__);     \
        std::printf("\n");            \
    } while (0)
#endif

#ifndef TILING_CHECK_NULL_WITH_CONTEXT
#define TILING_CHECK_NULL_WITH_CONTEXT(context, ptr)  \
    do {                                               \
        if ((ptr) == nullptr) {                        \
            return ge::GRAPH_FAILED;                   \
        }                                              \
    } while (0)
#endif

namespace optiling {
namespace {
constexpr uint32_t KV_BLOCK_TABLE_INDEX = 0;
constexpr uint32_t QUERY_START_LOC_INDEX = 1;
constexpr uint32_t SEQ_LENS_INDEX = 2;
constexpr uint32_t PER_TOKEN_SLOTS_INDEX = 0;

constexpr int64_t MAX_UINT32_VALUE = 0xFFFFFFFFLL;
constexpr int64_t MAX_INT32_VALUE = 0x7FFFFFFFLL;

constexpr uint32_t REQUESTS_PER_CORE_TARGET = 4;

uint32_t AlignUp(uint64_t value, uint32_t align)
{
    return static_cast<uint32_t>((value + align - 1) / align * align);
}

uint32_t CeilDiv(uint64_t lhs, uint64_t rhs)
{
    return static_cast<uint32_t>((lhs + rhs - 1) / rhs);
}
}  // namespace

static ge::graphStatus BuildDsparkSwaIndicesTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    TILING_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        aivCoreNum = ascendcPlatform.GetCoreNum();
    }
    if (aivCoreNum == 0) {
        TILING_OPS_LOGE(context->GetNodeName(), "Failed to get AIV core num.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        TILING_OPS_LOGE(context->GetNodeName(), "Failed to get UB size.");
        return ge::GRAPH_FAILED;
    }

    // --- Validate inputs ---
    auto blockTableShape = context->GetInputShape(KV_BLOCK_TABLE_INDEX);
    auto qslShape = context->GetInputShape(QUERY_START_LOC_INDEX);
    auto seqLensShape = context->GetInputShape(SEQ_LENS_INDEX);
    auto outputShape = context->GetOutputShape(PER_TOKEN_SLOTS_INDEX);
    TILING_CHECK_NULL_WITH_CONTEXT(context, blockTableShape);
    TILING_CHECK_NULL_WITH_CONTEXT(context, qslShape);
    TILING_CHECK_NULL_WITH_CONTEXT(context, seqLensShape);
    TILING_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    if (blockTableShape->GetStorageShape().GetDimNum() != 2) {
        TILING_OPS_LOGE(context->GetNodeName(), "kvBlockTable should be a 2D tensor.");
        return ge::GRAPH_FAILED;
    }
    if (qslShape->GetStorageShape().GetDimNum() != 1 || qslShape->GetStorageShape().GetDim(0) < 2) {
        TILING_OPS_LOGE(context->GetNodeName(), "queryStartLoc should be a 1D tensor with at least 2 elements.");
        return ge::GRAPH_FAILED;
    }
    if (seqLensShape->GetStorageShape().GetDimNum() != 1 || seqLensShape->GetStorageShape().GetDim(0) < 1) {
        TILING_OPS_LOGE(context->GetNodeName(), "seqLens should be a non-empty 1D tensor.");
        return ge::GRAPH_FAILED;
    }
    if (outputShape->GetStorageShape().GetDimNum() != 3) {
        TILING_OPS_LOGE(context->GetNodeName(), "perTokenSlots should be a 3D tensor [N, 1, W].");
        return ge::GRAPH_FAILED;
    }

    int64_t numReqs = qslShape->GetStorageShape().GetDim(0) - 1;
    int64_t numDecodeTokens = outputShape->GetStorageShape().GetDim(0);
    int64_t indexWidth = outputShape->GetStorageShape().GetDim(2);
    int64_t blockTableStride = blockTableShape->GetStorageShape().GetDim(1);
    int64_t blockTableRows = blockTableShape->GetStorageShape().GetDim(0);

    if (numReqs <= 0 || numReqs > MAX_UINT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "numReqs (queryStartLoc[0]-1) is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (numDecodeTokens <= 0 || numDecodeTokens > MAX_UINT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "numDecodeTokens (output dim 0) is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (indexWidth <= 0 || indexWidth > MAX_UINT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "indexWidth (output dim 2) is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (blockTableStride <= 0 || blockTableStride > MAX_UINT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "kvBlockTable stride (dim 1) is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (blockTableRows < numReqs) {
        TILING_OPS_LOGE(context->GetNodeName(), "kvBlockTable rows < numReqs.");
        return ge::GRAPH_FAILED;
    }
    if (seqLensShape->GetStorageShape().GetDim(0) < numReqs) {
        TILING_OPS_LOGE(context->GetNodeName(), "seqLens length < numReqs.");
        return ge::GRAPH_FAILED;
    }

    // --- Read attrs ---
    auto attrs = context->GetAttrs();
    TILING_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numSpecTokensPtr = attrs->GetInt(0);
    const int64_t* windowSizePtr = attrs->GetInt(1);
    const int64_t* blockSizePtr = attrs->GetInt(2);
    TILING_CHECK_NULL_WITH_CONTEXT(context, numSpecTokensPtr);
    TILING_CHECK_NULL_WITH_CONTEXT(context, windowSizePtr);
    TILING_CHECK_NULL_WITH_CONTEXT(context, blockSizePtr);
    if (*numSpecTokensPtr <= 0 || *numSpecTokensPtr > MAX_INT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "numSpeculativeTokens should be in (0, INT32_MAX].");
        return ge::GRAPH_FAILED;
    }
    if (*windowSizePtr < 0 || *windowSizePtr > MAX_INT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "windowSize should be in [0, INT32_MAX].");
        return ge::GRAPH_FAILED;
    }
    if (*blockSizePtr <= 0 || *blockSizePtr > MAX_INT32_VALUE) {
        TILING_OPS_LOGE(context->GetNodeName(), "blockSize should be in (0, INT32_MAX].");
        return ge::GRAPH_FAILED;
    }

    // --- Core allocation ---
    // R is small (1-256); target 4 requests per core for good parallelism
    // without over-fragmenting.
    uint32_t usedCoreNum = std::min(
        aivCoreNum,
        std::max(1U, CeilDiv(static_cast<uint64_t>(numReqs), REQUESTS_PER_CORE_TARGET)));

    // --- Fill tiling data ---
    BuildDsparkSwaIndicesTilingData tilingData;
    tilingData.set_numReqs(static_cast<uint32_t>(numReqs));
    tilingData.set_numDecodeTokens(static_cast<uint32_t>(numDecodeTokens));
    tilingData.set_numSpeculativeTokens(static_cast<uint32_t>(*numSpecTokensPtr));
    tilingData.set_windowSize(static_cast<uint32_t>(*windowSizePtr));
    tilingData.set_blockSize(static_cast<uint32_t>(*blockSizePtr));
    tilingData.set_indexWidth(static_cast<uint32_t>(indexWidth));
    tilingData.set_blockTableStride(static_cast<uint32_t>(blockTableStride));
    tilingData.set_usedCoreNum(usedCoreNum);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    TILING_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    *workspaceSize = 0;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);

    auto rawTilingData = context->GetRawTilingData();
    TILING_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBuildDsparkSwaIndices(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BuildDsparkSwaIndices)
    .Tiling(BuildDsparkSwaIndicesTilingFunc)
    .TilingParse<BuildDsparkSwaIndicesCompileInfo>(TilingParseForBuildDsparkSwaIndices);

}  // namespace optiling
