/**
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms of CANN Open Software License Agreement Version 2.0.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND. See LICENSE in the repository root for the full license text.
 */

#include "../op_kernel/triangle_paged_sparse_attention_tiling.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef OP_LOGE
#define OP_LOGE(nodeName, ...)                    \
    do {                                          \
        (void)(nodeName);                         \
        std::fprintf(stderr, "[TPSA] ");          \
        std::fprintf(stderr, __VA_ARGS__);        \
        std::fprintf(stderr, "\n");               \
    } while (0)
#endif

namespace {

constexpr uint32_t kTilingMagic = 0x54505341U;  // "TPSA"
constexpr uint32_t kAbiVersion = 2;
constexpr uint32_t kFastImplementation = 2;
constexpr int64_t kQueryHeads = 32;
constexpr int64_t kKvHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 128;
constexpr int64_t kQueryTile = 32;
constexpr int64_t kKvTile = 512;
constexpr int64_t kSinkTokens = 8;
constexpr int64_t kLocalWindow = 512;
constexpr int64_t kDenseTail = 128;
constexpr uint32_t kGroupSize =
    static_cast<uint32_t>(kQueryHeads / kKvHeads);
constexpr uint32_t kCubeRows =
    static_cast<uint32_t>(kQueryTile) * kGroupSize;
constexpr uint32_t kScoreBytes =
    kCubeRows * static_cast<uint32_t>(kKvTile) * sizeof(float);
constexpr uint32_t kProbabilityBytes =
    kCubeRows * static_cast<uint32_t>(kKvTile) * sizeof(uint16_t);
constexpr uint32_t kOutputTmpBytes =
    kCubeRows * static_cast<uint32_t>(kHeadDim) * sizeof(float);
constexpr uint32_t kOutputUpdateBytes = kOutputTmpBytes;
constexpr uint32_t kLseScratchBytes = kCubeRows * sizeof(float);
constexpr uint32_t kWorkspaceAlignment = 512;

constexpr uint32_t AlignUpU32(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

constexpr uint32_t kScoreOffsetBytes = 0;
constexpr uint32_t kProbabilityOffsetBytes =
    kScoreOffsetBytes + kScoreBytes;
constexpr uint32_t kOutputTmpOffsetBytes =
    kProbabilityOffsetBytes + kProbabilityBytes;
constexpr uint32_t kOutputUpdateOffsetBytes =
    kOutputTmpOffsetBytes + kOutputTmpBytes;
constexpr uint32_t kLseScratchOffsetBytes =
    kOutputUpdateOffsetBytes + kOutputUpdateBytes;
constexpr uint32_t kWorkspacePerCoreBytes = AlignUpU32(
    kLseScratchOffsetBytes + kLseScratchBytes,
    kWorkspaceAlignment);

enum InputIndex : size_t {
    kQuery = 0,
    kKeyCache = 1,
    kValueCache = 2,
    kBlockTable = 3,
};

bool HasShape(
    const gert::StorageShape* storageShape,
    std::initializer_list<int64_t> expectedTail)
{
    if (storageShape == nullptr) {
        return false;
    }
    const gert::Shape& shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() != expectedTail.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t expected : expectedTail) {
        if (expected >= 0 && shape.GetDim(index) != expected) {
            return false;
        }
        ++index;
    }
    return true;
}

bool SameShape(
    const gert::StorageShape* lhsStorage,
    const gert::StorageShape* rhsStorage)
{
    if (lhsStorage == nullptr || rhsStorage == nullptr) {
        return false;
    }
    const gert::Shape& lhs = lhsStorage->GetStorageShape();
    const gert::Shape& rhs = rhsStorage->GetStorageShape();
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace optiling {

struct TrianglePagedSparseAttentionCompileInfo {};

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto* queryShape = context->GetInputShape(kQuery);
    const auto* keyShape = context->GetInputShape(kKeyCache);
    const auto* valueShape = context->GetInputShape(kValueCache);
    const auto* blockTableShape = context->GetInputShape(kBlockTable);

    if (!HasShape(queryShape, {-1, kQueryHeads, kHeadDim})) {
        OP_LOGE(
            context->GetNodeName(),
            "query must be [Tq, 32, 128] BF16 (BSND without batch)");
        return ge::GRAPH_FAILED;
    }
    if (!HasShape(
            keyShape, {-1, kPageSize, kKvHeads, kHeadDim})) {
        OP_LOGE(
            context->GetNodeName(),
            "key_cache must be [num_pages, 128, 8, 128] BF16");
        return ge::GRAPH_FAILED;
    }
    if (!SameShape(keyShape, valueShape)) {
        OP_LOGE(
            context->GetNodeName(),
            "value_cache shape must exactly match key_cache");
        return ge::GRAPH_FAILED;
    }
    if (!HasShape(blockTableShape, {1, -1})) {
        OP_LOGE(
            context->GetNodeName(),
            "block_table must be batch-one [1, max_pages] INT32");
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputDesc(kQuery)->GetDataType() != ge::DT_BF16 ||
        context->GetInputDesc(kKeyCache)->GetDataType() != ge::DT_BF16 ||
        context->GetInputDesc(kValueCache)->GetDataType() != ge::DT_BF16 ||
        context->GetInputDesc(kBlockTable)->GetDataType() != ge::DT_INT32) {
        OP_LOGE(
            context->GetNodeName(),
            "fixed fast path requires BF16 Q/K/V and INT32 block_table");
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "attributes are required");
        return ge::GRAPH_FAILED;
    }
    const int64_t* queryStartPtr = attrs->GetInt(0);
    const int64_t* seqLenPtr = attrs->GetInt(1);
    const int64_t* promptLenPtr = attrs->GetInt(2);
    const float* scalePtr = attrs->GetFloat(3);
    const int64_t* queryTilePtr = attrs->GetInt(4);
    const int64_t* pageSizePtr = attrs->GetInt(5);
    const int64_t* sinkTokensPtr = attrs->GetInt(6);
    const int64_t* localWindowPtr = attrs->GetInt(7);
    const int64_t* denseTailPtr = attrs->GetInt(8);
    if (queryStartPtr == nullptr || seqLenPtr == nullptr ||
        promptLenPtr == nullptr || scalePtr == nullptr ||
        queryTilePtr == nullptr || pageSizePtr == nullptr ||
        sinkTokensPtr == nullptr || localWindowPtr == nullptr ||
        denseTailPtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "failed to read required attributes");
        return ge::GRAPH_FAILED;
    }

    if (*queryTilePtr != kQueryTile || *pageSizePtr != kPageSize ||
        *sinkTokensPtr != kSinkTokens ||
        *localWindowPtr != kLocalWindow || *denseTailPtr != kDenseTail) {
        OP_LOGE(
            context->GetNodeName(),
            "only q_tile=32, page_size=128, sink=8, window=512, "
            "dense_tail=128 is compiled");
        return ge::GRAPH_FAILED;
    }
    if (*queryStartPtr < 0 || *seqLenPtr <= 0 ||
        *promptLenPtr < *seqLenPtr || !std::isfinite(*scalePtr) ||
        *scalePtr <= 0.0F) {
        OP_LOGE(
            context->GetNodeName(),
            "invalid query_start/seq_len/prompt_len/scale attributes");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& query = queryShape->GetStorageShape();
    const gert::Shape& key = keyShape->GetStorageShape();
    const gert::Shape& blockTable = blockTableShape->GetStorageShape();
    const int64_t queryTokens = query.GetDim(0);
    const int64_t physicalPages = key.GetDim(0);
    const int64_t blockTableCapacity = blockTable.GetDim(1);
    const int64_t requiredPages =
        (*seqLenPtr + kPageSize - 1) / kPageSize;

    if (queryTokens <= 0 ||
        *queryStartPtr + queryTokens != *seqLenPtr) {
        OP_LOGE(
            context->GetNodeName(),
            "batch-one prefill requires query_start + Tq == seq_len");
        return ge::GRAPH_FAILED;
    }
    if (physicalPages <= 0 || blockTableCapacity < requiredPages) {
        OP_LOGE(
            context->GetNodeName(),
            "block_table lacks logical pages required by seq_len");
        return ge::GRAPH_FAILED;
    }
    if (*promptLenPtr > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(
            context->GetNodeName(),
            "first kernel ABI supports prompt lengths up to UINT32_MAX");
        return ge::GRAPH_FAILED;
    }

    auto* tiling =
        context->GetTilingData<TrianglePagedSparseAttentionTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context->GetNodeName(), "tiling buffer is unavailable");
        return ge::GRAPH_FAILED;
    }

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "platform info is unavailable");
        return ge::GRAPH_FAILED;
    }
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aicCores = platform.GetCoreNumAic();
    if (aicCores == 0) {
        OP_LOGE(context->GetNodeName(), "Ascend platform reports zero AICs");
        return ge::GRAPH_FAILED;
    }

    const uint32_t queryTileCount =
        (static_cast<uint32_t>(queryTokens) +
         static_cast<uint32_t>(kQueryTile) - 1U) /
        static_cast<uint32_t>(kQueryTile);
    const uint32_t taskCount =
        queryTileCount * static_cast<uint32_t>(kKvHeads);
    const uint32_t blockDim = std::min(aicCores, taskCount);
    const uint32_t sparseBegin =
        static_cast<uint32_t>(kSinkTokens + kLocalWindow + 1);
    const uint32_t sparseEnd = static_cast<uint32_t>(
        std::max<int64_t>(0, *promptLenPtr - kDenseTail));

    tiling->magic = kTilingMagic;
    tiling->abiVersion = kAbiVersion;
    tiling->implementationStatus = kFastImplementation;
    tiling->blockDim = blockDim;
    tiling->queryTokens = static_cast<uint32_t>(queryTokens);
    tiling->queryStart = static_cast<uint32_t>(*queryStartPtr);
    tiling->seqLen = static_cast<uint32_t>(*seqLenPtr);
    tiling->promptLen = static_cast<uint32_t>(*promptLenPtr);
    tiling->queryHeads = static_cast<uint32_t>(kQueryHeads);
    tiling->kvHeads = static_cast<uint32_t>(kKvHeads);
    tiling->headDim = static_cast<uint32_t>(kHeadDim);
    tiling->pageSize = static_cast<uint32_t>(kPageSize);
    tiling->physicalPageCount = static_cast<uint32_t>(physicalPages);
    tiling->blockTablePageCapacity =
        static_cast<uint32_t>(blockTableCapacity);
    tiling->queryTile = static_cast<uint32_t>(kQueryTile);
    tiling->taskCount = taskCount;
    tiling->sinkTokens = static_cast<uint32_t>(kSinkTokens);
    tiling->localWindow = static_cast<uint32_t>(kLocalWindow);
    tiling->denseTail = static_cast<uint32_t>(kDenseTail);
    tiling->sparseBegin = sparseBegin;
    tiling->sparseEnd = sparseEnd;
    tiling->reserved0 = 0;
    tiling->reserved1 = 0;
    tiling->reserved2 = 0;
    tiling->scale = *scalePtr;
    tiling->kvTile = static_cast<uint32_t>(kKvTile);
    tiling->groupSize = kGroupSize;
    tiling->queryTileCount = queryTileCount;
    tiling->activeAicCores = blockDim;
    tiling->workspacePerCoreBytes = kWorkspacePerCoreBytes;
    tiling->scoreOffsetBytes = kScoreOffsetBytes;
    tiling->probabilityOffsetBytes = kProbabilityOffsetBytes;
    tiling->outputTmpOffsetBytes = kOutputTmpOffsetBytes;
    tiling->outputUpdateOffsetBytes = kOutputUpdateOffsetBytes;
    tiling->lseScratchOffsetBytes = kLseScratchOffsetBytes;
    tiling->workspaceBytes = blockDim * kWorkspacePerCoreBytes;
    tiling->pipelineStages = 1;

    context->SetBlockDim(blockDim);
    // The non-templated build emits one dynamic MIX kernel with tiling key 0.
    context->SetTilingKey(0);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    const size_t libApiWorkspaceBytes =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    const size_t userWorkspaceBytes =
        static_cast<size_t>(tiling->workspaceBytes);
    if (libApiWorkspaceBytes >
        std::numeric_limits<size_t>::max() - userWorkspaceBytes) {
        OP_LOGE(
            context->GetNodeName(),
            "system and user workspace byte count overflows size_t");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = libApiWorkspaceBytes + userWorkspaceBytes;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForTrianglePagedSparseAttention(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TrianglePagedSparseAttention)
    .Tiling(TilingFunc)
    .TilingParse<TrianglePagedSparseAttentionCompileInfo>(
        TilingPrepareForTrianglePagedSparseAttention);

}  // namespace optiling
