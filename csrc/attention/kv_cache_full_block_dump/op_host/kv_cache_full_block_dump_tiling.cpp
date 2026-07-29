/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <cstdint>
#include <limits>

#include "kv_cache_full_block_dump_tiling.h"
#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int32_t SRC_CACHE_0_INPUT = 0;
constexpr int32_t SRC_CACHE_1_INPUT = 1;
constexpr int32_t DST_CACHE_0_INPUT = 2;
constexpr int32_t DST_CACHE_1_INPUT = 3;
constexpr int32_t SRC_IDS_INPUT = 4;
constexpr int32_t DST_IDS_INPUT = 5;
constexpr int64_t UB_ALIGNMENT = 32;
// Keep each ordinary copy at or above the efficient DMA range while exposing
// enough independent work for the common decode case with only one or two
// newly completed blocks. For the DSV3.2 BF16 MLA layout this produces four
// 32-KiB NoPE tasks and one 16-KiB RoPE task per full block.
constexpr int64_t TARGET_COPY_BUFFER_BYTES = 32 * 1024;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

inline int64_t AlignDown(int64_t value, int64_t alignment)
{
    return value / alignment * alignment;
}

bool MultiplyFitsInt64(int64_t lhs, int64_t rhs)
{
    return lhs >= 0 && rhs >= 0 &&
        (lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs);
}

ge::graphStatus GetShape(gert::TilingContext* context, int32_t index,
                         const char* name, gert::Shape& shape)
{
    auto inputShape = context->GetInputShape(index);
    OPS_ERR_IF(inputShape == nullptr,
        OPS_LOG_E(context->GetNodeName(), "get %s shape failed.", name),
        return ge::GRAPH_FAILED);
    shape = inputShape->GetStorageShape();
    return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus TilingKvCacheFullBlockDump(
    gert::TilingContext* context)
{
    gert::Shape srcCache0Shape;
    gert::Shape srcCache1Shape;
    gert::Shape dstCache0Shape;
    gert::Shape dstCache1Shape;
    gert::Shape srcIdsShape;
    gert::Shape dstIdsShape;
    if (GetShape(context, SRC_CACHE_0_INPUT, "src_cache_0", srcCache0Shape) != ge::GRAPH_SUCCESS ||
        GetShape(context, SRC_CACHE_1_INPUT, "src_cache_1", srcCache1Shape) != ge::GRAPH_SUCCESS ||
        GetShape(context, DST_CACHE_0_INPUT, "dst_cache_0", dstCache0Shape) != ge::GRAPH_SUCCESS ||
        GetShape(context, DST_CACHE_1_INPUT, "dst_cache_1", dstCache1Shape) != ge::GRAPH_SUCCESS ||
        GetShape(context, SRC_IDS_INPUT, "src_block_ids", srcIdsShape) != ge::GRAPH_SUCCESS ||
        GetShape(context, DST_IDS_INPUT, "dst_block_ids", dstIdsShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OPS_ERR_IF(srcCache0Shape.GetDimNum() != 3 || srcCache1Shape.GetDimNum() != 3 ||
                   dstCache0Shape.GetDimNum() != 3 || dstCache1Shape.GetDimNum() != 3,
        OPS_LOG_E(context->GetNodeName(),
                  "all KV cache dump payload tensors must be rank 3."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(srcIdsShape.GetDimNum() != 1 || dstIdsShape.GetDimNum() != 1,
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump block-id tensors must be rank 1."),
        return ge::GRAPH_FAILED);

    const int64_t rowCount = srcIdsShape.GetDim(0);
    const int64_t blockSize = srcCache0Shape.GetDim(1);
    const int64_t plane0Dim = srcCache0Shape.GetDim(2);
    const int64_t plane1Dim = srcCache1Shape.GetDim(2);
    OPS_ERR_IF(rowCount < 0,
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump row count must be non-negative."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(rowCount != dstIdsShape.GetDim(0),
        OPS_LOG_E(context->GetNodeName(),
                  "source and destination block-id rows must match."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockSize <= 0 || plane0Dim <= 0 || plane1Dim < 0,
        OPS_LOG_E(context->GetNodeName(),
                  "invalid block payload shape: block=%ld plane0=%ld plane1=%ld.",
                  blockSize, plane0Dim, plane1Dim),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(srcCache1Shape.GetDim(1) != blockSize ||
                   dstCache0Shape.GetDim(1) != blockSize ||
                   dstCache1Shape.GetDim(1) != blockSize,
        OPS_LOG_E(context->GetNodeName(),
                  "source and destination cache block sizes must match."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(dstCache0Shape.GetDim(2) != plane0Dim ||
                   dstCache1Shape.GetDim(2) != plane1Dim,
        OPS_LOG_E(context->GetNodeName(),
                  "source and destination payload dimensions must match."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(srcCache0Shape.GetDim(0) != srcCache1Shape.GetDim(0) ||
                   dstCache0Shape.GetDim(0) != dstCache1Shape.GetDim(0),
        OPS_LOG_E(context->GetNodeName(),
                  "cache-plane block counts must match within each tier."),
        return ge::GRAPH_FAILED);

    auto srcCache0Desc = context->GetInputDesc(SRC_CACHE_0_INPUT);
    auto srcCache1Desc = context->GetInputDesc(SRC_CACHE_1_INPUT);
    auto dstCache0Desc = context->GetInputDesc(DST_CACHE_0_INPUT);
    auto dstCache1Desc = context->GetInputDesc(DST_CACHE_1_INPUT);
    auto srcDesc = context->GetInputDesc(SRC_IDS_INPUT);
    auto dstDesc = context->GetInputDesc(DST_IDS_INPUT);
    OPS_ERR_IF(srcCache0Desc == nullptr || srcCache1Desc == nullptr ||
                   dstCache0Desc == nullptr || dstCache1Desc == nullptr ||
                   srcDesc == nullptr || dstDesc == nullptr,
        OPS_LOG_E(context->GetNodeName(),
                  "get KV cache dump input desc failed."),
        return ge::GRAPH_FAILED);
    const ge::DataType plane0Dtype = srcCache0Desc->GetDataType();
    const ge::DataType plane1Dtype = srcCache1Desc->GetDataType();
    OPS_ERR_IF(dstCache0Desc->GetDataType() != plane0Dtype ||
                   dstCache1Desc->GetDataType() != plane1Dtype ||
                   plane0Dtype != plane1Dtype,
        OPS_LOG_E(context->GetNodeName(),
                  "all KV cache dump payload tensors must have the same dtype."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(srcDesc->GetDataType() != ge::DT_INT32 ||
                   dstDesc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump block ids must be int32."),
        return ge::GRAPH_FAILED);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t vectorCoreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubBytesRaw = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytesRaw);
    const int64_t ubBytes = static_cast<int64_t>(ubBytesRaw);
    OPS_ERR_IF(vectorCoreNum <= 0 || ubBytes <= 0,
        OPS_LOG_E(context->GetNodeName(),
                  "invalid AIV core count or UB size."),
        return ge::GRAPH_FAILED);

    const int64_t plane0DtypeBytes = ge::GetSizeByDataType(plane0Dtype);
    const int64_t plane1DtypeBytes = ge::GetSizeByDataType(plane1Dtype);
    const int64_t copyBudget = AlignDown(
        std::min(TARGET_COPY_BUFFER_BYTES, ubBytes / 2), UB_ALIGNMENT);
    OPS_ERR_IF(plane0DtypeBytes <= 0 || plane1DtypeBytes <= 0 ||
                   copyBudget < UB_ALIGNMENT,
        OPS_LOG_E(context->GetNodeName(),
                  "invalid KV cache dump dtype size or UB copy budget."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(!MultiplyFitsInt64(plane0Dim, plane0DtypeBytes) ||
                   !MultiplyFitsInt64(plane1Dim, plane1DtypeBytes) ||
                   !MultiplyFitsInt64(blockSize, plane0Dim) ||
                   !MultiplyFitsInt64(blockSize, plane1Dim),
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump payload size overflows int64."),
        return ge::GRAPH_FAILED);

    const int64_t plane0ElementsPerBlock = blockSize * plane0Dim;
    const int64_t plane1ElementsPerBlock = blockSize * plane1Dim;
    const int64_t srcBlockNum = srcCache0Shape.GetDim(0);
    const int64_t dstBlockNum = dstCache0Shape.GetDim(0);
    OPS_ERR_IF(srcBlockNum <= 0 || dstBlockNum <= 0 ||
                   !MultiplyFitsInt64(srcBlockNum,
                                      plane0ElementsPerBlock) ||
                   !MultiplyFitsInt64(srcBlockNum,
                                      plane1ElementsPerBlock) ||
                   !MultiplyFitsInt64(dstBlockNum,
                                      plane0ElementsPerBlock) ||
                   !MultiplyFitsInt64(dstBlockNum,
                                      plane1ElementsPerBlock),
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump block capacity or address range is invalid."),
        return ge::GRAPH_FAILED);

    // Linearize each plane independently. Combining a wide NoPE slice and a
    // narrow RoPE slice in every task leaves a single decode block with too
    // little parallel work and repeatedly issues under-filled RoPE copies.
    // Independent contiguous chunks give the two planes balanced DMA tasks
    // without changing their physical block layout.
    const int64_t plane0ChunkElements = std::min(
        plane0ElementsPerBlock, copyBudget / plane0DtypeBytes);
    const int64_t plane1ChunkElements = plane1ElementsPerBlock == 0
        ? 0
        : std::min(plane1ElementsPerBlock,
                   copyBudget / plane1DtypeBytes);
    OPS_ERR_IF(plane0ChunkElements <= 0 ||
                   (plane1ElementsPerBlock > 0 &&
                    plane1ChunkElements <= 0),
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump copy budget cannot hold one payload element."),
        return ge::GRAPH_FAILED);
    const int64_t plane0TasksPerRow = CeilDiv(
        plane0ElementsPerBlock, plane0ChunkElements);
    const int64_t plane1TasksPerRow = plane1ElementsPerBlock == 0
        ? 0
        : CeilDiv(plane1ElementsPerBlock, plane1ChunkElements);
    const int64_t tasksPerRow = plane0TasksPerRow + plane1TasksPerRow;
    OPS_ERR_IF(tasksPerRow <= 0 ||
                   !MultiplyFitsInt64(rowCount, tasksPerRow),
        OPS_LOG_E(context->GetNodeName(),
                  "KV cache dump total task count is invalid."),
        return ge::GRAPH_FAILED);
    const int64_t taskCount = rowCount * tasksPerRow;

    // Linearize [row, plane, contiguous_chunk] into copy tasks. A decode block
    // can use several AIV cores, while all blocks produced by a prefill
    // naturally share one balanced task pool.
    const int64_t usedCoreNum = taskCount == 0
        ? 0
        : std::min(taskCount, vectorCoreNum);
    const int64_t maxChunkBytes = std::max(
        plane0ChunkElements * plane0DtypeBytes,
        plane1ChunkElements * plane1DtypeBytes);
    const int64_t bufferBytes = AlignUp(maxChunkBytes, UB_ALIGNMENT);

    KvCacheFullBlockDumpTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_taskCount(taskCount);
    tiling.set_srcBlockNum(srcBlockNum);
    tiling.set_dstBlockNum(dstBlockNum);
    tiling.set_plane0ElementsPerBlock(plane0ElementsPerBlock);
    tiling.set_plane1ElementsPerBlock(plane1ElementsPerBlock);
    tiling.set_plane0ChunkElements(plane0ChunkElements);
    tiling.set_plane1ChunkElements(plane1ChunkElements);
    tiling.set_plane0TasksPerRow(plane0TasksPerRow);
    tiling.set_tasksPerRow(tasksPerRow);
    tiling.set_bufferBytes(bufferBytes);

    context->SetTilingKey(0);
    context->SetBlockDim(usedCoreNum == 0 ? 1 : usedCoreNum);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
        OPS_LOG_E(context->GetNodeName(), "get workspace failed."),
        return ge::GRAPH_FAILED);
    workspaces[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct KvCacheFullBlockDumpCompileInfo {};

static ge::graphStatus TilingPrepareKvCacheFullBlockDump(
    gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheFullBlockDump)
    .Tiling(TilingKvCacheFullBlockDump)
    .TilingParse<KvCacheFullBlockDumpCompileInfo>(
        TilingPrepareKvCacheFullBlockDump);
}  // namespace optiling
