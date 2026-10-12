/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_scatter_copy_sparse_flash_attention_schedule.h
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_SCHEDULE_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_SCHEDULE_H
#include <cstdint>
#include "fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#ifdef __CCE_AICORE__
#define FQSCSFA_SCHEDULE_INLINE __aicore__ inline
#else
#define FQSCSFA_SCHEDULE_INLINE constexpr
#endif
namespace fused_quant_scatter_copy_sfa {
constexpr int32_t UNION_COPY_THRESHOLD = static_cast<int32_t>(FUSED_MISS_CAPACITY);
constexpr uint32_t GATHER_REGION_SIZE = FUSED_MISS_CAPACITY;
constexpr uint32_t GATHER_REGION_COUNT = TOPK_CAPACITY / GATHER_REGION_SIZE;
constexpr uint32_t GATHER_TILES_PER_REGION = GATHER_REGION_SIZE / CACHE_BLOCK_SIZE;
constexpr uint32_t GATHER_BATCH_PERIOD = 3;
constexpr uint32_t SOURCE_GATHER_ROWS = 16;
constexpr uint32_t SOURCE_GATHER_BUFFER_COUNT = 2;
constexpr uint32_t MISS_PREFETCH_ROWS = 4;
constexpr uint32_t BITMAP_WORD_BITS = 32;
constexpr uint32_t OWNER_BITMAP_WORD_COUNT =
    (COPY_CAPACITY + BITMAP_WORD_BITS - 1) / BITMAP_WORD_BITS;
constexpr uint32_t OWNER_SLOT_CAPACITY = OWNER_BITMAP_WORD_COUNT * BITMAP_WORD_BITS;
constexpr uint32_t NORMAL_AIV_SLICE_WIDTH = 64;
constexpr uint32_t SPLIT_HEAD_AIV_SLICE_WIDTH = 32;
constexpr uint32_t NORMAL_AIV_PARTITION_COUNT = CACHE_BLOCK_SIZE / NORMAL_AIV_SLICE_WIDTH;
constexpr uint32_t SPLIT_HEAD_AIV_PARTITION_COUNT = CACHE_BLOCK_SIZE / SPLIT_HEAD_AIV_SLICE_WIDTH;
constexpr uint16_t MISS_WRITEBACK_PAIR_ROWS = 2;

static_assert(TOPK_CAPACITY % GATHER_REGION_SIZE == 0,
    "TopK rows must contain complete gather regions");
static_assert(GATHER_REGION_SIZE % CACHE_BLOCK_SIZE == 0,
    "gather regions must contain complete S2 tiles");
static_assert(GATHER_BATCH_PERIOD <= GATHER_REGION_COUNT,
    "batch rotation must select a valid gather region");
static_assert(NORMAL_AIV_SLICE_WIDTH * NORMAL_AIV_PARTITION_COUNT == CACHE_BLOCK_SIZE,
    "normal head scheduling must split an S2 tile across two AIVs");
static_assert(SPLIT_HEAD_AIV_SLICE_WIDTH * SPLIT_HEAD_AIV_PARTITION_COUNT == CACHE_BLOCK_SIZE,
    "split-head scheduling must split an S2 tile across four AIVs");
static_assert(SOURCE_GATHER_BUFFER_COUNT == 2,
    "source gather uses xor to rotate two writeback buffers");
static_assert(OWNER_SLOT_CAPACITY >= COPY_CAPACITY,
    "owner bitmap must cover every request-level miss destination slot");
static_assert(OWNER_SLOT_CAPACITY - COPY_CAPACITY < BITMAP_WORD_BITS,
    "owner bitmap must not contain an unnecessary full word");
// Request-level union count, not per-query misses or the sum over the batch.
FQSCSFA_SCHEDULE_INLINE bool NeedsUnionCopy(int32_t unionCount)
{
    return unionCount > UNION_COPY_THRESHOLD;
}
struct MissWritebackPlan {
    uint16_t blockCount;
    uint32_t dstStride;
};
// Adjacent UB rows use the aligned row size, while GM stores only the packed payload.
// UB->GM DataCopyPad rounds its UB block step up to one data block (srcStride=0),
// while dstStride is the byte gap after the packed payload. Never reorder the UB rows.
FQSCSFA_SCHEDULE_INLINE MissWritebackPlan PlanMissWritebackPair(int64_t destination, int64_t nextDestination)
{
    if (destination < 0 || nextDestination <= destination) return {1, 0};
    const uint64_t distance = static_cast<uint64_t>(nextDestination) - static_cast<uint64_t>(destination);
    // Keep a uint32-representable gap for both existing and newer CANN headers.
    // Larger physical addresses themselves remain int64; only the gap is bounded.
    if (distance < PACKED_KV_ROW_BYTES || distance - PACKED_KV_ROW_BYTES > UINT32_MAX) return {1, 0};
    return {MISS_WRITEBACK_PAIR_ROWS, static_cast<uint32_t>(distance - PACKED_KV_ROW_BYTES)};
}
FQSCSFA_SCHEDULE_INLINE uint32_t GatherIndex(uint32_t batch, uint32_t index)
{
    if (index >= TOPK_CAPACITY) return index;
    const uint32_t region = index / GATHER_REGION_SIZE;
    const uint32_t regionOffset = index % GATHER_REGION_SIZE;
    const uint32_t rotatedRegion = (GATHER_REGION_COUNT - 1 + GATHER_REGION_COUNT -
        batch % GATHER_BATCH_PERIOD - region) % GATHER_REGION_COUNT;
    return rotatedRegion * GATHER_REGION_SIZE + regionOffset;
}
struct GatherBlockPlan {
    uint32_t indexBase;
    bool hasMiss;
};
struct OwnerScanPlan {
    uint32_t begin;
    bool reset;
};
// Reuse earlier-query ownership only within the same, forward-moving request.
FQSCSFA_SCHEDULE_INLINE OwnerScanPlan PlanOwnerScan(int32_t previousBatch, int32_t previousQuery,
    uint32_t batch, uint32_t queryBegin, uint32_t query)
{
    const bool reset = previousBatch != static_cast<int32_t>(batch) ||
        previousQuery < static_cast<int32_t>(queryBegin) || previousQuery > static_cast<int32_t>(query);
    return {reset ? queryBegin : static_cast<uint32_t>(previousQuery), reset};
}
// One S2 tile never crosses a gather-region permutation boundary.
// Its AIV slice [start,start+count) is contiguous after mapping as well.
FQSCSFA_SCHEDULE_INLINE GatherBlockPlan PlanGatherBlock(uint32_t batch, uint32_t tileBase,
    uint32_t start, uint32_t count, uint32_t misses)
{
    const uint32_t base = misses == 0 ? tileBase : GatherIndex(batch, tileBase);
    return {base, count != 0 && base + start < misses};
}
FQSCSFA_SCHEDULE_INLINE bool OwnsMissIndex(uint32_t index, uint32_t aiv, bool splitHeads)
{
    const uint32_t width = splitHeads ? SPLIT_HEAD_AIV_SLICE_WIDTH : NORMAL_AIV_SLICE_WIDTH;
    const uint32_t partitions = splitHeads ? SPLIT_HEAD_AIV_PARTITION_COUNT : NORMAL_AIV_PARTITION_COUNT;
    return (index % CACHE_BLOCK_SIZE) / width == aiv % partitions;
}
} // namespace fused_quant_scatter_copy_sfa
#undef FQSCSFA_SCHEDULE_INLINE
#endif
