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
 * \file fused_quant_scatter_copy_sparse_flash_attention_contract.h
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_CONTRACT_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_CONTRACT_H

#include <cstdint>

#ifdef __CCE_AICORE__
#define FQSCSFA_CONTRACT_INLINE __aicore__ inline
#else
#define FQSCSFA_CONTRACT_INLINE constexpr
#endif

namespace fused_quant_scatter_copy_sfa {
constexpr uint32_t TOPK_CAPACITY = 2048;
constexpr uint32_t COPY_CAPACITY = 32768;
constexpr uint32_t MAX_QUERY_COUNT = 16;
constexpr uint32_t CACHE_BLOCK_SIZE = 128;
constexpr uint32_t QUERY_TILE_SIZE = 64;
constexpr uint32_t KV_HEAD_COUNT = 1;
constexpr uint32_t PACKED_KV_ROW_BYTES = 656;
constexpr uint32_t ALIGNED_KV_ROW_BYTES = 672;
constexpr uint32_t KV_ROW_PADDING_BYTES = ALIGNED_KV_ROW_BYTES - PACKED_KV_ROW_BYTES;
constexpr uint32_t QUERY_DIM = 576;
constexpr uint32_t ATTENTION_OUTPUT_DIM = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t QUERY_ELEMENT_BYTES = 2;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t C8_QUANT_GROUP_SIZE = 128;
constexpr uint32_t SCALE_ELEMENT_BYTES = 4;
constexpr uint32_t FUSED_MISS_CAPACITY = 512;
constexpr uint32_t ATTENTION_PIPELINE_STAGE_COUNT = 3;
constexpr uint32_t MIN_QUERY_HEAD_COUNT = 8;
constexpr uint32_t MAX_QUERY_HEAD_COUNT = 128;
constexpr uint32_t SPLIT_HEAD_PARTITION_SIZE = MAX_QUERY_HEAD_COUNT / 2;
constexpr uint32_t MAX_BLOCKS_PER_REQUEST = 2048;

static_assert(QUERY_DIM == ATTENTION_OUTPUT_DIM + ROPE_DIM,
    "query dimension must contain the latent and RoPE dimensions");
static_assert(PACKED_KV_ROW_BYTES == ATTENTION_OUTPUT_DIM + ROPE_DIM * QUERY_ELEMENT_BYTES +
    ATTENTION_OUTPUT_DIM / C8_QUANT_GROUP_SIZE * SCALE_ELEMENT_BYTES,
    "packed C8 rows must contain latent, RoPE and scale payloads");
static_assert(ALIGNED_KV_ROW_BYTES >= PACKED_KV_ROW_BYTES,
    "aligned KV rows must contain the packed C8 payload");
static_assert(ALIGNED_KV_ROW_BYTES % DATA_BLOCK_BYTES == 0,
    "aligned KV rows must contain complete data blocks");
static_assert(FUSED_MISS_CAPACITY <= TOPK_CAPACITY,
    "fused misses must fit one TopK row");

FQSCSFA_CONTRACT_INLINE uint64_t AttentionWorkspaceBytes(uint32_t usedCoreCount)
{
    return static_cast<uint64_t>(usedCoreCount) * ATTENTION_PIPELINE_STAGE_COUNT *
        CACHE_BLOCK_SIZE * QUERY_DIM * QUERY_ELEMENT_BYTES;
}

FQSCSFA_CONTRACT_INLINE uint64_t MissPrefetchRowBytes()
{
    return static_cast<uint64_t>(FUSED_MISS_CAPACITY) * ALIGNED_KV_ROW_BYTES;
}

FQSCSFA_CONTRACT_INLINE uint64_t MissPrefetchRowOffset(uint32_t queryIndex)
{
    return static_cast<uint64_t>(queryIndex) * MissPrefetchRowBytes();
}

FQSCSFA_CONTRACT_INLINE uint64_t UserWorkspaceBytes(uint32_t usedCoreCount, uint32_t queryCount)
{
    return AttentionWorkspaceBytes(usedCoreCount) +
        static_cast<uint64_t>(queryCount) * MissPrefetchRowBytes();
}

FQSCSFA_CONTRACT_INLINE bool IsSupportedHeadCount(uint32_t heads)
{
    return heads >= MIN_QUERY_HEAD_COUNT && heads <= MAX_QUERY_HEAD_COUNT &&
        (heads & (heads - 1)) == 0;
}

FQSCSFA_CONTRACT_INLINE bool IsValidQuerySegment(
    int32_t previous, int32_t end, int32_t total)
{
    return end > previous && end <= total &&
        end - previous <= static_cast<int32_t>(MAX_QUERY_COUNT);
}

// A zero cache-token count is the non-offload sentinel. Offloaded requests
// expose a dense HBM logical address space [0, numCacheTokens), while
// non-offloaded requests retain their full logical KV length.
FQSCSFA_CONTRACT_INLINE int32_t AttentionKvLength(
    int32_t actualSeqLengthKv, int32_t numCacheTokens)
{
    return numCacheTokens == 0 ? actualSeqLengthKv : numCacheTokens;
}
} // namespace fused_quant_scatter_copy_sfa

#undef FQSCSFA_CONTRACT_INLINE

#endif
