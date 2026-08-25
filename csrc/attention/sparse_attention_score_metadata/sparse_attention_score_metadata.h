/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H

#include <cstdint>

namespace optiling {
namespace generic_block_sparse_attention_metadata {

using MetadataType = int32_t;

constexpr uint32_t METADATA_TOTAL_SIZE = 1024U;
constexpr uint32_t MAX_AIC_CORE_NUM = 36U;
constexpr uint32_t MAX_DECODE_CORE_NUM = 32U;
constexpr uint32_t MAX_COMBINE_TASK_NUM = 32U;
// FD scheduling capacity is independent from sparseBlockIdx.shape[-1].
constexpr uint32_t MAX_FD_ACTIVE_CORE_NUM = MAX_DECODE_CORE_NUM;
constexpr uint32_t MAX_FD_PARTIAL_PER_BASE_TASK = MAX_DECODE_CORE_NUM;
constexpr uint32_t MAX_SPARSE_BLOCK_CAPACITY = 16U;
constexpr MetadataType METADATA_MAGIC = 0x5341534D;
constexpr MetadataType METADATA_VERSION = 8;

enum FdScheduleFlag : uint32_t {
    FD_SCHEDULE_ENABLED = 1U << 0,
    FD_ACTUAL_BLOCK_PREFIX = 1U << 1,
};

enum HeaderIndex : uint32_t {
    MAGIC_INDEX = 0U,
    VERSION_INDEX = 1U,
    METADATA_USED_SIZE_INDEX = 2U,
    SA_USED_CORE_NUM_INDEX = 3U,
    SA_TOTAL_TASK_NUM_INDEX = 4U,
    FD_ACTIVE_CORE_NUM_INDEX = 5U,
    DECODE_PER_CORE_TASK_NUM_INDEX = 6U,
    COMBINE_TASK_NUM_INDEX = 7U,
    FD_SCHEDULE_FLAGS_INDEX = 8U,
    FD_TOTAL_FLAT_TASK_NUM_INDEX = 9U,
    FD_PARTIAL_TASK_NUM_INDEX = 10U,
    CONFIG_SIGNATURE_INDEX = 11U,
    HEADER_SIZE = 16U,
};

enum DecodeScheduleIndex : uint32_t {
    DECODE_BASE_TASK_START_INDEX = 0U,
    DECODE_BASE_TASK_END_INDEX = 1U,
    DECODE_FIRST_BLOCK_START_INDEX = 2U,
    DECODE_LAST_BLOCK_END_INDEX = 3U,
    DECODE_SCHEDULE_FIELD_NUM = 4U,
};

enum CombineScheduleIndex : uint32_t {
    COMBINE_BASE_TASK_INDEX = 0U,
    COMBINE_FIRST_CORE_INDEX = 1U,
    COMBINE_PARTIAL_START_INDEX = 2U,
    COMBINE_PARTIAL_COUNT_INDEX = 3U,
    COMBINE_SCHEDULE_FIELD_NUM = 4U,
};

constexpr uint32_t DECODE_SCHEDULE_OFFSET = HEADER_SIZE;
constexpr uint32_t COMBINE_SCHEDULE_OFFSET = DECODE_SCHEDULE_OFFSET + MAX_DECODE_CORE_NUM * DECODE_SCHEDULE_FIELD_NUM;
constexpr uint32_t METADATA_USED_SIZE = COMBINE_SCHEDULE_OFFSET + MAX_COMBINE_TASK_NUM * COMBINE_SCHEDULE_FIELD_NUM;

inline constexpr uint32_t GetDecodeScheduleIndex(uint32_t coreIdx, uint32_t fieldIdx)
{
    return DECODE_SCHEDULE_OFFSET + coreIdx * DECODE_SCHEDULE_FIELD_NUM + fieldIdx;
}

inline constexpr uint32_t GetCombineScheduleIndex(uint32_t combineTaskIdx, uint32_t fieldIdx)
{
    return COMBINE_SCHEDULE_OFFSET + combineTaskIdx * COMBINE_SCHEDULE_FIELD_NUM + fieldIdx;
}

static_assert(METADATA_USED_SIZE == 272U, "unexpected metadata used size");
static_assert(METADATA_USED_SIZE <= METADATA_TOTAL_SIZE, "metadata schedule exceeds output buffer");

} // namespace generic_block_sparse_attention_metadata
} // namespace optiling

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
