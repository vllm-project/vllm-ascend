/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_metadata.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_H
#define FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t FIASINK_META_SIZE = 1024;
constexpr uint32_t FIA_MAX_AIC_CORE_NUM = 26;
using FIASINK_METADATA_T = int32_t;

constexpr uint32_t AIC_METADATA_SIZE = 10;
constexpr uint32_t AIV_METADATA_SIZE = 3;
constexpr uint32_t BASE_METADATA_SIZE = 10;

// AIC Metadata Index Definitions
constexpr uint32_t AIC_CORE_ENABLE_INDEX = 0;
constexpr uint32_t BN2_END_PTR_INDEX = 1;
constexpr uint32_t GS1_END_PTR_INDEX = 2;
constexpr uint32_t S2_END_PTR = 3;
constexpr uint32_t BN2_IDX_OF_FD_HEAD_INDEX = 4;
constexpr uint32_t GS1_IDX_OF_FD_HEAD_INDEX = 5;
constexpr uint32_t S2_SPLIT_NUM_OF_FD_HEAD_INDEX = 6;
constexpr uint32_t S2_SPLIT_START_IDX_OF_CORE_INDEX = 7;
constexpr uint32_t GS1_SPLIT_NUM_OF_FD_HEAD_INDEX = 8;
constexpr uint32_t GS1_LAST_PART_SIZE_OF_FD_HEAD_INDEX = 9;

// AIV Metadata Index Definitions
constexpr uint32_t AIV_CORE_ENABLE_INDEX = 0;
constexpr uint32_t GS1_IDX_END_OF_FD_HEAD_INDEX = 1;
constexpr uint32_t GS1_IDX_END_OF_FD_HEAD_SPLIT_INDEX = 2;

// BASE Metadata Index Definitions
constexpr uint32_t M_BASE_SIZE_INDEX = 0;
constexpr uint32_t S_INNER_SIZE_INDEX = 1;
constexpr uint32_t M_FD_BASE_SIZE_INDEX = 2;
constexpr uint32_t NUM_OF_FD_INDEX = 3;
constexpr uint32_t USED_CORE_NUM_INDEX = 4;
constexpr uint32_t USED_VEC_NUM_OF_FD_INDEX = 5;
constexpr uint32_t S1_SIZE_INDEX = 6;
constexpr uint32_t S2_SIZE_INDEX = 7;
constexpr uint32_t ACUTAL_LEN_Q_DIM_INDEX = 8;
constexpr uint32_t ACUTAL_LEN_KV_DIM_INDEX = 9;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAICMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    return coreIdx * AIC_METADATA_SIZE + metaIdx;
}

__aicore__ inline uint32_t GetAIVMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    const uint32_t aicTotalSize = AIC_CORE_NUM * AIC_METADATA_SIZE;
    return aicTotalSize + coreIdx * AIV_METADATA_SIZE + metaIdx;
}

__aicore__ inline uint32_t GetBaseMetaAbsIndex(uint32_t metaIdx)
{
    const uint32_t aicTotalSize = AIC_CORE_NUM * AIC_METADATA_SIZE;
    const uint32_t aivTotalSize = AIV_CORE_NUM * AIV_METADATA_SIZE;
    return aicTotalSize + aivTotalSize + metaIdx;
}
#endif

namespace detail {
struct FiaSinkMetaData {
    uint32_t aicMetadata[AIC_CORE_NUM][AIC_METADATA_SIZE];
    uint32_t aivMetadata[AIV_CORE_NUM][AIV_METADATA_SIZE];
    uint32_t baseMetadata[BASE_METADATA_SIZE];
};
} // namespace detail

static_assert(FIASINK_META_SIZE * sizeof(FIASINK_METADATA_T) >= sizeof(detail::FiaSinkMetaData));
} // namespace optiling

#endif