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
* \file quant_lightning_indexer_v2_metadata.h
* \brief
*/

#ifndef QUANT_LIGHTNING_INDEXER_V2_METADATA_H
#define QUANT_LIGHTNING_INDEXER_V2_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
inline constexpr uint32_t AIC_CORE_MAX_NUM = 36;
inline constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t QLI_V2_METADATA_TOTAL_SIZE = 1024;
using QLI_V2_METADATA_T = int32_t;

inline constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
inline constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// LI Metadata Index Definitions
inline constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
inline constexpr uint32_t QLI_V2_M_START_INDEX = 2;
inline constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
inline constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
inline constexpr uint32_t QLI_V2_M_END_INDEX = 5;
inline constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
inline constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// LD Metadata Index Definitions
inline constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
inline constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
inline constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t QLD_V2_M_START_INDEX = 5;
inline constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

 /**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return QLI_V2_METADATA_SIZE * AIC_CORE_MAX_NUM + QLD_V2_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return QLI_V2_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
    struct QliV2Metadata {
        uint32_t qliV2Metadata[AIC_CORE_MAX_NUM][QLI_V2_METADATA_SIZE];
        uint32_t qldV2Metadata[AIV_CORE_MAX_NUM][QLD_V2_METADATA_SIZE];
    };
};

static_assert(QLI_V2_METADATA_TOTAL_SIZE * sizeof(QLI_V2_METADATA_T) >= sizeof(detail::QliV2Metadata));
};

#endif  // QUANT_LIGHTNING_INDEXER_V2_METADATA_H
