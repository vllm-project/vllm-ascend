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
* \file quant_lightning_indexer_metadata.h
* \brief
*/

#ifndef QUANT_LIGHTNING_INDEXER_METADATA_H
#define QUANT_LIGHTNING_INDEXER_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
inline constexpr uint32_t AIC_CORE_NUM = 36;
inline constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t QLI_META_SIZE = 1024;
using QLI_METADATA_T = int32_t;

inline constexpr uint32_t LI_METADATA_SIZE = 8;
inline constexpr uint32_t LD_METADATA_SIZE = 8;

// LI Metadata Index Definitions
inline constexpr uint32_t LI_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t LI_BN2_START_INDEX = 1;
inline constexpr uint32_t LI_M_START_INDEX = 2;
inline constexpr uint32_t LI_S2_START_INDEX = 3;
inline constexpr uint32_t LI_BN2_END_INDEX = 4;
inline constexpr uint32_t LI_M_END_INDEX = 5;
inline constexpr uint32_t LI_S2_END_INDEX = 6;
inline constexpr uint32_t LI_FIRST_LD_DATA_WORKSPACE_IDX_INDEX = 7;

// LD Metadata Index Definitions
inline constexpr uint32_t LD_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t LD_BN2_IDX_INDEX = 1;
inline constexpr uint32_t LD_M_IDX_INDEX = 2;
inline constexpr uint32_t LD_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t LD_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t LD_M_START_INDEX = 5;
inline constexpr uint32_t LD_M_NUM_INDEX = 6;

 /**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV=false)
{
    if (isAIV) {
        return LI_METADATA_SIZE * AIC_CORE_NUM + LD_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return LI_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
    struct QliMetaData {
        uint32_t LIMetadata[AIC_CORE_NUM][LI_METADATA_SIZE];
        uint32_t LDMetadata[AIV_CORE_NUM][LD_METADATA_SIZE];
    };
};

static_assert(QLI_META_SIZE * sizeof(QLI_METADATA_T) >= sizeof(detail::QliMetaData));
};

#endif