/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file kv_quant_sparse_attn_sharedkv_metadata.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t MAX_AIV_AIC_RATIO = 2;
constexpr uint32_t SCFA_META_SIZE = 2048;
using SCFA_METADATA_T = int32_t;

constexpr uint32_t CORE_METADATA_SIZE = 32;
constexpr uint32_t FA_METADATA_SIZE = 16;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_FD_VECTOR_NUM_INDEX = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FD_M_IDX_INDEX = 1;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FD_M_START_INDEX = 4;
constexpr uint32_t FD_M_NUM_INDEX = 5;

/**
 * @brief  获取属性的绝对索引
 * @details 此函数用于计算属性的绝对索引，输入参数包括属性类索引、属性实例索引和元数据索引。
 *
 * @param  aicIdx    aic序号: 0~AIC_CORE_NUM
 * @param  aivIdx    aiv序号: 根据C:V比例，0~1 or 1~2
 * @param  metaIdx   metadata中对应变量的INDEX
 *
 * @return 返回计算得到的属性绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t aicIdx, uint32_t metaIdx, bool isFDMeta=false, uint32_t aivIdx=0)
{
    uint32_t baseIndex = CORE_METADATA_SIZE * aicIdx + FD_METADATA_SIZE * aivIdx + metaIdx;
    return isFDMeta ? baseIndex + FA_METADATA_SIZE : baseIndex;
}
#endif

namespace detail {
    struct CoreMetadata{
        uint32_t faMetadata[FA_METADATA_SIZE];
        uint32_t fdMetadata0[FD_METADATA_SIZE];
        uint32_t fdMetadata1[FD_METADATA_SIZE];
    };

    struct SasMetaData {
        struct CoreMetadata coreMetadata[AIC_CORE_NUM];
    };
};

static_assert(SCFA_META_SIZE * sizeof(SCFA_METADATA_T) >= sizeof(detail::SasMetaData),
                "SCFA_META_SIZE is not large enough to hold SasMetaData");
};

#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H
