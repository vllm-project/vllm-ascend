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
 * \file flash_attn_metadata.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_H
#define FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
using FA_METADATA_T = uint32_t;
constexpr uint32_t METADATA_STRIDE = 16U;

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_AIC_NUM_INDEX = 4U;
constexpr uint32_t HEAD_AIV_NUM_INDEX = 5U;
constexpr uint32_t HEAD_OUTPUT_LAYOUT_INDEX = 6U;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

// FAG (Flash Attn Grad) Metadata Index Definitions — head[7] records FAG start offset
// (head[4]/[5]/[6] are taken by HEAD_AIC_NUM_INDEX/HEAD_AIV_NUM_INDEX/HEAD_OUTPUT_LAYOUT_INDEX)
constexpr uint32_t HEAD_FAG_START_OFFSET_INDEX = 7U;
constexpr uint32_t FAG_METADATA_SIZE = 121U;
constexpr uint32_t FAG_CORE_LIST_NUM = 36U;

// FAG Metadata field indices (relative to fagStartOffset)
constexpr uint32_t FAG_SPLIT_AXIS_INDEX = 0U;
constexpr uint32_t FAG_BLOCK_OUTER_INDEX = 1U;
constexpr uint32_t FAG_BLOCK_FACTOR_INDEX = 2U;
constexpr uint32_t FAG_S1_OUTER_INDEX = 3U;
constexpr uint32_t FAG_S2_OUTER_INDEX = 4U;
constexpr uint32_t FAG_LAYOUT_TYPE_INDEX = 5U;
constexpr uint32_t FAG_IS_SPARSE_INDEX = 6U;
constexpr uint32_t FAG_SPARSE_MODE_INDEX = 7U;
constexpr uint32_t FAG_BLOCK_STARTS_OFFSET = 8U;
constexpr uint32_t FAG_BLOCK_ENDS_OFFSET = 44U;
constexpr uint32_t FAG_TND_START_BIDX_OFFSET = 80U;
constexpr uint32_t FAG_MASK_MODE_INDEX = 116U;
constexpr uint32_t FAG_WIN_LEFT_INDEX = 117U;
constexpr uint32_t FAG_WIN_RIGHT_INDEX = 118U;
constexpr uint32_t FAG_MAX_SEQLEN_Q_INDEX = 119U;
constexpr uint32_t FAG_MAX_SEQLEN_KV_INDEX = 120U;

// FAG internal constants
constexpr uint32_t FAG_S1CV_RATIO_DEFAULT = 2U;
constexpr uint32_t FAG_S2CV_RATIO_DEFAULT = 1U;
constexpr int64_t FAG_ALIGN64 = 64;
constexpr uint32_t FAG_ARRAY_LENGTH = 3U;
constexpr uint32_t FAG_BATCH_MAX_SIZE = 2048U;
constexpr uint32_t FAG_MAX_S2_OUTER = 1024U;

// FAG layout type values (matching flash_attention_score_grad tiling common_regbase.h)
constexpr uint32_t FAG_INPUT_FORMAT_BS2N2GD = 1U; // BSND
constexpr uint32_t FAG_INPUT_FORMAT_BN2GS2D = 3U; // BNSD
constexpr uint32_t FAG_INPUT_FORMAT_TND = 4U;     // TND

// FAG sparse mode values (matching flash_attention_score_grad SparseMode enum)
constexpr uint32_t FAG_SPARSE_NO_MASK = 0U;
constexpr uint32_t FAG_SPARSE_RIGHT_DOWN_CAUSAL = 3U;
constexpr uint32_t FAG_SPARSE_BAND = 4U;

// FAG split axis values (matching SplitAxisEnum)
constexpr uint32_t FAG_SPLIT_AXIS_BN2GS1S2 = 0U;
constexpr uint32_t FAG_SPLIT_AXIS_BN2 = 1U;
constexpr uint32_t FAG_SPLIT_AXIS_BN2S2 = 2U;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    uint32_t aicNum;
    uint32_t aivNum;
    FA_METADATA_T *headMetadata; // [METADATA_STRIDE];
    FA_METADATA_T *faMetadata;   // [sectionNum][aicNum][METADATA_STRIDE];
    FA_METADATA_T *fdMetadata;   // [sectionNum][aivNum][METADATA_STRIDE];
    FaMetadata(uint32_t aicNum, uint32_t aivNum, uint32_t sectionNum, void *metadataPtr)
        : sectionNum(sectionNum),
          aicNum(aicNum),
          aivNum(aivNum),
          headMetadata(static_cast<FA_METADATA_T *>(metadataPtr)),
          faMetadata(headMetadata + METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * aicNum * METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicNum * METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aivNum * METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        faMetadata[sectionIdx * aicNum * METADATA_STRIDE + aicIdx * METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        return faMetadata[aicNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        fdMetadata[aivNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < METADATA_STRIDE);
        return fdMetadata[aivNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
