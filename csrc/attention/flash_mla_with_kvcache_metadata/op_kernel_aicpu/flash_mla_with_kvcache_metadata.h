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
 * \file flash_mla_with_kvcache_metadata.h
 * \brief
 */

#ifndef FLASH_MLA_WITH_KVCACHE_METADATA_H
#define FLASH_MLA_WITH_KVCACHE_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
constexpr uint32_t FA_META_SIZE = 1024U;
using FA_METADATA_T = uint32_t;

constexpr uint32_t HEAD_METADATA_STRIDE = 16U;
constexpr uint32_t FA_METADATA_STRIDE = 16U;
constexpr uint32_t FD_METADATA_STRIDE = 16U;

// R7: producer 侧布局不变量。三个 metadata 区每个 core 的 stride 必须恒为 16 words，
// 与 flash_mla kernel 消费点的 FLASH_ATTN_METADATA_SIZE / FA_FD_METADATA_SIZE == 16
// 断言同源一致（此断言位于 producer 布局头，是唯一权威来源）。
static_assert(FA_METADATA_STRIDE == 16U && FD_METADATA_STRIDE == 16U && HEAD_METADATA_STRIDE == 16U,
              "flash_mla metadata layout must stay 16 words/core");

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;

// MLA fixed base sizes (D17/B3): hard-forced into the head metadata by GenMetadata. The MLA kernel
// template asserts mBaseSize==64 && s2BaseSize==128 for D=576 (fia_block_cube_noquant_mla.h:106,
// ported as flash_mla_block_cube_noquant_mla.h) - that static_assert is the invariant source, so these
// MUST NOT be derived from fa_tiling_util::AdjustSinnerAndSouter.
constexpr uint32_t HEAD_M_BASE_SIZE_MLA = 64U;
constexpr uint32_t HEAD_S2_BASE_SIZE_MLA = 128U;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN2_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN2_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN2_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    FA_METADATA_T *headMetadata; // [HEAD_METADATA_STRIDE];
    FA_METADATA_T *faMetadata;   // [sectionNum][AIC_CORE_NUM][FA_METADATA_STRIDE];
    FA_METADATA_T *fdMetadata;   // [sectionNum][AIV_CORE_NUM][FD_METADATA_STRIDE];
    FaMetadata(void *metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum),
          headMetadata(static_cast<FA_METADATA_T *>(metadataPtr)),
          faMetadata(headMetadata + HEAD_METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * AIC_CORE_NUM * FA_METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < HEAD_METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * AIC_CORE_NUM * FA_METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * AIV_CORE_NUM * FD_METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        faMetadata[sectionIdx * AIC_CORE_NUM * FA_METADATA_STRIDE + aicIdx * FA_METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        return faMetadata[AIC_CORE_NUM * FA_METADATA_STRIDE * sectionIdx + FA_METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_STRIDE);
        fdMetadata[AIV_CORE_NUM * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_STRIDE);
        return fdMetadata[AIV_CORE_NUM * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
