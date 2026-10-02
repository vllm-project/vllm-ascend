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
 * \file flash_attn_tiling_data.h
 * \brief
 */

#ifndef FLASH_ATTN_TILING_DATA_H_
#define FLASH_ATTN_TILING_DATA_H_

namespace optiling {
// AICPU metadata format: 16 fields per core (FA and FD both)
constexpr uint32_t METADATA_STRIDE = 16U;
static constexpr uint32_t METADATA_HEADER_OFFSET = METADATA_STRIDE * sizeof(uint32_t);

// FA Metadata Index Definitions (0-based, matching AICPU flash_attn_metadata.h)
// HEAD
constexpr uint32_t METADATA_HEADER_SECTION_NUM_INDEX = 0;
constexpr uint32_t METADATA_HEADER_HAS_FD_INDEX = 1;
constexpr uint32_t METADATA_HEADER_AIC_NUM_INDEX = 4;
constexpr uint32_t METADATA_HEADER_AIV_NUM_INDEX = 5;

// No CORE_ENABLE field in AICPU format; inactive cores have all-zero data.
constexpr uint32_t FLASH_ATTN_BN2_START_INDEX = 0;
constexpr uint32_t FLASH_ATTN_M_START_INDEX = 1;
constexpr uint32_t FLASH_ATTN_S2_START_INDEX = 2;
constexpr uint32_t FLASH_ATTN_BN2_END_INDEX = 3;
constexpr uint32_t FLASH_ATTN_M_END_INDEX = 4;
constexpr uint32_t FLASH_ATTN_S2_END_INDEX = 5;
constexpr uint32_t FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

// FD Metadata Index Definitions (0-based, matching AICPU flash_attn_metadata.h)
// No CORE_ENABLE field; active state is indicated by FA_FD_M_NUM_INDEX > 0.
constexpr uint32_t FA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FA_FD_M_IDX_INDEX = 1;
constexpr uint32_t FA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FA_FD_M_START_INDEX = 4;
constexpr uint32_t FA_FD_M_NUM_INDEX = 5;

struct FlashAttnBaseParams {
    uint32_t bSize;
    uint32_t t1Size;
    uint32_t t2Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dSizeRope;
    uint32_t cuSeqLensQSize;
    uint32_t cuSeqLensKVSize;
    uint32_t seqUsedQSize;
    uint32_t seqUsedKvSize;
    float scaleValue;
    uint8_t iscuSeqLengthsNull;
    uint8_t iscuSeqLengthsKVNull;
    uint8_t isSoftMaxLseEnable;
    uint32_t coreNum;
    uint32_t outputLayout;
    bool needInitOutput;
    // kvcache non-contiguous strides (valid only for tensor v2 view inputs)
    uint64_t keyBnStride = 0;
    uint64_t keyN2Stride = 0;
    uint64_t valueBnStride = 0;
    uint64_t valueN2Stride = 0;
};

struct FlashAttnAttenMaskParams {
    uint8_t sparseMode;
    int32_t winLefts;
    int32_t winRights;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    uint8_t isExistRowInvalid = 0;
};

struct FlashAttnPageAttentionParams {
    uint8_t paLayoutType;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct FlashAttnWorkspaceParams {
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct FlashAttnS1OuterSplitCoreParams {
    bool enableS1OutSplit = 0;
    uint64_t totalSize;
};

class FlashAttnNoQuantTilingArch35 {
public:
    FlashAttnBaseParams flashAttnBaseParams;
    FlashAttnAttenMaskParams flashAttnAttenMaskParams;
    FlashAttnPageAttentionParams flashAttnPageAttentionParams;
    FlashAttnWorkspaceParams flashAttnWorkspaceParams;
    FlashAttnS1OuterSplitCoreParams flashAttnS1OuterSplitCoreParams;
};

class FlashAttnTilingData {
public:
    FlashAttnNoQuantTilingArch35 baseTiling;
};

} // namespace optiling
#endif
