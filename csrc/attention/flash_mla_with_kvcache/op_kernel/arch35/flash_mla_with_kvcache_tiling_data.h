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
 * \file flash_mla_with_kvcache_tiling_data.h
 * \brief flash_mla_with_kvcache arch35 kernel 侧 tiling struct 与 metadata 消费常量
 *
 * Tiling struct：字段布局与 FIA NoQuantTilingArch35 / FusedInferAttentionScoreTilingData 一致，
 * 仅改名（Fia* -> FlashMlaWithKvcache*），并移除内嵌 fiaMetaData —— metadata 以 GM 输入 tensor 传入
 * （flash_attn 式 AICPU 多 section 布局）。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_
#define FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_

namespace optiling {

// 数组长度（对齐 FIA 命名；kernel 侧仅用 AIC/AIV 核数做索引数学）
constexpr uint32_t FA_AIC_CORE_NUM = 36;
constexpr uint32_t FA_AIV_CORE_NUM = 72;

// ============================= metadata 消费常量 =============================
// 消费 flash_mla_metadata（flash_attn_metadata 改名克隆）产出的 AICPU 多 section 布局：（producer
// 旧名：flash_mla_metadata，历史原因保留）
//   [16-word header][sectionNum * 36 * 16 (FA)][sectionNum * 72 * 16 (FD)]
// AICPU producer 侧对应常量 FA_METADATA_STRIDE=16 / FD_METADATA_STRIDE=16
// （flash_mla_metadata/op_kernel_aicpu/flash_mla_metadata.h，字节布局与 flash_attn_metadata 一致）。（producer
// 旧名：flash_mla_metadata，历史原因保留） 名称沿用 flash_attn kernel 侧常量（flash_attn_tiling_data.h / 编译期即暴露与
// producer 的偏差）。
constexpr uint32_t FLASH_ATTN_METADATA_SIZE = 16;
constexpr uint32_t FA_FD_METADATA_SIZE = 16;
// header 区 16 个 uint32（sectionNum/isFd/mBaseSize/s2BaseSize + 12 保留）
constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);

// FA Metadata Index Definitions（0 基、无 CORE_ENABLE，对应 AICPU producer；非活跃核全零）
constexpr uint32_t FLASH_ATTN_BN2_START_INDEX = 0;
constexpr uint32_t FLASH_ATTN_M_START_INDEX = 1;
constexpr uint32_t FLASH_ATTN_S2_START_INDEX = 2;
constexpr uint32_t FLASH_ATTN_BN2_END_INDEX = 3;
constexpr uint32_t FLASH_ATTN_M_END_INDEX = 4;
constexpr uint32_t FLASH_ATTN_S2_END_INDEX = 5;
constexpr uint32_t FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

// FD Metadata Index Definitions（0 基、无 CORE_ENABLE；核活跃由 FA_FD_M_NUM_INDEX > 0 判定）
constexpr uint32_t FA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FA_FD_M_IDX_INDEX = 1;
constexpr uint32_t FA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FA_FD_M_START_INDEX = 4;
constexpr uint32_t FA_FD_M_NUM_INDEX = 5;

struct stridesParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;

    void set_bnStride(uint64_t bnStride)
    {
        this->bnStride = bnStride;
    }
    uint64_t get_bnStride() const
    {
        return bnStride;
    }
    void set_n2Stride(uint64_t n2Stride)
    {
        this->n2Stride = n2Stride;
    }
    uint64_t get_n2Stride() const
    {
        return n2Stride;
    }
};

struct FlashMlaWithKvcacheBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t t2Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeRope = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float scaleValue = 0.0f;
    uint8_t isKvContinuous = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
    // 增加strides参数
    stridesParams keyStrides;
    stridesParams valueStrides;
    stridesParams kRopeStrides;
    stridesParams kScaleStrides;
    stridesParams vScaleStrides;
};

struct FlashMlaWithKvcacheAttenMaskParams {
    uint8_t sparseMode = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t isRowInvalidOpen = 0;
    uint8_t isExistRowInvalid = 0;
};

struct FlashMlaWithKvcachePseParams {
    uint8_t pseShiftByBatch = 0;
    uint32_t pseS1Size = 0;
    uint32_t pseS2Size = 0;
    uint32_t pseStride = 0;
    uint32_t qStartIdx = 0;
    uint32_t kvStartIdx = 0;
};

struct FlashMlaWithKvcacheSystemPrefixParams {
    uint8_t isActualSharedPrefixLenNull = 0;
    uint32_t prefixSeqInnerSize = 0;
};

struct FlashMlaWithKvcachePageAttentionParams {
    uint8_t paLayoutType = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
};

struct FlashMlaWithKvcacheLeftPaddingParams {
    uint8_t isQHasLeftPadding = 0;
    uint8_t isKVHasLeftPadding = 0;
};

struct FlashMlaWithKvcachePostQuantParams {
    uint8_t isPostQuantPerChnl = 0;
    uint8_t isPostQuantBF16 = 0;
};

struct FlashMlaWithKvcacheWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct FlashMlaWithKvcacheEmptyTensorParams {
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

// FIA FusedInferAttentionScoreTilingData 的字段布局改名保留；FiaMetaData 成员已移除
// （metadata 走 GM 输入，见 flash_mla_with_kvcache_kernel_noquant_mla.h）
class FlashMlaWithKvcacheNoQuantTilingArch35 {
public:
    FlashMlaWithKvcacheBaseParams flashMlaWithKvcacheBaseParams;
    FlashMlaWithKvcacheAttenMaskParams flashMlaWithKvcacheAttenMaskParams;
    FlashMlaWithKvcachePseParams flashMlaWithKvcachePseParams;
    FlashMlaWithKvcacheSystemPrefixParams flashMlaWithKvcacheSystemPrefixParams;
    FlashMlaWithKvcachePageAttentionParams flashMlaWithKvcachePageAttentionParams;
    FlashMlaWithKvcacheLeftPaddingParams flashMlaWithKvcacheLeftPaddingParams;
    FlashMlaWithKvcachePostQuantParams flashMlaWithKvcachePostQuantParams;
    FlashMlaWithKvcacheWorkspaceParams flashMlaWithKvcacheWorkspaceParams;
    FlashMlaWithKvcacheEmptyTensorParams flashMlaWithKvcacheEmptyTensorParams;
};

class FlashMlaWithKvcacheTilingData {
public:
    FlashMlaWithKvcacheNoQuantTilingArch35 baseTiling;
};

} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_
