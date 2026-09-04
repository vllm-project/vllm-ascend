/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling_struct.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_STRUCT_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_STRUCT_H_


#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {
struct IncreFlashAttentionCompileInfo {
    int64_t core_num;
};

struct ActualSeqInfo {
    std::vector<int64_t> actualSeqQ;
    int64_t maxActualseqkv;

    ActualSeqInfo(uint32_t bSize, int64_t maxVal) : actualSeqQ(bSize), maxActualseqkv(maxVal) {};
    ActualSeqInfo() = delete;
};

struct SeqTilingInfo {
    std::vector<uint32_t> s1OuterNum;                           // S1方向，切了多少个基本块
    std::vector<uint32_t> s2OuterNum;                           // S2方向，切了多少个基本块
    uint32_t lastValidBIdx = 0U;
    uint64_t avgS2Length = 1U;

    explicit SeqTilingInfo(uint32_t bSize) : s1OuterNum(bSize), s2OuterNum(bSize) {};
    SeqTilingInfo() = delete;
};

struct BalancedSplitTilingInfo
{
    std::vector<uint64_t> coreLoad;
    uint64_t accumS2Length = 0U;       // 当前累积的权重
    uint32_t currCoreIdx = 0U;
    bool needUpdate = false;
    uint32_t maxKvSplitPart = 0U;
    uint32_t tndFDCoreArrLen = 0U;

    explicit BalancedSplitTilingInfo(uint32_t coreNum) : coreLoad(coreNum) {};

    BalancedSplitTilingInfo() = delete;
};

struct IfaWorkSpaceSizeParams {
    uint32_t mmResElemSize = 4U;         // 4:fp32
    uint32_t vec1ResElemSize = 2U;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4U;       // 4:fp32
    uint32_t vec2ResElemSize = 4U;       // 4:fp32
    uint32_t qPreProcResElemSize = 0U;   // 普通场景不涉及Q预处理
    uint32_t nUpdateElemSize = 4U;   // 4:int32
    uint32_t softmaxSumElemSize = 4U;   // 4:int32
    float kvDtypeRatio = 1.0;
};

struct KvLayoutInfo {
    uint8_t kvLayoutVal = 0U;
    uint8_t amlaMode = 0U;
};

struct TilingIndexes {
    uint32_t bIdx;
    uint32_t s1Idx;
    uint32_t s2Idx;

    TilingIndexes() : bIdx(0U), s1Idx(0U), s2Idx(0U) {}
    TilingIndexes(uint32_t b, uint32_t s1, uint32_t s2) : bIdx(b), s1Idx(s1), s2Idx(s2) {}
};

enum class KvCacheLayout : uint32_t {
    KV_CACHE_BSH = 0,
    KV_CACHE_BNSD = 1,
    KV_CACHE_NZ = 2,
};

enum class TilingInOutMode : uint32_t {
    IO_INVALID = 0,
    INT8_INT8 = 1,
    FP16_INT8 = 2,
    INT8_FP16 = 3,
    FP16_FP16 = 4,
    BF16_BF16 = 5,
    FP32_FP32 = 6,
    FP16_FP16_SPLITKV = 7,
    BF16_INT8 = 8,
    INT8_BF16 = 9,
    FP16_FP8_E4M3FN = 10,
    FP16_HIFLOAT8 = 12,
    BF16_FP8_E4M3FN = 13,
    BF16_HIFLOAT8 = 15,
};

enum class IfaPerfMode : uint32_t {
    NORMAL = 0,
    BMM_ALL_BY_VEC, // 1
    C1_V1, // 2
    CUBE_VIEW_MM, // 3
    CUBE_VIEW_MM_FULL_LOAD, // 4
    CUBE_VIEW_MM_MLA,  // 5
    CUBE_VIEW_MM_DD  // 6
};

enum class IfaSocVersion : uint32_t {
    SOC_ASCEND_910B = 0,
    SOC_ASCEND_310P = 1,
    SOC_ASCEND_950 = 2,
};


enum class IfaLayout : uint32_t {
    BSH_BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class IfaAmlaMode : uint32_t {
    DISABLE_AMLA = 0, // 不使能AMLA
    AMLA = 1,
    AMLA_3BUF = 2
};

enum class IfaMaskType : uint32_t {
    NO_MASK = 0,
    MASK_NORM = 1,
    MASK_SWA_NORM = 2,
    MASK_SWA_COMPRESS = 3,
};

enum class IfaPseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE = 2,
    PSE_1_N2_G_SLOPE = 3
};

enum class IfaPseType : int64_t {
    PSE_OUTER_MUL_ADD_TYPE = 0,
    PSE_INNER_MUL_ADD_TYPE = 2,
    PSE_INNER_MUL_ADD_SQRT_TYPE = 3,
};

} // namespace optiling
#endif