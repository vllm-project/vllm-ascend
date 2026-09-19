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
 * \file flash_attn_common_def.h
 * \brief
 */

#ifndef FLASH_ATTN_COMMON_DEF_H_
#define FLASH_ATTN_COMMON_DEF_H_

// 提供flash_attn算子tiling和kernel共用的相关定义

// BUFFER的字节数
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;

#define ASCENDC_TPL_3_BW 3

enum class FA_LAYOUT : uint32_t {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class inferFaLayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};
enum class inferPFALayoutTypeEnum {
    LAYOUT_BSH = 0,
    LAYOUT_BNSD = 1,
    LAYOUT_TND = 2,
    LAYOUT_NTD = 3
};
enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

// q_out layout → (inputLayout, outputLayout)
//   0=BSND → (BSND, BSND)
//   1=BNSD → (BNSD, BNSD)
//   2=TND  → (TND, TND)
//   3=BNSD_BSND → (BNSD, BSND)
static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[5][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_NTD},
};

static constexpr inferPFALayoutTypeEnum InOutLayoutPFATypeValue[5][2] = {
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BNSD},
    {inferPFALayoutTypeEnum::LAYOUT_BSH, inferPFALayoutTypeEnum::LAYOUT_BSH},
    {inferPFALayoutTypeEnum::LAYOUT_TND, inferPFALayoutTypeEnum::LAYOUT_TND},
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BSH}, // bsndout占位
    {inferPFALayoutTypeEnum::LAYOUT_NTD, inferPFALayoutTypeEnum::LAYOUT_NTD},
};

// q_out layout
//   0: BSND (BSND排布与BSH一样)
//   1: BNSD
//   2: TND
//   3: BNSD_BSND
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_BNSD_BSND 3

// KvLayoutType
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

// templateId
#define FA_Template_ND 0
#define FA_Template_DN 1

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

// config:
//   config=0: sOuter=64, sInner=128 → S1=128, S2=128, D=64
//   config=1: sOuter=32, sInner=256 → S1=64,  S2=256, D=64
//   config=2: sOuter=64, sInner=128 → S1=128, S2=128, D=128
//   config=3: sOuter=32, sInner=256 → S1=64,  S2=256, D=128
//   config=4: sOuter=64, sInner=128 → S1=128, S2=128, D=256
//   config=5: sOuter=32, sInner=256 → S1=64,  S2=256, D=256
//   config=6: sOuter=64, sInner=128 → S1=128, S2=128, D=192, DV=128
//   config=7: sOuter=32, sInner=256 → S1=64,  S2=256, D=192, DV=128
static constexpr ConfigParams ConfigValue[] = {
    // D=64 configurations
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // config=0
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64}, // config=1

    // D=128 configurations
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // config=2
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128}, // config=3

    // D=256 configurations
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // config=4
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256}, // config=5

    // D=192, DV=128 configurations (MLA qk != v)
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128}, // config=6
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned192,
     inferDTemplateType::Aligned128}, // config=7
};

// Config macro definitions for D=64
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64 0
#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64 1

// Config macro definitions for D=128
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128 2
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128 3

// Config macro definitions for D=256
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 4
#define Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 5

// Config macro definitions for D=192, DV=128 (MLA qk != v)
#define Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128 6
#define Config_S1Aligned64_S2Aligned256_DAligned192_DVAligned128 7

// PseMode
#define PSE_MODE_PSE_OUTER_MUL_ADD_TYPE 0
#define PSE_MODE_PSE_OUTER_ADD_MUL_TYPE 1
#define PSE_MODE_PSE_INNER_MUL_ADD_TYPE 2
#define PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE 3
#define PSE_MODE_PSE_INVALID_TYPE 4
#define PSE_MODE_PSE_NONE_TYPE 9

// QuantModeEnum
#define NoQuantMode 31

// bool
#define false 0
#define true 1

namespace FlashAttnKernel {

struct FDparams {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfo {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;     // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;     // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;
    // uint32_t curS2LoopTimes = 0;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块
    bool isFirstFdBlock = false;

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t dSizeRope;
    uint32_t gSize; /* g轴的大小 */
    uint32_t n2Size;
    uint64_t s1Size;          /* s1总大小 */
    uint64_t s2Size;          /* s2总大小 */
    uint64_t cuSeqLensQSize;  /* 用户输入的actualseq的长度 */
    uint64_t cuSeqLensKVSize; /* 用户输入的actualseq_kv的长度 */
    uint64_t seqUsedQSize;    /* 用户输入的seq used q的长度 */
    uint64_t seqUsedKvSize;   /* 用户输入的seq used kv的长度 */

    /* FA kernel meta */
    uint32_t bN2Start;
    uint32_t bN2End;
    uint32_t gS1OStart;
    uint32_t gS1OEnd;
    uint32_t s2OStart;
    uint32_t s2OEnd;
    uint32_t coreFirstTmpOutWsPos;

    uint32_t sparseMode; // sparse
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isExistRowInvalid;
    float scaleValue;

    /* 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    /* 输出shape */
    FA_LAYOUT outputLayout;
    bool needInitOutput;
    bool enableFlashDecode;
};

struct PAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
    /* kvcache非连续场景的strides, 0表示连续 */
    uint64_t keyBnStride = 0;
    uint64_t keyN2Stride = 0;
    uint64_t valueBnStride = 0;
    uint64_t valueN2Stride = 0;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable;
};

struct SinkConstInfo {
    bool learnableSinkFlag = false;
};

struct ConstInfo_t : CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo {};

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_COMMON_DEF_H_
