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
 * \file prompt_flash_attention_template_tiling_key_enum.h
 * \brief
 */


#ifndef PROMPT_FLASH_ATTENTION_TEMPLATE_TILING_KEY_ENUM_H_
#define PROMPT_FLASH_ATTENTION_TEMPLATE_TILING_KEY_ENUM_H_

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

#if (__CCE_AICORE__ == 310)
#define PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, ...)                                                  \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);  \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);                     \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);                     \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);                         \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)

#define PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...)                                                    \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);  \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);                     \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);                     \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);                         \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)

#define PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, ...)                                                  \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);  \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);                     \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);                     \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);                         \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)

#endif

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

enum class inferImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

#define ImplMode_AA_HIGH_PRECISION 0
#define ImplMode_AA_HIGH_PERFORMANCE 1
#define ImplMode_AA_INVALID_LINE_HIGH_PRECISION 2

static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[6][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_NTD},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_TND},
};

static constexpr inferPFALayoutTypeEnum InOutLayoutPFATypeValue[6][2] = {
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BNSD},
    {inferPFALayoutTypeEnum::LAYOUT_BSH, inferPFALayoutTypeEnum::LAYOUT_BSH},
    {inferPFALayoutTypeEnum::LAYOUT_TND, inferPFALayoutTypeEnum::LAYOUT_TND},
    {inferPFALayoutTypeEnum::LAYOUT_BNSD, inferPFALayoutTypeEnum::LAYOUT_BSH}, // bsndout占位
    {inferPFALayoutTypeEnum::LAYOUT_NTD, inferPFALayoutTypeEnum::LAYOUT_NTD},
    {inferPFALayoutTypeEnum::LAYOUT_NTD, inferPFALayoutTypeEnum::LAYOUT_TND},
};

#define InOutLayoutType_BNSD_BNSD 0
#define InOutLayoutType_BSH_BSH 1 // BSND由于排布与BSH一样，因此两个会编译成相同取值
#define InOutLayoutType_TND_TND 2
#define InOutLayoutType_BNSD_BSND 3
#define InOutLayoutType_NTD_NTD 4
#define InOutLayoutType_NTD_TND 5

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

// Config
static constexpr ConfigParams ConfigValue[] ={
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //0
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //1
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //2
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //3
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned192, inferDTemplateType::Aligned128}, //4
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned128}, //5
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //6
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //7
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //8
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned576, inferDTemplateType::Aligned512}, //9
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //10
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned64, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //11
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned1024, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //12
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //13
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //14
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //15
   {inferS1TemplateType::Aligned16, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //16
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //17
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //18
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //19
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //20
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256, inferDTemplateType::Aligned256}, //21
   {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned512, inferDTemplateType::Aligned512}, //22
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned128, inferDTemplateType::Aligned64}, //23
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned64, inferDTemplateType::Aligned128}, //24
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128, inferDTemplateType::Aligned64}, //25
   {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned64, inferDTemplateType::Aligned128}, //26
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64, inferDTemplateType::Aligned64}, //27
   {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}, //28
};

#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64 0
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64 2
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128 3
#define Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128 4
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128 5
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 6
#define Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512 7
#define Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64 8
#define Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 9
#define Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256 10
#define Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512 11
#define Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64 12
#define Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128 13
#define Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256 14
#define Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512 15
#define Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64 16
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 17
#define Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 18
#define Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64 19
#define Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128 20
#define Config_S1Aligned32_S2Aligned256_DAligned256_DVAligned256 21
#define Config_S1Aligned32_S2Aligned128_DAligned512_DVAligned512 22
#define Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64 23
#define Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128 24
#define Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64 25
#define Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128 26
#define Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64 27
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 28


// PseMode
#define PSE_MODE_PSE_OUTER_MUL_ADD_TYPE 0
#define PSE_MODE_PSE_OUTER_ADD_MUL_TYPE 1
#define PSE_MODE_PSE_INNER_MUL_ADD_TYPE 2
#define PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE 3
#define PSE_MODE_PSE_INVALID_TYPE 4
#define PSE_MODE_PSE_NONE_TYPE 9

// QuantModeEnum
#define AntiquantMode_PER_CHANNEL 0
#define AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN 2
#define AntiquantMode_K_PER_TOKEN 6 // 表示其他伪量化类型
#define FULLQUANT_MODE_QKV_PERBLOCK 17
#define FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR 18
#define FULLQUANT_MODE_QKV_MXFP8_PREFILL 19
#define FULLQUANT_MODE_QKV_MXFP8_DECODE 20
#define FULLQUANT_MODE_QK_PER_TOKEN_HEAD_V_PER_HEAD 21
#define FullQuantMode 30
#define NoQuantMode 31

// KvLayoutType
/*
    To ensure template compatibility and reduce total tilingkey, KvLayoutType is assigned different optional values for
   kernel templates before/after refactoring.
    1. For refactored template(e.g. GQA non quant), KvLayoutType is template parameterized, KvLayoutType_NO_PA,
   KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ are required.
    2. For unrefactored templates, KvLayoutType_NO_PA and KvLayoutType_ENABLE_PA are recommended to reduce tilingkey and
   keep compatibility, KvLayoutType has same meaning as the original isPa.
*/
#define KvLayoutType_NO_PA 0
#define KvLayoutType_ENABLE_PA 1
#define KvLayoutType_PA_BBH 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

// SplitCoreModeEnum
#define SplitCoreMode_SPLIT_NBS_VECTOR 0
#define SplitCoreMode_SPLIT_NBS_CUBE 1
#define SplitCoreMode_SPLIT_ONEN_VECTOR 2
#define SplitCoreMode_SPLIT_ONEN_CUBE 3
#define SplitCoreMode_BALANCE_VECTOR 4
#define SplitCoreMode_BALANCE_CUBE 5

// bool
#define false 0
#define true 1

#endif