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
 * \file prompt_flash_attention_template_tiling_key.h
 * \brief
 */
 

#ifndef PROMPT_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H_
#define PROMPT_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H_


#include "prompt_flash_attention_template_tiling_key_enum.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_FLOAT16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_FLOAT16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_FLOAT16)
#endif

ASCENDC_TPL_ARGS_DECL(PromptFlashAttention, 
    //    bit 8-1 InOutLayoutType(InputLayoutType-OutputLayoutType)
    //    0: InOutLayoutType_BNSD_BNSD
    //    1: InOutLayoutType_BSH_BSH
    //    2: InOutLayoutType_TND_TND
    //    3: InOutLayoutType_BNSD_BSND
    //    4: InOutLayoutType_NTD_NTD
    ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
    //    bit 18-9 Config(S1,S2,D,DV)
    //    0: Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64
    //    1: Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128
    //    2: Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64
    //    3: Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128
    //    4: Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128
    //    5: Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128
    //    6: Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256
    //    7: Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512
    //    8: Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64
    //    9: Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512
    //   10: Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256
    //   11: Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512
    //   12: Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64
    //   13: Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128
    //   14: Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256
    //   15: Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512
    //   16: Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64
    //   17: Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128
    //   18: Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256
    ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
    //    bit 22-19 PseMode
    //    0: PSE_MODE_PSE_OUTER_MUL_ADD_TYPE
    //    1: PSE_MODE_PSE_OUTER_ADD_MUL_TYPE
    //    2: PSE_MODE_PSE_INNER_MUL_ADD_TYPE
    //    3: PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE
    //    4: PSE_MODE_PSE_INVALID_TYPE
    //    9: PSE_MODE_PSE_NONE_TYPE
    ASCENDC_TPL_UINT_DECL(PseMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15),
    //    bit 27-23 QuantMode
    //    0: AntiquantMode_PER_CHANNEL
    //    1: AntiquantMode_PER_TOKEN
    //    2: AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN
    //    3: AntiquantMode_PER_TOKEN_HEAD
    //    4: AntiquantMode_PER_TOKEN_PAGE_ATTENTION
    //    5: AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION
    //   17: FULLQUANT_MODE_QKV_PERBLOCK
    //   18: FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR
    //   30: FullQuantMode
    //   31: NoQuantMode
    ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
    //    bit 28 HasAttenMask
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),
    //    bit 29 HasRope
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasRope, false, true),
    //    bit 31-30 KvLayoutType
    //    0: KvLayoutType_NO_PA
    //    1: KvLayoutType_PA_BBH
    //    2: KvLayoutType_PA_BNBD
    //    3: KvLayoutType_PA_NZ
    ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
    //    bit 32 IsFd
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(IsFd, false, true),
    //    bit 33 EmptyTensor
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EmptyTensor, false, true),
    //    bit 39 EnableKVPrefix
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EnableKVPrefix, false, true),
    //    bit 40 EnableS1OutSplit
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EnableS1OutSplit, false, true),
);

ASCENDC_TPL_SEL(
#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, true),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,                            
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_INNER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128,
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64,
                            Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, true),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_INNER_MUL_ADD_TYPE,
                            PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, true),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_OUTER_MUL_ADD_TYPE, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 
                            Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),
        ASCENDC_TPL_BOOL_SEL(HasRope, false),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
        ASCENDC_TPL_BOOL_SEL(IsFd, false),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
     // fp8 per-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FULLQUANT_MODE_QKV_PERBLOCK),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
        ASCENDC_TPL_BOOL_SEL(HasRope, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, 0),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    // fp8 per-block
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FULLQUANT_MODE_QKV_PERBLOCK),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
        ASCENDC_TPL_BOOL_SEL(HasRope, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, 0),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
    ),
#endif
ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0),
    ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
    ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
    ASCENDC_TPL_BOOL_SEL(HasRope, 0),
    ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
    ASCENDC_TPL_BOOL_SEL(IsFd, 0),
    ASCENDC_TPL_BOOL_SEL(EmptyTensor, 1),
    ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, 0),
    ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
    ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)
),
);
#endif