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
 * \file prompt_flash_attention_entry_regbase.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_ENTRY_ARCH38_H_
#define PROMPT_FLASH_ATTENTION_ENTRY_ARCH38_H_

#include "../common/arch38/flash_attention_score_kernel_infer_regbase_v2.h"
#include "prompt_flash_attention_dummy.h"
namespace optiling {};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define PFA_REGBASE_COPY_TILING_DATA(tiling)                                                                                                \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);                                           \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn;                                                 \
 
#define INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(templateClass, ...)                                 \
    do {                                                                                                                                \
        if (query == nullptr) {return;}                                                                                                 \
        PFA_REGBASE_COPY_TILING_DATA(tiling);                                                                                           \
        TPipe tPipe;                                                                                                                    \
        using CubeBlockType = BaseApi::FABlockCube<__VA_ARGS__>; \
        using VecBlockType = BaseApi::FABlockVecInfer<__VA_ARGS__>; \
        templateClass<CubeBlockType, VecBlockType> op;                                                                                  \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths,                             \
            actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, nullptr, nullptr, deq_scale2,                                 \
            deq_scale1, quant_scale1, postQuantScale, postQuantOffset, queryRope, keyRope,                                                           \
            nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe);                                             \
        op.Process();                                                                                                                   \
    } while (0)

// kv is empty tensor, return zero output
#define INVOKE_PFA_DUMMY(templateClass, ...)                                                            \
    do {                                                                                                                                \
        TPipe tPipe;                                                                                        \
        PFA_REGBASE_COPY_TILING_DATA(tiling);                                                               \
        PromptFlashAttentionDummy<half> op;                                                                 \
        op.Init(attentionOut, tilingData);                                                                  \
        op.Process();                                                                                       \
        return                                                                                              \
    } while (0)


inline __aicore__ void prompt_flash_attention_FIAS_regbase(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
    __gm__ uint8_t* deq_scale2, __gm__ uint8_t* postQuantScale, __gm__ uint8_t* postQuantOffset,
    __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* key_antiquant_scale,
    __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
    __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
    __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
#if (__NPU_ARCH__ == 5102)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_IS(1002312000040001212);
    TILING_KEY_IS(1002312000040021212);
    TILING_KEY_IS(1001311000000001212);
    TILING_KEY_IS(1001311000000021212);
    TILING_KEY_IS(1002122000000021012);
    TILING_KEY_IS(1000000000000000090);
    #if TILING_KEY_VAR == 1002312000040001212
        // BNSD layout HighPerformance, No mask, No pse, 常量化, D128, useDn
        INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
            LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128,
            DTemplateType::Aligned128, DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
    #elif TILING_KEY_VAR == 1002312000040021212
        // BNSD layout HighPerformance, No mask, No pse, 常量化, D128, useDn
        INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
            LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128,
            DTemplateType::Aligned128, DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
    #elif TILING_KEY_VAR == 1001311000000001212
        // BNSD layout HighPerformance, No mask, No pse, 常量化, D128
        INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
            LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128,
            DTemplateType::Aligned64, DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
    #elif TILING_KEY_VAR == 1001311000000021212
        // BNSD layout HighPerformance, No mask, No pse, 常量化, D64
        INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
            LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128,
            DTemplateType::Aligned64, DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
    #elif TILING_KEY_VAR == 1002122000000021012
        // BNSD layout HighPerformance, No mask, No pse, 常量化, D128
        INVOKE_PFA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
            LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned64, S2TemplateType::Aligned256,
            DTemplateType::Aligned128, DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
    #elif TILING_KEY_VAR == 1000000000000000090
        INVOKE_PFA_DUMMY();
    #endif
#endif
}
#endif // end of PROMPT_FLASH_ATTENTION_ENTRY_ARCH38_H_