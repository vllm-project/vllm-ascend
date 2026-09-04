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
 * \file prompt_flash_attention_arch22.h
 * \brief
 */

#include "prompt_flash_attention_tilingkey.h"
#if (__CCE_AICORE__ > 200)
#include "./arch22/prompt_flash_attention_base.h"
#include "./arch22/prompt_flash_attention_bnstilling_n_s_no_tail.h"
#include "./arch22/prompt_flash_attention_bnstilling_n_s_tail.h"
#include "./arch22/prompt_flash_attention_bnstilling_n_s_no_tailWBNSD.h"
#include "./arch22/prompt_flash_attention_bnstilling_n_s_tailWBNSD.h"
#include "./arch22/prompt_flash_attention_s1s2_bns1_x910.h"
#include "./arch22/prompt_flash_attention_base_api.h"
#include "./arch22/prompt_flash_attention_base_api_high_precision_no_mask.h"
#include "./arch22/prompt_flash_attention_base_mla.h"
#include "./arch22/prompt_flash_attention_base_mla_high_precision.h"
#include "./arch22/prompt_flash_attention_s1s2_bns1_mla.h"
#include "./arch22/prompt_flash_attention_var_len_score_sab.h"
#include "./arch22/prompt_flash_attention_s1s2_bns1_mla_baseapi.h"
#include "./arch22/prompt_flash_attention_var_len_score_sab_baseapi.h"
#include "./arch22/prompt_flash_attention_empty_tensor.h"
#else
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "unpad_flash_attention_common.h"
#include "prompt_attention_prefill.h"
#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "prompt_flash_attention_s1s2_bns1_x310.h"
#endif

constexpr uint32_t FLOATBYTENUM = 8;
constexpr uint32_t FLOAT16BYTENUM = 16;
constexpr uint32_t INT8BYTENUM = 32;

#define INVOKE_PFA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
            kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);                     \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_INT8_OP_IMPL(templateClass, ...)                                                                     \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
            kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);            \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_KVANTIQUANT_OP_IMPL(templateClass, ...)                                                              \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
            kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitKvAntiquant(antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                              \
        matmul::l1Global = l1Array;                                                                                     \
        matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                      \
        op.Init(query, key, value, attenMask, attentionOut, softmaxLse, user, tilingData, &tPipe);                                  \
        op.Process();                                                                                                   \
    } while (0)
// PFA TND
#define INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2, bmm2tiling); \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, attentionOut, softmaxLse, user, tilingData, &tPipe);   \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, blocktable, learnableSink, \
                     attentionOut, softmaxLse, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, \
                deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, nullptr);                                    \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_MLA(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, nullptr); \
    } while (0)
#define INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling)                                                                      \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                         \
    const MLAGeneralTilingData *__restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr
#ifdef __DAV_C220_CUBE__
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm1TilingDataRect, bmm1TilingData, tiling);                \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm2TilingDataRect, bmm2TilingData, tiling);                \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;                                                        \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;                                                        \
    const PromptFlashAttentionTilingData* __restrict tiling_data = nullptr

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                             \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);                           \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);                           \
    const MLAGeneralTilingData * __restrict tilingData = nullptr;                                                      \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingDataVar
#else
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionTilingData, tiling_data_in, tiling);                               \
    const PromptFlashAttentionTilingData* __restrict tiling_data = &tiling_data_in;                                    \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                     \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                              \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                          \
    const MLAGeneralTilingData * __restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData);                                           \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData)

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in
#endif
#define INVOKE_PFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        if (query == nullptr) {return;}                                                                              \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                                \
        templateClass<__VA_ARGS__> op;                                                                          \
        op.Init(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, attentionOut, user, tiling_data);                                 \
        op.Process();                                                                                                  \
    } while (0)

inline __aicore__ void prompt_flash_attention_FIAS_arch22(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                        __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                        __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                        __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                        __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset,
                                                        __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                        __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                        __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
                                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                        __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* learnableSink,
                                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                        __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
#if (__CCE_AICORE__ > 200)
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
    auto maskByteNum = baseParams.maskTypeByteNum;

    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4)

        #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_BASE_API_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_NORMAL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_NORMAL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_NORMAL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_NORMAL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_BASICAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING);

            #if TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING
                // Non-BNSD layout, split NS no tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, half, CubeFormat::ND, half);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING
                // Non-BNSD layout, split NS with tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, half, CubeFormat::ND, half);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING
                // BNSD layout, split NS no tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, half, CubeFormat::ND,
                                            half);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, bool, CubeFormat::ND,
                                            half);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING
                // BNSD layout, split NS with tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, half, CubeFormat::ND, half);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool>);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t>);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_BASE_API_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, half, half, half, half, float>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPerformance, PFATypeNew<PromptFlashAttentionBaseApiTilingData, half, half, half, half, half, float, half>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLA, PFAMLAType<PromptFlashAttentionBaseApiTilingData>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout HighPrecision, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix BSH layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                // BSH layout HighPrecision, enable L1 reuse
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout HighPrecision, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix BNSD layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_NORMAL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_NORMAL_CUBEVECTORDIFF_NEWTILING
                // Prefix BNSD layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_NORM, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                // BNSD layout HighPrecision
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in half out, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in half out, enable L1 reuse
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BSH, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BNSD, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                }
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_NORMAL_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in half out, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_NORMAL_CUBEVECTORDIFF_NEWTILING  // enable prefix
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_NORM, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_BASICAPI_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV, PFATypeNew<PromptFlashAttentionBaseApiTilingData, half, half, half, float, half, float, half, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, half, false>);

            #endif
        #endif

        #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            #if TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout HighPrecision
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix BSH layout HighPrecision
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout HighPrecision
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix BNSD layout HighPrecision
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BSH, half in half out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BSH, half in half out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BNSD, half in half out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BNSD, half in half out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD
                // BSH layout fp16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout fp16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTFP16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout fp16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout fp16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA_ANTIQUANT>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t>);
            #endif
        #endif

        #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);

            #if TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #endif
        #endif

        #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVINT8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            #if TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BSH_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QFP16_KVINT8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING
                // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
                INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD
                // BSH layout fp16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout fp16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout fp16 in int8 out cvdiff, enable MSD
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QFP16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout fp16 in int8 out cvdiff, enable MSD
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #endif
        #endif
    
    #endif

    #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY != DT_INT4)

        #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
            TILING_KEY_IS(QBF16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_BASE_API_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_MDL_TAIL_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_VALUED192_HIGHPERFORMANCE_MDL_NOTAIL_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_PA_ND_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_ND_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_PA_ND_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_ND_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_PA_NZ_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_NZ_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_PA_NZ_MDL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_NZ_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            #if TILING_KEY_VAR == QBF16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING
                // Non-BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING
                // Non-BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING
                // BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING
                // BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_BASE_API_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, float>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV,
                                                    PFATypeNew<PromptFlashAttentionBaseApiTilingData, bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t, float, bfloat16_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPERFORMANCE_BASICAPI_MLA_CUBEVECTORDIFF_NEWTILING
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, bfloat16_t, true>);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_MDL_TAIL_OLDTILING
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_VALUED192_HIGHPERFORMANCE_MDL_NOTAIL_OLDTILING
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_PA_ND_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_ND_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_PA_ND_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_ND_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_HIGHPERFORMANCE_PA_NZ_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_NZ_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_HIGHPERFORMANCE_PA_NZ_MDL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_NTD_TND_ENABLEATTNETIONMASK_HIGHPERFORMANCE_PA_NZ_MDL_NOTAIL_CUBEVECTORDIFF_OLDTILING
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 cvdiff, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 cvdiff, enable L1 reuse
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
                // BSH layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 cvdiff, enable PA
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 cvdiff, enable L1 reuse
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
                // BNSD layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #endif
        #endif

        #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
            TILING_KEY_IS(QBF16_KVFP8_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            
            #if TILING_KEY_VAR == QBF16_KVFP8_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD
                // BSH layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD
                // BNSD layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTBF16_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTBF16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 cvdiff
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #endif

        #endif

        #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
            #if TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 in int8 out cvdiff, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
                // BSH layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 in int8 out cvdiff, enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == QBF16_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
                // BNSD layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #endif
        #endif

        #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
            TILING_KEY_IS(QBF16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            TILING_KEY_IS(QBF16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
            #if TILING_KEY_VAR == QBF16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD, enable MSD
                // BSH layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MSD_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix, enable MSD
                // BNSD layout bf16 in int8 out cvdiff
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTINT8_BSH_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BSH layout bf16 in int8 out cvdiff, enable MSD
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #elif TILING_KEY_VAR == QBF16_KVFP8_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
                // BNSD layout bf16 in int8 out cvdiff, enable MSD
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
            #endif
        #endif
    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_INT8)
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        #if TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, int8_t);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, int8_t);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, int8_t);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, int8_t);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_BNSD_NEWTILING  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_BNSD_NEWTILING  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
        #endif
    #endif

    #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8)
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING);
        TILING_KEY_IS(QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_BNSD_NEWTILING);
        #if TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, half);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, half);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, half);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, half);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_NEWTILING  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_BNSD_NEWTILING
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_PA_ND_MDL_CUBEVECTORDIFF_BNSD_NEWTILING  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_NEWTILING  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
        #elif TILING_KEY_VAR == QINT8_KVFP16_OUTFP16_HIGHPRECISION_HIGHLEVELAPI_PREFIX_MDL_CUBEVECTORDIFF_BNSD_NEWTILING  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
        #endif
    #endif

    TILING_KEY_IS(QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_EMPTYKVTILINGKEY);
    #if TILING_KEY_VAR == QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_TAIL_EMPTYKVTILINGKEY
        TPipe tPipe;
        INVOKE_PFA_TILING_DATA(tiling);
        PromptFlashAttentionEmptyTensor<half> op;
        op.Init(attentionOut, tiling_data, &tPipe);
        op.Process();
        return;
    #endif
#else // 310P
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
    auto maskByteNum = baseParams.maskTypeByteNum;

    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TILING_KEY_IS(QINT8_KVFP16_OUTBF16_BSH_HIGHLEVELAPI_MDL_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BSH_HIGHLEVELAPI_MDL_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    #if TILING_KEY_VAR == QINT8_KVFP16_OUTBF16_BSH_HIGHLEVELAPI_MDL_310TILING
        INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BSH_HIGHLEVELAPI_MDL_310TILING
        INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
    #elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
    #elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
    #endif
#endif
}