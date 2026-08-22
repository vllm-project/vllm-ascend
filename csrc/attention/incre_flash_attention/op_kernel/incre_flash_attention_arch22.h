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
 * \file incre_flash_attention_arch22.h
 * \brief
 */
#include "incre_flash_attention_tilingkey.h"
#include "./incre_flash_attention_allvec_new.h"
#if (__CCE_AICORE__ == 200)
#include "./arch20/incre_flash_attention_cube_310P_kvquant.h"
#endif
#if (__CCE_AICORE__ > 200)
#include "./arch22/incre_flash_attention_split_Bbn2s2_Us2.h"
#include "./arch22/incre_flash_attention_preload.h"
#include "./arch22/incre_flash_attention_preload_dd.h"
#include "./arch22/paged_attention_antiquantkv.h"

#ifdef FIA_ENABLE_MLA
// mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'IncreFlashAttentionTilingDataMla'
#if ((ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8))
#include "./arch22/incre_flash_attention_preload_mla.h"
#endif
#endif // FIA_ENABLE_MLA

#else
#include "./arch20/unpad_paged_attention_decoder.h"
#endif

#define NEED_CUBE_TILING (true)
#define NOT_NEED_CUBE_TILING (false)

#define INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_PREFIX(tiling, NEED_CUBE_TILING);                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling, op.mm1Sp,              \
                          bmm1tilingPrefix, op.mm2Sp, bmm2tilingPrefix);                                               \
        op.InitPrefix(query, keySharedPrefix, valueSharedPrefix, pseShift, attenMask, actualSharedPrefixLen,           \
                      blocktable, kvPaddingSize, attentionOut, softmaxLse, user, &tiling_data->tilingPrefix, tiling,   \
                      &tPipe);                                                                                         \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
        SyncAll(); /* workspace改为每个核单独使用即可去掉此处同步 */                                                      \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, &tiling_data->tilingBase, tiling, nullptr);                                          \
        op.Process();                                                                                                  \
        op.ProcessSysPrefixCombine();                                                                                  \
    } while (0)

#define INVOKE_IFA_GENERAL_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA(tiling, NEED_CUBE_TILING);                                                                    \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                       \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, tiling, &tPipe);                                                        \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_ALL_VEC_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_NO_CUBE(tiling);                                                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, &tPipe);                                                                \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     user);                                                                                            \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_NO_KFC_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_ALL(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, tiling, &tPipe);                                                        \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_NO_KFC_DD_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_ALL(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
                attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_ANTIQUANT_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingAtbDataV2, tiling_data_in, tiling);                       \
        const IncreFlashAttentionTilingAtbDataV2 *__restrict tiling_data = &tiling_data_in;                            \
        op.Init(query, key, value, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, attentionOut, user,     \
                tiling_data);                                                                                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#ifdef FIA_ENABLE_MLA
#define INVOKE_IFA_NO_KFC_MLA_OP_IMPL(templateClass, ...)                                                              \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_MLA(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
            queryRope, keyRope, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                 \
                     keyRopeAntiquantScale, dequantScaleQuery, user);                                                  \
        op.Process();                                                                                                  \
    } while (0)

#else
#define INVOKE_IFA_NO_KFC_MLA_OP_IMPL(templateClass, ...)  do {} while (0)
#endif

#define COPY_TILING_DATA_ALL(tiling)                                                                                   \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr

#define INVOKE_IFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingAtbDataV2, tiling_data_in, tiling);                       \
        const IncreFlashAttentionTilingAtbDataV2 *__restrict tiling_data = &tiling_data_in;                            \
        op.Init(query, key, value, pseShift, actualSeqLengths, blocktable, attentionOut, user, tiling_data);           \
        op.Process();                                                                                                  \
    } while (0)

#define COPY_TILING_DATA_MLA(tiling)                                                                             \
    GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingDataMla, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingDataMla *__restrict tiling_data = &tiling_data_in

#ifdef __DAV_C220_CUBE__
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);     \
    const IncreFlashAttentionTilingData *__restrict tiling_data = nullptr;                                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingPrefix.base.bmm1TilingData, bmm1TilingDataVarPrefix, \
                        tiling);                                                                                    \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingPrefix.base.bmm2TilingData, bmm2TilingDataVarPrefix, \
                        tiling);                                                                                    \
    const IncreFlashAttentionTilingDataV2 *__restrict tiling_data = nullptr;                                           \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm1tilingPrefix = &bmm1TilingDataVarPrefix;                                         \
    const TCubeTiling *__restrict bmm2tilingPrefix = &bmm2TilingDataVarPrefix
#define COPY_TILING_DATA_NO_CUBE(tiling) COPY_TILING_DATA(tiling, NOT_NEED_CUBE_TILING)

#else
#if (__CCE_AICORE__ != 310) && (!defined (__DAV_310R6__))
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingDataV2, tiling_data_in, tiling);                              \
    const IncreFlashAttentionTilingDataV2 *__restrict tiling_data = &tiling_data_in;                                   \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm1tilingPrefix = nullptr;                                                          \
    const TCubeTiling *__restrict bmm2tilingPrefix = nullptr

#define COPY_TILING_DATA_NO_CUBE(tiling)                                                                               \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in
#endif
#endif

inline __aicore__ void incre_flash_attention_FIAS_arch22(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
#if (__CCE_AICORE__ > 200)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_VALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_VALL_TILING);
	TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING);
	TILING_KEY_IS(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_ATB_C1V2_TILING);
    #if TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, false, true, LAYOUT::BSH);
    #endif
#if (__CCE_AICORE__ > 200)
    #if TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, false, false, 
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, half, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, false,
                                   LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, half, half, false, true,
                                   LAYOUT::BSH, 0, true);
    #endif
#else
    #if TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, false, false, 
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionMulAttenCube310P, half, half, half, half, false, false, 
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionMulAttenCube310P, half, half, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, half, half, half, true, true,
                                   LAYOUT::BSH);
	#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING
		INVOKE_IFA_NEW_GQA_OP_IMPL(PagedAttentionDecoderMask, half, half, half, half, true, false, LAYOUT::BNSD,
                                    false, false, LAYOUT::BNSD, AMLAMODE::NORMAL, false, false, IncreFlashAttentionTilingAtbDataV2);
	#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_ATB_C1V2_TILING
		INVOKE_IFA_NEW_GQA_OP_IMPL(PagedAttentionDecoderMask, half, half, half, half, true, false, LAYOUT::BSND,
                                    false, false, LAYOUT::BSND, AMLAMODE::NORMAL, false, false, IncreFlashAttentionTilingAtbDataV2);
    #endif
#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8)
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_VALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING);
#if (__CCE_AICORE__ > 200)
    #if TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, false,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                  LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, true,
                                  LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                  LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, true,
                                  LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                  LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_KVDEPART_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_KVDEPART_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, true,
                                  LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false,
                                          false, LAYOUT::BSH, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, half, half, false, true,
                                          LAYOUT::BSH, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_CALL_TILING 
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING
        INVOKE_IFA_ANTIQUANT_OP_IMPL(PagedAttentionAntiquant, half, int8_t, half, half, true, false, LAYOUT::TND, false,
                                     false, LAYOUT::BSND, AMLAMODE::NORMAL, false, false, IncreFlashAttentionTilingAtbDataV2);
    #endif
#else
    #if TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_VALL_TILING // kvDeq
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionMulAttenCube310P, half, int8_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_VALL_TILING // pageAttention + kvDeq
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionMulAttenCube310P, half, int8_t, half, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_VALL_TILING
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, half, int8_t, half, half, true, true,
                                   LAYOUT::BSH);
    #endif
#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT4)
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
#if (__CCE_AICORE__ > 200)
    #if TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, true, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, true, false,
                                   LAYOUT::BSH, 1);                                  
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_KVDEPART_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT4_OUTF16_KVDEPART_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, false,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false, true,
                                   LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int4b_t, half, half, false, true,
                                  LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          false, LAYOUT::BSH, 2, true);
    #elif TILING_KEY_VAR == QF16_KVINT4_OUTF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int4b_t, half, half, false,
                                          true, LAYOUT::BSH, 2, true);
    #endif
#endif
#endif

#if (__CCE_AICORE__ > 200) // new template

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
     #if TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_CALL_TILING // A16W16 fp16, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING // A16W16 fp16, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING // A16W16 fp16, out int8, FD
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, false, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING // A16W16 fp16, out int8, FD
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, true, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING // A16W16 fp16, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING // A16W16 fp16, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, false, true,
                                  LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, half, int8_t, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING // A16W16 fp16, out int8, FD
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING // A16W16 fp16, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, half, int8_t, half, false,
                                          true, LAYOUT::BSH, 0, true);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_INT8)
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    #if TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V2_TILING // A16W16 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_CALL_TILING // A16W16 fp16, out int
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING // A16W8 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING // A16W8 fp16, out int
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, true,
                                  LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING // A16W8 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING // A16W8 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, true,
                                  LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BSH);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V2_TILING // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_CALL_TILING // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, true,
                                  LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_CALL_TILING // A16W8 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING // A16W8 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, false,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, false, true,
                                  LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, true, true,
                                   LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, half, int8_t, int8_t, half, true, true,
                                  LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING // A16W8 fp16, out int
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING // A16W8 fp16, pertoken, out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING // A16W8 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, half, int8_t, int8_t, half, false,
                                          true, LAYOUT::BSH, 1, true);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    #if TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t, 
                                  bfloat16_t, true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t,
                                          bfloat16_t, bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8)
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_TND_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_TND_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_TND_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_TND_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_PREDD_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING);
    #if TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  true, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                   true, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                  true, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t,
                                    bfloat16_t, true, false, LAYOUT::BNSD, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t,
                                bfloat16_t, true, false, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_TND_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t,
                                bfloat16_t, true, false, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::BNSD, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_TND_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, false, LAYOUT::BNSD, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, false, LAYOUT::BSH, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_TND_PAGEDCACHE_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, false, LAYOUT::TND, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::BNSD, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::BSH, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_TND_PAGEDCACHE_FLASHDECODING_PREDD_TILING
        INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t,
                                    true, true, LAYOUT::TND, true, false, LAYOUT::NZ, AMLAMODE::NORMAL);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_ATB_C1V2_TILING
        INVOKE_IFA_ANTIQUANT_OP_IMPL(PagedAttentionAntiquant, bfloat16_t, int8_t, bfloat16_t, bfloat16_t, true, false,
                                     LAYOUT::TND, false, false, LAYOUT::BSND, AMLAMODE::NORMAL, false, false,
                                     IncreFlashAttentionTilingAtbDataV2);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT4)
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    #if TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t, 
                                    true, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t, 
                                    true, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t, 
                                    true, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_KVDEPART_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT4_OUTBF16_KVDEPART_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int4b_t, bfloat16_t, bfloat16_t,
                                  false, true, LAYOUT::BSH, 2);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 2, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 2, true);
    #elif TILING_KEY_VAR == QBF16_KVINT4_OUTBF16_KVDEPART_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int4b_t, bfloat16_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 2, true);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_BF16)
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    #if TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_C1V2_TILING // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                  true, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                   true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, bfloat16_t, int8_t, bfloat16_t,
                                  true, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING // A16W16 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVBF16_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, bfloat16_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY == DT_INT8)
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING);
    TILING_KEY_IS(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING);
    #if TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V2_TILING // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_CALL_TILING // A16W8 BF16 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_CALL_TILING // A16W8 BF16 out int8
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                  true, LAYOUT::BNSD);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                  false, true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                  true, LAYOUT::BSH);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V2_TILING // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_CALL_TILING // A16W8 BF16 out int8 per token
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_CALL_TILING // A16W8 BF16 out int8 per token
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                  false, true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                  true, LAYOUT::BNSD, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING
        KERNEL_TASK_TYPE(QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_C1V1_TILING, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   false, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                   false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t,
                                  false, true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                   true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_PAGEDCACHE_FLASHDECODING_CALL_TILING
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, bfloat16_t, int8_t, int8_t, bfloat16_t, true,
                                  true, LAYOUT::BSH, 1);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_SHAPREFIX_C1V2_TILING // A16W8 BF16 out int8
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERCHANNEL_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 0, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_SHAPREFIX_C1V2_TILING // A16W8 BF16 out int8 per token
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BNSD, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, false, LAYOUT::BSH, 1, true);
    #elif TILING_KEY_VAR == QBF16_KVINT8_OUTINT8_ANTIPERTOKEN_BSH_FLASHDECODING_SHAPREFIX_C1V2_TILING
        INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(IncreFlashAttentionAttenSplitBbn2s2Us2, bfloat16_t, int8_t, int8_t,
                                          bfloat16_t, false, true, LAYOUT::BSH, 1, true);
    #endif

#endif

#if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8)
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // sparseMode == 9 (tree mask) variants, enableTree isolated via ENABLE_TREE template parameter
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_LIST(SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING);
    KERNEL_TASK_TYPE(SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    #if (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::BSH, false, false, LAYOUT::NZ);
    #elif (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::BSH, false, false, LAYOUT::NZ);
    #elif (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true);
    #elif (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true);
    #elif (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true);
    #elif (TILING_KEY_VAR == KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING) || (TILING_KEY_VAR == C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLA_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, false, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLA_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, false, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_BSH_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::BSH, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        false, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true, true);
    #elif (TILING_KEY_VAR == SPARSE9_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING) || (TILING_KEY_VAR == SPARSE9_C1V1_KVINT8_OUTBF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_MLANORMAL_TILING)
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, int8_t, int8_t, bfloat16_t, bfloat16_t, true,
                                        true, LAYOUT::TND, false, false, LAYOUT::NZ, AMLAMODE::NORMAL, true, true);
    #endif
#endif

#endif // new template
}