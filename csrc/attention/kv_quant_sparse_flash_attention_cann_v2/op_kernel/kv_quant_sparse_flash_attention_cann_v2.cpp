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
 * \file kv_quant_sparse_flash_attention_cann_v2.cpp
 * \brief
 */

#define KVQSFA_VERSION 2

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kv_quant_sparse_flash_attention_cann_v2_template_tiling_key.h"
#include "arch35/kv_quant_sparse_flash_attention_cann_v2_kernel_mla_arch35.h"

using namespace AscendC;

#if defined(__DAV_C310_CUBE__)
#define QSFA_V2_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QSFAMatmulService<__VA_ARGS__>, \
                                      BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockTypeV2 = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, \
                                      BaseApi::QSFAVectorService<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockTypeV2> opV2; \
        opV2.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable, actualSeqLengthsQuery, \
                  actualSeqLengthsKV, sinks, attentionOut, softmaxMax, softmaxSum, user, nullptr, &tPipe); \
        opV2.Process(); \
    } while (0)
#else
#define QSFA_V2_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QSFAMatmulService<__VA_ARGS__>, \
                                      BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockTypeV2 = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, \
                                      BaseApi::QSFAVectorService<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockTypeV2> opV2; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tilingDataInV2, tiling); \
        const tilingdataClass *__restrict tilingDataV2 = &tilingDataInV2; \
        opV2.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable, actualSeqLengthsQuery, \
                  actualSeqLengthsKV, sinks, attentionOut, softmaxMax, softmaxSum, user, tilingDataV2, &tPipe); \
        opV2.Process(); \
    } while (0)
#endif

template <int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G,
          int IS_VEC_S2PHYADDR>
__aicore__ inline void DispatchKernelDtype310V2(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *keyScale,
                                                __gm__ uint8_t *valueScale, __gm__ uint8_t *blocktable,
                                                __gm__ uint8_t *actualSeqLengthsQuery,
                                                __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *sinks,
                                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *user,
                                                __gm__ uint8_t *tiling, TPipe &tPipe)
{
    constexpr auto kvDtype = ORIG_DTYPE_KEY;
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && kvDtype == DT_FLOAT8_E4M3FN &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, bfloat16_t,
                        fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION,
                        static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
                        static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G, IS_VEC_S2PHYADDR);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && kvDtype == DT_HIFLOAT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, bfloat16_t,
                        hifloat8_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION, static_cast<QSFA_LAYOUT>(LAYOUT_T),
                        static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G,
                        IS_VEC_S2PHYADDR);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && kvDtype == DT_INT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, bfloat16_t,
                        int8_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION, static_cast<QSFA_LAYOUT>(LAYOUT_T),
                        static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G,
                        IS_VEC_S2PHYADDR);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && kvDtype == DT_FLOAT8_E4M3FN &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, half,
                        fp8_e4m3fn_t, float, half, FLASH_DECODE, PAGE_ATTENTION, static_cast<QSFA_LAYOUT>(LAYOUT_T),
                        static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G,
                        IS_VEC_S2PHYADDR);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && kvDtype == DT_HIFLOAT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, half,
                        hifloat8_t, float, half, FLASH_DECODE, PAGE_ATTENTION, static_cast<QSFA_LAYOUT>(LAYOUT_T),
                        static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G,
                        IS_VEC_S2PHYADDR);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && kvDtype == DT_INT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_V2_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionCannV2Mla, KvQuantSparseFlashAttentionCannV2TilingDataMla, half, int8_t,
                        float, half, FLASH_DECODE, PAGE_ATTENTION, static_cast<QSFA_LAYOUT>(LAYOUT_T),
                        static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G,
                        IS_VEC_S2PHYADDR);
    }
}

template <int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G,
          int IS_VEC_S2PHYADDR>
__global__ __aicore__ void kv_quant_sparse_flash_attention_cann_v2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *sinks,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    DispatchKernelDtype310V2<FLASH_DECODE, PAGE_ATTENTION, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G,
                             IS_VEC_S2PHYADDR>(query, key, value, sparseIndices, keyScale, valueScale, blocktable,
                                               actualSeqLengthsQuery, actualSeqLengthsKV, sinks, attentionOut,
                                               softmaxMax, softmaxSum, user, tiling, tPipe);
}
