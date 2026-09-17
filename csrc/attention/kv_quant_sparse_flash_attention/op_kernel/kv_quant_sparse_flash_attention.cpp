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
 * \file kv_quant_sparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kv_quant_sparse_flash_attention_template_tiling_key.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/kv_quant_sparse_flash_attention_kernel_mla.h"
#else
#include "kv_quant_sparse_flash_attention_kernel_mla.h"
#endif

using namespace AscendC;

#if (__CCE_AICORE__ == 310)
#if defined(__DAV_C310_CUBE__)
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                                 \
    do {                                                                                                  \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                       \
            BaseApi::QSFAMatmulService<__VA_ARGS__>, BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                        \
            BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, BaseApi::QSFAVectorService<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                    \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,                       \
            actualSeqLengthsQuery, actualSeqLengthsKV,                                                    \
	    attentionOut, user, nullptr, &tPipe);                                                             \
        op.Process();                                                                                     \
    } while (0)
#else
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                                 \
    do {                                                                                                  \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                       \
            BaseApi::QSFAMatmulService<__VA_ARGS__>, BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                        \
            BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, BaseApi::QSFAVectorService<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                    \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tilingDataIn, tiling);                               \
        const tilingdataClass *__restrict tilingData = &tilingDataIn;                                     \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,                       \
            actualSeqLengthsQuery, actualSeqLengthsKV,                                                    \
	    attentionOut, user, tilingData, &tPipe);                                                          \
        op.Process();                                                                                     \
    } while (0)
#endif
#else
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                         \
    do {                                                                                          \
        templateClass<QSFAType<__VA_ARGS__>> op;                                                  \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,               \
            actualSeqLengthsQuery, actualSeqLengthsKV,                                            \
	    attentionOut, softmaxMax, softmaxSum, user, tiling_data, tiling, &tPipe);                 \
        op.Process();                                                                             \
    } while (0)
#endif

#if (__CCE_AICORE__ == 310)
template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
__aicore__ inline void DispatchKernelDtype310(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *user, __gm__ uint8_t *tiling, TPipe &tPipe)
{
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            bfloat16_t, hifloat8_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            bfloat16_t, int8_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            half, fp8_e4m3fn_t, float, half, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            half, hifloat8_t, float, half, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 &&
                         ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla,
            half, int8_t, float, half, FLASH_DECODE, PAGE_ATTENTION,
            static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<QSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
    }
}
#endif

template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
 __global__ __aicore__ void
kv_quant_sparse_flash_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale, __gm__ uint8_t* valueScale,
                       __gm__ uint8_t *blocktable, __gm__ uint8_t *actualSeqLengthsQuery,
                       __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
                       __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                       __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
#if (__CCE_AICORE__ == 310)
    DispatchKernelDtype310<FLASH_DECODE, PAGE_ATTENTION, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G>(
        query, key, value, sparseIndices, keyScale, valueScale, blocktable,
        actualSeqLengthsQuery, actualSeqLengthsKV, attentionOut, user, tiling, tPipe);
#else
    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_OP_IMPL(KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla, half, int8_t,
            half, FLASH_DECODE, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            TEMPLATE_MODE);
    } else { // bf16
        QSFA_OP_IMPL(KvQuantSparseFlashAttentionMla, KvQuantSparseFlashAttentionTilingDataMla, bfloat16_t, int8_t,
            bfloat16_t, FLASH_DECODE, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
            TEMPLATE_MODE);
    }
#endif
}