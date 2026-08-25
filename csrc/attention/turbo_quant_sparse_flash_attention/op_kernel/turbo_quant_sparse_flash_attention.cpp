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
 * \file turbo_quant_sparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "turbo_quant_sparse_flash_attention_template_tiling_key.h"
#include "turbo_quant_sparse_flash_attention_kernel_mla.h"

using namespace AscendC;
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        templateClass<QSFAType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable, actualSeqLengthsQuery, \
                actualSeqLengthsKV, attentionOut, softmaxMax, softmaxSum, user, tiling_data, tiling, &tPipe); \
        op.Process(); \
    } while (0)

template <int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
__global__ __aicore__ void turbo_quant_sparse_flash_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    // 算子仅声明支持 BFLOAT16，FP16 分支不会被实例化，故不保留。
    QSFA_OP_IMPL(TurboQuantSparseFlashAttentionMla, TurboQuantSparseFlashAttentionTilingDataMla, bfloat16_t, int8_t,
                 bfloat16_t, FLASH_DECODE, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
                 TEMPLATE_MODE);
}
