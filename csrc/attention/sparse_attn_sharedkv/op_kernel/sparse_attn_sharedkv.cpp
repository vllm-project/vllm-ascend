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
 * \file sparse_attn_sharedkv.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sparse_attn_sharedkv_template_tiling_key.h"
#include "arch22/sparse_attn_sharedkv_scfa_kernel.h"
#include "arch22/sparse_attn_sharedkv_swa_kernel.h"
#include "sparse_attn_sharedkv_metadata.h"

using namespace AscendC;
using namespace optiling::detail;
using namespace SASKernel;

#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<SASType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                                          \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                                               \
        op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,                       \
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV,                                                   \
                sinks, metadata, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                     \
        op.Process();                                                                                                  \
    } while (0)


template <int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void
sparse_attn_sharedkv(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                     __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                     __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                     __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV,
                     __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_ORI_KV == DT_FLOAT16 && ORIG_DTYPE_ATTN_OUT == DT_FLOAT16) {
        if constexpr (TEMPLATE_MODE == SCFA_TEMPLATE) {
            SAS_OP_IMPL(SparseAttnSharedkvScfa, SparseAttnSharedkvTilingData, half, half, half, FLASH_DECODE,
                        static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE);
        } else {
            SAS_OP_IMPL(SparseAttnSharedkvSwa, SparseAttnSharedkvTilingData, half, half, half, FLASH_DECODE,
                        static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE);
        }
    }
    if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_ORI_KV == DT_BF16 && ORIG_DTYPE_ATTN_OUT == DT_BF16) {
        if constexpr (TEMPLATE_MODE == SCFA_TEMPLATE) {
            SAS_OP_IMPL(SparseAttnSharedkvScfa, SparseAttnSharedkvTilingData, bfloat16_t, bfloat16_t, bfloat16_t,
                        FLASH_DECODE, static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
                        TEMPLATE_MODE);
        } else {
            SAS_OP_IMPL(SparseAttnSharedkvSwa, SparseAttnSharedkvTilingData, bfloat16_t, bfloat16_t, bfloat16_t,
                        FLASH_DECODE, static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
                        TEMPLATE_MODE);
        }
    }
}