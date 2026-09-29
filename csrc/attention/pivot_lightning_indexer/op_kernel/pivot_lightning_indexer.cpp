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
 * \file pivot_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pivot_lightning_indexer_template_tiling_key.h"

#include "arch22/pivot_lightning_indexer_kernel.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/pivot_lightning_indexer_pivot_reuse.h"
#endif

using namespace LIKernel;

#if (__CCE_AICORE__ == 220)
#define TRY_PIVOT_REUSE(...)                                                                                           \
    if (TryPivotReuse<LIType<__VA_ARGS__>>(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable,        \
            sparseIndices, sparseValues, user, tiling_data_in, tPipe)) {                                                \
        break;                                                                                                         \
    }
#else
#define TRY_PIVOT_REUSE(...)
#endif

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<LIType<__VA_ARGS__>> op;                                                                         \
        GET_TILING_DATA_WITH_STRUCT(PivotLITilingData, tiling_data_in, tiling);                                         \
        TRY_PIVOT_REUSE(__VA_ARGS__);                                                                                  \
        const PivotLITilingData *__restrict tiling_data = &tiling_data_in;                                              \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, sparseIndices, sparseValues, user, \
                tiling_data, &tPipe);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void pivot_lightning_indexer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                             __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                             __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
                                             __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16 && DT_OUT == LI_TPL_INT32) {
        INVOKE_LI_NO_KFC_OP_IMPL(PivotLightningIndexerKernel, half, half, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(PivotLightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG);
    }
}
