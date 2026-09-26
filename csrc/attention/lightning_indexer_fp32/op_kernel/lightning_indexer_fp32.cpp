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
 * \file lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_fp32_template_tiling_key.h"

#if __CCE_AICORE__ != 220
#error "LightningIndexerFp32 supports the Ascend 910B/arch22 target only"
#endif

// The package flattens op_kernel; source builds retain it.
#if __has_include("../../lightning_indexer/op_kernel/arch22/lightning_indexer_kernel.h")
#include "../../lightning_indexer/op_kernel/arch22/lightning_indexer_kernel.h"
#else
#include "../lightning_indexer/arch22/lightning_indexer_kernel.h"
#endif

using namespace LIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<LIType<__VA_ARGS__>> op;                                                                         \
        GET_TILING_DATA_WITH_STRUCT(LITilingData, tiling_data_in, tiling);                                             \
        const LITilingData *__restrict tiling_data = &tiling_data_in;                                                  \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, sparseIndices, sparseValues, user,             \
                tiling_data, &tPipe);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void lightning_indexer_fp32(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                             __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                             __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
                                             __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16 && DT_OUT == LI_TPL_INT32) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, half, half, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG, float);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG, float);
    }
}
