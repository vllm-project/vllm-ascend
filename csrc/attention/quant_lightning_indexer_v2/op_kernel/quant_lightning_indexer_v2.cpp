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
 * \file quant_lightning_indexer_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/quant_lightning_indexer_v2_kernel_arch35.h"
#else
#include "arch22/quant_lightning_indexer_v2_kernel_arch22.h"
#endif
#include "quant_lightning_indexer_v2_template_tiling_key.h"
using namespace QLIV2Kernel;
using namespace optiling::detail;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<QLIV2Type<__VA_ARGS__>> op; \
        op.Init(query, key, weights, queryScale, keyScale, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, \
                blockTable, outputIdxOffset, metadata, sparseIndices, sparseValues, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer_v2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
    __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(QLIV2TilingData, tiling_data_in, tiling);
    const QLIV2TilingData *__restrict tiling_data = &tiling_data_in;

#if (__CCE_AICORE__ == 310)
    constexpr uint32_t QUANT_MODE_FP8 = 1;
    constexpr uint32_t QUANT_MODE_INT8 = 2;
    constexpr uint32_t QUANT_MODE_MXFP8 = 3;
    constexpr uint32_t QUANT_MODE_HIFLOAT8 = 4;
    constexpr uint32_t QUANT_MODE_MXFP4 = 5;
    if (tiling_data->quantMode == QUANT_MODE_FP8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), float, float, float);
    } else if (tiling_data->quantMode == QUANT_MODE_MXFP8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), fp8_e8m0_t, float, float);
    } else if (tiling_data->quantMode == QUANT_MODE_HIFLOAT8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, hifloat8, hifloat8, float, uint16_t, int32_t, PAGE_ATTENTION,
                                 LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), float, float, float);
    } else if (tiling_data->quantMode == QUANT_MODE_MXFP4) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, uint16_t, int32_t,
                                 PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), fp8_e8m0_t, float,
                                 float);
    } else if (tiling_data->quantMode == QUANT_MODE_INT8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, int8_t, int8_t, int32_t, uint16_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                half, half, int32_t);
    }

#else
    INVOKE_LI_NO_KFC_OP_IMPL(QLIV2Preload, int8_t, int8_t, float, uint16_t, int32_t, PAGE_ATTENTION,
                             LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
#endif
}
