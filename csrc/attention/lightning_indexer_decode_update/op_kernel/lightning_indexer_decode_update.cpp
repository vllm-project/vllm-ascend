/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_decode_update.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_decode_update_template_tiling_key.h"
#include "lightning_indexer_decode_update_kernel.h"

using namespace LIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<LIType<__VA_ARGS__>> op;                                                                         \
        LI_COPY_TILING_DATA(LIU2TilingData, tiling);                                                                  \
        op.Init(query, key, weights, reqPoolEntries, cacheSlots, rowModes, actualSeqLengths, blocktable,          \
                topkIndex, topkSlots, missCount, tailInfo,                                                        \
                user, tiling_data, &tPipe);                                                                           \
        op.Process();                                                                                                  \
    } while (0)

#define LI_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                  \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;


template <int DT>
__global__ __aicore__ void lightning_indexer_decode_update(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                             __gm__ uint8_t *rowModes,
                                             __gm__ uint8_t *actualSeqLengths,
                                             __gm__ uint8_t *blocktable,
                                             __gm__ uint8_t *topkIndex, __gm__ uint8_t *topkSlots,
                                             __gm__ uint8_t *missCount, __gm__ uint8_t *tailInfo,
                                             __gm__ uint8_t *workspace,
                                             __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if constexpr (DT == LI_TPL_FP16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, half);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(LIPreload, bfloat16_t);
    }
#endif
}
