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
 * \file paged_select_attention_apt.cpp
 * \brief Specialized sparse paged decode kernel entry for apt builds.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "paged_select_attention_tilingkey.h"
#include "paged_select_attention_kernel_bridge.h"

extern "C" __global__ __aicore__ void paged_select_attention(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *blocktable,
    __gm__ uint8_t *selectedKvIndices,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    FIA_HARD_FAIL_IF(blocktable == nullptr, "paged_select_attention requires blocktable");
    FIA_HARD_FAIL_IF(selectedKvIndices == nullptr, "paged_select_attention requires selectedKvIndices");
    FIA_HARD_FAIL_IF(actualSeqLengths == nullptr, "paged_select_attention requires actualSeqLengths");
    FIA_HARD_FAIL_IF(actualSeqLengthsKV == nullptr, "paged_select_attention requires actualSeqLengthsKV");

    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PAGED_SELECT_COPY_TILING_DATA(PagedSelectAttentionTilingData, tiling);

    FIA_HARD_FAIL_IF(tiling_data->maxNumBlocksPerBatch == 0U,
        "paged_select_attention requires paged block metadata");
    FIA_HARD_FAIL_IF(tiling_data->sparseLaunchEnabled == 0U,
        "paged_select_attention no longer supports dense fallback");
    FIA_HARD_FAIL_IF(tiling_data->sparseKMax == 0U,
        "paged_select_attention requires a non-zero selected_kv_indices width");

    if (TILING_KEY_IS(PAGED_SELECT_ATTENTION_TILING_FP16)) {
        PagedSelectAttentionKernel::Run<half, float>(
            query, key, value, blocktable, selectedKvIndices, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling);
        return;
    }
    if (TILING_KEY_IS(PAGED_SELECT_ATTENTION_TILING_BF16)) {
        PagedSelectAttentionKernel::Run<bfloat16_t, float>(
            query, key, value, blocktable, selectedKvIndices, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling);
        return;
    }

    FIA_HARD_FAIL_IF(true, "unsupported paged_select_attention tiling key=%llu",
        static_cast<unsigned long long>(TILING_KEY_VAR));
}
