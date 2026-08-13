/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_attention_score_tilingkey.h"
#include "sparse_attention_score_kernel_interface.cpp"

extern "C" __global__ __aicore__ void sparse_attention_score(
    __gm__ uint8_t* query,
    __gm__ uint8_t* key,
    __gm__ uint8_t* value,
    __gm__ uint8_t* selectIdx,
    __gm__ uint8_t* blockTable,
    __gm__ uint8_t* selectNumIdx,
    __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv,
    __gm__ uint8_t* qDequantScale,
    __gm__ uint8_t* kDequantScale,
    __gm__ uint8_t* vDequantScale,
    __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* softmaxLse,
    __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR >= SASA_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        TILING_KEY_IS(SASA_FP8_D128_TILING);
        TILING_KEY_IS(SASA_FP16_D128_TILING);
        TILING_KEY_IS(SASA_BF16_D128_TILING);

#if TILING_KEY_VAR == SASA_FP8_D128_TILING  // gitleaks:allow
        SasaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, SasaKernelArch35::Format::TND>(
            query, key, value, selectIdx, blockTable, selectNumIdx,
            actualSeqLengths, actualSeqLengthsKv,
            qDequantScale, kDequantScale, vDequantScale,
            attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == SASA_FP16_D128_TILING  // gitleaks:allow
        SasaInferIntfRegular<half, half, float, SasaKernelArch35::Format::TND>(
            query, key, value, selectIdx, blockTable, selectNumIdx,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == SASA_BF16_D128_TILING  // gitleaks:allow
        SasaInferIntfRegular<bfloat16_t, bfloat16_t, float, SasaKernelArch35::Format::TND>(
            query, key, value, selectIdx, blockTable, selectNumIdx,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#endif
#elif (__CCE_AICORE__ == 220)
        TILING_KEY_IS(SASA_FP16_D128_ARCH22_TILING);
        TILING_KEY_IS(SASA_BF16_D128_ARCH22_TILING);

#if TILING_KEY_VAR == SASA_FP16_D128_ARCH22_TILING
        SasaInferIntfRegularArch22<half, float>(
            query, key, value, selectIdx, blockTable, selectNumIdx,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#elif TILING_KEY_VAR == SASA_BF16_D128_ARCH22_TILING
        SasaInferIntfRegularArch22<bfloat16_t, float>(
            query, key, value, selectIdx, blockTable, selectNumIdx,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#endif
#endif
    }
}
