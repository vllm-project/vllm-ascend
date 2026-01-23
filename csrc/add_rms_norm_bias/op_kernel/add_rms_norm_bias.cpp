/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias.cpp
 * \brief
 */
#include "add_rms_norm_bias.h"
#include "add_rms_norm_bias_split_d.h"
#include "add_rms_norm_bias_merge_n.h"
#include "add_rms_norm_bias_multi_n.h"
#include "add_rms_norm_bias_single_n.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)              \
    do {                                                 \
        templateClass<__VA_ARGS__> op(&pipe);            \
        op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData); \
        op.Process();                                    \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_bias(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, half);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, float);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBias, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, half);
    } else if (TILING_KEY_IS(21)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, float);
    } else if (TILING_KEY_IS(31)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(12)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, half);
    } else if (TILING_KEY_IS(22)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, float);
    } else if (TILING_KEY_IS(32)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(13)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, half);
    } else if (TILING_KEY_IS(23)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, float);
    } else if (TILING_KEY_IS(33)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(14)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMultiN, half);
    } else if (TILING_KEY_IS(34)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMultiN, bfloat16_t);
    }
}