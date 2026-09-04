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
 * \file prompt_flash_attention_zero_output.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_ZERO_OUTPUT_H
#define PROMPT_FLASH_ATTENTION_ZERO_OUTPUT_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
using namespace AscendC;

template<typename T>
class PromptFlashAttentionZeroOutPut {
public:
    __aicore__ inline PromptFlashAttentionZeroOutPut() {};
    __aicore__ inline void Init(__gm__ uint8_t*  attentionOut,
                                const PromptFlashAttentionTilingData* __restrict tiling);
    __aicore__ inline void Process();

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionZeroOutPut<T>::Init(__gm__ uint8_t*  attentionOut,
                                                               const PromptFlashAttentionTilingData* __restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionZeroOutPut<T>::Process() {
        uint32_t tmp_block_idx = GetBlockIdx();
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<T>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
}
#endif  // PROMPT_FLASH_ATTENTION_ZERO_OUTPUT_H