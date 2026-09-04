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
 * \file prompt_flash_attention_empty_tensor.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H
#define PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../kernel_data_copy_transpose.h"

template<typename T>
class PromptFlashAttentionEmptyTensor {
public:
    __aicore__ inline PromptFlashAttentionEmptyTensor() {};
    __aicore__ inline void Init(__gm__ uint8_t *attentionOut, const PromptFlashAttentionTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    TPipe *pipe;
    const PromptFlashAttentionTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionEmptyTensor<T>::Init(__gm__ uint8_t*  attentionOut,
                                                                const PromptFlashAttentionTilingData* __restrict tiling,
                                                                TPipe *tPipe) {
    pipe = tPipe;
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionEmptyTensor<T>::Process() {
    uint32_t tmp_block_idx = GetBlockIdx();
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    int32_t tailSize = (int32_t)initParams.totalOutputSize - tmp_block_idx * (int32_t)initParams.singleCoreSize;
    if (tailSize > 0) {
        uint32_t singleInitOutputSize =
            tailSize < initParams.singleCoreSize ? static_cast<uint32_t>(tailSize) : initParams.singleCoreSize;
        InitOutput<T>(attentionOutGm[tmp_block_idx * (int64_t)initParams.singleCoreSize], singleInitOutputSize, 0);
    }
}
#endif  // PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H