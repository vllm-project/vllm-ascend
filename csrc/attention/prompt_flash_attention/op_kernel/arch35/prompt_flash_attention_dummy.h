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
 * \file prompt_flash_attention_dummy.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_DUMMY_H
#define PROMPT_FLASH_ATTENTION_DUMMY_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

template<typename T>
class PromptFlashAttentionDummy {
public:
    __aicore__ inline PromptFlashAttentionDummy() {};
    __aicore__ inline void Init(__gm__ uint8_t* attentionOut, const FlashAttentionScoreSimplifiedTilingData* __restrict tiling);
    __aicore__ inline void Process();

protected:
    const FlashAttentionScoreSimplifiedTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionDummy<T>::Init(__gm__ uint8_t *attentionOut,
    const FlashAttentionScoreSimplifiedTilingData* __restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionDummy<T>::Process() {
    uint32_t blockIdx = GetBlockIdx();
}
#endif  // PROMPT_FLASH_ATTENTION_DUMMY_H