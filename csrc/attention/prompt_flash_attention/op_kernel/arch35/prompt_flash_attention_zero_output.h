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
#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h")
#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#else
#include "../../common/arch35/flash_attention_score_tiling_regbase.h"
#endif

template<typename T>
class PromptFlashAttentionZeroOutPut {
public:
    __aicore__ inline PromptFlashAttentionZeroOutPut() {};
    __aicore__ inline void Init(__gm__ uint8_t*  attentionOut,
                                __gm__ uint8_t*  softmaxLse,
                                const optiling::FlashAttentionScoreSimplifiedTilingData* __restrict tiling);
    __aicore__ inline void Process();
    static constexpr bool POST_QUANT = !IsSameType<T, half>::value;

protected:
    const optiling::FlashAttentionScoreSimplifiedTilingData* __restrict tilingData;
    GlobalTensor<half> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionZeroOutPut<T>::Init(__gm__ uint8_t*  attentionOut,
                                                               __gm__ uint8_t*  softmaxLse,
                                                               const optiling::FlashAttentionScoreSimplifiedTilingData* __restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ half*)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionZeroOutPut<T>::Process() {
    uint32_t tmp_block_idx = GetBlockIdx();
    auto &initParams = tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if (singleInitOutputSize > 0) {
        if constexpr (POST_QUANT){
            InitOutput<half>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0);
        } else {
            InitOutput<half>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
        }
    }

    auto mte3ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);

    int64_t coreNum = GetBlockNum() * GetTaskRation();
    if (coreNum != 0 && tmp_block_idx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (tmp_block_idx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        if (singleCoreLseSize > 0) {
            InitOutput<float>(softmaxLseGm[tmp_block_idx * (initParams.totalSoftMaxLseOutputSize / coreNum)], 
                singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
        }
    }

    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}
#endif  // PROMPT_FLASH_ATTENTION_ZERO_OUTPUT_H