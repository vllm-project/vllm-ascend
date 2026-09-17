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
 * \file key_pool.cpp
 * \brief
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/key_pool_kernel_perf.h"
#include "arch22/key_pool_kernel_full_load.h"
#else
#include "arch35/key_pool_kernel.h"
#include "arch35/key_pool_kernel_full_load.h"
#endif

using namespace KeyPool;

#define INVOKE_KEY_POOL_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData); \
        op.Init(hidden_states, wk, gateWeight, normWeight, normBias, stateCache, ape, cacheBlockTable, seqLens, \
                seqused, startPos, pooledKeyOut, workspace); \
        op.Process(); \
    } while (0)

template <uint8_t HiddenStatesLayout, uint8_t HiddenStatesDtype, uint8_t TemplateId>
__global__ __aicore__ void key_pool(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk, __gm__ uint8_t *gateWeight,
                                    __gm__ uint8_t *ape, __gm__ uint8_t *stateCache, __gm__ uint8_t *cacheBlockTable,
                                    __gm__ uint8_t *startPos, __gm__ uint8_t *normWeight, __gm__ uint8_t *normBias,
                                    __gm__ uint8_t *cos, __gm__ uint8_t *sin, __gm__ uint8_t *seqLens,
                                    __gm__ uint8_t *seqused, __gm__ uint8_t *pooledKeyOut, __gm__ uint8_t *workspace,
                                    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::KeyPoolTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::KeyPoolTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_HIDDEN_STATES) {
        return;
    }
    const optiling::KeyPoolTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto hiddenStatesLayout = static_cast<HIDDEN_STATES_LAYOUT>(HiddenStatesLayout);
    constexpr auto hiddenStatesDtype = static_cast<HIDDEN_STATES_DTYPE>(HiddenStatesDtype);
#if (__CCE_AICORE__ == 220)
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_KEY_POOL_GENERAL_OP_IMPL(KeyPoolKernelFullLoad, hiddenStatesLayout, hiddenStatesDtype);
    } else {
        INVOKE_KEY_POOL_GENERAL_OP_IMPL(KeyPoolKernelPerf, hiddenStatesLayout, hiddenStatesDtype);
    }
#else
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_KEY_POOL_GENERAL_OP_IMPL(KeyPoolKernelFullLoad, hiddenStatesLayout, hiddenStatesDtype);
    } else {
        INVOKE_KEY_POOL_GENERAL_OP_IMPL(KeyPoolKernel, hiddenStatesLayout, hiddenStatesDtype);
    }
#endif
}
