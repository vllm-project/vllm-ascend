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
 * \file compressor.cpp
 * \brief
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/compressor_v2_kernel_full_load.h"
#include "arch22/compressor_v2_kernel_perf.h"
#else
#include "arch35/compressor_v2_kernel.h"
#endif

using namespace CompressorV2;

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, isFullLoad, ...) \
    do { \
        templateClass<COMPType<__VA_ARGS__>, isFullLoad> op(&pipe, tilingData); \
        op.Init(x, wKv, wGate, stateCache, stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut, workspace); \
        op.Process(); \
    } while (0)

template <uint8_t XLayout, uint8_t XDType, uint8_t TemplateId>
__global__ __aicore__ void compressor_v2(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                         __gm__ uint8_t *stateCache, __gm__ uint8_t *stateBlockTable,
                                         __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                         __gm__ uint8_t *cmpKvOut, __gm__ uint8_t *stateCacheOut,
                                         __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    (void)stateCacheOut; // state_cache 为 in-place，kernel 直接写 stateCache(input)
    REGISTER_TILING_DEFAULT(optiling::CompressorV2TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorV2TilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::CompressorV2TilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
#if (__CCE_AICORE__ == 220)
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        CompressorV2KernelFullLoad<COMPType<xLayout, xDtype>> op(&pipe, tilingData);
        op.Init(x, wKv, wGate, stateCache, stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);
        op.Process();
    } else {
        CompressorV2KernelPerf<COMPType<xLayout, xDtype>> op(&pipe, tilingData);
        op.Init(x, wKv, wGate, stateCache, stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);
        op.Process();
    }
#else
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorV2Kernel, true, xLayout, xDtype);
    } else {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorV2Kernel, false, xLayout, xDtype);
    }
#endif
}
