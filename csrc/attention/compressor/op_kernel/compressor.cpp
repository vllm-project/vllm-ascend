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
 * \file compressor.cpp
 * \brief
 */

#include "compressor_kernel.h"
#if (__CCE_AICORE__ == 220)
#include "compressor_kernel_perf.h"
#endif
 
using namespace Compressor;

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData);                                                    \
        op.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, kvBlockTable, scoreBlockTable,  \
                cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);                                                    \
        op.Process();                                                                                                  \
    } while (0)

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t TemplateId>
__global__ __aicore__ void compressor(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *kvState,
    __gm__ uint8_t *scoreState,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *kvBlockTable,
    __gm__ uint8_t *scoreBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *kvStateOut,
    __gm__ uint8_t *scoreStateOut,
    __gm__ uint8_t *wkvProjOut,
    __gm__ uint8_t *softmaxResOut,
    __gm__ uint8_t *normXOut,
    __gm__ uint8_t *normRstdOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto rotaryMode = static_cast<ROTARY_MODE>(RotaryMode);
#if (__CCE_AICORE__ == 220)
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::PERF) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, xLayout, xDtype, coff, rotaryMode);
    } else {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode);
    }
#else
    INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode);
#endif
}