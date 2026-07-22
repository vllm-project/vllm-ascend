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

#include "compressor_kernel.h"
// #if (__CCE_AICORE__ == 220)
#include "compressor_kernel_perf.h"
// #endif

using namespace Compressor;

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId,
         uint8_t RopeDType>
__aicore__ inline void RunCompressorKernel(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *slotMapping,
    __gm__ uint8_t *pagedKvCache,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }

    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;

    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::PERF) {
        CompressorKernelPerf<COMPType<static_cast<X_LAYOUT>(XLayout), static_cast<X_DTYPE>(XDType),
            static_cast<ROPE_DTYPE>(RopeDType), static_cast<COFF>(Coff), static_cast<ROTARY_MODE>(RotaryMode)>> op(
            &pipe, tilingData);
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens,
                seqUsed, startPos, slotMapping, pagedKvCache, cmpKvOut, workspace);
        op.Process();
    } else {
        CompressorKernel<COMPType<static_cast<X_LAYOUT>(XLayout), static_cast<X_DTYPE>(XDType),
            static_cast<ROPE_DTYPE>(RopeDType), static_cast<COFF>(Coff), static_cast<ROTARY_MODE>(RotaryMode)>> op(
            &pipe, tilingData);
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens,
                seqUsed, startPos, slotMapping, pagedKvCache, cmpKvOut, workspace);
        op.Process();
    }
}

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId, uint8_t RopeDType>
__global__ __aicore__ void compressor(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    RunCompressorKernel<XLayout, XDType, Coff, RotaryMode, CacheMode, TemplateId, RopeDType>(
        x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed,
        startPos, nullptr, nullptr, cmpKvOut, workspace, tiling);
}

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId, uint8_t RopeDType>
__global__ __aicore__ void fused_compressor_and_scatter_nd_update_v2(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *slotMapping,
    __gm__ uint8_t *pagedKvCache,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    __gm__ uint8_t *pagedKvCacheOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    RunCompressorKernel<XLayout, XDType, Coff, RotaryMode, CacheMode, TemplateId, RopeDType>(
        x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed,
        startPos, slotMapping, pagedKvCache, cmpKvOut, workspace, tiling);
}
