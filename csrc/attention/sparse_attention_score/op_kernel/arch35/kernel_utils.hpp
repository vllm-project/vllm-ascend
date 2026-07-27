/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SASA_ARCH35_KERNEL_UTILS
#define SASA_ARCH35_KERNEL_UTILS

#include "../attn_infra/base_defs.hpp"
#include "../attn_infra/arch/arch.hpp"
#include "../attn_infra/layout/layout.hpp"

#include "../attn_infra/gemm/block/block_mmad.hpp"
#include "../attn_infra/gemm/dispatch_policy.hpp"
#include "../attn_infra/gemm/gemm_type.hpp"

#include "../attn_infra/arch/cross_core_sync.hpp"
#include "../attn_infra/arch/resource.hpp"
#include "../attn_infra/epilogue/block/block_epilogue.hpp"
#include "../attn_infra/epilogue/dispatch_policy.hpp"
#include "../tla/tensor.hpp"
#include "../tla/layout.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SasaKernelArch35 {

enum class Format {
    TND = 0,
    BNSD = 1,
    BSND = 2
};

__aicore__ inline
uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    uint32_t xBlockNum = static_cast<uint32_t>((curQSeqlen + blockShapeX - 1) / blockShapeX);
    uint32_t qSTileNumPerFullXBlock = (blockShapeX + qBaseTile - 1) / qBaseTile;
    uint32_t lastXBlockSize = curQSeqlen - (xBlockNum - 1) * blockShapeX;
    uint32_t lastXBlockQSTileNum = (lastXBlockSize + qBaseTile - 1) / qBaseTile;
    return (xBlockNum - 1) * qSTileNumPerFullXBlock + lastXBlockQSTileNum;
}

struct SasaKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR selectIdx;
    GM_ADDR blockTable;
    GM_ADDR selectNumIdx;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR softmaxLse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    __aicore__ inline
    SasaKernelParamsArch35() {}
    __aicore__ inline
    SasaKernelParamsArch35(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_,
        GM_ADDR selectIdx_, GM_ADDR blockTable_, GM_ADDR selectNumIdx_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
        GM_ADDR o_, GM_ADDR softmaxLse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), selectIdx(selectIdx_), blockTable(blockTable_),
        selectNumIdx(selectNumIdx_), actualQseqlen(actualQseqlen_),
        actualKvseqlen(actualKvseqlen_), o(o_), softmaxLse(softmaxLse_),
        workSpace(workSpace_), tiling(tiling_) {}
};

struct SasaFullQuantKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR selectIdx;
    GM_ADDR blockTable;
    GM_ADDR selectNumIdx;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR qDequantScale;
    GM_ADDR kDequantScale;
    GM_ADDR vDequantScale;
    GM_ADDR o;
    GM_ADDR softmaxLse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    __aicore__ inline
    SasaFullQuantKernelParamsArch35() {}
    __aicore__ inline
    SasaFullQuantKernelParamsArch35(
        GM_ADDR q_, GM_ADDR k_, GM_ADDR v_,
        GM_ADDR selectIdx_, GM_ADDR blockTable_, GM_ADDR selectNumIdx_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
        GM_ADDR qDequantScale_, GM_ADDR kDequantScale_, GM_ADDR vDequantScale_,
        GM_ADDR o_, GM_ADDR softmaxLse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), selectIdx(selectIdx_), blockTable(blockTable_),
          selectNumIdx(selectNumIdx_), actualQseqlen(actualQseqlen_),
          actualKvseqlen(actualKvseqlen_), qDequantScale(qDequantScale_),
          kDequantScale(kDequantScale_), vDequantScale(vDequantScale_),
          o(o_), softmaxLse(softmaxLse_), workSpace(workSpace_), tiling(tiling_) {}
};

__aicore__ inline
uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline
int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline
uint32_t RoundUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

}

#endif
