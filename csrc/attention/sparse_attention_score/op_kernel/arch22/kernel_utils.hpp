/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SASA_ARCH22_KERNEL_UTILS
#define SASA_ARCH22_KERNEL_UTILS

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
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
namespace SasaKernelArch22 {

struct SasaKernelParamsArch22 {
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
    SasaKernelParamsArch22() {}
    __aicore__ inline
    SasaKernelParamsArch22(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_,
        GM_ADDR selectIdx_, GM_ADDR blockTable_, GM_ADDR selectNumIdx_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
        GM_ADDR o_, GM_ADDR softmaxLse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), selectIdx(selectIdx_), blockTable(blockTable_),
        selectNumIdx(selectNumIdx_), actualQseqlen(actualQseqlen_),
        actualKvseqlen(actualKvseqlen_), o(o_), softmaxLse(softmaxLse_),
        workSpace(workSpace_), tiling(tiling_) {}
};

}  // namespace SasaKernelArch22

#endif  // SASA_ARCH22_KERNEL_UTILS