/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gumbel_sample.h"
#include "gumbel_sample_tiling_key.h"

// ===========================================================================
// GumbelSample kernel 入口
//   参数顺序：5 输入 + 1 输出 + workspace + tiling
//   TilingKey 单维：applyTemp ∈ {0, 1}（0=不缩放, 1=z/τ 缩放）
// ===========================================================================
template <uint32_t applyTemp>
__global__ __aicore__ void gumbel_sample(
    GM_ADDR logits, GM_ADDR temperature, GM_ADDR seeds, GM_ADDR pos,
    GM_ADDR idxMapping,
    GM_ADDR sampled, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(GumbelSampleTilingData, tilingData, tiling);

    // [opt-1] TPipe 在核函数入口创建，传指针给 Op 类（减少头尾开销）
    TPipe pipe;
    if constexpr (applyTemp == 1) {
        NsGumbelSample::GumbelSampleOp<true> op;
        op.Init(logits, temperature, seeds, pos, idxMapping, sampled, workspace, tilingData, &pipe);
        op.Process();
    } else {
        NsGumbelSample::GumbelSampleOp<false> op;
        op.Init(logits, temperature, seeds, pos, idxMapping, sampled, workspace, tilingData, &pipe);
        op.Process();
    }
}
