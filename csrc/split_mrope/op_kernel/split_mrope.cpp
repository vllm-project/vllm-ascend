/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lib/matmul_intf.h"
#include "kernel_operator.h"

#include "split_mrope.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void split_mrope(
    GM_ADDR positions,
    GM_ADDR in_qkv,
    GM_ADDR in_cos_sin_cache,
    GM_ADDR out_query,
    GM_ADDR out_key,
    GM_ADDR out_value,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;

    if (TILING_KEY_IS(20)) {
        SplitMrope::SplitMropeBF16<bfloat16_t> op;
        op.Init(positions, in_qkv, in_cos_sin_cache, out_query, out_key, out_value, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(21)) {
        SplitMrope::SplitMropeBF16<half> op;
        op.Init(positions, in_qkv, in_cos_sin_cache, out_query, out_key, out_value, &tiling_data, &pipe);
        op.Process();
    }
}