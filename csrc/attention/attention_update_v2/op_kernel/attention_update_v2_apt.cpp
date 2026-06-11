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
* \file attention_update_v2_apt.cpp
* \brief
*/
#include "arch35/attention_update_v2_regbase.h"

#define TILING_KEY_GO_FP32_WITHOUT_MAX_OUT 20010
#define TILING_KEY_GO_FP32_WITH_MAX_OUT 20011
#define TILING_KEY_GO_FP16_WITHOUT_MAX_OUT 20020
#define TILING_KEY_GO_FP16_WITH_MAX_OUT 20021
#define TILING_KEY_GO_BF16_WITHOUT_MAX_OUT 20030
#define TILING_KEY_GO_BF16_WITH_MAX_OUT 20031

using namespace AscendC;
using namespace AttentionUpdateV2Opt;

extern "C" __global__ __aicore__ void attention_update_v2(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax, GM_ADDR workSpace, GM_ADDR tiling) {
   TPipe pipe;
   GET_TILING_DATA_WITH_STRUCT(AttentionUpdateV2TilingData, tilingData, tiling);
   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
   if (TILING_KEY_IS(TILING_KEY_GO_FP32_WITHOUT_MAX_OUT)) {
       AttentionUpdateV2<float, float, false> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   } else if (TILING_KEY_IS(TILING_KEY_GO_FP32_WITH_MAX_OUT)) {
       AttentionUpdateV2<float, float, true> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   } else if (TILING_KEY_IS(TILING_KEY_GO_FP16_WITHOUT_MAX_OUT)) {
       AttentionUpdateV2<float, half, false> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   } else if (TILING_KEY_IS(TILING_KEY_GO_FP16_WITH_MAX_OUT)) {
       AttentionUpdateV2<float, half, true> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   }else if (TILING_KEY_IS(TILING_KEY_GO_BF16_WITHOUT_MAX_OUT)) {
       AttentionUpdateV2<float, bfloat16_t, false> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   } else if (TILING_KEY_IS(TILING_KEY_GO_BF16_WITH_MAX_OUT)) {
       AttentionUpdateV2<float, bfloat16_t, true> op(&pipe, &tilingData);
       op.Init(lse, go, out, outLseMax, workSpace);
       op.Process();
   }
}