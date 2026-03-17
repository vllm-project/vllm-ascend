/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swi_glu_dynamic_quant.cpp
 * \brief
 */

#include "swi_glu_dynamic_quant.h"
#include "swi_glu_dynamic_quant_static.h"
using namespace AscendC;
using namespace SwiGluDynamicQuantOpt;

extern "C" __global__ __aicore__ void swi_glu_dynamic_quant(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR offsets,
                                                    GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  GET_TILING_DATA(tiling_data, tiling);
  if (GetBlockIdx() >= tiling_data.realCoreNum) {
    return;
  }
  TPipe pipe;
  #if (!(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)) && (ORIG_DTYPE_X == DT_BF16)
     if (TILING_KEY_IS(206)) {
      SwiGluDynamicQuant<bfloat16_t, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(204)) {
      SwiGluDynamicQuantStatic<bfloat16_t, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(205)) {
      SwiGluDynamicQuantStatic<bfloat16_t, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT16
    if (TILING_KEY_IS(106)) {
      SwiGluDynamicQuant<half, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
    else if (TILING_KEY_IS(104)) {
      SwiGluDynamicQuantStatic<half, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(105)) {
      SwiGluDynamicQuantStatic<half, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT
    if (TILING_KEY_IS(306)) {
      SwiGluDynamicQuant<float, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(304)) {
      SwiGluDynamicQuantStatic<float, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(305)) {
      SwiGluDynamicQuantStatic<float, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, &tiling_data);
      op.Process();
    }
  #endif
}
