/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_grouped_matmul.h"
#include "kernel_operator.h"

#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat formatWeight = CubeFormat::NZ;
#else
constexpr CubeFormat formatWeight = CubeFormat::ND;
#endif

//using namespace matmul;
#define GMM_CUBE_IMP(transWeight)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        GET_TILING_DATA(tiling_data, tiling);                                                                      \
        AscendC::TPipe pipe;                                                                                       \
        KernelMoeGMMNoQuant<DTYPE_X, DTYPE_GROUP_LIST, formatWeight, transWeight> op(&pipe);                       \
        op.Init(x, weight, group_list, y, &tiling_data);                                                     \
        op.Process();                                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void moe_grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y,
                   GM_ADDR workSpace, GM_ADDR tiling) {

  if (TILING_KEY_IS(10UL)) {
    GMM_CUBE_IMP(false);
  } else if (TILING_KEY_IS(11UL)) {
    GMM_CUBE_IMP(true);
  }
}
