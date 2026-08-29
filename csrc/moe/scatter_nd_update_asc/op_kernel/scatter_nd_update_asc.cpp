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
 * \file scatter_nd_update_asc.cpp
 * \brief scatter_nd_update_asc
 */

#include "scatter_nd_update_asc_pure_copy.h"

using namespace ScatterNdUpdateAsc;
using namespace AscendC;

#define TILING_KEY_PURE    0

extern "C" __global__ __aicore__ void scatter_nd_update_asc(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                      GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_PURE)) {
        // 与 ops-nn a2(arch22) 方案一致：isViewStride0 作为编译期模板参数，运行时分发
        const bool isView = tilingData.isViewStride0 != 0;
        if (isView) {
            ScatterNdUpdateAscPureCopy<DTYPE_UPDATES, DTYPE_INDICES, true> op(&pipe);
            op.Init(var, indices, updates, y, &tilingData);
            op.Process();
        } else {
            ScatterNdUpdateAscPureCopy<DTYPE_UPDATES, DTYPE_INDICES, false> op(&pipe);
            op.Init(var, indices, updates, y, &tilingData);
            op.Process();
        }
    }
}