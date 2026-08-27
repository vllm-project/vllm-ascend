/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "matmul_blaze_kernel.h"

extern "C" void matmul_blaze_launch(
    aclrtStream stream,
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData tilingData,
    bool transA, bool transB)
{
    if (transA && transB) {
        matmul_blaze_kernel<true, true, Blaze::Gemm::NONE_FULL_LOAD_MODE>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
    } else if (transA && !transB) {
        matmul_blaze_kernel<true, false, Blaze::Gemm::NONE_FULL_LOAD_MODE>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
    } else if (!transA && transB) {
        matmul_blaze_kernel<false, true, Blaze::Gemm::NONE_FULL_LOAD_MODE>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
    } else {
        matmul_blaze_kernel<false, false, Blaze::Gemm::NONE_FULL_LOAD_MODE>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);
    }
}
