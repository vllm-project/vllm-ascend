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
 * \file turbo_quant_compress_latent.cpp
 * \brief
 */
#include "turbo_quant_compress_latent.h"

extern "C" __global__ __aicore__ void turbo_quant_compress_latent(GM_ADDR latent, GM_ADDR centroids, GM_ADDR slot,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TurboQuantCompressLatent::KernelTurboQuantCompressLatent op;
    op.Init(latent, centroids, slot, tilingData.numTokens, tilingData.tokensPerCore, tilingData.headDim,
            tilingData.slotSize, tilingData.tokensPerBatch, tilingData.outputMode);
    op.Process();
}
