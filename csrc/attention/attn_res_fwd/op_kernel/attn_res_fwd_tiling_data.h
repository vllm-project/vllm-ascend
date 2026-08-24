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
 * \file attn_res_fwd_tiling_data.h
 * \brief AttnResFwd tiling data structure
 */
#ifndef ATTN_RES_FWD_TILING_DATA_H
#define ATTN_RES_FWD_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace AttnResFwd {
#pragma pack(push, 8)
struct alignas(8) AttnResFwdTilingData {
    uint32_t numTokens;
    uint32_t numBlocks;
    uint32_t hiddenSize;
    uint32_t tokensPerCore;
    uint32_t usedCoreNum;
    float normEps;
    float invHiddenSize;
    uint64_t wsSizePerToken;
    uint32_t blockCount;
    uint32_t needBackward;     // 0/1
    uint32_t stagingBytes;     // 512 倍数；false 时为 0
    uint32_t tokensPerFlush;   // staging 可攒 token 数；false 时为 0
    uint32_t elemsPerToken;    // 2*blockCount；false 时可写 0
};
#pragma pack(pop)
} // namespace AttnResFwd

#endif // ATTN_RES_FWD_TILING_DATA_H
