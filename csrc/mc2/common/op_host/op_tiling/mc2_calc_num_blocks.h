/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_CALC_NUM_BLOCKS_H__
#define __MC2_CALC_NUM_BLOCKS_H__
#include <algorithm>
#include <cstdint>
#include "mc2_log.h"

namespace mc2tiling {
// 根据ascendc提供的aic aiv数量，按照aic:aiv=1:2比例设置逻辑核数量，以aic和aiv/2较小的值为基点
inline uint64_t GetNumBlocks(const uint64_t aicNum, const uint64_t aivNum, const char *debugDesc)
{
    uint64_t halfAivNum = aivNum / 2; // aic:aiv按照1：2配比
    uint64_t numBlocks = std::min(aicNum, halfAivNum);
    if (numBlocks != aicNum || numBlocks != aivNum * 2) { // aic:aiv按照1：2配比
        OP_LOGI(debugDesc,
                "aicNum is %lu, aivNum is %lu. Since aicNum != 2 * aivNum, "
                "the actual number of enabled AIC cores is %lu, and the number of AIV cores is %lu.",
                aicNum, aivNum, numBlocks, numBlocks * 2); // aic:aiv按照1：2配比
    }
    OP_TILING_CHECK((numBlocks == 0U),
                    OP_LOGE_FOR_INVALID_VALUE(debugDesc, "aicNum, aivNum",
                                              (std::to_string(aicNum) + ", " + std::to_string(aivNum)).c_str(),
                                              "numBlocks=min(aicNum, aivNum/2) should not be 0"),
                    return ge::GRAPH_FAILED);
    return numBlocks;
}
} // namespace mc2tiling
#endif
