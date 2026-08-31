/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
#define FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H

#include <cstdint>

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;

// layout 枚举值，与 FiaLayout 一致，供外部算子使用
constexpr uint32_t LAYOUT_BSH = 0;
constexpr uint32_t LAYOUT_BSND = 1;
constexpr uint32_t LAYOUT_BNSD = 2;
constexpr uint32_t LAYOUT_TND = 4;

// tiling 切块常量
constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param gSize      GQA 的 group 数（n1/n2）
 * @param maxSeqQ    Q 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maxSeqKv   KV 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maskMode   mask 模式（0/2/4 等）
 * @param winLeft    左侧窗口
 * @param winRight   右侧窗口
 * @param qLayout    Q 的 layout（使用 LAYOUT_BSH / LAYOUT_BSND / LAYOUT_TND 等）
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t gSize, int64_t maxSeqQ, int64_t maxSeqKv,
                                  int32_t maskMode, int64_t winLeft, int64_t winRight, uint32_t qLayout,
                                  uint32_t &sOuterFactor, uint32_t &sInnerFactor)
{
    if (maxSeqQ == -1) {
        maxSeqQ = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqKv == -1) {
        maxSeqKv = MAX_SEQ_LEN_DEFAULT;
    }
    sOuterFactor = SOUTER_64;
    sInnerFactor = SINNER_128;

    bool checkQueryAndValueS = maxSeqQ <= SOUTER_64 && maxSeqKv > SINNER_128;

    if (vHeadDim <= DSIZE_128) {
        int64_t winLeftTmp = winLeft;
        int64_t winRightTmp = winRight;
        if (maskMode == 0) {
            winLeftTmp = (winLeftTmp > 0) ? 0 : winLeftTmp;
        } else if (maskMode == 4) {
            winRightTmp = (winRightTmp > 0) ? 0 : winRightTmp;
        }
        bool checkSparseMode = (maskMode != 2 && winLeftTmp + winRightTmp > 128);
        if (checkQueryAndValueS && checkSparseMode) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        }
    }
    if (vHeadDim == DSIZE_256) {
        if (gSize * maxSeqQ < SOUTER_64) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        } else {
            sOuterFactor = SOUTER_64;
            sInnerFactor = SINNER_128;
        }
    }
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
