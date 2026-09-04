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
 * \file const_def.h
 * \brief Common constants
 */
#ifndef CONST_DEF_H
#define CONST_DEF_H

namespace AttentionCommon {
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);

enum class TASK_DEAL_MODE : uint32_t
{
    DEAL_ZERO = 0,      // actualSeqLen为0，需清零输出
    SKIP = 1,           // 整行S2无需处理
    CREATE_TASK = 2,    // 正常创建任务
    SKIP_S1OUT = 3,     // 跳过不属于当前核的任务，仅S1外切分场景
    SKIP_ZERO = 4,      // 跳过非首个actualSeqLen为0的块
    S2_END = 5,         // 有效行且S2已全部计算完
    NOT_START = 6,      // 有效行且S2还未开始计算
    SKIP_REMAINING_S2 = 7, // 跳过S2上尚未遍历的任务
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

} // namespace const_def
#endif
