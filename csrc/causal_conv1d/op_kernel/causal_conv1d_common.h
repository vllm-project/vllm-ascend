/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_common.h
 * \brief Common utilities and constants for CausalConv1D prefill kernel.
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MAX_BLOCK_DIM = 4096;
constexpr int32_t RING_SLOTS = 5;

__aicore__ inline int32_t SlotCurr(int32_t t)
{
    return (t + 3) % RING_SLOTS;
}

__aicore__ inline int32_t SlotHist(int32_t t, int32_t i)
{
    return (t + 3 - i) % RING_SLOTS;
}

__aicore__ inline int32_t SlotPrefetch(int32_t t)
{
    return (t + 4) % RING_SLOTS;
}

} // namespace NsCausalConv1dCommon

#endif // CAUSAL_CONV1D_COMMON_H