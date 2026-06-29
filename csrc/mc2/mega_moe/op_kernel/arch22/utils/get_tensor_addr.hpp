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
 * \file get_tensor_addr.hpp
 * \brief
 */

#ifndef GET_TENSOR_ADDR_HPP
#define GET_TENSOR_ADDR_HPP
#include "kernel_operator.h"

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <typename T>
FORCE_INLINE_AICORE __gm__ T *GetTensorAddr(uint32_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(retPtr + index));
}

#endif // GET_TENSOR_ADDR_HPP