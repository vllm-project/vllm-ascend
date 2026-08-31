/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file array.h
 * \brief
 */
#ifndef ARRAY_H
#define ARRAY_H

#include "kernel_operator.h"

template <typename T, uint32_t N, uint64_t Stride=0>
class Array {
public:
    __aicore__ inline T& operator[](uint32_t i)
    {
        return elems[i];
    }

private:
    T elems[N];
};

template <typename T, uint32_t N, uint64_t Stride>
class Array<LocalTensor<T>, N, Stride> {
public:
    __aicore__ inline void Init(const LocalTensor<T>& tensor)
    {
         this->tensor = tensor;
    }
    __aicore__ inline LocalTensor<T> operator[](uint32_t i)
    {
        return tensor[i * Stride];
    }

private:
    LocalTensor<T> tensor;
};

template <typename T, uint32_t N>
class Array<GlobalTensor<T>, N> {
public:
    __aicore__ inline void Init(const GlobalTensor<T>& tensor, uint64_t splitSize)
    {
        this->tensor = tensor;
        this->stride = splitSize;
    }
    __aicore__ inline GlobalTensor<T> operator[](uint32_t i)
    {
        return tensor[i * stride];
    }

private:
    GlobalTensor<T> tensor;
    uint64_t stride = 0;
};
#endif // ARRAY_H