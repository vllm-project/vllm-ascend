/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

template <typename T, uint32_t N, uint64_t Stride=0>
class Array {
private:
    T elems[N];

public:
    __aicore__ inline T& operator[](uint32_t i)
    {
        return elems[i];
    }
};

template <typename T, uint32_t N, uint64_t Stride>
class Array<LocalTensor<T>, N, Stride> {
private:
    LocalTensor<T> tensor;

public:
    __aicore__ inline void Init(const LocalTensor<T>& tensor)
    {
         this->tensor = tensor;
    }
    __aicore__ inline LocalTensor<T> operator[](uint32_t i)
    {
        return tensor[i * Stride];
    }
};

template <typename T, uint32_t N>
class Array<GlobalTensor<T>, N> {
private:
    GlobalTensor<T> tensor;
    uint64_t stride = 0;

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
};
#endif // ARRAY_H
