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
 * \file static_buffer.h
 * \brief 静态 tensor buffer 管理：StaticBuffer 携带显式 bufferId，RingBuffer 提供轮转。
 *        与 TPipe 管理的 TBuf/Buffer 不同，本文件的 buffer 直接从首地址偏移排布，
 *        由使用者手动指定地址与 bufferId。
 */
#ifndef SPARSE_FLASH_MLA_STATIC_BUFFER_H
#define SPARSE_FLASH_MLA_STATIC_BUFFER_H

#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

namespace fa_base_matmul {

template <typename ElemT>
struct StaticBuffer {
    LocalTensor<ElemT> tensor;
    uint32_t idx;
};

template <typename ElemT>
struct RingBuffer {
    StaticBuffer<ElemT> *bufs;
    uint32_t bufNum;
    uint32_t curId;

    __aicore__ inline RingBuffer()
        : bufs(nullptr),
          bufNum(0),
          curId(0)
    {}
    __aicore__ inline RingBuffer(StaticBuffer<ElemT> *b, uint32_t n)
        : bufs(b),
          bufNum(n),
          curId(n - 1)
    {}

    __aicore__ inline StaticBuffer<ElemT> &GetNext()
    {
        curId = (curId + 1) % bufNum;
        return bufs[curId];
    }
};

} // namespace fa_base_matmul
#endif // SPARSE_FLASH_MLA_STATIC_BUFFER_H
