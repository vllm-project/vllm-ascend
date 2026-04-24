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
 * \file buffer_manager.h
 * \brief buffer内存管理
 */
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "buffer.h"

// L1  TPosition::A1
// L0A TPosition::A2
// L0B TPosition::B2
// L0C TPosition::CO1
// UB  TPosition::VECIN
namespace fa_base_matmul {
template<BufferType bufferType>
class BufferManager {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size) {
        static_assert(bufferType != BufferType::GM, "GM should use workspace.");
        TBuf<BufferInfo<bufferType>::Position> tbuf;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    __aicore__ inline void Init(__gm__ uint8_t* workspace) {
        static_assert(bufferType == BufferType::GM, "BufferType should be GM.");
        mem_.SetGlobalBuffer((__gm__ uint8_t*)workspace);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline Buffer<bufferType, syncType> AllocBuffer(uint32_t size) {
        TensorType temp = mem_[offset_];
        offset_ += size;
        return Buffer<bufferType, syncType>(temp, size);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline void FreeBuffer(Buffer<bufferType, syncType> &buffer){
    }
private:
    uint32_t offset_ = 0;
    TensorType mem_;
};
}
#endif