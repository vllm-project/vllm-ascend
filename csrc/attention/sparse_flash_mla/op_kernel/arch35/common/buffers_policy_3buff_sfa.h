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
 * \file buffer_policy_sfa.h
 * \brief
 */
#ifndef BUFFER_POLICY_SFA_H
#define BUFFER_POLICY_SFA_H

#if __has_include("../../../common/op_kernel/buffers_policy.h")
#include "../../../common/op_kernel/buffers_policy.h"
#elif __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/op_kernel/buffers_policy.h"
#endif

namespace fa_base_matmul {
// 申请3个buffer, 轮转
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicy3buffSFA {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size, uint32_t aId = 0U,
                                uint32_t bId = 0U, uint32_t cId = 0U)
    {
        a_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        b_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        c_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);

        if constexpr (idSource == IdSource::INTERNAL) {
            a_.template Init<idSource>();
            b_.template Init<idSource>();
            c_.template Init<idSource>();
        } else if constexpr (idSource == IdSource::EXTERNAL) {
            a_.template Init<idSource>(aId);
            b_.template Init<idSource>(bId);
            c_.template Init<idSource>(cId);
        }
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager)
    {
        a_.template UnInit<idSource>();
        b_.template UnInit<idSource>();
        c_.template UnInit<idSource>();

        bufferManager.template FreeBuffer<syncType, syncMode>(a_);
        bufferManager.template FreeBuffer<syncType, syncMode>(b_);
        bufferManager.template FreeBuffer<syncType, syncMode>(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode> &Get()
    {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else {
            flag1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode> &Get(uint32_t id)
    {
        uint32_t flag = id % 3;
        if (flag == 0) {
            return a_;
        } else if (flag == 1) {
            return b_;
        } else {
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode> &GetVec()
    { // mixcore architecture
        if (flag1_vec1_ == 0) {
            flag1_vec1_ = 1;
            return a_;
        } else if (flag1_vec1_ == 1) {
            flag1_vec1_ = NUM_2;
            return b_;
        } else {
            flag1_vec1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode> &GetCube()
    { // mixcore architecture
        if (flag1_bmm2_ == 0) {
            flag1_bmm2_ = 1;
            return a_;
        } else if (flag1_bmm2_ == 1) {
            flag1_bmm2_ = NUM_2;
            return b_;
        } else {
            flag1_bmm2_ = 0;
            return c_;
        }
    }

    // Q复用
    __aicore__ inline Buffer<bufferType, syncType, syncMode> &GetPre()
    {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    // KV复用
    __aicore__ inline Buffer<bufferType, syncType, syncMode> &GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1) {
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }

private:
    Buffer<bufferType, syncType, syncMode> a_;
    Buffer<bufferType, syncType, syncMode> b_;
    Buffer<bufferType, syncType, syncMode> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};

} // namespace fa_base_matmul
#endif
