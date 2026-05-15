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
 * \file buffers_policy.h
 * \brief 综合管理buffer的内存和同步
 */
#ifndef BUFFERS_POLICY_H
#define BUFFERS_POLICY_H

#include "buffer_manager.h"
#include "buffer.h"
#define NUM_2 2
#define NUM_3 3
#define NUM_4 4
// Q复用 KV复用
// 申请单块buffer
namespace fa_base_matmul {
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        buffer_ = bufferManager.template AllocBuffer<syncType>(size);
        buffer_.Init();
    }
 
    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        buffer_.UnInit();
        bufferManager.FreeBuffer(buffer_);
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &Get(){
        return buffer_;
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &GetPre(){
        return Get();
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &GetReused(){
        return Get();
    }
private:
    Buffer<bufferType, syncType> buffer_;
};

// 申请2个buffer，乒乓轮转
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        ping_.UnInit();
        pong_.UnInit();

        bufferManager.FreeBuffer(ping_);
        bufferManager.FreeBuffer(pong_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_) { // 1
            flag1_ = 0;
            return ping_;
        } else { // 0
            flag1_ = 1;
            return pong_;
        }
    }

    // 需要与Get联用， 首次调用Get，第二次调用GetPre(Q复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_) { // 0->1
            return pong_;
        } else { // 1->0
            return ping_;
        }
    }

    // 需要与Get,GetPre联用， 首次调用Get，第二次调用GetPre,第三次复用时GetReused(KV复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else { 
            flag2_ = 0;
            return ping_;
        }
    }
 
    // 针对
    __aicore__ inline Buffer<bufferType, syncType> &GetReused(bool isNextS2IdxNoChange) {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }
 
private:
    Buffer<bufferType, syncType> ping_;
    Buffer<bufferType, syncType> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};

// 申请3个buffer, 轮转
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
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

    __aicore__ inline Buffer<bufferType, syncType> &GetVec() { // mixcore architecture
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

    __aicore__ inline Buffer<bufferType, syncType> &GetCube() { // mixcore architecture
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
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) { 
            return a_;
        } else {
            return b_;
        }
    }

    // KV复用
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) { 
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1){ 
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }
private:
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};

// 申请4个buffer + kv复用
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy4buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);
        d_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
        d_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
        d_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
        bufferManager.FreeBuffer(d_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get(uint32_t id) {
        uint32_t flag = id % 4;
        if (flag == 0) {
            return a_;
        } else if (flag == 1) {
            return b_;
        } else if (flag == 2) { // 2:c_
            return c_;
        } else {
            return d_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        auto& buffer = Get(head_);
        head_++;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        auto& buffer = Get(used_);
        used_ = (used_ - tail_ + 1) % (head_ - tail_) + tail_;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetFree() {
        if (tail_ == used_) {
            used_++;
        }
        auto& buffer = Get(tail_);
        tail_++;
        return buffer;
    }
private:
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    Buffer<bufferType, syncType> d_;
    uint32_t tail_ = 0; // 表示当前正在使用的buffer队列队尾
    uint32_t head_ = 0; // 表示当前正在使用的buffer队列队首+1
    uint32_t used_ = 0; // 表示当前正在使用的buffer，于首尾间，左闭右开
};

template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Matrix2x2BufferPolicy { // 4buffer
// 二维buffer管理，地址行优先，使用列优先
// MracBuffer:memory address with row first, alloc/use/free with column first
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        bufferM0k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM0k1_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k1_ = bufferManager.template AllocBuffer<syncType>(size);

        bufferM0k0_.Init();
        bufferM0k1_.Init();
        bufferM1k0_.Init();
        bufferM1k1_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        bufferM0k0_.UnInit();
        bufferM0k1_.UnInit();
        bufferM1k0_.UnInit();
        bufferM1k1_.UnInit();

        bufferManager.FreeBuffer(bufferM0k0_);
        bufferManager.FreeBuffer(bufferM0k1_);
        bufferManager.FreeBuffer(bufferM1k0_);
        bufferManager.FreeBuffer(bufferM1k1_);
    }

    __aicore__ inline void SetMExtent(int32_t mExtent) {
        aIdx_ = -1;
        amIdx_ = (amIdx_ + mSize_ - 1) % mSize_; // 翻转 0->1, 1->0
        akIdx_ = 0;

        uIdx_ = -1;
        umIdx_ = (umIdx_ + mSize_ - 1) % mSize_;
        ukIdx_ = 0;

        fIdx_ = -1;
        fmIdx_ = (fmIdx_ + mSize_ - 1) % mSize_;
        fkIdx_ = 0;

        mExtent_ = mExtent;
    }

    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekNextK() { // 在Alloc阶段使用，k方向取下一个
        return PeekBuffer(amIdx_, (1 - akIdx_)); // k翻转
    }
private:
    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {
        // xIdx为入参，表示当前alloc/use/free的idx，mIdx和kIdx为下标出参，移动到下一个buffer并获取
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if (mIdx == 1 && kIdx == 0) {
            return bufferM1k0_;
        } else { // 该分支条件为：mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekBuffer(int32_t mIdx, int32_t kIdx) {
        // 只访问buffer，不进行下标移动
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if ((mIdx == 1) && (kIdx == 0)) {
            return bufferM1k0_;
        } else { // mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    Buffer<bufferType, syncType> bufferM0k0_;
    Buffer<bufferType, syncType> bufferM0k1_;
    Buffer<bufferType, syncType> bufferM1k0_;
    Buffer<bufferType, syncType> bufferM1k1_;
    int32_t mSize_ = 2; // m的总buffer数
    int32_t kSize_ = 2; // k的总buffer数

    // Alloc
    int32_t aIdx_ = -1; // 当前第几次Alloc Buffer
    int32_t amIdx_ = 0; // 当前Alloc Buffer的m下标
    int32_t akIdx_ = 0; // 当前Alloc Buffer的k下标

    // Reuse
    int32_t uIdx_ = -1;
    int32_t umIdx_ = 0;
    int32_t ukIdx_ = 0;

    // Free
    int32_t fIdx_ = -1;
    int32_t fmIdx_ = 0;
    int32_t fkIdx_ = 0;

    int32_t mExtent_ = 0; // m实际使用的大小，可以为1或者2
};
}
#endif