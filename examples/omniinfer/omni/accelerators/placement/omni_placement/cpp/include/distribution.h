// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "tensor.h"
#include <assert.h>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

const int QUEUE_SIZE = 5;

// void ACLCHECK(aclError ret);
// void HCCLCHECK(HcclResult ret);
#define ACLCHECK(ret)                                                          \
    do {                                                                       \
        if (ret != ACL_SUCCESS) {                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, \
                   __LINE__, ret);                                             \
        }                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                         \
    do {                                                                       \
        if (ret != HCCL_SUCCESS) {                                             \
            printf("hccl interface return err %s:%d, retcode: %d \n",          \
                   __FILE__, __LINE__, ret);                                   \
        }                                                                      \
    } while (0)

#define TIME_IT_LABEL(label, code)                                             \
    do {                                                                       \
        auto start = std::chrono::high_resolution_clock::now();                \
        code;                                                                  \
        auto end = std::chrono::high_resolution_clock::now();                  \
        auto duration =                                                        \
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start) \
                .count();                                                      \
        std::cout << "Time taken by [" << label << "]: " << duration           \
                  << " milliseconds" << std::endl;                             \
    } while (0)

#define TIME_IT_LABEL_RETURN(label, result, code)                              \
    do {                                                                       \
        auto start = std::chrono::high_resolution_clock::now();                \
        (result) = (code);                                                     \
        auto end = std::chrono::high_resolution_clock::now();                  \
        auto duration =                                                        \
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start) \
                .count();                                                      \
        std::cout << "Time taken by [" << label << "]: " << duration           \
                  << " milliseconds" << std::endl;                             \
    } while (0)

typedef struct {
    std::vector<void *> address; // 多个权重的地址
    std::vector<size_t> lengths; // 多个权重的长度
    std::vector<size_t> sizes;
    std::vector<std::string> dtypes; // 多个权重的类型
    std::vector<void *> recv_buffs;  // 中转空间站
    size_t t_rank;
    size_t localExpertPositionOfsset;

} TransDesc;

// 与线程交互的队列, 队头队尾两个计数
class ThreadQueue {
  private:
    TransDesc *mTransDescQueue[QUEUE_SIZE];
    int mCurDescFront = 0;
    int mCurDescRear = -1;
    int currentSize = 0; // 当前队列大小
    std::mutex mMutex;
    std::condition_variable mCond;     // 通知队列不空
    std::condition_variable not_full_; // 通知队列不满
  public:
    ThreadQueue() {
        for (int i = 0; i < QUEUE_SIZE; ++i) {
            mTransDescQueue[i] = new TransDesc;
        }
        not_full_.notify_one(); // 通知等待入队的线程
    }
    int getCurrentSize() const { return currentSize; }
    ThreadQueue(const ThreadQueue &other) {
        mCurDescFront = other.mCurDescFront;
        mCurDescRear = other.mCurDescRear;
        // 深拷贝 mTransDescQueue
        for (int i = 0; i < QUEUE_SIZE; ++i) {
            mTransDescQueue[i] = new TransDesc(*other.mTransDescQueue[i]);
        }
    }
    ~ThreadQueue() {
        for (int i = 0; i < QUEUE_SIZE; ++i) {
            delete mTransDescQueue[i];
        }
    }
    bool IsFull() {
        // std::lock_guard<std::mutex> lock(mMutex);
        return currentSize == QUEUE_SIZE;
    }

    bool IsEmpty() {
        // std::lock_guard<std::mutex> lock(mMutex);
        return currentSize == 0;
    }
    void Enqueue(TransDesc *desc) {

        // if(IsFull()){
        //     return false;
        // }

        std::unique_lock<std::mutex> lock(mMutex);
        // 等待直到队列不满
        not_full_.wait(lock, [this]() { return !IsFull(); }); // 释放锁

        mCurDescRear = (mCurDescRear + 1) % QUEUE_SIZE;
        TransDesc *transDesc = mTransDescQueue[mCurDescRear];
        if (desc == nullptr) {
            // 用于触发线程退出
            transDesc = nullptr;
        } else {
            transDesc->address = std::move(desc->address);
            transDesc->lengths = std::move(desc->lengths);
            transDesc->dtypes = std::move(desc->dtypes);
            transDesc->sizes = std::move(desc->sizes);
            transDesc->recv_buffs = std::move(desc->recv_buffs);
            transDesc->t_rank = desc->t_rank;
            transDesc->localExpertPositionOfsset =
                desc->localExpertPositionOfsset;
        }
        currentSize++;
        lock.unlock();
        mCond.notify_one();
    }
    void Dequeue() {
        if (IsEmpty()) {
            std::cout
                << "The Swap Queue is Empty, Ignore the Dequeue Operation!";
            return;
        }
        std::unique_lock<std::mutex> lock(mMutex);
        mCurDescFront = (mCurDescFront + 1) % QUEUE_SIZE;
        currentSize--;
        lock.unlock();
        not_full_.notify_one(); // 通知等待入队的线程
        return;
    }
    TransDesc *GetFront() {
        std::unique_lock<std::mutex> lock(mMutex);
        mCond.wait(lock, [this]() { return !IsEmpty(); });
        return mTransDescQueue[mCurDescFront];
    }
    TransDesc *GetRear(size_t position_offset) {
        std::unique_lock<std::mutex> lock(mMutex);
        if (IsEmpty()) {
            return nullptr;
        }

        // 从队头遍历到队尾，寻找目标recv_buffs
        TransDesc *result = nullptr;
        for (int i = 0; i < currentSize; ++i) {
            int index = (mCurDescFront + i) % QUEUE_SIZE;
            if (mTransDescQueue[index]->localExpertPositionOfsset ==
                position_offset) {
                result = mTransDescQueue[index];
            }
        }
        return result; // 如果没有找到，返回null
    }
    size_t getEnqueueIdx() { return (mCurDescRear + 1) % QUEUE_SIZE; }
};

enum class HcclCommInitType { RootInfoString, RankTableFile };

class Distribution {
  private:
    HcclComm hcclComm_;
    uint32_t rank_;
    uint32_t world_size_;
    ThreadQueue completedSynchronizeQueue_;
    aclrtStream stream_;
    aclrtStream memcopy_stream_;
    size_t expert_size_;
    void *recv_buff_ = nullptr;
    std::vector<int> hostHandshakeStatus_; // 用于在host侧获取所有的握手信息
    Tensor deviceHandshakeStatus_;
    Tensor deviceCurrentStatus_;
    size_t info_length_ =
        3; // 握手信息的信息量 <t_rank, global_position_id, expert_id>;

  public:
    Distribution(size_t rank, const char *rankTableFile);
    Distribution(size_t rank, size_t world_size, const char *infoStr,
                 HcclCommInitType type);
    ~Distribution();
    void enqueue(TransDesc *desc, size_t t_rank,
                 bool need_enqueue_recv_buff); // 选择放到哪个队列当中
    void swap(void *src_addr, void *recv_addr, size_t length, std::string dtype,
              uint32_t t_rank, bool send_first, aclrtStream stream);
    void allgather(void *src_addr, void *recv_addr, size_t length,
                   std::string dtype);
    void printCommInfo();
    void warmup();
    void copyFromCompletedQueueToHBM();
    void copyAllFromCompletedQueueToHBM();
    void set_stream(aclrtStream stream);
    void allocate_recv_buffs(size_t expert_size);
    void release_recv_buffs();
    void *get_recv_buff_address();
    size_t getCompletedQueueEnqueuePosition() {
        return completedSynchronizeQueue_.getEnqueueIdx();
    }
    bool isCompletedQueueFull();
    bool performGlobalHandshake(int t_rank, int position_offset, int expert_id);
    std::vector<int> getHandshakeStatus() const { return hostHandshakeStatus_; }
    size_t get_info_length() const { return info_length_; }
    int getCompletedQueueSize() const {
        return completedSynchronizeQueue_.getCurrentSize();
    }
};
#endif // ACL_CHECK_H