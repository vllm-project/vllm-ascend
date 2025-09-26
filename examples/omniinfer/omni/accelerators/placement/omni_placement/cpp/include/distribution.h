// SPDX-License-Identifier: MIT
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

enum SwapDirection {
    SEND = 0x01,  // 0001
    RECV = 0x02,  // 0010
    LOCAL = 0x04, // 0100
    RESERVE = 0x10
};

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
    std::vector<SwapDirection> sendRecvType;
    std::vector<void *> address; // 多个权重的地址
    std::vector<size_t> lengths; // 多个权重的长度
    std::vector<size_t> sizes;
    std::vector<std::string> dtypes; // 多个权重的类型
    std::vector<void *> recv_buffs;  // 中转空间站
    std::vector<size_t> t_rank;
} TransDesc;

enum class HcclCommInitType { RootInfoString, RankTableFile };

class Distribution {
  private:
    HcclComm hcclComm_;
    uint32_t rank_;
    uint32_t world_size_;
    aclrtStream stream_;
    aclrtStream memcopy_stream_;
    size_t expert_size_;
    void *recv_buff_ = nullptr;
    std::vector<TransDesc *> completed_sync_queue_;
    size_t queue_size_;
    size_t queue_item_num_;
    size_t max_hccl_batch_size_;
    std::vector<int> host_round_status_;
    Tensor device_round_status_;
    Tensor device_cur_round_status_;
    size_t round_info_length_ = 2;
    TransDesc send_recv_info_;
    uint32_t recv_buff_cur_ = 0;
    size_t hccl_batch_idx_;
    size_t expert_weight_num_ = 0;
    size_t num_deploy_experts_per_rank_;

  public:
    Distribution(size_t rank, const char *rankTableFile);
    Distribution(size_t num_deploy_experts_per_rank, size_t rank,
                 size_t world_size, const char *infoStr, HcclCommInitType type);
    ~Distribution();
    void allgather(void *src_addr, void *recv_addr, size_t length,
                   std::string dtype);
    void printCommInfo();
    void warmup();
    void set_stream(aclrtStream stream);
    void allocate_recv_buffs(size_t expert_size);
    void release_recv_buffs();
    void copy_from_queue_to_hbm();
    void init_hccl_buffs(size_t expert_size);
    void clear_hccl_buffs();
    void prepare_batch(bool need_enqueue, TransDesc &desc);
    void add_to_batch(TransDesc &desc, int direction);
    void hccl_batch_send();
    size_t hccl_buffs_size();
    void *get_recv_buff_address(bool need_enqueue, size_t size);
    size_t get_recv_buff_maxsize();
    bool sync_round_shakehand(int curLayer, int curBatch);
    void enqueue(TransDesc *desc) {
        if (queue_item_num_ >= queue_size_) {
            throw std::runtime_error("queue can not be full!");
        }
        completed_sync_queue_[queue_item_num_]->address =
            std::move(desc->address);
        completed_sync_queue_[queue_item_num_]->lengths =
            std::move(desc->lengths);
        completed_sync_queue_[queue_item_num_]->dtypes =
            std::move(desc->dtypes);
        completed_sync_queue_[queue_item_num_]->sizes = std::move(desc->sizes);
        completed_sync_queue_[queue_item_num_]->recv_buffs =
            std::move(desc->recv_buffs);
        completed_sync_queue_[queue_item_num_]->t_rank =
            std::move(desc->t_rank);
        queue_item_num_++;
    }
    void clear_queue() { queue_item_num_ = 0; }
    void reset_buff_cur() { recv_buff_cur_ = 0; }
    uint32_t get_buff_cur() { return recv_buff_cur_; }
    size_t queue_size() { return queue_size_; }
};
#endif // ACL_CHECK_H