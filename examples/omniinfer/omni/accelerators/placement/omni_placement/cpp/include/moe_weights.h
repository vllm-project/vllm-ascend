// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef MOEWEIGHTS_H
#define MOEWEIGHTS_H

#include "distribution.h"
#include "tensor.h"
#include <assert.h>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

class ExpertWeights {
    // 专家类， 包含多个权重信息
  public:
    // 构造函数
    ExpertWeights() {}
    ExpertWeights(int expert_id, std::vector<Tensor> weights)
        : expert_id_(expert_id), weights_(weights) {
        total_size_ = 0;
        for (auto &weight : weights_) {
            total_size_ += weight.get_total_size();
        }
    }
    ExpertWeights(std::vector<Tensor> weights) : weights_(weights) {
        total_size_ = 0;

        if (weights.size() == 0)
            throw std::runtime_error("Invalid ExpertWeights.weights.size: 0");

        for (auto &weight : weights_) {
            total_size_ += weight.get_total_size();
        }
    }
    void enqueueSwapInformation(Distribution *dist_ptr, size_t t_rank);
    void enqueueSwapInformation(Distribution *dist_ptr, size_t t_rank,
                                void *recv_buff, bool need_enqueue_recv_buff,
                                size_t localExpertPositionOfsset);
    void swap(Distribution *dist_ptr, size_t t_rank, bool send_first,
              aclrtStream stream);
    // 新增公共方法获取私有成员
    int get_expert_id() const {
        return expert_id_;
    } // TODO: 不再维护，后期废弃， 统一在placement_mapping中维护
    size_t get_total_size() const { return total_size_; }
    aclError to_host(char *host_ptr) const {
        aclError ret;
        if (host_ptr == nullptr) {
            throw std::runtime_error("Target memory (host_ptr) is null");
        }
        for (const auto &weight : weights_) {
            size_t data_size = weight.get_total_size();
            ret = weight.to_host((void *)(host_ptr));
            if (ret != ACL_ERROR_NONE) {
                throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                         std::to_string(ret));
            }
            host_ptr += data_size;
        }
        return ret;
    };
    aclError to_device(char *host_ptr) {
        aclError ret;
        if (host_ptr == nullptr) {
            throw std::runtime_error("Target memory (host_ptr) is null");
        }
        for (auto &weight : weights_) {
            size_t data_size = weight.get_total_size();
            ret = weight.to_device((void *)(host_ptr));
            if (ret != ACL_ERROR_NONE) {
                throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                         std::to_string(ret));
            }
            host_ptr += data_size;
        }
        return ret;
    };

    void info() {
        std::cout << "One Expert has " + std::to_string(weights_.size()) +
                         " weights"
                  << std::endl;
    }

  private:
    int expert_id_; // TODO: 不再维护，后期废弃， 统一在placement_mapping中维护
    std::vector<Tensor> weights_; //该专家的多个权重，包含 bias， weight等信息
    size_t total_size_; // 该专家权重参数数量
};

struct CountData {
    std::atomic<int> completed_processes;
    std::atomic<int> init_flag; // 用于标记是否已初始化
};

class MoEWeights {
  private:
    std::vector<std::vector<ExpertWeights>> npu_weights_;

    void *count_ptr_; // 共享引用计数

    std::string shm_name_ = "/omni_moe_weights"; // Shared memory name
    void *shm_ptr_;                              // Pointer to shared DRAM
    size_t shm_size_;                            // Total size in bytes

    size_t rank_;
    size_t world_size_; //总进程数，用于分析共享内存的拷贝是否全部完成
    size_t num_layers_;
    size_t num_experts_;
    size_t num_deploy_experts_per_rank_;
    mutable std::mutex mtx_;
    bool is_initilized_ = false;
    // Initialize shared memory
    void init_shared_memory(size_t shm_size);
    void replicate_to_shared_memory();

    // 创建或附加共享内存
    void *create_or_attach_shmem(const std::string &name, size_t size);

    // 权重分布式通信变量
    Distribution *dist_ptr_ = nullptr;

    // 根据Queue_size, 分配一块显存;
    void *recv_buff_;

  public:
    MoEWeights(size_t num_experts);
    MoEWeights(size_t num_experts, size_t world_size);
    MoEWeights(size_t num_experts, size_t rank, size_t world_size);
    MoEWeights(size_t num_experts, size_t rank, size_t world_size,
               const char *rankTableFile);
    MoEWeights(size_t num_experts, size_t rank, size_t world_size,
               Distribution *dist_ptr);
    ~MoEWeights();

    void init_weights(
        const std::vector<std::vector<std::vector<Tensor>>> &npu_weights,
        bool init_shm);

    void replacement(Distribution *dist_ptr, size_t layer_idx, size_t rank_a,
                     size_t expert_position_a, size_t rank_b,
                     size_t expert_position_b);
    void replacement(Distribution *dist_ptr, size_t layer_idx, size_t rank_a,
                     size_t expert_position_a, size_t rank_b,
                     size_t expert_position_b, void *recv_buff_start_address,
                     bool need_enqueue_recv_buff);

    void replacement(Distribution *dist_ptr, aclrtStream stream,
                     size_t layer_idx, size_t local_expert_idx, size_t t_rank);

    void replacement(size_t layer_idx, size_t src_global_expert_idx,
                     size_t dst_local_expert_idx);
    std::vector<std::vector<ExpertWeights>> getNpuWeights() const {
        return npu_weights_;
    }
    ExpertWeights getExpert(size_t layer_id, size_t local_expert_idx) {
        return npu_weights_[layer_id][local_expert_idx];
    }
    size_t getLocalExpertPositionOffset(size_t layer_id,
                                        size_t local_expert_idx) const {
        // Rank 内部的专家位置 便宜 < layer_id*num_deployed_expert_per_rank +
        // local_expert_idx
        size_t num_deployed_expert_per_rank = npu_weights_[0].size();
        return layer_id * num_deployed_expert_per_rank + local_expert_idx;
    }
    size_t getNumLayers() const { return num_layers_; }
    size_t getNumExperts() const { return num_experts_; }
    size_t getNumDeployExpertsPerRank() const {
        return num_deploy_experts_per_rank_;
    }
    void *getShmPtr() const { return shm_ptr_; }
    bool isHbmInitialized() const;
    std::unique_lock<std::mutex> acquire_lock() const {
        return std::unique_lock<std::mutex>(mtx_);
    }
    bool isShmInitialized() const;
    size_t getShmSize() const { return shm_size_; }
    std::string getShmName() const { return shm_name_; }
    void unittest_for_init_shared_memory(size_t shm_size) {
        init_shared_memory(shm_size);
    } // 仅供Unitest调用
    size_t get_expert_size();
};
#endif // MOEWEIGHTS_H
