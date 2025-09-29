// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef PLACEMENT_H
#define PLACEMENT_H

#include "expert_activation.h"
#include "moe_weights.h"
#include "placement_mapping.h"
#include "placement_optimizer.h"
#include <assert.h>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

class Placement {
  private:
    MoEWeights *moe_weight_;
    int rank_;       // global device id
    int world_size_; // global device number
    int hccl_comm_world_size_;
    int num_devices_per_host_;
    ClusterActivation *activations_;
    PlacementMapping *mapping_;

    PlacementOptimizer *optimizer_;
    int num_layers_;

    int num_experts_; // num of logic expert, e.g. 256

    int num_deploy_experts_;          // num of physic deploy expert, e.g. 256 +
                                      // 16(redundant)
    int num_deploy_experts_per_rank_; // num of physic deploy expert per ran,
                                      // e.g. 17 + 1(redundant)

    std::thread worker_thread_; // Main worker thread
    std::thread init_thread_;   // Initialization thread
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> should_stop_init_{false};
    ; // 新增，用于 init_thread_
    std::atomic<bool> should_pause_{false};
    mutable std::mutex mtx_;
    std::atomic<bool> sub_thread_is_changing_{false};
    // 权重分布式通信变量
    Distribution *dist_ptr_ = nullptr;
    std::vector<bool> is_layer_update;
    bool enable_dynamic_ = false;

  public:
    Placement()
        : moe_weight_(nullptr), rank_(0), world_size_(0),
          num_devices_per_host_(0), activations_(nullptr), mapping_(nullptr),
          optimizer_(nullptr), num_layers_(0), num_experts_(0),
          num_deploy_experts_(0), num_deploy_experts_per_rank_(0) {}

    mutable size_t dump_count_ = 0;
    void dump_count_addone() const { dump_count_ = dump_count_ + 1; }

    Placement(int rank, int world_size, int hccl_comm_world_size,
              int num_devices_per_host, ClusterActivation *activation,
              PlacementMapping *placement_mapping, char *root_info,
              bool enable_dynamic);

    Placement(int rank, int world_size, int num_devices_per_host,
              ClusterActivation *activation, size_t expert_mapping_ptr,
              std::vector<int64_t> shape, int dtype,
              size_t placement_pattern_ptr,
              std::vector<int64_t> placement_shape, int placement_dtype);

    ~Placement();

    bool get_subthread_is_changing() const { return sub_thread_is_changing_; }

    void initialize_components(char *root_info);
    void initialize_components(size_t expert_mapping_ptr,
                               std::vector<int64_t> shape, int dtype,
                               size_t placement_pattern_ptr,
                               std::vector<int64_t> placement_shape,
                               int placement_dtype);
    void check_shm_weights();
    void placement_manager(aclrtContext currentContext);
    void replace_redundant_experts(int layer_id);

    // Thread control related operations
    void start_thread();
    void stop_thread();
    void reset_layer_update() {
        for (size_t idx = 0; idx < is_layer_update.size(); ++idx) {
            is_layer_update[idx] = false;
        }
    }
    std::vector<bool> get_layer_update() const { return is_layer_update; }
    ClusterActivation *get_activations() const { return activations_; }
    PlacementMapping *get_mapping() const { return mapping_; }
    PlacementOptimizer *get_optimizer() const { return optimizer_; }

    int get_num_layers() const { return num_layers_; }
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    int get_num_experts() const { return num_experts_; }
    int get_num_deploy_experts() const { return num_deploy_experts_; }
    int get_num_deploy_experts_per_rank() const {
        return num_deploy_experts_per_rank_;
    }
    int get_num_devices_per_host() const { return num_devices_per_host_; }
    bool is_redundant_share_expert_rank() const { return rank_ >= world_size_; }

    std::thread &get_worker_thread() {
        return worker_thread_;
    } // 返回引用，因为 std::thread 不可拷贝
    std::thread &get_init_thread() {
        return init_thread_;
    } // 返回引用，因为 std::thread 不可拷贝
    bool get_should_stop() const {
        return should_stop_.load();
    } // std::atomic 需要用 load() 获取值
    MoEWeights *get_moe_weights() const { return moe_weight_; }
    std::unique_lock<std::mutex> acquire_lock() const {
        return std::unique_lock<std::mutex>(mtx_);
    }
    Distribution *get_distribution() const { return dist_ptr_; }
};

#endif // PLACEMENT_H