// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef EXPERT_ACTIVATION_H
#define EXPERT_ACTIVATION_H

#include "distribution.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "placement_mapping.h"
#include "tensor.h"
#include <acl/acl.h>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex> // 用于 std::unique_lock 和 std::mutex
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>  // New include for POSIX functions
#include <sys/types.h> // New include for POSIX types
#include <thread>      // 用于 std::this_thread::sleep_for
#include <unistd.h>
#include <vector>

class ExpertActivation {
  private:
    size_t length_ = 20;
    std::vector<int64_t> activated_values;
    size_t idx = 0;

  public:
    ExpertActivation() { activated_values.resize(length_, 0); }
    void update(int64_t value) {
        activated_values[idx] = value;
        idx = (idx + 1) % length_;
    }
    int64_t get_last_value() {
        size_t tmp_index = (length_ + idx - 1) % length_;
        return activated_values[tmp_index];
    }
    int64_t getTotalValue() {
        int64_t result = 0;
        for (size_t idx = 0; idx < activated_values.size(); ++idx) {
            result = result + activated_values[idx];
        }
        return result;
    }
};

class ClusterActivation {
  private:
    Tensor npu_count_;
    int64_t max_activation_count_;
    size_t num_layers_;
    size_t num_deploy_experts_;
    size_t num_experts_;
    size_t num_deploy_experts_per_rank_;
    int activation_window_size_;
    size_t world_size_;
    size_t hccl_comm_world_size_;
    size_t rank_;
    void *total_count_ptr_;

    std::thread thread_; // 工作线程
    bool enable_dump_ = false;
    std::string dump_dir_ =
        ""; // Fixed: Removed the reference, initialized an empty string
    Tensor *expert_activation_counts_ = nullptr; // activation all rank reduced
    void *deployed_experts_counts_host_;
    void *last_count_ptr_;
    void *delta_experts_counts_;
    std::vector<ExpertActivation> all_logit_experts_activation_;

    enum ThreadState {
        INIT,
        RUNNING,
        STOPPING,
        STOPPED
    } thread_state_ = ThreadState::INIT;

    // Activation共享内存描述符
    std::string act_shm_name_ = "/omni_moe_activations";
    ExpertActivation *act_shm_ptr_;
    size_t act_shm_size_;

    void *create_or_attach_shmem(const std::string &name, size_t size);
    void init_activation_hbm();
    void free_activation_hbm();
    bool is_enbale_dump() const { return enable_dump_; }

  public:
    ClusterActivation(Tensor npu_count, int64_t max_activation_count,
                      size_t num_layers, size_t num_deploy_experts_per_rank,
                      int activation_window_size, size_t world_size,
                      size_t hccl_comm_world_size, size_t rank);
    ~ClusterActivation();
    int64_t getLayerActivationCount(size_t layer_id = 0);
    int64_t getClusterTotalActivationCount(size_t layer_idx,
                                           size_t deploy_expert_idx);
    int64_t getExpertActivationCount(int32_t layer_idx,
                                     int32_t deploy_expert_idx,
                                     void *activation_ptr);
    int64_t getExpertActivationCount(int32_t layer_idx,
                                     int32_t deploy_expert_idx);
    void updateDeltaActivationCount();
    void print_activations();
    void setDumpDir(const std::string &dump_dir);
    void stopDump();
    void dumpActivationCounts(size_t dump_count);
    void dumpActivationCountsPerRank(size_t dump_count,
                                     int64_t *total_count_ptr);
    size_t get_num_layers() const { return num_layers_; }
    size_t get_num_deploy_experts() const { return num_deploy_experts_; }
    size_t get_num_deploy_experts_per_rank() const {
        return num_deploy_experts_per_rank_;
    }
    size_t get_rank() const { return rank_; }
    size_t get_world_size() const { return world_size_; }
    size_t get_hccl_comm_world_size() const { return hccl_comm_world_size_; }
    void set_params(size_t num_experts);
    int64_t getLogitExpertShiftActivateion(int32_t layer_id, int32_t expert_id,
                                           int this_expert_deployed_num);
    void updateShiftWindows(PlacementMapping *placement_mapping);

    // For Unittest
    Tensor &get_npu_count() { return npu_count_; }
    void *get_total_count_ptr() { return total_count_ptr_; }
    void *get_last_count_ptr() { return last_count_ptr_; }

    // 线程控制相关操作
    void collect_from_txt(const std::string &txt_path);
    void collect(Distribution *dist_ptr, aclrtStream stream);
    void dump_and_collect(Distribution *dist_ptr, aclrtStream stream,
                          size_t dump_count);
    void start_thread();
    void stop_thread();
};

#endif // EXPERT_ACTIVATION_H