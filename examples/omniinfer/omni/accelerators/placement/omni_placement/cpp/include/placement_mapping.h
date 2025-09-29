// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef PLACEMENT_MAPPING_H
#define PLACEMENT_MAPPING_H
#include "distribution.h"
#include "tensor.h"
#include <acl/acl.h>
#include <cmath>
#include <map>
#include <memory> // 用于 std::unique_ptr
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

/**
 * @class PlacementMapping
 * @brief Manages the mapping and placement of expert models across distributed
 * computing devices
 *
 * This class handles the distribution and redundancy of expert models in a
 * multi-device environment for large language model inference, with support for
 * redundancy and remapping.
 */
class PlacementMapping {
  protected:
    int rank_;        // Global device ID in the distributed system
    int world_size_;  // Total number of devices in the distributed system
    int num_experts_; // Number of logical experts (e.g., 256)
    int num_deploy_experts_; // Number of physical deployed experts including
                             // redundant ones (e.g., 256 + 16)

    int num_devices_per_host_;   // Number of devices on each host/node
    int num_layers_;             // Number of model layers that have experts
    int num_experts_per_device_; // Number of experts assigned to each device
                                 // (without redundancy)
    int num_deploy_experts_per_device_; // Number of experts assigned to each
                                        // device (with redundancy)

    Tensor selector_;
    std::vector<int32_t> selector_host_;

    int32_t *placement_pattern_;  // DRAM pointer to placement pattern data
    int placement_pattern_dtype_; // Data type of placement pattern tensor

    int32_t
        max_redundant_per_rank_; // The number of redundant expert of each Rank
    int32_t max_redundant_per_expert_;

    std::vector<std::vector<std::vector<int>>>
        placement_pattern_vector_; // C++ representation of placement pattern

    std::vector<int32_t>
        globalDeployedPositionToLogisticsIdMappingHost_; // DRAM-全局部署专家对应逻辑专家映射
                                                         // shape: [num_layers,
                                                         // world_size_,
                                                         // num_deploy_experts_per_device_]

    aclrtStream stream_; // 用于mapping同步更新的Stream
    std::vector<int32_t> expert_id_deployed_nums_host_;
    Tensor expert_id_deployed_nums_;

    bool enable_rank_round_robin_ = false;

  private:
    /**
     * @brief Converts tensor data to 3D vector of integers
     * @return 3D vector containing placement pattern data
     */
    std::vector<std::vector<std::vector<int>>>
    torch_tensor_to_3d_vector_int32();

    /**
     * @brief Load placement pattern from binary file
     * @param filename Path to the binary file
     * @return 3D vector containing the loaded placement pattern
     */
    std::vector<std::vector<std::vector<int>>>
    load_placement_pattern_from_file(const char *filename);

    void init_placement_pattern(const std::string &placement_pattern_filename,
                                size_t placement_pattern_ptr,
                                std::vector<int64_t> placement_shape);

  public:
    /**
     * @brief Constructor for PlacementMapping
     * @param placement_pattern_filename Path to binary file containing
     * placement pattern (used if placement_pattern_ptr is empty)
     * @param rank Current device's rank in the distributed system
     * @param num_devices_per_host Number of devices per host/node
     * @param max_num_deployed_expert Number of deployed experts per layer
     * @param placement_pattern_ptr Pointer to placement pattern data in memory
     * (can be nullptr if using file)
     * @param placement_shape Shape of the placement pattern tensor (used if
     * placement_pattern_ptr is not nullptr)
     * @param selector_ptr Mapping of expert_id 2 its global position
     */
    PlacementMapping(const std::string &placement_pattern_filename, int rank,
                     int num_devices_per_host, int max_redundant_per_expert,
                     int max_num_deployed_expert, size_t placement_pattern_ptr,
                     std::vector<int64_t> pattern_shape, size_t selector_ptr,
                     bool enable_rank_round_robin,
                     size_t num_redundant_per_expert_ptr);

    /**
     * @brief Destructor for PlacementMapping
     */
    virtual ~PlacementMapping();

    // Getter methods for class properties
    int get_rank() { return rank_; }
    int get_world_size() { return world_size_; }
    int get_num_layers() { return num_layers_; }
    int get_num_experts() { return num_experts_; }
    int get_num_deploy_experts() { return num_deploy_experts_; }
    int get_num_devices_per_host() { return num_devices_per_host_; }

    int32_t get_redundant_count(int32_t layer_id, int32_t expert_id);
    int32_t get_position_expert_id(int32_t layer_id, int32_t position_id);

    /**
     * @brief 根绝物理位置查询逻辑ID的相关操作
     */
    void init_DeployedPositionToLogisticsIdMapping();
    void release_DeployedPositionToLogisticsIdMapping();

    /**
     * @param t_rank 握手信息， -1 不发生交换
     * @param layer_id layer_id
     * @param global_position_id_this_layer  当前层部署位置要发生替换
     * @param expert_id 要替换为的逻辑专家ID
     * @param dist_ptr 通信工具
     */
    bool update_globalDeployedPositionToLogisticsIdMapping(
        std::vector<int> global_synchronize_mapping_info, size_t num_info,
        std::vector<bool> &is_layer_update); // 子线程调用
    bool checkUpdateIsValied(size_t layer_id, int expert_id, int ops);

    int getGlobalPositionOffset(int layer_id,
                                int global_position_id_this_layer);
    int get_num_redundant_per_rank() { return max_redundant_per_rank_; }
    void printDeployedPositionToLogisticsIdMapping();
    bool checkPositionIsConsistency(size_t layer_id, size_t global_position,
                                    int expert_id);

    // 新增 getter 方法
    std::vector<int32_t>
    get_global_deployed_position_to_logistics_id_mapping() const {
        return globalDeployedPositionToLogisticsIdMappingHost_;
    }

    void update_globalDeployedPositionToLogisticsIdMapping(int layer_id,
                                                           size_t offset,
                                                           int expert_id);

    int32_t get_expert_id(int layer_id, int global_position_id_this_layer) {
        size_t global_offset =
            getGlobalPositionOffset(layer_id, global_position_id_this_layer);
        return globalDeployedPositionToLogisticsIdMappingHost_[global_offset];
    }

    int get_max_redundant_per_expert() const {
        return max_redundant_per_expert_;
    }
    void update_selector(std::vector<bool> is_layer_update);
    void update_selector_layer(
        int layer_id, std::vector<int> finish_table,
        std::vector<std::vector<int32_t>> same_host_candidates,
        std::vector<std::vector<int32_t>> distant_candidates);
    void update_selector_layer(int layer_id);
    Tensor get_selector() { return selector_; }
    void set_rank(int rank) { rank_ = rank; }
    void init_selector(size_t selector_ptr);
};

#endif // PLACEMENT_MAPPING_H