// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_optimizer.h"
#include <iostream>        
#include <stdexcept>        
#include <unordered_set>    
#include <numeric>          
#include "expert_activation.h" 
#include "placement_optimizer.h" 
#include <algorithm>

/**
 * @brief PlacementOptimizer 的构造函数。
 *
 * 使用 PlacementMapping 和 ClusterActivation 初始化优化器，并创建 ExpertLoadBalancer 实例。
 *
 * @param placement_mapping PlacementMapping 对象的指针。
 * @param clusterActivation ClusterActivation 对象的指针。
 * @throws std::runtime_error 如果任一指针为空或参数无效。
 */
PlacementOptimizer::PlacementOptimizer(PlacementMapping* placement_mapping, ClusterActivation* clusterActivation)
    : placement_mapping_(placement_mapping),
      clusterActivation_(clusterActivation),
      num_layers_(placement_mapping ? placement_mapping->get_num_layers() : 0),
      rank_(placement_mapping ? placement_mapping->get_rank() : 0),
      world_size_(placement_mapping ? placement_mapping->get_world_size() : 0),
      num_experts_(placement_mapping ? placement_mapping->get_num_experts() : 0),
      num_devices_per_host_(placement_mapping ? placement_mapping->get_num_devices_per_host() : 0),
      num_redundant_per_rank_(placement_mapping ? placement_mapping->get_num_redundant_per_rank() : 0),
      expert_redundant_limit_(placement_mapping ? placement_mapping->get_max_redundant_per_expert() - 1 : 0),
      load_balancer_(nullptr) {
    if (!placement_mapping_ || !clusterActivation_) {
        throw std::runtime_error("无效的初始化参数：placement_mapping 或 clusterActivation 为空");
    }
    if (num_layers_ <= 0 || world_size_ <= 0 || num_experts_ <= 0) {
        throw std::runtime_error("无效的初始化参数：层数、秩数或专家数不合法");
    }
    if (expert_redundant_limit_ < 0) {
        throw std::runtime_error("专家冗余上限不合法：max_redundant_count 必须大于或等于 1");
    }

    // 计算 num_experts_per_rank_
    if (num_experts_ % world_size_ != 0) {
        throw std::runtime_error("专家数 " + std::to_string(num_experts_) + " 不能被秩数 " +
                                 std::to_string(world_size_) + " 整除");
    }
    num_experts_per_rank_ = num_experts_ / world_size_;
    if (num_experts_per_rank_ <= 0) {
        throw std::runtime_error("每秩专家数不合法");
    }

    // 创建 ExpertLoadBalancer 实例
    load_balancer_ = new ExpertLoadBalancer(
        num_layers_,
        world_size_,
        num_experts_per_rank_,
        num_redundant_per_rank_,
        expert_redundant_limit_,
        rank_
    );

    greedy_load_balancer_ = new GreedyExpertLoadBalancer(
        num_layers_,
        world_size_, 
        num_experts_, 
        placement_mapping ? placement_mapping->get_num_deploy_experts() : 0, 
        expert_redundant_limit_, 
        rank_);
}

/**
 * @brief PlacementOptimizer 的析构函数。
 *
 * 释放 load_balancer_ 的资源。
 */
PlacementOptimizer::~PlacementOptimizer() {
    delete load_balancer_;
}

/**
 * @brief 从 PlacementMapping 和 ClusterActivation 中提取所有层的输入数据。
 *
 * @param placement 放置向量。
 * @param activations 激活向量。
 */
void PlacementOptimizer::extract_input_data(
    std::vector<int>& placement,
    std::vector<int64_t>& activations) {
    // 计算每秩的最大槽位数
    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    int expected_size = num_layers_ * world_size_ * max_slots_per_rank;

    // 获取 globalDeployedPositionToLogisticsIdMappingHost_ 作为 placement
    auto global_mapping = placement_mapping_->get_global_deployed_position_to_logistics_id_mapping();
    if (static_cast<int>(global_mapping.size()) != expected_size) {
        throw std::runtime_error("globalDeployedPositionToLogisticsIdMappingHost_ 大小 " +
                                 std::to_string(global_mapping.size()) +
                                 " 与预期大小 " + std::to_string(expected_size) + " 不匹配");
    }

    // 直接使用 global_mapping 作为 placement
    placement = global_mapping;

    // 初始化 activations 向量
    activations.assign(expected_size, 0);

    clusterActivation_->updateShiftWindows(placement_mapping_);

    // 填充 activations
    std::string log = "";
    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        int layer_offset = layer_id * world_size_ * max_slots_per_rank;
        for (int rank = 0; rank < world_size_; ++rank) {
            for (int pos = 0; pos < max_slots_per_rank; ++pos) {
                int idx = layer_offset + rank * max_slots_per_rank + pos;
                int physical_pos = rank * max_slots_per_rank + pos;

                // int expert_id = placement_mapping_->get_expert_id(layer_id, physical_pos);
                // int this_logit_expert_deployed_nums = placement_mapping_->get_redundant_count_easy(layer_id,expert_id);
                // activations[idx] = clusterActivation_->getLogitExpertShiftActivateion(layer_id, expert_id, this_logit_expert_deployed_nums);

                activations[idx] = clusterActivation_->getExpertActivationCount(layer_id, physical_pos); // 旧接口， 依旧保留使用，无滑动窗口
                log += std::to_string(activations[idx])+", ";
            }
        }
        log += "\n";
    }
}


void save_vector(const std::vector<int>& vec, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << " ";
            }
        }
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}

void save_vector(const std::vector<int64_t>& vec, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << " ";
            }
        }
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}
/**
 * @brief 为所有层优化专家放置。
 *
 * 调用 ExpertLoadBalancer 的优化函数生成所有层的更改指令。
 *
 * @return std::vector<ChangeInstruction> 所有层的优化更改指令向量。
 */
std::vector<ChangeInstruction> PlacementOptimizer::optimize() {
    std::vector<int> placement;
    std::vector<int64_t> activations;
    extract_input_data(placement, activations);
    // if (rank_==0){
    //     save_vector(placement, "/data/kww/vllm-09/debug-2/placement-"+std::to_string(record_)+".txt");
    //     save_vector(activations, "/data/kww/vllm-09/debug-2/activations-"+std::to_string(record_)+".txt");
    // }
    // record_++;
    return greedy_load_balancer_->optimize_and_generate_instructions(placement, activations);
    // return load_balancer_->optimize_and_generate_instructions(placement, activations);
}

std::vector<ChangeInstruction> PlacementOptimizer::optimize(std::vector<int> placement, std::vector<int64_t> activations) {
    return greedy_load_balancer_->optimize_and_generate_instructions(placement, activations);
    // return load_balancer_->optimize_and_generate_instructions(placement, activations);
}