// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_mapping.h"
#include <algorithm> // for std::find
#include <assert.h>
#include <iostream>
#include <string.h>
#include <tensor.h>
#include <tuple>

const int global_ScalarType_Int = 3;
PlacementMapping::PlacementMapping(
    const std::string &placement_pattern_filename, int rank,
    int num_devices_per_host, int max_redundant_per_expert,
    int max_num_deployed_expert, size_t placement_pattern_ptr,
    std::vector<int64_t> placement_shape, size_t selector_ptr,
    bool enable_rank_round_robin, size_t num_redundant_per_expert_ptr)
    : rank_(rank), num_devices_per_host_(num_devices_per_host),
      max_redundant_per_expert_(max_redundant_per_expert),
      enable_rank_round_robin_(enable_rank_round_robin) {

    // init_placement_pattern
    init_placement_pattern(placement_pattern_filename, placement_pattern_ptr,
                           placement_shape);

    assert(world_size_ >= 1);
    assert(num_layers_ >= 1);
    assert(num_experts_ >= 1);

    // Calculate experts per device
    num_experts_per_device_ = num_experts_ / world_size_;
    assert(num_experts_per_device_ >= 1);

    num_deploy_experts_ = max_num_deployed_expert;
    num_deploy_experts_per_device_ = num_deploy_experts_ / world_size_;
    assert(num_deploy_experts_ >= num_experts_);
    assert(num_deploy_experts_per_device_ >= num_experts_per_device_);

    max_redundant_per_rank_ =
        num_deploy_experts_per_device_ - num_experts_per_device_;
    assert(max_redundant_per_rank_ >= 1);

    // Construct mappings
    init_DeployedPositionToLogisticsIdMapping();
    expert_id_deployed_nums_ =
        Tensor((uint64_t)num_redundant_per_expert_ptr,
               get_num_layers() * get_num_experts(), sizeof(int32_t), "int32_t",
               "redundant_nums");

    // Mapping更新到HBM用的Stream
    ACLCHECK(aclrtCreateStream(&stream_));

    // Init selector
    init_selector(selector_ptr);
}

PlacementMapping::~PlacementMapping() { aclrtDestroyStream(stream_); }

void PlacementMapping::init_placement_pattern(
    const std::string &placement_pattern_filename, size_t placement_pattern_ptr,
    std::vector<int64_t> placement_shape) {
    // 检查是否提供了文件名且指针为空
    bool use_file = (!placement_pattern_filename.empty() &&
                     placement_pattern_filename[0] != '\0');
    if (use_file) {
        // 从文件加载 placement_pattern
        placement_pattern_vector_ = load_placement_pattern_from_file(
            placement_pattern_filename.c_str());

        // 根据读取的数据计算参数
        world_size_ = placement_pattern_vector_.size();
        num_layers_ = placement_pattern_vector_[0].size();
        num_experts_ = placement_pattern_vector_[0][0].size();

        // placement_pattern_ 指针设为 nullptr，因为我们直接使用 vector
        placement_pattern_ = nullptr;
        placement_pattern_dtype_ =
            global_ScalarType_Int; // 假设文件中的数据是 int32 类型
    } else {
        // 原来的方式：从内存指针构造
        if (placement_pattern_ptr == 0) {
            throw std::invalid_argument(
                "Both placement_pattern_ptr and "
                "placement_pattern_filename are invalid");
        }
        placement_pattern_dtype_ = 3;
        placement_pattern_ = (int32_t *)placement_pattern_ptr;

        // 初始化基于 placement pattern 的属性
        world_size_ = placement_shape[0];
        num_layers_ = placement_shape[1];
        num_experts_ = placement_shape[2];

        // 转换 placement pattern tensor 为 3D vector
        placement_pattern_vector_ = torch_tensor_to_3d_vector_int32();
    }
}

// 从文件加载 placement pattern 的辅助方法
std::vector<std::vector<std::vector<int>>>
PlacementMapping::load_placement_pattern_from_file(const char *filename) {
    std::ifstream file(filename); // Open in text mode
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open placement pattern file: " +
                                 std::string(filename));
    }

    // Read shape
    int32_t shape[3]; // num_devices, num_layers, num_experts
    for (int i = 0; i < 3; ++i) {
        if (!(file >> shape[i])) {
            file.close();
            throw std::runtime_error(
                "Failed to read placement pattern shape from file");
        }
    }

    // Initialize 3D array
    int num_devices = shape[0];
    int num_layers = shape[1];
    int num_experts = shape[2];
    std::vector<std::vector<std::vector<int>>> pattern(
        num_devices, std::vector<std::vector<int>>(
                         num_layers, std::vector<int>(num_experts)));

    // Read subsequent 0s and 1s
    long long expected_count =
        static_cast<long long>(num_devices) * num_layers * num_experts;
    long long read_count = 0;
    int value;
    for (int i = 0; i < num_devices; ++i) {
        for (int j = 0; j < num_layers; ++j) {
            for (int k = 0; k < num_experts; ++k) {
                if (!(file >> value)) {
                    file.close();
                    throw std::runtime_error(
                        "Failed to read sufficient placement "
                        "pattern data; file may be incomplete");
                }
                if (value != 0 && value != 1) {
                    file.close();
                    throw std::runtime_error("Placement pattern contains "
                                             "invalid value (not 0 or 1)");
                }
                pattern[i][j][k] = value;
                ++read_count;
            }
        }
    }

    // Check for excess data
    if (file >> value) {
        std::cerr << "Warning: File contains excess data, not read"
                  << std::endl;
    }

    // Verify read count
    if (read_count != expected_count) {
        file.close();
        throw std::runtime_error(
            "Number of read values does not match expected shape");
    }

    file.close();
    return pattern;
}

std::vector<std::vector<std::vector<int>>>
PlacementMapping::torch_tensor_to_3d_vector_int32() {
    std::vector<std::vector<std::vector<int>>> result(
        world_size_, std::vector<std::vector<int>>(
                         num_layers_, std::vector<int>(num_experts_)));
    if (placement_pattern_dtype_ == global_ScalarType_Int) {
        int32_t *data_ptr = placement_pattern_;
        for (int i = 0; i < world_size_; i++) {
            for (int j = 0; j < num_layers_; j++) {
                for (int k = 0; k < num_experts_; k++) {
                    int idx =
                        i * (num_layers_ * num_experts_) + j * num_experts_ + k;
                    result[i][j][k] = data_ptr[idx];

                    assert(result[i][j][k] >= 0);
                    assert(result[i][j][k] <= 1);
                }
            }
        }
    } else {
        throw std::runtime_error(
            "Unsupported data type in torch_tensor_to_3d_vector_int32: " +
            std::to_string(placement_pattern_dtype_));
    }
    return result;
}

int32_t PlacementMapping::get_position_expert_id(int32_t layer_id,
                                                 int32_t position_id) {
    // 边界检查
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " +
                                std::to_string(layer_id));
    }
    if (position_id < 0 || position_id >= get_num_deploy_experts()) {
        throw std::out_of_range("position out of range: " +
                                std::to_string(get_num_deploy_experts()));
    }

    // 计算三维数组的线性化偏移量
    int64_t offset = (layer_id * get_num_deploy_experts()) + position_id;

    return globalDeployedPositionToLogisticsIdMappingHost_[offset];
}

int32_t PlacementMapping::get_redundant_count(int32_t layer_id,
                                              int32_t expert_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " +
                                std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " +
                                std::to_string(expert_id));
    }

    int32_t offset = layer_id * num_experts_ + expert_id;
    return expert_id_deployed_nums_host_[offset];
}

void PlacementMapping::init_DeployedPositionToLogisticsIdMapping() {

    expert_id_deployed_nums_host_.resize(get_num_layers() * get_num_experts(),
                                         0);

    size_t length = num_layers_ * num_deploy_experts_;
    globalDeployedPositionToLogisticsIdMappingHost_.resize(length, -1);

    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        for (int32_t rank = 0; rank < world_size_; ++rank) {
            int32_t local_position_id = 0;
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                if (placement_pattern_vector_[rank][layer][expert] == 1) {
                    expert_id_deployed_nums_host_[layer * num_experts_ +
                                                  expert]++;
                    // 确保冗余计数不超过最大允许值
                    assert(expert_id_deployed_nums_host_[layer * num_experts_ +
                                                         expert] <
                           max_redundant_per_expert_);

                    // 计算全局位置ID：使用设备编号乘以每设备最大专家数作为基础偏移，再加上当前位置
                    int32_t global_position_id_per_layer =
                        num_deploy_experts_per_device_ * rank +
                        local_position_id; // offset + cumsum;
                    size_t offset = layer * num_deploy_experts_ +
                                    global_position_id_per_layer;

                    // Set the expert_id on offset Position
                    globalDeployedPositionToLogisticsIdMappingHost_[offset] =
                        expert;

                    local_position_id++;
                }
            }
        }
    }
}

void PlacementMapping::printDeployedPositionToLogisticsIdMapping() {
    // 打印 vector 的大小
    std::cout << "globalDeployedPositionToLogisticsIdMappingHost_ size: "
              << globalDeployedPositionToLogisticsIdMappingHost_.size()
              << std::endl;

    // 打印 vector 的所有值
    std::cout << "globalDeployedPositionToLogisticsIdMappingHost_ values: ";

    std::string log_info = "";
    for (int32_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        for (int32_t position_id_this_layer = 0;
             position_id_this_layer < num_deploy_experts_;
             ++position_id_this_layer) {
            int offset =
                getGlobalPositionOffset(layer_id, position_id_this_layer);
            log_info +=
                std::to_string(position_id_this_layer) + ":" +
                std::to_string(
                    globalDeployedPositionToLogisticsIdMappingHost_[offset]) +
                ", ";
        }
        log_info += "\n";
        break;
    }
    std::cout << log_info << std::endl;
}

void PlacementMapping::update_globalDeployedPositionToLogisticsIdMapping(
    int layer_id, size_t offset, int expert_id) {
    if (globalDeployedPositionToLogisticsIdMappingHost_[offset] != -1) {
        checkUpdateIsValied(
            layer_id, globalDeployedPositionToLogisticsIdMappingHost_[offset],
            -1);
    }
    if (expert_id != -1) {
        checkUpdateIsValied(layer_id, expert_id, 1);
    }
    globalDeployedPositionToLogisticsIdMappingHost_[offset] = expert_id;
}

bool PlacementMapping::checkPositionIsConsistency(size_t layer_id,
                                                  size_t global_position,
                                                  int expert_id) {
    size_t position_offset =
        layer_id * get_num_deploy_experts() + global_position;
    if (globalDeployedPositionToLogisticsIdMappingHost_[position_offset] ==
        expert_id) {
        return true;
    } else {
        std::cout
            << "[Error]-layer[" << layer_id << "]-global_position["
            << global_position << "]-expert_id["
            << globalDeployedPositionToLogisticsIdMappingHost_[position_offset]
            << "]-InstructionExpertId[" << expert_id << "]\n";
        return false;
    }
}

bool PlacementMapping::checkUpdateIsValied(size_t layer_id, int expert_id,
                                           int ops) {
    // 判断更新是否符合
    if (expert_id == -1)
        throw std::runtime_error("[Error]-rank[" + std::to_string(rank_) +
                                 "]  checkUpdateIsValied");
    size_t offset = get_num_experts() * layer_id + expert_id;
    expert_id_deployed_nums_host_[offset] += ops;
    if (expert_id_deployed_nums_host_[offset] < 0) {
        std::cout << "[Error] rank[" << rank_
                  << "] on Deployed Mapping Update !!!!!!!!!! with\t layer_id["
                  << layer_id << "] expert_id[" << expert_id << "] nums["
                  << expert_id_deployed_nums_host_[offset] << "]" << std::endl;
        ;
        return false;
    }
    return true;
}

bool PlacementMapping::update_globalDeployedPositionToLogisticsIdMapping(
    std::vector<int> global_synchronize_mapping_info, size_t num_info,
    std::vector<bool> &is_layer_update) {
    bool isUpdateValied;
    for (int32_t rank = 0; rank < world_size_; ++rank) {
        size_t offset = rank * num_info;
        int t_rank = global_synchronize_mapping_info[offset];
        int position_offset = global_synchronize_mapping_info[offset + 1]; //
        int expert_id = global_synchronize_mapping_info[offset + 2];

        if (t_rank == -1 || t_rank == world_size_)
            continue;
        // 对端与当前Rank握手，才发生更新
        if (position_offset != -1 &&
            global_synchronize_mapping_info[t_rank * num_info] == rank) {
            size_t layer_id = position_offset / get_num_deploy_experts();
            is_layer_update[layer_id] = true;
            update_globalDeployedPositionToLogisticsIdMapping(
                layer_id, position_offset, expert_id);
        }
    }
    return true;
}

int PlacementMapping::getGlobalPositionOffset(
    int layer_id, int global_position_id_this_layer) {
    return layer_id * num_deploy_experts_ + global_position_id_this_layer;
}

void PlacementMapping::init_selector(size_t selector_ptr) {
    size_t size;
    if (enable_rank_round_robin_) {
        size = num_layers_ * num_experts_;
    } else {
        size = num_layers_ * get_max_redundant_per_expert() * num_experts_;
    }

    selector_host_.resize(size, 0);
    selector_ = Tensor((uint64_t)selector_ptr, size, sizeof(int32_t), "int32_t",
                       "selector");
    std::vector<bool> tmp(num_layers_, true);
    update_selector(tmp);
}

void PlacementMapping::update_selector_layer(
    int layer_id, std::vector<int> finish_table,
    std::vector<std::vector<int32_t>> same_host_candidates,
    std::vector<std::vector<int32_t>> distant_candidates) {
    size_t position_offset = layer_id * this->num_deploy_experts_;
    size_t logits_offset = layer_id * this->num_experts_;

    for (int i = 0; i < num_experts_; ++i) {
        finish_table[i] = 0;
        same_host_candidates[i].clear();
        distant_candidates[i].clear();
    }

    for (int pos_id = 0; pos_id < num_deploy_experts_; ++pos_id) {

        int logit_expert_id =
            globalDeployedPositionToLogisticsIdMappingHost_[position_offset +
                                                            pos_id];

        if (logit_expert_id == -1)
            continue;

        if (get_redundant_count(layer_id, logit_expert_id) == 1) {
            selector_host_[logit_expert_id + logits_offset] = pos_id;
            finish_table[logit_expert_id] = true;
            continue;
        }

        int phy_device = pos_id / num_deploy_experts_per_device_;
        int phy_host = phy_device / num_devices_per_host_;
        int rank_host = rank_ / num_devices_per_host_;

        if (finish_table[logit_expert_id] == 2)
            continue;

        if (phy_device == rank_) {
            selector_host_[logit_expert_id + logits_offset] = pos_id;
            finish_table[logit_expert_id] = 2;
        } else if (finish_table[logit_expert_id] < 2 && phy_host == rank_host) {
            same_host_candidates[logit_expert_id].push_back(pos_id);
            selector_host_[logit_expert_id + logits_offset] =
                same_host_candidates
                    [logit_expert_id]
                    [rank_ % same_host_candidates[logit_expert_id].size()];
            finish_table[logit_expert_id] = 1;
        } else if (finish_table[logit_expert_id] < 1) {
            distant_candidates[logit_expert_id].push_back(pos_id);
            selector_host_[logit_expert_id + logits_offset] =
                distant_candidates[logit_expert_id]
                                  [rank_ %
                                   distant_candidates[logit_expert_id].size()];
        }
    }
}

void PlacementMapping::update_selector_layer(int layer_id) {
    // For disable topological affinity selector

    size_t position_offset = layer_id * this->num_deploy_experts_;
    size_t logits_offset =
        layer_id * get_max_redundant_per_expert() * this->num_experts_;

    std::vector<int32_t> cur_count(get_num_experts(), 0);

    for (int pos_id = 0; pos_id < num_deploy_experts_; ++pos_id) {
        int logit_expert_id =
            globalDeployedPositionToLogisticsIdMappingHost_[position_offset +
                                                            pos_id];
        if (logit_expert_id == -1)
            continue;

        int32_t redundant_idx = cur_count[logit_expert_id];
        size_t offset = logits_offset +
                        logit_expert_id * get_max_redundant_per_expert() +
                        redundant_idx;
        selector_host_[offset] = pos_id;
        cur_count[logit_expert_id]++;
    }
}

void PlacementMapping::update_selector(std::vector<bool> is_layer_update) {

    std::vector<int> finish_table(num_experts_, 0);
    std::vector<std::vector<int32_t>> same_host_candidates(
        num_experts_, std::vector<int32_t>());
    std::vector<std::vector<int32_t>> distant_candidates(
        num_experts_, std::vector<int32_t>());

    for (int i = 0; i < num_experts_; ++i) {
        same_host_candidates[i].reserve(max_redundant_per_expert_);
        distant_candidates[i].reserve(max_redundant_per_expert_);
    }

    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        if (is_layer_update[layer_id])
            if (enable_rank_round_robin_) {
                update_selector_layer(layer_id, finish_table,
                                      same_host_candidates, distant_candidates);
            } else {
                update_selector_layer(layer_id);
            }
    }

    selector_.to_device(selector_host_.data(), stream_);
    expert_id_deployed_nums_.to_device(expert_id_deployed_nums_host_.data(),
                                       stream_);
}
