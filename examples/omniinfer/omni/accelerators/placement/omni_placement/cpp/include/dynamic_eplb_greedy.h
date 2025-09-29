// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef DYNAMIC_EPLB_GREEDY_H
#define DYNAMIC_EPLB_GREEDY_H

#include "expert_load_balancer.h"
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <vector>

class LogitExpertInformation {
  private:
    std::vector<int> global_position_ids;
    std::vector<int> activated_values;

  public:
    LogitExpertInformation() {}
    void remove(int global_position_id) {
        for (size_t idx = 0; idx < global_position_ids.size(); ++idx) {
            if (global_position_ids[idx] == global_position_id) {
                global_position_ids.erase(global_position_ids.begin() + idx);
                activated_values.erase(activated_values.begin() + idx);
                break;
            }
        }
    }

    void update(int global_position_id, int value, float keep_ratio) {
        int idx = find_position(global_position_id);
        if (idx == -1) {
            global_position_ids.emplace_back(global_position_id);
            activated_values.emplace_back((1 - keep_ratio) * value);
        } else {
            activated_values[idx] =
                (1 - keep_ratio) * value + activated_values[idx] * keep_ratio;
        }
    }

    // step1. 平均一份
    // step2. 原来各份合并出新增的这一份
    // step3. 计算原来各份减少多少
    void update() {
        int value = getActivateValue();
        int update_value = value / activated_values.size(); // 新增均分
        if (activated_values.size() <= 1)
            throw std::runtime_error(
                "Invalid activated_values.size() is less than 2 , could split");
        int decrease = update_value / (activated_values.size() - 1);
        update_value = 0;
        for (size_t idx = 0; idx < activated_values.size() - 1; ++idx) {
            if (decrease > activated_values[idx]) {
                activated_values[idx] = 0;
                update_value = update_value + activated_values[idx];
            } else {
                activated_values[idx] = activated_values[idx] - decrease;
                update_value = update_value + decrease;
            }
        }
        activated_values[activated_values.size() - 1] = update_value;
    }

    void update(int value) {
        // 把value均分到剩下的冗余专家
        int update_value = value / activated_values.size();
        for (size_t idx = 0; idx < activated_values.size(); ++idx) {
            activated_values[idx] = activated_values[idx] + update_value;
        }
    }

    bool is_redundant() { return global_position_ids.size() > 1; }

    int find_position(int global_position_id) {
        for (size_t idx = 0; idx < global_position_ids.size(); ++idx) {
            if (global_position_ids[idx] == global_position_id)
                return idx;
        }
        return -1;
    }

    int getNumRedundants() const { return global_position_ids.size(); }

    std::vector<int> getRedundantRanks(int num_deployed_expert_per_rank) {
        std::vector<int> results;
        for (size_t idx = 0; idx < global_position_ids.size(); ++idx) {
            int rank = global_position_ids[idx] / num_deployed_expert_per_rank;
            results.emplace_back(rank);
        }
        return results;
    }

    std::vector<int> getLocalPositionIdx(int num_deployed_expert_per_rank) {
        std::vector<int> results;
        for (size_t idx = 0; idx < global_position_ids.size(); ++idx) {
            int localpositionidx =
                global_position_ids[idx] % num_deployed_expert_per_rank;
            results.emplace_back(localpositionidx);
        }
        return results;
    }

    std::vector<int> getGlobalPositionIdx() { return global_position_ids; }

    int getActivateValue(int global_position_id) {
        int idx = find_position(global_position_id);
        if (idx == -1) {
            throw std::runtime_error("this logit expert is not deployed on "
                                     "global_position_per_layer[" +
                                     std::to_string(global_position_id) + "]");
        }
        return activated_values[idx];
    }

    int getActivateValue() {
        int value = 0;
        for (size_t idx = 0; idx < activated_values.size(); ++idx) {
            value = value + activated_values[idx];
        }
        return value;
    }
    void print() {
        std::cout << "global_position_ids[";
        for (size_t idx = 0; idx < activated_values.size(); ++idx) {
            std::cout << global_position_ids[idx] << ", ";
        }
        std::cout << "] with activated_values[";
        for (size_t idx = 0; idx < activated_values.size(); ++idx) {
            std::cout << activated_values[idx] << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

class RankActivateInformation {
  private:
    std::vector<int> deployed_to_logits_mapping_;
    int rank_;
    std::vector<int> pos_activations_; // 该卡的当前负载
  public:
    RankActivateInformation(int rank, int num_deployed_experts_per_rank) {
        deployed_to_logits_mapping_.resize(num_deployed_experts_per_rank, -1);
        rank_ = rank;
        pos_activations_.resize(num_deployed_experts_per_rank, 0);
    }
    bool is_empty(int position_id) {
        return deployed_to_logits_mapping_[position_id] == -1;
    }
    int getExpertID(int position_id) {
        return deployed_to_logits_mapping_[position_id];
    }
    int get_value() const {
        int value = 0;
        for (size_t idx = 0; idx < pos_activations_.size(); ++idx) {
            value = value + pos_activations_[idx];
        }
        return value;
    }
    int get_value(int local_position_idx) const {
        return pos_activations_[local_position_idx];
    }
    int getNumPosition() const { return deployed_to_logits_mapping_.size(); }
    void update(int position_id, int expert_id, int value) {
        deployed_to_logits_mapping_[position_id] = expert_id;
        if (expert_id == -1 && value != 0) {
            throw std::runtime_error("expert_id is -1 while value[" +
                                     std::to_string(value) +
                                     "] is not equal to 0");
        }
        pos_activations_[position_id] = value;
    }
    int get_rank_id() const { return rank_; }

    void print() {
        std::cout << "rank[" << rank_ << "] ";
        for (size_t idx = 0; idx < pos_activations_.size(); ++idx) {
            std::cout << pos_activations_[idx] << " ";
        }
        std::cout << std::endl;
    }
};

class GreedyExpertLoadBalancer {
  private:
    int num_layers_;
    int world_size_;
    int num_experts_;
    int num_deployed_experts_;
    int num_deployed_experts_per_rank_;
    int expert_redundant_limit_;
    int rank_;
    std::vector<LogitExpertInformation *>
        logit_expert_info_ptrs_; // 所有layer的逻辑专家部署以及激活信息
    std::vector<RankActivateInformation *>
        rankinfo_ptrs_; // 所有layer的各个Rank的激活信息
    float alpha_ = 0.6; // 0.6; // activation 保留速率
    void init_infomation(int layer_id, const std::vector<int> &placement,
                         const std::vector<int64_t> &activations);
    float getUnbalancedRatio(const std::vector<int> &all_ranks_values);
    std::vector<int> getAllRanksValue(int layer_id);
    int getBestEPValue(const std::vector<int> &all_ranks_values);
    LogitExpertInformation *getLogitExpertInfo(int layer_id, int expert_id) {
        if (expert_id < 0 || expert_id >= num_experts_) {
            throw std::out_of_range("expert_id " + std::to_string(expert_id) +
                                    " is out of valid range (0, " +
                                    std::to_string(num_experts_) + ")");
        }
        int offset = layer_id * num_experts_ + expert_id;
        return logit_expert_info_ptrs_[offset];
    }
    RankActivateInformation *getRankInfos(int layer_id, int rank) {
        int offset = layer_id * world_size_ + rank;
        return rankinfo_ptrs_[offset];
    }
    int getTheHighestOffload(const std::vector<int> &subset_ranks_values,
                             int bestEPValue,
                             const std::vector<int> &include_ranks);
    std::vector<ChangeInstruction>
    generate_remove_instructions_per_layer(int layer_id, int bestEP_value);
    RankActivateInformation *
    getTheHighestOffloadRank(int layer_id,
                             const std::vector<int> &exclude_ranks);
    RankActivateInformation *
    getTheLowestOffloadRank(int layer_id,
                            const std::vector<int> &exclude_ranks);
    void update(ChangeInstruction instruction);
    ChangeInstruction optimizeTheHighestOffload(int layer_id);
    ChangeInstruction optimizeTheLowestOffload(int layer_id);
    std::vector<ChangeInstruction>
    generate_swap_add_instructions_per_layer(int layer_id);
    bool isInvalidOffloadRatio(float ratio) const { return ratio > 300; }

  public:
    GreedyExpertLoadBalancer(int num_layers, int world_size, int num_experts,
                             int num_deployed_experts,
                             int expert_redundant_limit, int rank);
    ~GreedyExpertLoadBalancer();
    std::vector<ChangeInstruction>
    optimize_and_generate_instructions(const std::vector<int> &placement,
                                       const std::vector<int64_t> &activations);
};
#endif // DYNAMIC_EPLB_GREEDY_H