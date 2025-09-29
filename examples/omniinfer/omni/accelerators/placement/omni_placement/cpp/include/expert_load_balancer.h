// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef EXPERT_LOAD_BALANCER_H
#define EXPERT_LOAD_BALANCER_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

enum class OperationType {
    SWAP = 0,
    ADD = 1,
    REMOVE = 2,
    EMPTY = 3,
};

struct ChangeInstruction {
    int layer_idx;              // 层索引
    OperationType type;         // 操作类型：SWAP、ADD、REMOVE
    int source_rank;            // 源 rank
    int source_expert_id;       // 源专家 ID
    int source_global_position; // 源全局位置
    int target_rank;            // 目标 rank
    int target_expert_id;       // 目标专家 ID（对于 ADD 通常为 -1）
    int target_global_position; // 目标全局位置
    int prior;                  // 指令优先级
    bool operator<(const ChangeInstruction &other) const {
        return prior < other.prior;
    }
};

struct ExpertInformation {
    int layer_idx;       // 层索引
    int rank_id;         // 设备 rank 标识
    int expert_id;       // 专家标识
    int64_t activations; // 激活值
    int global_position; // 全局位置
    int total_count;     // 该专家在层中的总出现次数
};

// 修改后的 generate_bucket_balanced_placement 返回类型
struct BucketBalancedResult {
    std::vector<ChangeInstruction> instructions;
    std::vector<int> placement;
    std::vector<std::pair<ChangeInstruction, int>> instruction_depths;
};

// expert_load_balancer.h
struct LayerDebugInfo {
    int layer_idx;
    std::vector<int> input_placement;
    std::vector<int64_t> input_activations;
    std::vector<int> initial_placement;
    std::vector<int> optimized_placement;
    std::vector<int> output_placement;
    std::vector<ChangeInstruction> instructions;
    std::vector<ChangeInstruction> adjustment_instructions;
    std::vector<std::tuple<double, int, ChangeInstruction>>
        candidate_adjustments;
    std::vector<std::pair<ChangeInstruction, int>>
        instruction_depths; // 新增：存储指令和深度
    int total_diff_count;
};

class ExpertLoadBalancer {
  public:
    // 构造函数，初始化优化器参数
    ExpertLoadBalancer(int num_layers, int num_ranks, int num_experts_per_rank,
                       int num_redundant_per_rank, int expert_redundant_limit,
                       int rank);

    // 主优化函数，返回优化指令
    std::vector<ChangeInstruction> optimize_and_generate_instructions(
        const std::vector<int> &input_placement,
        const std::vector<int64_t> &input_activations);

    // Getter 方法：基本属性
    int get_num_layers() const { return num_layers_; }
    int get_num_ranks() const { return num_ranks_; }
    int get_num_experts_per_rank() const { return num_experts_per_rank_; }
    int get_num_redundant_per_rank() const { return num_redundant_per_rank_; }
    int get_expert_redundant_limit() const { return expert_redundant_limit_; }
    int get_num_experts() const { return num_ranks_ * num_experts_per_rank_; }
    int get_max_slots_per_rank() const {
        return num_experts_per_rank_ + num_redundant_per_rank_;
    }

    // Getter 方法：私有函数的公共接口
    std::vector<std::set<int>>
    get_compute_rank_sets(const std::vector<int> &placement, int num_ranks,
                          int max_slots_per_rank) {
        return compute_rank_sets(placement, num_ranks, max_slots_per_rank);
    }
    int get_find_position_with_expert(const std::vector<int> &placement, int r,
                                      int k, int max_slots_per_rank) {
        return find_position_with_expert(placement, r, k, max_slots_per_rank);
    }
    int get_find_empty_position(const std::vector<int> &placement, int r,
                                int max_slots_per_rank) {
        return find_empty_position(placement, r, max_slots_per_rank);
    }
    std::unordered_map<int, int>
    get_compute_expert_counts(const std::vector<int> &placement, int num_ranks,
                              int max_slots_per_rank) {
        return compute_expert_counts(placement, num_ranks, max_slots_per_rank);
    }
    bool get_validate_input_size(const std::vector<int> &placement,
                                 const std::vector<int64_t> &activations,
                                 int num_layers, int num_ranks,
                                 int max_slots_per_rank) {
        return validate_input_size(placement, activations, num_layers,
                                   num_ranks, max_slots_per_rank);
    }
    bool get_validate_unique_expert_ids(const std::vector<int> &placement,
                                        int layer_idx, int num_ranks,
                                        int max_slots_per_rank) {
        return validate_unique_expert_ids(placement, layer_idx, num_ranks,
                                          max_slots_per_rank);
    }
    bool get_validate_all_experts_present(const std::vector<int> &placement,
                                          int layer_idx, int num_ranks,
                                          int max_slots_per_rank,
                                          int num_experts) {
        return validate_all_experts_present(placement, layer_idx, num_ranks,
                                            max_slots_per_rank, num_experts);
    }
    std::unordered_map<int, double>
    get_compute_expert_loads(const std::vector<ExpertInformation> &experts,
                             int num_experts) {
        return compute_expert_loads(experts, num_experts);
    }
    std::vector<int64_t> get_compute_expert_total_activations(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        int num_layers, int num_ranks, int num_experts_per_rank) {
        return compute_expert_total_activations(
            layer_experts, num_layers, num_ranks, num_experts_per_rank);
    }
    double
    get_compute_placement_ratio(const std::vector<int> &placement,
                                const std::vector<int64_t> &expert_activations,
                                int layer_idx, int num_ranks,
                                int max_slots_per_rank, int num_experts) {
        return compute_placement_ratio(placement, expert_activations, layer_idx,
                                       num_ranks, max_slots_per_rank,
                                       num_experts);
    }
    double get_simulate_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts) {
        return simulate_placement_ratio(placement, expert_loads, layer_idx,
                                        num_ranks, max_slots_per_rank,
                                        num_experts);
    }
    std::vector<ExpertInformation> get_extract_layer_expert_info(
        const std::vector<int> &placement,
        const std::vector<int64_t> &activations, int layer_idx, int num_ranks,
        int max_slots_per_rank, int num_experts, int expert_redundant_limit) {
        return extract_layer_expert_info(placement, activations, layer_idx,
                                         num_ranks, max_slots_per_rank,
                                         num_experts, expert_redundant_limit);
    }
    std::vector<std::vector<ExpertInformation>> get_extract_expert_info(
        const std::vector<int> &placement,
        const std::vector<int64_t> &activations, int num_layers, int num_ranks,
        int num_experts_per_rank, int num_redundant_per_rank,
        int expert_redundant_limit) {
        return extract_expert_info(
            placement, activations, num_layers, num_ranks, num_experts_per_rank,
            num_redundant_per_rank, expert_redundant_limit);
    }
    bool get_validate_expert_counts(const std::vector<int> &placement,
                                    int num_ranks, int max_slots_per_rank,
                                    int num_experts_per_rank,
                                    int num_redundant_per_rank, int layer_idx) {
        return validate_expert_counts(placement, num_ranks, max_slots_per_rank,
                                      num_experts_per_rank,
                                      num_redundant_per_rank, layer_idx);
    }

    void print_debug_info(
        const std::vector<LayerDebugInfo> &layer_info, bool enable_debug,
        const std::chrono::high_resolution_clock::time_point &start_time = {});

    // 新增：simulate_adjusted_placement_ratio 的公共接口
    double get_simulate_adjusted_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads,
        const std::vector<int> &remaining_hot_ranks, double avg_load,
        int layer_idx, int num_ranks, int max_slots_per_rank, int num_experts) {
        return simulate_adjusted_placement_ratio(
            placement, expert_loads, remaining_hot_ranks, avg_load, layer_idx,
            num_ranks, max_slots_per_rank, num_experts);
    }

    BucketBalancedResult get_generate_bucket_balanced_placement(
        const std::vector<int> &layer_placement,
        const std::vector<int64_t> &layer_activations, int layer_idx,
        int num_ranks, int num_experts_per_rank) {
        return generate_bucket_balanced_placement(
            layer_placement, layer_activations, layer_idx, num_ranks,
            num_experts_per_rank);
    }

  private:
    // 数据成员
    int rank_;
    int num_layers_;
    int num_ranks_;
    int num_experts_per_rank_;
    int num_redundant_per_rank_;
    int expert_redundant_limit_;
    int comm_limit_;                    // 每卡通信次数上限
    std::vector<int> rank_comm_counts_; // 每卡通信次数

    // 私有成员函数
    std::vector<std::set<int>>
    compute_rank_sets(const std::vector<int> &placement, int num_ranks,
                      int max_slots_per_rank);
    int find_position_with_expert(const std::vector<int> &placement, int r,
                                  int k, int max_slots_per_rank);
    int find_empty_position(const std::vector<int> &placement, int r,
                            int max_slots_per_rank);
    std::unordered_map<int, int>
    compute_expert_counts(const std::vector<int> &placement, int num_ranks,
                          int max_slots_per_rank);
    bool validate_input_size(const std::vector<int> &placement,
                             const std::vector<int64_t> &activations,
                             int num_layers, int num_ranks,
                             int max_slots_per_rank);
    bool validate_unique_expert_ids(const std::vector<int> &placement,
                                    int layer_idx, int num_ranks,
                                    int max_slots_per_rank);
    bool validate_all_experts_present(const std::vector<int> &placement,
                                      int layer_idx, int num_ranks,
                                      int max_slots_per_rank, int num_experts);
    std::vector<ExpertInformation> extract_layer_expert_info(
        const std::vector<int> &placement,
        const std::vector<int64_t> &activations, int layer_idx, int num_ranks,
        int max_slots_per_rank, int num_experts, int expert_redundant_limit);
    std::vector<std::vector<ExpertInformation>>
    extract_expert_info(const std::vector<int> &placement,
                        const std::vector<int64_t> &activations, int num_layers,
                        int num_ranks, int num_experts_per_rank,
                        int num_redundant_per_rank, int expert_redundant_limit);
    std::vector<int64_t> compute_expert_total_activations(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        int num_layers, int num_ranks, int num_experts_per_rank);

    double
    compute_placement_ratio(const std::vector<int> &placement,
                            const std::vector<int64_t> &expert_activations,
                            int layer_idx, int num_ranks,
                            int max_slots_per_rank, int num_experts);

    double simulate_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts);

    std::unordered_map<int, double>
    compute_expert_loads(const std::vector<ExpertInformation> &experts,
                         int num_experts);

    bool validate_expert_counts(const std::vector<int> &placement,
                                int num_ranks, int max_slots_per_rank,
                                int num_experts_per_rank,
                                int num_redundant_per_rank, int layer_idx);

    // 新增：辅助函数，寻找两个 buckets 间的最佳交换对
    std::optional<std::pair<int, int>>
    find_best_swap(std::vector<int> &placement1,
                   std::vector<int64_t> &activations1,
                   std::vector<int> &placement2,
                   std::vector<int64_t> &activations2, int max_slots_per_rank);

    void buck_balance(
        std::vector<int> &placement, std::vector<int64_t> &activations,
        int left, int right, int layer_idx, int max_slots_per_rank,
        int num_ranks,
        std::vector<std::pair<ChangeInstruction, int>> &instruction_depths,
        int depth = 0);

    BucketBalancedResult generate_bucket_balanced_placement(
        const std::vector<int> &layer_placement,
        const std::vector<int64_t> &layer_activations, int layer_idx,
        int num_ranks, int num_experts_per_rank);

    // 新增：模拟调整后的排布比率，排除未优化的最热 rank
    double simulate_adjusted_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads,
        const std::vector<int> &remaining_hot_ranks, double avg_load,
        int layer_idx, int num_ranks, int max_slots_per_rank, int num_experts);

    // 修改后的 generate_constrained_placement 声明
    std::vector<int> generate_constrained_placement(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        const std::vector<int> &input_placement,
        const std::vector<int64_t> &input_activations, int layer_idx,
        int num_ranks, int num_experts_per_rank, int num_redundant_per_rank,
        int expert_redundant_limit, int diff_limit,
        std::vector<ChangeInstruction> &instructions,
        std::vector<std::tuple<double, int, ChangeInstruction>>
            &candidate_adjustments);

    std::vector<std::tuple<int, int, int, int, int, int, int, double, int,
                           ChangeInstruction>>
    find_next_candidate(std::vector<int> &g, int layer_idx,
                        std::vector<double> &rank_loads,
                        std::unordered_map<int, double> &expert_loads,
                        std::unordered_map<int, int> &expert_deployments,
                        std::unordered_map<int, int> &expert_counts,
                        std::vector<int> &rank_expert_counts,
                        std::vector<int> &rank_comm_counts, int num_ranks,
                        int max_slots_per_rank, int num_experts,
                        int num_experts_per_rank, int num_redundant_per_rank,
                        int expert_redundant_limit, int comm_limit,
                        int total_diff_count, int diff_limit);

    void print_placement(const std::vector<int> &placement, int layer_idx,
                         const std::string &label);
    void print_instruction(const ChangeInstruction &instr, int depth = -1);
    void print_activations(const std::vector<int64_t> &activations,
                           int layer_idx);

    // 新增：transfer 函数，用于对指令进行优先级排序
    void transfer(std::vector<ChangeInstruction> &changeInstructions);
};

#endif // EXPERT_LOAD_BALANCER_H