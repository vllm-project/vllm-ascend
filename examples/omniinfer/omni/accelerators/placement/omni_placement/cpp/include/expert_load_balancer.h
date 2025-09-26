// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#ifndef EXPERT_LOAD_BALANCER_H
#define EXPERT_LOAD_BALANCER_H

#include <chrono>
#include <cstdint>
#include <iostream>
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
    int layer_idx;              // Layer index
    OperationType type;         // Operation type: SWAP, ADD, REMOVE
    int source_rank;            // Source rank
    int source_expert_id;       // Source expert ID
    int source_global_position; // Source global position
    int target_rank;            // Target rank
    int target_expert_id;       // Target expert ID (typically -1 for ADD)
    int target_global_position; // Target global position
    int prior;                  // Instruction priority
    int round;     // Generation round per layer: from 1 to max_slots_per_rank
    int batch = 0; // Transfer batch
    bool operator<(const ChangeInstruction &other) const {
        return prior < other.prior;
    }
    void print() const {
        std::cout << "ChangeInstruction {"
                  << "  layer: " << layer_idx << ", "
                  << "  batch: " << batch << ", "
                  << "  round: " << round << ", "
                  << "  type: " << (int)type << ", "
                  << "  source_rank: " << source_rank << ", "
                  << "  source_expert_id: " << source_expert_id << ", "
                  << "  source_global_position: " << source_global_position
                  << ", "
                  << "  target_rank: " << target_rank << ", "
                  << "  target_expert_id: " << target_expert_id << ", "
                  << "  target_global_position: " << target_global_position
                  << "}\n";
    }
};

struct ExpertInformation {
    int layer_idx;       // Layer index
    int rank_id;         // Device rank identifier
    int expert_id;       // Expert identifier
    int64_t activations; // Activation count
    int global_position; // Global position
    int total_count;     // Total occurrences of this expert in the layer
};

class ExpertLoadBalancer {
  public:
    // Constructor, initialize optimizer parameters
    ExpertLoadBalancer(int num_layers, int num_ranks, int num_experts_per_rank,
                       int num_redundant_per_rank, int expert_redundant_limit,
                       int rank = 0, double input_ratio_threshold = 1.1,
                       double improvement_threshold = 0.05,
                       int num_ranks_per_host = 16,
                       double high_low_ratio_threshold = 300.0);
    // Main optimization function - now returns optimized placement instead of
    // instructions
    std::vector<int>
    optimize_placement(const std::vector<int> &input_placement,
                       const std::vector<int64_t> &input_activations);

    // Getter methods for UT: basic properties
    int ut_num_layers() const { return num_layers_; }
    int ut_num_ranks() const { return num_ranks_; }
    int ut_num_experts_per_rank() const { return num_experts_per_rank_; }
    int ut_num_redundant_per_rank() const { return num_redundant_per_rank_; }
    int ut_expert_redundant_limit() const { return expert_redundant_limit_; }
    int ut_num_experts() const { return num_ranks_ * num_experts_per_rank_; }
    int ut_max_slots_per_rank() const {
        return num_experts_per_rank_ + num_redundant_per_rank_;
    }
    int ut_num_ranks_per_host() const { return num_ranks_per_host_; }
    double ut_high_low_ratio_threshold() const {
        return high_low_ratio_threshold_;
    }

    // Getter methods for unit testing: public interfaces for private functions
    std::vector<std::set<int>>
    ut_compute_rank_sets(const std::vector<int> &placement, int num_ranks,
                         int max_slots_per_rank) {
        return compute_rank_sets(placement, num_ranks, max_slots_per_rank);
    }

    int ut_find_position_with_expert(const std::vector<int> &placement, int r,
                                     int k, int max_slots_per_rank) {
        return find_position_with_expert(placement, r, k, max_slots_per_rank);
    }

    int ut_find_empty_position(const std::vector<int> &placement, int r,
                               int max_slots_per_rank) {
        return find_empty_position(placement, r, max_slots_per_rank);
    }

    std::unordered_map<int, int>
    ut_compute_expert_counts(const std::vector<int> &placement, int num_ranks,
                             int max_slots_per_rank) {
        return compute_expert_counts(placement, num_ranks, max_slots_per_rank);
    }

    bool ut_validate_input_size(const std::vector<int> &placement,
                                const std::vector<int64_t> &activations,
                                int num_layers, int num_ranks,
                                int max_slots_per_rank) {
        return validate_input_size(placement, activations, num_layers,
                                   num_ranks, max_slots_per_rank);
    }

    bool ut_validate_unique_expert_ids(const std::vector<int> &placement,
                                       int layer_idx, int num_ranks,
                                       int max_slots_per_rank) {
        return validate_unique_expert_ids(placement, layer_idx, num_ranks,
                                          max_slots_per_rank);
    }

    bool ut_validate_all_experts_present(const std::vector<int> &placement,
                                         int layer_idx, int num_ranks,
                                         int max_slots_per_rank,
                                         int num_experts) {
        return validate_all_experts_present(placement, layer_idx, num_ranks,
                                            max_slots_per_rank, num_experts);
    }

    std::unordered_map<int, double>
    ut_compute_expert_loads(const std::vector<ExpertInformation> &experts,
                            int num_experts) {
        return compute_expert_loads(experts, num_experts);
    }

    std::vector<int> ut_allocate_expert_deployments(
        const std::unordered_map<int, double> &expert_loads, int num_experts,
        int budget_limit, int expert_redundant_limit) {
        return allocate_expert_deployments(
            expert_loads, num_experts, budget_limit, expert_redundant_limit);
    }

    double ut_compute_placement_ratio_combined(
        const std::vector<int> &placement,
        const std::vector<int64_t> &expert_activations, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts, int type) {
        return compute_placement_ratio_combined(
            placement, expert_activations, layer_idx, num_ranks,
            max_slots_per_rank, num_experts, type);
    }

    double ut_simulate_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts) {
        return simulate_placement_ratio(placement, expert_loads, layer_idx,
                                        num_ranks, max_slots_per_rank,
                                        num_experts);
    }

    std::vector<int> ut_generate_layer_placement(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        int layer_idx, int num_ranks, int num_experts_per_rank,
        int num_redundant_per_rank, int expert_redundant_limit,
        int budget_limit) {
        return generate_layer_placement(
            layer_experts, layer_idx, num_ranks, num_experts_per_rank,
            num_redundant_per_rank, expert_redundant_limit, budget_limit);
    }

    std::vector<int> ut_select_best_layer_placement(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        const std::vector<int> &input_placement,
        const std::vector<int64_t> &input_activations, int layer_idx,
        int num_ranks, int num_experts_per_rank, int num_redundant_per_rank,
        int expert_redundant_limit) {
        return select_best_layer_placement(
            layer_experts, input_placement, input_activations, layer_idx,
            num_ranks, num_experts_per_rank, num_redundant_per_rank,
            expert_redundant_limit);
    }

    std::vector<ExpertInformation> ut_extract_layer_expert_info(
        const std::vector<int> &placement,
        const std::vector<int64_t> &activations, int layer_idx, int num_ranks,
        int max_slots_per_rank, int num_experts, int expert_redundant_limit) {
        return extract_layer_expert_info(placement, activations, layer_idx,
                                         num_ranks, max_slots_per_rank,
                                         num_experts, expert_redundant_limit);
    }

    std::vector<std::vector<ExpertInformation>>
    ut_extract_expert_info(const std::vector<int> &placement,
                           const std::vector<int64_t> &activations,
                           int num_layers, int num_ranks,
                           int num_experts_per_rank, int num_redundant_per_rank,
                           int expert_redundant_limit) {
        return extract_expert_info(
            placement, activations, num_layers, num_ranks, num_experts_per_rank,
            num_redundant_per_rank, expert_redundant_limit);
    }

    std::pair<double, std::vector<int>> ut_distribute_experts_to_ranks(
        const std::unordered_map<int, double> &expert_loads,
        const std::vector<int> &deployments, int num_ranks, int num_experts,
        int max_slots_per_rank, int current_experts_per_rank, int layer_idx) {
        return distribute_experts_to_ranks(expert_loads, deployments, num_ranks,
                                           num_experts, max_slots_per_rank,
                                           current_experts_per_rank, layer_idx);
    }

  private:
    // Data members
    int rank_;
    int num_layers_;
    int num_ranks_;
    int num_experts_per_rank_;
    int num_redundant_per_rank_;
    int expert_redundant_limit_;
    int num_ranks_per_host_;
    double input_ratio_threshold_; // Threshold for input ratio comparison
    double
        improvement_threshold_; // Threshold for improvement in placement ratio
    double high_low_ratio_threshold_;

    // Private member functions
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

    double compute_placement_ratio_combined(
        const std::vector<int> &placement,
        const std::vector<int64_t> &expert_activations, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts, int type);

    double simulate_placement_ratio(
        const std::vector<int> &placement,
        const std::unordered_map<int, double> &expert_loads, int layer_idx,
        int num_ranks, int max_slots_per_rank, int num_experts);

    std::vector<int> select_best_layer_placement(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        const std::vector<int> &input_placement,
        const std::vector<int64_t> &input_activations, int layer_idx,
        int num_ranks, int num_experts_per_rank, int num_redundant_per_rank,
        int expert_redundant_limit);

    std::unordered_map<int, double>
    compute_expert_loads(const std::vector<ExpertInformation> &experts,
                         int num_experts);
    std::vector<int> allocate_expert_deployments(
        const std::unordered_map<int, double> &expert_loads, int num_experts,
        int budget_limit, int expert_redundant_limit);
    std::pair<double, std::vector<int>> distribute_experts_to_ranks(
        const std::unordered_map<int, double> &expert_loads,
        const std::vector<int> &deployments, int num_ranks, int num_experts,
        int max_slots_per_rank, int current_experts_per_rank, int layer_idx);
    std::vector<int> generate_layer_placement(
        const std::vector<std::vector<ExpertInformation>> &layer_experts,
        int layer_idx, int num_ranks, int num_experts_per_rank,
        int num_redundant_per_rank, int expert_redundant_limit,
        int budget_limit);
};

#endif // EXPERT_LOAD_BALANCER_H