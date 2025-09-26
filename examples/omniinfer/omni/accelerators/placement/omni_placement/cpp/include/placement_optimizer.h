// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef PLACEMENT_OPTIMIZER_H
#define PLACEMENT_OPTIMIZER_H

#include "dynamic_eplb_greedy.h"
#include "expert_activation.h"
#include "expert_load_balancer.h"
#include "placement_mapping.h"
#include <stdexcept>
#include <string>
#include <vector>

// Debug information structure
struct DebugInfo {
    int layer_idx;
    std::vector<int> input_placement;
    std::vector<int64_t> input_activations;
    std::vector<int> initial_placement;
    std::vector<int> optimized_placement;
    std::vector<int> output_placement;
    std::vector<ChangeInstruction> instructions;
};

class PlacementOptimizer {
  private:
    PlacementMapping *placement_mapping_;  // Pointer to placement mapping data
    ClusterActivation *clusterActivation_; // Pointer to cluster activation data
    ExpertLoadBalancer *load_balancer_ = nullptr; // Load balancer instance
    GreedyExpertLoadBalancer *greedy_load_balancer_ =
        nullptr;                 // Greedy load balancer instance
    int num_layers_;             // Number of layers in the model
    int rank_;                   // Rank of the current process
    int world_size_;             // Total number of processes (world size)
    int num_experts_;            // Total number of logical experts in the model
    int num_devices_per_host_;   // Number of devices per host
    int num_experts_per_rank_;   // Number of experts per rank
    int num_redundant_per_rank_; // Number of redundant experts per rank
    int expert_redundant_limit_; // Expert redundancy limit

    // Extracts input data for placement and activations
    void extract_input_data(std::vector<int> &placement,
                            std::vector<int64_t> &activations);

    // Validates if all experts are present in the placement
    bool validate_all_experts_present(const std::vector<int> &placement,
                                      int num_experts);

    // Constructs rearrange_g vector for a specific rank
    void construct_rearrange_g_for_rank(std::vector<int> &rearrange_g,
                                        const std::vector<int> &current_f,
                                        const std::vector<int> &g, int rank,
                                        int max_slots_per_rank,
                                        int num_experts);

    // Validates equivalence between two placement vectors for a rank
    bool validate_equivalence(const std::vector<int> &placement1,
                              const std::vector<int> &placement2, int rank,
                              int max_slots_per_rank, int num_experts);

    // Selects source rank and position for a target expert
    bool select_source_for_expert(int target_expert, int layer_idx,
                                  const std::vector<int> &current_f,
                                  int num_ranks, int max_slots_per_rank,
                                  std::vector<int> &rank_comm, int &source_rank,
                                  int &source_pos);
    void print_debug_info(const std::vector<DebugInfo> &debug_info);
    // Generates instructions for a single round of placement changes
    void generate_round_instructions(
        std::vector<ChangeInstruction> &instructions,
        std::vector<int> &tmp_placement, const std::vector<int> &current_f,
        const std::vector<int> &rearrange_g, int layer_idx, int round,
        int num_ranks, int max_slots_per_rank, int num_experts,
        std::vector<int> &rank_comm);

    // Generates layer instructions for placement optimization
    std::pair<std::vector<ChangeInstruction>, std::vector<int>>
    generate_layer_instructions(std::vector<int> &current_f,
                                const std::vector<int> &g, int layer_idx,
                                int num_ranks, int max_slots_per_rank,
                                int num_experts);

    // Generate instructions from two placements
    std::vector<ChangeInstruction> generate_instructions_from_placements(
        const std::vector<int> &current_placement,
        const std::vector<int> &target_placement);

  public:
    PlacementOptimizer(PlacementMapping *placement_mapping,
                       ClusterActivation *clusterActivation);
    ~PlacementOptimizer();

    std::vector<ChangeInstruction> optimize();
    std::vector<ChangeInstruction> optimize(std::vector<int> placement,
                                            std::vector<int64_t> activations);

    int get_num_layers() const { return num_layers_; }
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    int get_num_experts() const { return num_experts_; }
    int get_num_devices_per_host() const { return num_devices_per_host_; }
    int get_num_experts_per_rank() const { return num_experts_per_rank_; }
    int get_num_redundant_per_rank() const { return num_redundant_per_rank_; }
    int get_expert_redundant_limit() const { return expert_redundant_limit_; }
};

#endif // PLACEMENT_OPTIMIZER_H