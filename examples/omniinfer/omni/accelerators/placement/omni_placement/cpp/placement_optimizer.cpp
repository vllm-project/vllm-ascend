// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_optimizer.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#define ENABLE_DEBUG 0

// Constructor for PlacementOptimizer
PlacementOptimizer::PlacementOptimizer(PlacementMapping *placement_mapping,
                                       ClusterActivation *clusterActivation)
    : placement_mapping_(placement_mapping),
      clusterActivation_(clusterActivation) {
    if (!placement_mapping_ || !clusterActivation_) {
        throw std::runtime_error(
            "Invalid initialization parameters: placement_mapping or "
            "clusterActivation is null");
    }

    num_layers_ = placement_mapping->get_num_layers();
    rank_ = placement_mapping->get_rank();
    world_size_ = placement_mapping->get_world_size();
    num_experts_ = placement_mapping->get_num_experts();
    num_devices_per_host_ = placement_mapping->get_num_devices_per_host();
    num_redundant_per_rank_ = placement_mapping->get_num_redundant_per_rank();
    expert_redundant_limit_ =
        placement_mapping->get_max_redundant_per_expert() - 1;

    if (num_layers_ <= 0 || world_size_ <= 0 || num_experts_ <= 0) {
        throw std::runtime_error("Invalid initialization parameters: number of "
                                 "layers, ranks, or experts is invalid");
    }
    if (expert_redundant_limit_ < 0) {
        throw std::runtime_error("Invalid expert redundancy limit: "
                                 "max_redundant_count must be at least 1");
    }

    if (num_experts_ % world_size_ != 0) {
        throw std::runtime_error(
            "Number of experts " + std::to_string(num_experts_) +
            " is not divisible by world size " + std::to_string(world_size_));
    }
    num_experts_per_rank_ = num_experts_ / world_size_;
    if (num_experts_per_rank_ <= 0) {
        throw std::runtime_error("Invalid number of experts per rank");
    }

    load_balancer_ = new ExpertLoadBalancer(
        num_layers_, world_size_, num_experts_per_rank_,
        num_redundant_per_rank_, expert_redundant_limit_, rank_);

    greedy_load_balancer_ = new GreedyExpertLoadBalancer(
        num_layers_, world_size_, num_experts_,
        placement_mapping ? placement_mapping->get_num_deploy_experts() : 0,
        expert_redundant_limit_, rank_);
}

// Destructor for PlacementOptimizer
PlacementOptimizer::~PlacementOptimizer() {
    delete load_balancer_;
    delete greedy_load_balancer_;
}

// Validates if all experts are present in the placement vector
bool PlacementOptimizer::validate_all_experts_present(
    const std::vector<int> &placement, int num_experts) {
    std::vector<int> expert_counts(num_experts, 0);
    for (int expert_id : placement) {
        if (expert_id != -1) {
            if (expert_id < 0 || expert_id >= num_experts) {
                return false;
            }
            expert_counts[expert_id]++;
        }
    }
    return std::all_of(expert_counts.begin(), expert_counts.end(),
                       [](int count) { return count >= 1; });
}

// Extracts input data for placement and activations
void PlacementOptimizer::extract_input_data(std::vector<int> &placement,
                                            std::vector<int64_t> &activations) {
    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    int expected_size = num_layers_ * world_size_ * max_slots_per_rank;

    auto global_mapping =
        placement_mapping_
            ->get_global_deployed_position_to_logistics_id_mapping();
    if (static_cast<int>(global_mapping.size()) != expected_size) {
        throw std::runtime_error(
            "globalDeployedPositionToLogisticsIdMappingHost_ size " +
            std::to_string(global_mapping.size()) +
            " does not match expected size " + std::to_string(expected_size));
    }

    placement = global_mapping;
    activations.assign(expected_size, 0);

    clusterActivation_->updateShiftWindows(placement_mapping_);

    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        int layer_offset = layer_id * world_size_ * max_slots_per_rank;
        for (int rank = 0; rank < world_size_; ++rank) {
            for (int pos = 0; pos < max_slots_per_rank; ++pos) {
                int idx = layer_offset + rank * max_slots_per_rank + pos;
                int physical_pos = rank * max_slots_per_rank + pos;
                activations[idx] = clusterActivation_->getExpertActivationCount(
                    layer_id, physical_pos);
            }
        }
    }

    if (!validate_all_experts_present(placement, num_experts_)) {
        std::cerr << "Warning: Input placement is missing some experts or "
                     "contains invalid IDs"
                  << std::endl;
    }
}

// Constructs rearrange_g vector for a specific rank
void PlacementOptimizer::construct_rearrange_g_for_rank(
    std::vector<int> &rearrange_g, const std::vector<int> &current_f,
    const std::vector<int> &g, int rank, int max_slots_per_rank,
    int num_experts) {
    std::vector<int> count_f(num_experts, 0);
    std::vector<int> count_g(num_experts, 0);
    int start_idx = rank * max_slots_per_rank;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (current_f[i] != -1)
            count_f[current_f[i]]++;
        if (g[i] != -1)
            count_g[g[i]]++;
    }

    std::vector<int> intersection_count(num_experts, 0);
    std::vector<int> placed_intersection(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert) {
        intersection_count[expert] = std::min(count_f[expert], count_g[expert]);
    }

    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        int expert = current_f[i];
        if (expert != -1 &&
            placed_intersection[expert] < intersection_count[expert]) {
            rearrange_g[i] = expert;
            placed_intersection[expert]++;
        }
    }

    std::vector<int> remaining_experts;
    for (int expert = 0; expert < num_experts; ++expert) {
        int extra = count_g[expert] - count_f[expert];
        if (extra > 0) {
            for (int k = 0; k < extra; ++k) {
                remaining_experts.push_back(expert);
            }
        }
    }

    size_t expert_idx = 0;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (rearrange_g[i] == -1 && expert_idx < remaining_experts.size()) {
            rearrange_g[i] = remaining_experts[expert_idx++];
        }
    }
}

// Validates equivalence between two placement vectors for a rank
bool PlacementOptimizer::validate_equivalence(
    const std::vector<int> &placement1, const std::vector<int> &placement2,
    int rank, int max_slots_per_rank, int num_experts) {
    std::vector<int> count1(num_experts, 0);
    std::vector<int> count2(num_experts, 0);
    int start_idx = rank * max_slots_per_rank;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (placement1[i] != -1)
            count1[placement1[i]]++;
        if (placement2[i] != -1)
            count2[placement2[i]]++;
    }
    return count1 == count2;
}

// Selects source rank and position for an optimized expert
bool PlacementOptimizer::select_source_for_expert(
    int optimized_expert, int layer_idx, const std::vector<int> &current_f,
    int num_ranks, int max_slots_per_rank, std::vector<int> &rank_comm,
    int &source_rank, int &source_pos) {
    std::vector<std::pair<int, int>> candidates;
    for (int r = 0; r < num_ranks; ++r) {
        int pos = -1;
        int start_idx = r * max_slots_per_rank;
        for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
            if (current_f[i] == optimized_expert) {
                pos = i;
                break;
            }
        }
        if (pos != -1) {
            candidates.emplace_back(rank_comm[r], r);
        }
    }
    if (candidates.empty()) {
        std::cerr << "Error: Could not find position for expert "
                  << optimized_expert << " in layer " << layer_idx << std::endl;
        return false;
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    source_rank = candidates[0].second;
    for (int i = source_rank * max_slots_per_rank;
         i < (source_rank + 1) * max_slots_per_rank; ++i) {
        if (current_f[i] == optimized_expert) {
            source_pos = i;
            return true;
        }
    }
    return false;
}

// Generates instructions for a single round of placement changes
void PlacementOptimizer::generate_round_instructions(
    std::vector<ChangeInstruction> &instructions,
    std::vector<int> &tmp_placement, const std::vector<int> &current_f,
    const std::vector<int> &rearrange_g, int layer_idx, int round,
    int num_ranks, int max_slots_per_rank, int num_experts,
    std::vector<int> &rank_comm) {
    for (int rank = 0; rank < num_ranks; ++rank) {
        int target_global_position = rank * max_slots_per_rank + round;
        int current_expert = current_f[target_global_position];
        int optimized_expert = rearrange_g[target_global_position];

        if (current_expert != optimized_expert) {
            if (optimized_expert != -1) {
                ChangeInstruction instr;
                instr.layer_idx = layer_idx;
                instr.type = OperationType::ADD;
                instr.source_expert_id = optimized_expert;
                instr.target_expert_id = current_expert;
                instr.target_rank = rank;
                instr.target_global_position = target_global_position;
                instr.round = round + 1;

                int source_rank, source_pos;
                if (!select_source_for_expert(optimized_expert, layer_idx,
                                              current_f, num_ranks,
                                              max_slots_per_rank, rank_comm,
                                              source_rank, source_pos)) {
                    instructions.clear();
                    tmp_placement.clear();
                    return;
                }

                instr.source_rank = source_rank;
                instr.source_global_position = source_pos;
                rank_comm[rank]++;
                rank_comm[source_rank]++;

                instructions.push_back(instr);
                tmp_placement[target_global_position] = optimized_expert;
            } else {
                ChangeInstruction instr;
                instr.layer_idx = layer_idx;
                instr.type = OperationType::REMOVE;
                instr.source_rank = -1;
                instr.source_expert_id = -1;
                instr.source_global_position = -1;
                instr.target_rank = rank;
                instr.target_expert_id = current_expert;
                instr.target_global_position = target_global_position;
                instr.round = round + 1;

                instructions.push_back(instr);
                tmp_placement[target_global_position] = -1;
            }
        } else {
            tmp_placement[target_global_position] = current_expert;
        }
    }
}

// Generates layer instructions for placement optimization
std::pair<std::vector<ChangeInstruction>, std::vector<int>>
PlacementOptimizer::generate_layer_instructions(std::vector<int> &current_f,
                                                const std::vector<int> &g,
                                                int layer_idx, int num_ranks,
                                                int max_slots_per_rank,
                                                int num_experts) {
    std::vector<ChangeInstruction> instructions;
    std::vector<int> tmp_placement(num_ranks * max_slots_per_rank, -1);
    std::vector<int> rearrange_g(num_ranks * max_slots_per_rank, -1);

    // Construct rearrange_g for all ranks
    for (int rank = 0; rank < num_ranks; ++rank) {
        construct_rearrange_g_for_rank(rearrange_g, current_f, g, rank,
                                       max_slots_per_rank, num_experts);
    }

    // Validate rearrange_g
    if (!validate_all_experts_present(rearrange_g, num_experts)) {
        std::cerr << "Error: rearrange_g is missing some experts in layer "
                  << layer_idx << std::endl;
        return {{}, {}};
    }

    // Validate equivalence of rearrange_g with g for each rank
    for (int rank = 0; rank < num_ranks; ++rank) {
        if (!validate_equivalence(g, rearrange_g, rank, max_slots_per_rank,
                                  num_experts)) {
            std::cerr << "Error: rearrange_g is not equivalent to g (different "
                         "counts) for rank "
                      << rank << " in layer " << layer_idx << std::endl;
            return {{}, {}};
        }
    }

    // Generate instructions for each round
    std::vector<int> rank_comm(num_ranks, 0);
    for (int round = 0; round < max_slots_per_rank; ++round) {
        generate_round_instructions(instructions, tmp_placement, current_f,
                                    rearrange_g, layer_idx, round, num_ranks,
                                    max_slots_per_rank, num_experts, rank_comm);
        if (instructions.empty() && tmp_placement.empty()) {
            return {{}, {}};
        }
    }

    // Validate tmp_placement
    for (int rank = 0; rank < num_ranks; ++rank) {
        if (!validate_equivalence(g, tmp_placement, rank, max_slots_per_rank,
                                  num_experts)) {
            std::cerr << "Error: tmp_placement is not equivalent to g "
                         "(different counts) for rank "
                      << rank << " in layer " << layer_idx << std::endl;
            return {{}, {}};
        }
    }

    if (!validate_all_experts_present(tmp_placement, num_experts)) {
        std::cerr << "Error: tmp_placement is missing some experts in layer "
                  << layer_idx << std::endl;
        return {{}, {}};
    }

    return {instructions, tmp_placement};
}

// New function to generate instructions from two placements
std::vector<ChangeInstruction>
PlacementOptimizer::generate_instructions_from_placements(
    const std::vector<int> &current_placement,
    const std::vector<int> &target_placement) {
    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    std::vector<ChangeInstruction> all_instructions;

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        int layer_offset = layer_idx * world_size_ * max_slots_per_rank;

        // Extract current and target placements for this layer
        std::vector<int> current_f(current_placement.begin() + layer_offset,
                                   current_placement.begin() + layer_offset +
                                       world_size_ * max_slots_per_rank);
        std::vector<int> g(target_placement.begin() + layer_offset,
                           target_placement.begin() + layer_offset +
                               world_size_ * max_slots_per_rank);

        // Generate instructions for this layer
        auto [layer_instructions, tmp_placement] =
            generate_layer_instructions(current_f, g, layer_idx, world_size_,
                                        max_slots_per_rank, num_experts_);

        // Add instructions directly without batch processing
        all_instructions.insert(all_instructions.end(),
                                layer_instructions.begin(),
                                layer_instructions.end());
    }

    return all_instructions;
}

// Print debug information
void PlacementOptimizer::print_debug_info(
    const std::vector<DebugInfo> &debug_info) {
    if (!ENABLE_DEBUG || rank_ != 0)
        return;

    std::cout << "=== PlacementOptimizer Debug Information ===" << std::endl;
    std::cout << "=== Step 1: Extract input data ===" << std::endl;

    for (const auto &info : debug_info) {
        int layer_idx = info.layer_idx;
        std::cout << "\nProcessing layer " << layer_idx << std::endl;

        // Print input placement
        std::cout << "Input Placement for layer " << layer_idx << ":"
                  << std::endl;
        for (size_t i = 0; i < info.input_placement.size(); ++i) {
            std::cout << info.input_placement[i] << " ";
            if ((i + 1) % (world_size_ *
                           (num_experts_per_rank_ + num_redundant_per_rank_)) ==
                0)
                std::cout << std::endl;
        }

        // Print input activations
        std::cout << "Input Activations for layer " << layer_idx << ":"
                  << std::endl;
        for (size_t i = 0; i < info.input_activations.size(); ++i) {
            std::cout << info.input_activations[i] << " ";
            if ((i + 1) % (world_size_ *
                           (num_experts_per_rank_ + num_redundant_per_rank_)) ==
                0)
                std::cout << std::endl;
        }

        // Print initial and optimized placements
        std::cout << "Initial Placement (f) for layer " << layer_idx << ":"
                  << std::endl;
        for (size_t i = 0; i < info.initial_placement.size(); ++i) {
            std::cout << info.initial_placement[i] << " ";
            if ((i + 1) % (num_experts_per_rank_ + num_redundant_per_rank_) ==
                0)
                std::cout << std::endl;
        }

        std::cout << "Optimized Placement (g) for layer " << layer_idx << ":"
                  << std::endl;
        for (size_t i = 0; i < info.optimized_placement.size(); ++i) {
            std::cout << info.optimized_placement[i] << " ";
            if ((i + 1) % (num_experts_per_rank_ + num_redundant_per_rank_) ==
                0)
                std::cout << std::endl;
        }

        // Print instructions
        std::cout << "\nLayer " << layer_idx << " Instructions:" << std::endl;
        if (info.instructions.empty()) {
            std::cout << "  No instructions needed." << std::endl;
        } else {
            for (const auto &instr : info.instructions) {
                std::cout << "  Instruction: Round=" << instr.round
                          << ", Type=" << static_cast<int>(instr.type)
                          << ", Source=(rank=" << instr.source_rank
                          << ", id=" << instr.source_expert_id
                          << ", pos=" << instr.source_global_position << ")"
                          << ", Target=(rank=" << instr.target_rank
                          << ", id=" << instr.target_expert_id
                          << ", pos=" << instr.target_global_position << ")"
                          << std::endl;
            }
        }

        // Print output placement
        std::cout << "Output Placement for layer " << layer_idx << ":"
                  << std::endl;
        for (size_t i = 0; i < info.output_placement.size(); ++i) {
            std::cout << info.output_placement[i] << " ";
            if ((i + 1) % (num_experts_per_rank_ + num_redundant_per_rank_) ==
                0)
                std::cout << std::endl;
        }
    }
}

// Modified optimize() function
std::vector<ChangeInstruction> PlacementOptimizer::optimize() {
    std::vector<DebugInfo> debug_info;

    std::vector<int> placement;
    std::vector<int64_t> activations;
    extract_input_data(placement, activations);

    // Get optimized placement from load_balancer
    std::vector<int> optimized_placement =
        load_balancer_->optimize_placement(placement, activations);

    // // Get optimized placement from greedy_load_balancer
    // std::vector<int> optimized_placement =
    //     greedy_load_balancer_->optimize_placement(placement, activations);

    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    std::vector<ChangeInstruction> all_instructions;

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        DebugInfo info;
        info.layer_idx = layer_idx;

        int layer_offset = layer_idx * world_size_ * max_slots_per_rank;
        std::vector<int> current_f(placement.begin() + layer_offset,
                                   placement.begin() + layer_offset +
                                       world_size_ * max_slots_per_rank);
        std::vector<int64_t> layer_activations(
            activations.begin() + layer_offset,
            activations.begin() + layer_offset +
                world_size_ * max_slots_per_rank);
        std::vector<int> g(optimized_placement.begin() + layer_offset,
                           optimized_placement.begin() + layer_offset +
                               world_size_ * max_slots_per_rank);

        info.input_placement = current_f;
        info.input_activations = layer_activations;
        info.initial_placement = current_f;
        info.optimized_placement = g;

        // Generate instructions for this layer
        auto [layer_instructions, tmp_placement] =
            generate_layer_instructions(current_f, g, layer_idx, world_size_,
                                        max_slots_per_rank, num_experts_);

        info.instructions = layer_instructions;
        info.output_placement = tmp_placement;

        all_instructions.insert(all_instructions.end(),
                                layer_instructions.begin(),
                                layer_instructions.end());

        debug_info.push_back(info);
    }

    // Print debug information
    print_debug_info(debug_info);

    return all_instructions;
}

// Modified optimize() function with parameters for unittest
std::vector<ChangeInstruction>
PlacementOptimizer::optimize(std::vector<int> placement,
                             std::vector<int64_t> activations) {
    // Get optimized placement from load_balancer
    std::vector<int> optimized_placement =
        load_balancer_->optimize_placement(placement, activations);

    // Generate instructions from input placement to optimized placement
    return generate_instructions_from_placements(placement,
                                                 optimized_placement);
}