// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_load_balancer.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

using namespace std;

// Constructor: Initialize optimizer parameters with validation
ExpertLoadBalancer::ExpertLoadBalancer(
    int num_layers, int num_ranks, int num_experts_per_rank,
    int num_redundant_per_rank, int expert_redundant_limit, int rank,
    double input_ratio_threshold, double improvement_threshold,
    int num_ranks_per_host, double high_low_ratio_threshold)
    : num_layers_(num_layers), num_ranks_(num_ranks),
      num_experts_per_rank_(num_experts_per_rank),
      num_redundant_per_rank_(num_redundant_per_rank),
      expert_redundant_limit_(expert_redundant_limit), rank_(rank),
      num_ranks_per_host_(num_ranks_per_host),
      input_ratio_threshold_(input_ratio_threshold),
      improvement_threshold_(improvement_threshold),
      high_low_ratio_threshold_(high_low_ratio_threshold) {
    if (num_layers <= 0 || num_ranks <= 0 || num_experts_per_rank <= 0 ||
        num_redundant_per_rank < 0 || expert_redundant_limit < 0 ||
        num_ranks_per_host <= 0 || high_low_ratio_threshold <= 0) {
        throw runtime_error("Invalid initialization parameters");
    }
}

// Get expert sets for each rank
vector<set<int>>
ExpertLoadBalancer::compute_rank_sets(const vector<int> &placement,
                                      int num_ranks, int max_slots_per_rank) {
    vector<set<int>> rank_sets(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        int start_idx = r * max_slots_per_rank;
        for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
            if (placement[i] != -1) {
                rank_sets[r].insert(placement[i]);
            }
        }
    }
    return rank_sets;
}

// Find position of a specific expert in a rank
int ExpertLoadBalancer::find_position_with_expert(const vector<int> &placement,
                                                  int r, int k,
                                                  int max_slots_per_rank) {
    int start_idx = r * max_slots_per_rank;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (placement[i] == k)
            return i;
    }
    return -1;
}

// Find an empty position in a rank
int ExpertLoadBalancer::find_empty_position(const vector<int> &placement, int r,
                                            int max_slots_per_rank) {
    int start_idx = r * max_slots_per_rank;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (placement[i] == -1)
            return i;
    }
    return -1;
}

// Compute expert counts across ranks
unordered_map<int, int> ExpertLoadBalancer::compute_expert_counts(
    const vector<int> &placement, int num_ranks, int max_slots_per_rank) {
    unordered_map<int, int> expert_counts;
    for (int r = 0; r < num_ranks; ++r) {
        int start_idx = r * max_slots_per_rank;
        for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
            if (placement[i] != -1) {
                expert_counts[placement[i]]++;
            }
        }
    }
    return expert_counts;
}

// Validate input vector sizes
bool ExpertLoadBalancer::validate_input_size(const vector<int> &placement,
                                             const vector<int64_t> &activations,
                                             int num_layers, int num_ranks,
                                             int max_slots_per_rank) {
    int64_t expected_size = num_layers * num_ranks * max_slots_per_rank;
    if (placement.size() != expected_size ||
        activations.size() != expected_size) {
        cerr << "Error: Input vector sizes (placement: " << placement.size()
             << ", activations: " << activations.size()
             << ") does not match expected size (" << expected_size << ")"
             << endl;
        return false;
    }
    return true;
}

bool ExpertLoadBalancer::validate_unique_expert_ids(
    const vector<int> &placement, int layer_idx, int num_ranks,
    int max_slots_per_rank) {
    for (int r = 0; r < num_ranks; ++r) {
        set<int> expert_ids;
        int rank_offset = r * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            int expert_id = placement[rank_offset + i];
            if (expert_id != -1) {
                if (expert_ids.count(expert_id)) {
                    cerr << "Error: Duplicate expert ID " << expert_id
                         << " in layer " << layer_idx << ", rank " << r << endl;
                    return false;
                }
                expert_ids.insert(expert_id);
            }
        }
    }
    return true;
}

bool ExpertLoadBalancer::validate_all_experts_present(
    const vector<int> &placement, int layer_idx, int num_ranks,
    int max_slots_per_rank, int num_experts) {
    set<int> present_experts;
    for (int r = 0; r < num_ranks; ++r) {
        int rank_offset = r * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            if (placement[rank_offset + i] != -1) {
                present_experts.insert(placement[rank_offset + i]);
            }
        }
    }
    if (present_experts.size() != num_experts) {
        cerr << "Error: Layer " << layer_idx
             << " does not contain all logical experts (expected "
             << num_experts << ", actual " << present_experts.size() << ")"
             << endl;
        return false;
    }
    return true;
}

// Extract ExpertInformation for a single layer
vector<ExpertInformation> ExpertLoadBalancer::extract_layer_expert_info(
    const vector<int> &placement, const vector<int64_t> &activations,
    int layer_idx, int num_ranks, int max_slots_per_rank, int num_experts,
    int expert_redundant_limit) {
    vector<ExpertInformation> experts;

    vector<int> layer_placement(
        placement.begin() + layer_idx * num_ranks * max_slots_per_rank,
        placement.begin() + (layer_idx + 1) * num_ranks * max_slots_per_rank);
    unordered_map<int, int> expert_counts =
        compute_expert_counts(layer_placement, num_ranks, max_slots_per_rank);
    int layer_offset = layer_idx * num_ranks * max_slots_per_rank;

    for (int r = 0; r < num_ranks; ++r) {
        int rank_offset = layer_offset + r * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            int idx = rank_offset + i;
            int expert_id = placement[idx];
            if (expert_id == -1)
                continue;
            if (expert_id < 0 || expert_id >= num_experts) {
                cerr << "Warning: Invalid expert ID " << expert_id
                     << " in layer " << layer_idx << ", rank " << r
                     << ", position " << i << endl;
                continue;
            }

            ExpertInformation info;
            info.layer_idx = layer_idx;
            info.rank_id = r;
            info.expert_id = expert_id;
            info.activations = activations[idx];
            info.global_position = idx;
            info.total_count = expert_counts[expert_id];
            experts.push_back(info);
        }
    }

    return experts;
}

// Extract ExpertInformation for all layers
vector<vector<ExpertInformation>> ExpertLoadBalancer::extract_expert_info(
    const vector<int> &placement, const vector<int64_t> &activations,
    int num_layers, int num_ranks, int num_experts_per_rank,
    int num_redundant_per_rank, int expert_redundant_limit) {
    int max_slots_per_rank = num_experts_per_rank + num_redundant_per_rank;
    int num_experts = num_ranks * num_experts_per_rank;

    if (!validate_input_size(placement, activations, num_layers, num_ranks,
                             max_slots_per_rank)) {
        return {};
    }

    vector<vector<ExpertInformation>> layer_experts(num_layers);
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        vector<int> layer_placement(
            placement.begin() + layer_idx * num_ranks * max_slots_per_rank,
            placement.begin() +
                (layer_idx + 1) * num_ranks * max_slots_per_rank);
        if (!validate_unique_expert_ids(layer_placement, layer_idx, num_ranks,
                                        max_slots_per_rank)) {
            return {};
        }
        if (!validate_all_experts_present(layer_placement, layer_idx, num_ranks,
                                          max_slots_per_rank, num_experts)) {
            return {};
        }
        layer_experts[layer_idx] = extract_layer_expert_info(
            placement, activations, layer_idx, num_ranks, max_slots_per_rank,
            num_experts, expert_redundant_limit);
    }
    return layer_experts;
}

// Compute placement ratio (max/avg or max/min) for a single layer
// type: 0 for max/avg, 1 for max/min
double ExpertLoadBalancer::compute_placement_ratio_combined(
    const vector<int> &placement, const vector<int64_t> &expert_activations,
    int layer_idx, int num_ranks, int max_slots_per_rank, int num_experts,
    int type) {
    // Validate input size
    if (placement.size() != num_ranks * max_slots_per_rank ||
        expert_activations.size() != num_ranks * max_slots_per_rank) {
        throw std::runtime_error("Invalid input size for "
                                 "compute_placement_ratio_combined in layer " +
                                 std::to_string(layer_idx));
    }

    vector<double> rank_activations(num_ranks, 0.0);

    // Single pass: accumulate activations for each rank
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = placement[idx];
            if (expert_id != -1 && expert_id >= 0 && expert_id < num_experts) {
                rank_activations[rank] +=
                    static_cast<double>(expert_activations[idx]);
            }
        }
    }

    // Compute max, min, and average activations
    double max_activation = 0.0;
    double min_activation = std::numeric_limits<double>::max();
    double total_activation = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        max_activation = std::max(max_activation, rank_activations[rank]);
        min_activation = std::min(min_activation, rank_activations[rank]);
        total_activation += rank_activations[rank];
    }
    double avg_activation = total_activation / num_ranks;

    // Handle zero activation case
    if (avg_activation == 0) {
        return (max_activation == 0) ? 1.0
                                     : std::numeric_limits<double>::infinity();
    }

    // Return max/avg or max/min based on type
    if (type == 0) { // max/avg
        return max_activation / avg_activation;
    } else { // max/min
        min_activation =
            std::max(min_activation, 1.0); // Prevent division by zero
        return max_activation / min_activation;
    }
}

// Compute expert loads
unordered_map<int, double> ExpertLoadBalancer::compute_expert_loads(
    const vector<ExpertInformation> &experts, int num_experts) {
    unordered_map<int, double> expert_loads;
    for (const auto &info : experts) {
        if (info.expert_id < 0 || info.expert_id >= num_experts)
            continue;
        expert_loads[info.expert_id] += static_cast<double>(info.activations);
    }
    return expert_loads;
}

// Allocate expert deployments
vector<int> ExpertLoadBalancer::allocate_expert_deployments(
    const unordered_map<int, double> &expert_loads, int num_experts,
    int budget_limit, int expert_redundant_limit) {
    vector<int> deployments(num_experts, 1);
    int remaining_budget = budget_limit;
    int max_deployments_per_expert =
        std::min(1 + expert_redundant_limit, this->num_ranks_);

    if (remaining_budget > 0) {
        auto cmp = [](const pair<double, int> &a, const pair<double, int> &b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second > b.second;
        };
        priority_queue<pair<double, int>, vector<pair<double, int>>,
                       decltype(cmp)>
            heap(cmp);

        for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
            double load = expert_loads.count(expert_id)
                              ? expert_loads.at(expert_id)
                              : 0.0;
            double priority =
                load == 0.0 ? 0.0 : -load / deployments[expert_id];
            if (deployments[expert_id] < max_deployments_per_expert) {
                heap.push({priority, expert_id});
            }
        }

        int deployments_added = 0;
        while (deployments_added < remaining_budget && !heap.empty()) {
            auto p = heap.top();
            int expert_id = p.second;
            heap.pop();
            if (deployments[expert_id] < max_deployments_per_expert) {
                deployments[expert_id]++;
                deployments_added++;
                double load = expert_loads.count(expert_id)
                                  ? expert_loads.at(expert_id)
                                  : 0.0;
                double new_priority =
                    load == 0.0 ? 0.0 : -load / deployments[expert_id];
                if (deployments[expert_id] < max_deployments_per_expert) {
                    heap.push({new_priority, expert_id});
                }
            }
        }
    }
    return deployments;
}

// Distribute experts to ranks
pair<double, vector<int>> ExpertLoadBalancer::distribute_experts_to_ranks(
    const unordered_map<int, double> &expert_loads,
    const vector<int> &deployments, int num_ranks, int num_experts,
    int max_slots_per_rank, int current_experts_per_rank, int layer_idx) {
    vector<pair<double, int>> expert_instances;
    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        double load =
            (expert_loads.count(expert_id) && deployments[expert_id] > 0)
                ? expert_loads.at(expert_id) / deployments[expert_id]
                : 0.0;
        for (int j = 0; j < deployments[expert_id]; ++j) {
            expert_instances.emplace_back(load, expert_id);
        }
    }
    stable_sort(expert_instances.begin(), expert_instances.end(),
                [](const auto &a, const auto &b) {
                    if (a.first != b.first)
                        return a.first > b.first;
                    return a.second < b.second;
                });

    vector<int> target_placement(num_ranks * max_slots_per_rank, -1);
    vector<int> rank_expert_counts(num_ranks, 0);
    vector<double> rank_loads(num_ranks, 0.0);

    int start_rank = 0;
    for (const auto &p : expert_instances) {
        double load = p.first;
        int expert_id = p.second;

        vector<pair<double, int>> candidate_ranks;
        for (int i = 0; i < num_ranks; ++i) {
            int r = start_rank + i;
            bool can_place = true;
            int start_idx = r * max_slots_per_rank;
            for (int j = start_idx; j < start_idx + max_slots_per_rank; ++j) {
                if (target_placement[j] == expert_id) {
                    can_place = false;
                    break;
                }
            }
            if (can_place && rank_expert_counts[r] < current_experts_per_rank) {
                candidate_ranks.emplace_back(rank_loads[r], r);
            }
        }

        if (candidate_ranks.empty()) {
            cerr << "Error: Cannot allocate position for expert " << expert_id
                 << " in layer " << layer_idx << endl;
            return {0.0, {}};
        }

        stable_sort(candidate_ranks.begin(), candidate_ranks.end(),
                    [](const auto &a, const auto &b) {
                        return a.first < b.first ||
                               (a.first == b.first && a.second < b.second);
                    });

        int best_rank = candidate_ranks[0].second;
        vector<int> available_positions;
        int start_idx = best_rank * max_slots_per_rank;
        for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
            if (target_placement[i] == -1) {
                available_positions.push_back(i);
            }
        }
        if (available_positions.empty()) {
            cerr << "Error: No available position in rank " << best_rank
                 << " in layer " << layer_idx << endl;
            return {0.0, {}};
        }

        stable_sort(available_positions.begin(), available_positions.end());
        int pos = available_positions[0];
        target_placement[pos] = expert_id;
        rank_expert_counts[best_rank]++;
        rank_loads[best_rank] += load;
    }

    for (int r = 0; r < num_ranks; ++r) {
        set<int> rank_experts;
        int start_idx = r * max_slots_per_rank;
        for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
            if (target_placement[i] != -1) {
                if (rank_experts.count(target_placement[i])) {
                    cerr << "Error: Layer " << layer_idx << " rank " << r
                         << " contains duplicate expert ID "
                         << target_placement[i] << endl;
                    return {0.0, {}};
                }
                rank_experts.insert(target_placement[i]);
            }
        }
    }

    // The set of rank loads does not change during reordering, only their
    // order. Therefore, the maximum load value is constant from this point
    // forward.
    double max_load = 0.0;
    for (double load : rank_loads) {
        max_load = max(max_load, load);
    }

    // --- CORRECTLY IMPLEMENTED HOST-LEVEL RANK REORDERING ---
    int num_hosts = (num_ranks + num_ranks_per_host_ - 1) / num_ranks_per_host_;
    if (num_hosts <= 1) {
        // If there's only one host, no reordering is needed.
        // Return the pre-calculated max_load.
        return {max_load, target_placement};
    }

    vector<int> final_placement(num_ranks * max_slots_per_rank, -1);
    vector<int> host_rank_counters(num_hosts, 0);

    for (int logical_rank = 0; logical_rank < num_ranks; ++logical_rank) {
        int host_id = logical_rank % num_hosts;
        int rank_on_host = host_rank_counters[host_id];
        int physical_rank = host_id * num_ranks_per_host_ + rank_on_host;

        if (physical_rank >= num_ranks) {
            cerr << "Warning: Host-level reordering failed due to incompatible "
                    "rank/host configuration. "
                 << "Falling back to non-reordered placement for layer "
                 << layer_idx << endl;
            // Return the pre-calculated max_load.
            return {max_load, target_placement};
        }

        int src_start_idx = logical_rank * max_slots_per_rank;
        int dest_start_idx = physical_rank * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            final_placement[dest_start_idx + i] =
                target_placement[src_start_idx + i];
        }

        host_rank_counters[host_id]++;
    }

    // The final_placement is ready, return it with the pre-calculated max_load.
    return {max_load, final_placement};
}

// Generate placement for a single layer
vector<int> ExpertLoadBalancer::generate_layer_placement(
    const vector<vector<ExpertInformation>> &layer_experts, int layer_idx,
    int num_ranks, int num_experts_per_rank, int num_redundant_per_rank,
    int expert_redundant_limit, int budget_limit) {
    int max_slots_per_rank = num_experts_per_rank + num_redundant_per_rank;
    int num_experts = num_ranks * num_experts_per_rank;
    int current_experts_per_rank =
        num_experts_per_rank + (budget_limit / num_ranks);

    auto expert_loads =
        compute_expert_loads(layer_experts[layer_idx], num_experts);
    auto deployments = allocate_expert_deployments(
        expert_loads, num_experts, budget_limit, expert_redundant_limit);

    auto [max_load, target_placement] = distribute_experts_to_ranks(
        expert_loads, deployments, num_ranks, num_experts, max_slots_per_rank,
        current_experts_per_rank, layer_idx);
    if (target_placement.empty()) {
        return {};
    }

    if (!validate_all_experts_present(target_placement, 0, num_ranks,
                                      max_slots_per_rank, num_experts)) {
        return {};
    }

    return target_placement;
}

// Simulate placement ratio based on physical placement and expert loads
double ExpertLoadBalancer::simulate_placement_ratio(
    const vector<int> &placement,
    const unordered_map<int, double> &expert_loads, int layer_idx,
    int num_ranks, int max_slots_per_rank, int num_experts) {
    // Validate input size
    if (placement.size() != num_ranks * max_slots_per_rank) {
        throw runtime_error(
            "Invalid placement size for simulate_placement_ratio in layer " +
            to_string(layer_idx));
    }

    vector<double> rank_activations(num_ranks, 0.0);
    unordered_map<int, int> expert_deployments;

    // Compute deployment count for each expert
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = placement[idx];
            if (expert_id != -1 && expert_id >= 0 && expert_id < num_experts) {
                expert_deployments[expert_id]++;
            }
        }
    }

    // Compute load for each rank
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = placement[idx];
            if (expert_id != -1 && expert_loads.count(expert_id)) {
                int num_deployments = expert_deployments[expert_id];
                if (num_deployments > 0) {
                    rank_activations[rank] +=
                        expert_loads.at(expert_id) / num_deployments;
                }
            }
        }
    }

    // Compute max and average activation values
    double max_activation = 0.0;
    double total_activation = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        max_activation = max(max_activation, rank_activations[rank]);
        total_activation += rank_activations[rank];
    }
    double avg_activation = total_activation / num_ranks;

    // Handle zero activation case
    if (avg_activation == 0) {
        return (max_activation == 0) ? 1.0 : numeric_limits<double>::infinity();
    }
    return max_activation / avg_activation;
}

// Select best placement for a layer
vector<int> ExpertLoadBalancer::select_best_layer_placement(
    const vector<vector<ExpertInformation>> &layer_experts,
    const vector<int> &input_placement,
    const vector<int64_t> &input_activations, int layer_idx, int num_ranks,
    int num_experts_per_rank, int num_redundant_per_rank,
    int expert_redundant_limit) {
    int max_slots_per_rank = num_experts_per_rank + num_redundant_per_rank;
    int num_experts = num_ranks * num_experts_per_rank;

    // Extract single layer input placement and activations
    int layer_offset = layer_idx * num_ranks * max_slots_per_rank;
    const vector<int> input_layer_placement(
        input_placement.begin() + layer_offset,
        input_placement.begin() + layer_offset +
            num_ranks * max_slots_per_rank);
    const vector<int64_t> input_layer_activations(
        input_activations.begin() + layer_offset,
        input_activations.begin() + layer_offset +
            num_ranks * max_slots_per_rank);

    // Compute expert loads for ratio calculation of candidate placements
    auto expert_loads =
        compute_expert_loads(layer_experts[layer_idx], num_experts);

    // Compute ratio for initial placement
    const double input_ratio = compute_placement_ratio_combined(
        input_layer_placement, input_layer_activations, layer_idx, num_ranks,
        max_slots_per_rank, num_experts, 0);

    // If input_ratio is less than threshold, directly return
    // input_layer_placement
    if (input_ratio < input_ratio_threshold_) {
        return input_layer_placement;
    }

    const double high_low_ratio = compute_placement_ratio_combined(
        input_layer_placement, input_layer_activations, layer_idx, num_ranks,
        max_slots_per_rank, num_experts, 1);

    if (high_low_ratio > high_low_ratio_threshold_) {
        return input_layer_placement;
    }

    // Collect candidate placements and their ratios (excluding input_placement)
    vector<tuple<double, vector<int>, int>> placement_ratios;
    for (int budget = 0; budget <= num_ranks * num_redundant_per_rank;
         budget += num_ranks) {
        auto layer_placement = generate_layer_placement(
            layer_experts, layer_idx, num_ranks, num_experts_per_rank,
            num_redundant_per_rank, expert_redundant_limit, budget);
        if (layer_placement.empty()) {
            continue;
        }
        double ratio = simulate_placement_ratio(
            layer_placement, expert_loads, layer_idx, num_ranks,
            max_slots_per_rank, num_experts);
        placement_ratios.emplace_back(ratio, layer_placement, budget);
    }

    // If no candidate placements, return input_layer_placement
    if (placement_ratios.empty()) {
        return input_layer_placement;
    }

    // Sort by ratio (only sort candidates, not input_placement)
    stable_sort(
        placement_ratios.begin(), placement_ratios.end(),
        [](const auto &a, const auto &b) { return get<0>(a) < get<0>(b); });

    // Get minimum candidate ratio
    double min_candidate_ratio = get<0>(placement_ratios[0]);

    // If minimum candidate ratio is less than input_ratio -
    // improvement_threshold_, accept the candidate placement
    if (min_candidate_ratio < input_ratio - improvement_threshold_) {
        return get<1>(placement_ratios[0]);
    }

    // Otherwise return input_layer_placement
    return input_layer_placement;
}

vector<int> ExpertLoadBalancer::optimize_placement(
    const vector<int> &input_placement,
    const vector<int64_t> &input_activations) {
    auto layer_experts =
        extract_expert_info(input_placement, input_activations, num_layers_,
                            num_ranks_, num_experts_per_rank_,
                            num_redundant_per_rank_, expert_redundant_limit_);
    if (layer_experts.empty()) {
        cerr << "Error: Failed to extract expert information" << endl;
        return input_placement;
    }

    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    vector<int> optimized_placement;

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        vector<int> best_layer_placement = select_best_layer_placement(
            layer_experts, input_placement, input_activations, layer_idx,
            num_ranks_, num_experts_per_rank_, num_redundant_per_rank_,
            expert_redundant_limit_);

        if (best_layer_placement.empty()) {
            cerr << "Error: Failed to generate optimized placement for layer "
                 << layer_idx << endl;
            // Use input placement for this layer
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank;
            best_layer_placement.assign(input_placement.begin() + layer_offset,
                                        input_placement.begin() + layer_offset +
                                            num_ranks_ * max_slots_per_rank);
        }

        optimized_placement.insert(optimized_placement.end(),
                                   best_layer_placement.begin(),
                                   best_layer_placement.end());
    }

    return optimized_placement;
}