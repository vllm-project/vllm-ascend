// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#define ENABLE_DEBUG 1
#include "expert_load_balancer.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

using namespace std;

// 构造函数：初始化优化器参数并进行合法性检查
ExpertLoadBalancer::ExpertLoadBalancer(int num_layers, int num_ranks,
                                       int num_experts_per_rank,
                                       int num_redundant_per_rank,
                                       int expert_redundant_limit, int rank)
    : num_layers_(num_layers), num_ranks_(num_ranks),
      num_experts_per_rank_(num_experts_per_rank),
      num_redundant_per_rank_(num_redundant_per_rank),
      expert_redundant_limit_(expert_redundant_limit), rank_(rank),
      comm_limit_(3), // 默认通信上限
      rank_comm_counts_(num_ranks, 0) {
    if (num_layers <= 0 || num_ranks <= 0 || num_experts_per_rank <= 0 ||
        num_redundant_per_rank < 0 || expert_redundant_limit < 0) {
        throw runtime_error("Invalid initialization parameters");
    }
}

// 新增：transfer 函数，用于对指令进行优先级排序
void ExpertLoadBalancer::transfer(
    std::vector<ChangeInstruction> &changeInstructions) {
    int num_experts = num_ranks_ * num_experts_per_rank_;
    std::vector<int> expert(num_experts, 0);
    for (size_t i = 0; i < changeInstructions.size(); ++i) {
        changeInstructions[i].prior = 0;
        expert[changeInstructions[i].source_expert_id]++;
        expert[changeInstructions[i].target_expert_id]++;
        changeInstructions[i].prior =
            std::max(expert[changeInstructions[i].source_expert_id],
                     expert[changeInstructions[i].target_expert_id]);
        expert[changeInstructions[i].source_expert_id] =
            changeInstructions[i].prior;
        expert[changeInstructions[i].target_expert_id] =
            changeInstructions[i].prior;
    }
    std::stable_sort(changeInstructions.begin(), changeInstructions.end());
}

// 验证每个 rank 的专家数量约束
bool ExpertLoadBalancer::validate_expert_counts(
    const vector<int> &placement, int num_ranks, int max_slots_per_rank,
    int num_experts_per_rank, int num_redundant_per_rank, int layer_idx) {
    int min_experts = num_experts_per_rank;
    int max_experts = num_experts_per_rank + num_redundant_per_rank;
    for (int rank = 0; rank < num_ranks; ++rank) {
        int count = 0;
        int offset = rank * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            if (placement[offset + i] != -1)
                count++;
        }
        if (count < min_experts || count > max_experts) {
            cerr << "Error: Layer " << layer_idx << ", rank " << rank << " has "
                 << count << " experts, not in range [" << min_experts << ", "
                 << max_experts << "]\n";
            return false;
        }
    }
    return true;
}

// 获取专家集合
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

// 查找专家位置
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

// 查找空槽位
int ExpertLoadBalancer::find_empty_position(const vector<int> &placement, int r,
                                            int max_slots_per_rank) {
    int start_idx = r * max_slots_per_rank;
    for (int i = start_idx; i < start_idx + max_slots_per_rank; ++i) {
        if (placement[i] == -1)
            return i;
    }
    return -1;
}

// 计算专家计数
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

// 验证输入大小
bool ExpertLoadBalancer::validate_input_size(const vector<int> &placement,
                                             const vector<int64_t> &activations,
                                             int num_layers, int num_ranks,
                                             int max_slots_per_rank) {
    int64_t expected_size = num_layers * num_ranks * max_slots_per_rank;
    if (placement.size() != expected_size ||
        activations.size() != expected_size) {
        cerr << "Error: Input vector sizes (placement: " << placement.size()
             << ", activations: " << activations.size()
             << ") does not match expected size (" << expected_size << ")\n";
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
                         << " in layer " << layer_idx << ", rank " << r
                         << ", position " << (rank_offset + i) << "\n";
                    return false;
                }
                expert_ids.insert(expert_id);
            }
        }
    }
    return true;
}

// 验证所有专家存在
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
             << num_experts << ", actual " << present_experts.size() << ")\n";
        return false;
    }
    return true;
}

// 提取单层专家信息
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
                     << ", position " << i << "\n";
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

// 提取所有层专家信息
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
                                        max_slots_per_rank) ||
            !validate_all_experts_present(layer_placement, layer_idx, num_ranks,
                                          max_slots_per_rank, num_experts)) {
            return {};
        }
        layer_experts[layer_idx] = extract_layer_expert_info(
            placement, activations, layer_idx, num_ranks, max_slots_per_rank,
            num_experts, expert_redundant_limit);
    }
    return layer_experts;
}

// 计算每层专家总激活值
vector<int64_t> ExpertLoadBalancer::compute_expert_total_activations(
    const vector<vector<ExpertInformation>> &layer_experts, int num_layers,
    int num_ranks, int num_experts_per_rank) {
    int num_experts = num_ranks * num_experts_per_rank;
    vector<int64_t> total_activations(num_layers * num_experts, 0);

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        unordered_map<int, int64_t> expert_activations;
        for (const auto &info : layer_experts[layer_idx]) {
            if (info.expert_id < 0 || info.expert_id >= num_experts)
                continue;
            expert_activations[info.expert_id] += info.activations;
        }

        int layer_offset = layer_idx * num_experts;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
            total_activations[layer_offset + expert_id] =
                expert_activations[expert_id];
        }
    }
    return total_activations;
}

// 计算 placement 比率
double ExpertLoadBalancer::compute_placement_ratio(
    const vector<int> &placement, const vector<int64_t> &expert_activations,
    int layer_idx, int num_ranks, int max_slots_per_rank, int num_experts) {
    if (placement.size() != num_ranks * max_slots_per_rank ||
        expert_activations.size() != num_ranks * max_slots_per_rank) {
        throw runtime_error(
            "Invalid input size for compute_placement_ratio in layer " +
            to_string(layer_idx));
    }

    vector<double> rank_activations(num_ranks, 0.0);
    unordered_map<int, int64_t> expert_total_activations;
    unordered_map<int, int> expert_deployments;

    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = placement[idx];
            if (expert_id != -1 && expert_id >= 0 && expert_id < num_experts) {
                expert_total_activations[expert_id] += expert_activations[idx];
                expert_deployments[expert_id]++;
            }
        }
    }

    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = placement[idx];
            if (expert_id != -1 && expert_total_activations.count(expert_id)) {
                int num_deployments = expert_deployments[expert_id];
                if (num_deployments > 0) {
                    rank_activations[rank] +=
                        static_cast<double>(
                            expert_total_activations[expert_id]) /
                        num_deployments;
                }
            }
        }
    }

    double max_activation = 0.0;
    double total_activation = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        max_activation = max(max_activation, rank_activations[rank]);
        total_activation += rank_activations[rank];
    }
    double avg_activation = total_activation / num_ranks;

    if (avg_activation == 0) {
        cerr << "Warning: Average activation is zero in layer " << layer_idx
             << "\n";
        return (max_activation == 0) ? 1.0 : numeric_limits<double>::infinity();
    }
    return max_activation / avg_activation;
}

// 模拟 placement 比率
double ExpertLoadBalancer::simulate_placement_ratio(
    const vector<int> &placement,
    const unordered_map<int, double> &expert_loads, int layer_idx,
    int num_ranks, int max_slots_per_rank, int num_experts) {
    if (placement.size() != num_ranks * max_slots_per_rank) {
        throw runtime_error(
            "Invalid placement size for simulate_placement_ratio in layer " +
            to_string(layer_idx));
    }

    vector<double> rank_activations(num_ranks, 0.0);
    unordered_map<int, int> expert_deployments;

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

    double max_activation = 0.0;
    double total_activation = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        max_activation = max(max_activation, rank_activations[rank]);
        total_activation += rank_activations[rank];
    }
    double avg_activation = total_activation / num_ranks;

    if (avg_activation == 0) {
        cerr << "Warning: Average activation is zero in layer " << layer_idx
             << "\n";
        return (max_activation == 0) ? 1.0 : numeric_limits<double>::infinity();
    }
    return max_activation / avg_activation;
}

// 新增：模拟调整后的排布比率，排除未优化的最热 rank，使用全局 avg_load 作为分母
double ExpertLoadBalancer::simulate_adjusted_placement_ratio(
    const std::vector<int> &placement,
    const std::unordered_map<int, double> &expert_loads,
    const std::vector<int> &remaining_hot_ranks, double avg_load, int layer_idx,
    int num_ranks, int max_slots_per_rank, int num_experts) {
    // 验证输入
    if (placement.size() != num_ranks * max_slots_per_rank) {
        throw runtime_error("层 " + to_string(layer_idx) + " 的排布大小无效");
    }

    // 计算每个 rank 的负载和专家部署次数
    std::vector<double> rank_activations(num_ranks, 0.0);
    std::unordered_map<int, int> expert_deployments;

    // 计算专家部署次数
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

    // 计算每个 rank 的负载
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

    // 计算调整后的最大负载，排除 remaining_hot_ranks
    double adjusted_max_activation = 0.0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        if (std::find(remaining_hot_ranks.begin(), remaining_hot_ranks.end(),
                      rank) == remaining_hot_ranks.end()) {
            adjusted_max_activation =
                std::max(adjusted_max_activation, rank_activations[rank]);
        }
    }

    // 使用传入的全局 avg_load 作为分母
    if (avg_load == 0) {
        cerr << "警告：层 " << layer_idx << " 的平均负载为零\n";
        return (adjusted_max_activation == 0)
                   ? 1.0
                   : numeric_limits<double>::infinity();
    }
    return adjusted_max_activation / avg_load;
}

// 计算专家负载
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

std::vector<std::tuple<int, int, int, int, int, int, int, double, int,
                       ChangeInstruction>>
ExpertLoadBalancer::find_next_candidate(
    std::vector<int> &g, int layer_idx, std::vector<double> &rank_loads,
    std::unordered_map<int, double> &expert_loads,
    std::unordered_map<int, int> &expert_deployments,
    std::unordered_map<int, int> &expert_counts,
    std::vector<int> &rank_expert_counts, std::vector<int> &rank_comm_counts,
    int num_ranks, int max_slots_per_rank, int num_experts,
    int num_experts_per_rank, int num_redundant_per_rank,
    int expert_redundant_limit, int comm_limit, int total_diff_count,
    int diff_limit) {
    // 存储所有候选调整
    std::vector<std::tuple<int, int, int, int, int, int, int, double, int,
                           ChangeInstruction>>
        selected_adjustments;
    selected_adjustments.reserve(num_ranks); // 最多 k 条指令，k <= num_ranks

    // 计算当前负载统计
    double total_load = 0.0;
    double max_load = rank_loads[0];
    for (double load : rank_loads) {
        total_load += load;
        max_load = std::max(max_load, load);
    }
    double avg_load = total_load / num_ranks;

    // 识别最热卡（负载接近最大值的卡）
    std::vector<int> hot_ranks;
    const double load_threshold =
        0.05 * (max_load - avg_load); // 容忍 10% 的浮点误差
    for (int r = 0; r < num_ranks; ++r) {
        if (rank_comm_counts[r] >= comm_limit) {
            if (ENABLE_DEBUG) {
                std::cout << "Skipping rank " << r
                          << " due to communication limit reached: "
                          << rank_comm_counts[r] << std::endl;
            }
            continue;
        }
        if (std::abs(rank_loads[r] - max_load) <= load_threshold) {
            hot_ranks.push_back(r);
        }
    }

    if (ENABLE_DEBUG) {
        std::cout << "Number of hot ranks: " << hot_ranks.size() << std::endl;
    }

    // 收集最冷卡（负载低于平均值的卡）
    std::vector<std::pair<double, int>> cold_ranks;
    for (int r = 0; r < num_ranks; ++r) {
        if (rank_comm_counts[r] >= comm_limit) {
            if (ENABLE_DEBUG) {
                std::cout << "Skipping rank " << r
                          << " due to communication limit reached: "
                          << rank_comm_counts[r] << std::endl;
            }
            continue;
        }
        if (rank_loads[r] < avg_load) {
            cold_ranks.emplace_back(rank_loads[r], r);
        }
    }
    std::sort(cold_ranks.begin(), cold_ranks.end());

    // 为每个最热卡生成一条优化指令
    for (int rank_a : hot_ranks) {
        std::vector<std::tuple<int, int, int, int, int, int, int, double, int,
                               ChangeInstruction>>
            adjustments;
        adjustments.reserve(100);
        int rank_a_offset = rank_a * max_slots_per_rank;

        // 计算排除其他未优化最热卡后的 max_load 和 avg_load
        std::vector<int> remaining_hot_ranks;

        double adjusted_max_load = 0.0;
        for (int r = 0; r < num_ranks; ++r) {
            if (std::find(remaining_hot_ranks.begin(),
                          remaining_hot_ranks.end(),
                          r) == remaining_hot_ranks.end()) {
                adjusted_max_load = std::max(adjusted_max_load, rank_loads[r]);
            }
        }

        // REMOVE 操作：尝试移除最热卡的冗余专家
        std::vector<std::pair<double, std::pair<int, int>>> experts_a;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            int pos_a = rank_a_offset + i;
            int expert_a = g[pos_a];
            if (expert_a != -1 &&
                expert_counts.at(expert_a) > 1) { // 仅考虑冗余专家
                double load = expert_loads.count(expert_a)
                                  ? expert_loads.at(expert_a) /
                                        expert_deployments.at(expert_a)
                                  : 0.0;
                experts_a.emplace_back(load, std::make_pair(pos_a, expert_a));
            }
        }
        std::sort(experts_a.begin(), experts_a.end(), std::greater<>());
        for (size_t i = 0; i < std::min<size_t>(5, experts_a.size()); ++i) {
            int pos_a = experts_a[i].second.first;
            int expert_a = experts_a[i].second.second;

            // 临时移除专家
            std::vector<int> temp_g = g;
            temp_g[pos_a] = -1;

            // 更新专家负载
            std::unordered_map<int, double> temp_expert_loads = expert_loads;
            std::unordered_map<int, int> temp_deployments = expert_deployments;
            temp_deployments[expert_a]--;
            if (temp_deployments[expert_a] > 0) {
                temp_expert_loads[expert_a] = expert_loads.at(expert_a) *
                                              expert_deployments.at(expert_a) /
                                              temp_deployments[expert_a];
            } else {
                temp_expert_loads.erase(expert_a);
            }

            if (!validate_unique_expert_ids(temp_g, layer_idx, num_ranks,
                                            max_slots_per_rank) ||
                !validate_all_experts_present(temp_g, layer_idx, num_ranks,
                                              max_slots_per_rank,
                                              num_experts) ||
                !validate_expert_counts(temp_g, num_ranks, max_slots_per_rank,
                                        num_experts_per_rank,
                                        num_redundant_per_rank, layer_idx)) {
                if (ENABLE_DEBUG) {
                    std::cout << "REMOVE candidate skipped: invalid placement "
                                 "in layer "
                              << layer_idx << std::endl;
                }
                continue;
            }

            double new_ratio = simulate_adjusted_placement_ratio(
                temp_g, temp_expert_loads, remaining_hot_ranks, avg_load,
                layer_idx, num_ranks, max_slots_per_rank, num_experts);
            int diff_increment = 1;
            if (total_diff_count + diff_increment <= diff_limit) {
                ChangeInstruction instr = {layer_idx, OperationType::REMOVE,
                                           rank_a,    expert_a,
                                           pos_a,     rank_a,
                                           -1,        pos_a};
                adjustments.emplace_back(1, rank_a, pos_a, expert_a, rank_a,
                                         pos_a, -1, new_ratio, diff_increment,
                                         instr);
                if (ENABLE_DEBUG) {
                    std::cout
                        << "Generated REMOVE candidate: ratio=" << new_ratio
                        << ", source=(rank=" << rank_a << ", id=" << expert_a
                        << ", pos=" << pos_a << ")" << std::endl;
                }
                break; // 每张最热卡至多一条 REMOVE 指令
            }
        }

        // SWAP 和 ADD 操作：遍历最冷卡
        for (const auto &[_, rank_b] : cold_ranks) {
            int rank_b_offset = rank_b * max_slots_per_rank;

            // SWAP 操作
            if (rank_comm_counts[rank_a] < comm_limit &&
                rank_comm_counts[rank_b] < comm_limit) {
                std::vector<std::pair<double, std::pair<int, int>>> experts_a;
                std::vector<std::pair<double, std::pair<int, int>>> experts_b;
                for (int i = 0; i < max_slots_per_rank; ++i) {
                    int pos_a = rank_a_offset + i;
                    int expert_a = g[pos_a];
                    if (expert_a != -1) {
                        double load = expert_loads.count(expert_a)
                                          ? expert_loads.at(expert_a) /
                                                expert_deployments.at(expert_a)
                                          : 0.0;
                        experts_a.emplace_back(load,
                                               std::make_pair(pos_a, expert_a));
                    }
                    int pos_b = rank_b_offset + i;
                    int expert_b = g[pos_b];
                    if (expert_b != -1) {
                        double load = expert_loads.count(expert_b)
                                          ? expert_loads.at(expert_b) /
                                                expert_deployments.at(expert_b)
                                          : 0.0;
                        experts_b.emplace_back(load,
                                               std::make_pair(pos_b, expert_b));
                    }
                }
                std::sort(experts_a.begin(), experts_a.end(), std::greater<>());
                std::sort(experts_b.begin(), experts_b.end(), std::greater<>());
                for (size_t i = 0; i < std::min<size_t>(5, experts_a.size());
                     ++i) {
                    int pos_a = experts_a[i].second.first;
                    int expert_a = experts_a[i].second.second;
                    for (size_t j = 0;
                         j < std::min<size_t>(5, experts_b.size()); ++j) {
                        int pos_b = experts_b[j].second.first;
                        int expert_b = experts_b[j].second.second;
                        if (expert_a == expert_b)
                            continue;

                        std::vector<int> temp_g = g;
                        std::swap(temp_g[pos_a], temp_g[pos_b]);
                        if (!validate_unique_expert_ids(temp_g, layer_idx,
                                                        num_ranks,
                                                        max_slots_per_rank) ||
                            !validate_all_experts_present(
                                temp_g, layer_idx, num_ranks,
                                max_slots_per_rank, num_experts)) {
                            if (ENABLE_DEBUG) {
                                std::cout << "SWAP candidate skipped: invalid "
                                             "placement in layer "
                                          << layer_idx << std::endl;
                            }
                            continue;
                        }

                        double new_ratio = simulate_adjusted_placement_ratio(
                            temp_g, expert_loads, remaining_hot_ranks, avg_load,
                            layer_idx, num_ranks, max_slots_per_rank,
                            num_experts);

                        int diff_increment = 4;
                        int comm_increment = (rank_a != rank_b) ? 2 : 1;
                        if (total_diff_count + diff_increment <= diff_limit &&
                            rank_comm_counts[rank_a] + 1 <= comm_limit &&
                            rank_comm_counts[rank_b] + 1 <= comm_limit) {
                            ChangeInstruction instr = {
                                layer_idx, OperationType::SWAP,
                                rank_a,    expert_a,
                                pos_a,     rank_b,
                                expert_b,  pos_b};
                            adjustments.emplace_back(
                                0, rank_a, pos_a, expert_a, rank_b, pos_b,
                                expert_b, new_ratio, diff_increment, instr);
                            if (ENABLE_DEBUG) {
                                std::cout
                                    << "Generated SWAP candidate: ratio="
                                    << new_ratio << ", source=(rank=" << rank_a
                                    << ", id=" << expert_a << ", pos=" << pos_a
                                    << ")"
                                    << ", target=(rank=" << rank_b
                                    << ", id=" << expert_b << ", pos=" << pos_b
                                    << ")" << std::endl;
                            }
                            break; // 每张最热卡至多一条 SWAP 指令
                        }
                    }
                    if (!adjustments.empty() &&
                        std::get<0>(adjustments.back()) == 0)
                        break; // 已有 SWAP 指令
                }
            }

            // ADD 操作（空槽位）
            if (rank_expert_counts[rank_b] <
                    num_experts_per_rank + num_redundant_per_rank &&
                rank_comm_counts[rank_b] < comm_limit) {
                int empty_pos =
                    find_empty_position(g, rank_b, max_slots_per_rank);
                if (empty_pos != -1) {
                    std::vector<std::pair<double, std::pair<int, int>>>
                        expert_load_list;
                    for (const auto &[expert_id, load] : expert_loads) {
                        if (expert_counts.at(expert_id) <
                            expert_redundant_limit + 1) {
                            for (int r = 0; r < num_ranks; ++r) {
                                int source_pos = find_position_with_expert(
                                    g, r, expert_id, max_slots_per_rank);
                                if (source_pos != -1) {
                                    expert_load_list.emplace_back(
                                        load,
                                        std::make_pair(expert_id, source_pos));
                                    break;
                                }
                            }
                        }
                    }
                    std::sort(expert_load_list.begin(), expert_load_list.end(),
                              std::greater<>());
                    for (size_t i = 0;
                         i < std::min<size_t>(5, expert_load_list.size());
                         ++i) {
                        int expert_id = expert_load_list[i].second.first;
                        int source_pos = expert_load_list[i].second.second;
                        int source_rank = source_pos / max_slots_per_rank;

                        std::vector<int> temp_g = g;
                        temp_g[empty_pos] = expert_id;
                        if (!validate_unique_expert_ids(temp_g, layer_idx,
                                                        num_ranks,
                                                        max_slots_per_rank) ||
                            !validate_all_experts_present(
                                temp_g, layer_idx, num_ranks,
                                max_slots_per_rank, num_experts) ||
                            !validate_expert_counts(
                                temp_g, num_ranks, max_slots_per_rank,
                                num_experts_per_rank, num_redundant_per_rank,
                                layer_idx)) {
                            if (ENABLE_DEBUG) {
                                std::cout << "ADD candidate skipped: invalid "
                                             "placement in layer "
                                          << layer_idx << std::endl;
                            }
                            continue;
                        }

                        double new_ratio = simulate_adjusted_placement_ratio(
                            temp_g, expert_loads, remaining_hot_ranks, avg_load,
                            layer_idx, num_ranks, max_slots_per_rank,
                            num_experts);
                        int diff_increment = 1;
                        int comm_increment = (source_rank != rank_b) ? 2 : 1;
                        if (total_diff_count + diff_increment <= diff_limit &&
                            rank_comm_counts[source_rank] + 1 <= comm_limit &&
                            rank_comm_counts[rank_b] + 1 <= comm_limit) {
                            ChangeInstruction instr = {
                                layer_idx, OperationType::ADD, source_rank,
                                expert_id, source_pos,         rank_b,
                                -1,        empty_pos};
                            adjustments.emplace_back(2, source_rank, source_pos,
                                                     expert_id, rank_b,
                                                     empty_pos, -1, new_ratio,
                                                     diff_increment, instr);
                            if (ENABLE_DEBUG) {
                                std::cout << "Generated ADD candidate: ratio="
                                          << new_ratio
                                          << ", source=(rank=" << source_rank
                                          << ", id=" << expert_id
                                          << ", pos=" << source_pos << ")"
                                          << ", target=(rank=" << rank_b
                                          << ", pos=" << empty_pos << ")"
                                          << std::endl;
                            }
                            break; // 每张最热卡至多一条 ADD 指令
                        }
                    }
                    if (!adjustments.empty() &&
                        std::get<0>(adjustments.back()) == 2)
                        break; // 已有 ADD 指令
                }
            }

            // ADD 操作（替换低负载专家）
            if (rank_comm_counts[rank_b] < comm_limit) {
                std::vector<std::pair<double, std::pair<int, int>>>
                    low_load_experts;
                for (int i = 0; i < max_slots_per_rank; ++i) {
                    int pos_b = rank_b_offset + i;
                    int expert_b = g[pos_b];
                    if (expert_b == -1 || expert_counts.at(expert_b) <= 1)
                        continue;
                    double load = expert_loads.count(expert_b)
                                      ? expert_loads.at(expert_b) /
                                            expert_deployments.at(expert_b)
                                      : 0.0;
                    low_load_experts.emplace_back(
                        load, std::make_pair(pos_b, expert_b));
                }
                std::sort(low_load_experts.begin(), low_load_experts.end());
                std::vector<std::pair<double, std::pair<int, int>>> hot_experts;
                for (int i = 0; i < max_slots_per_rank; ++i) {
                    int pos_a = rank_a_offset + i;
                    int expert_a = g[pos_a];
                    if (expert_a == -1)
                        continue;
                    double load = expert_loads.count(expert_a)
                                      ? expert_loads.at(expert_a) /
                                            expert_deployments.at(expert_a)
                                      : 0.0;
                    hot_experts.emplace_back(load,
                                             std::make_pair(pos_a, expert_a));
                }
                std::sort(hot_experts.begin(), hot_experts.end(),
                          std::greater<>());
                for (size_t i = 0; i < std::min<size_t>(5, hot_experts.size());
                     ++i) {
                    int pos_a = hot_experts[i].second.first;
                    int expert_a = hot_experts[i].second.second;
                    if (expert_counts.at(expert_a) >=
                        expert_redundant_limit + 1)
                        continue;
                    for (size_t j = 0;
                         j < std::min<size_t>(5, low_load_experts.size());
                         ++j) {
                        int pos_b = low_load_experts[j].second.first;
                        int expert_b = low_load_experts[j].second.second;
                        if (expert_a == expert_b)
                            continue;

                        std::vector<int> temp_g = g;
                        temp_g[pos_b] = expert_a;
                        std::unordered_map<int, double> temp_expert_loads =
                            expert_loads;
                        std::unordered_map<int, int> temp_deployments =
                            expert_deployments;
                        temp_deployments[expert_a]++;
                        temp_deployments[expert_b]--;
                        temp_expert_loads[expert_a] =
                            expert_loads.at(expert_a) *
                            expert_deployments.at(expert_a) /
                            temp_deployments[expert_a];
                        if (temp_deployments[expert_b] > 0) {
                            temp_expert_loads[expert_b] =
                                expert_loads.at(expert_b) *
                                expert_deployments.at(expert_b) /
                                temp_deployments[expert_b];
                        } else {
                            temp_expert_loads.erase(expert_b);
                        }

                        if (!validate_unique_expert_ids(temp_g, layer_idx,
                                                        num_ranks,
                                                        max_slots_per_rank) ||
                            !validate_all_experts_present(
                                temp_g, layer_idx, num_ranks,
                                max_slots_per_rank, num_experts) ||
                            !validate_expert_counts(
                                temp_g, num_ranks, max_slots_per_rank,
                                num_experts_per_rank, num_redundant_per_rank,
                                layer_idx)) {
                            if (ENABLE_DEBUG) {
                                std::cout << "ADD (replace) candidate skipped: "
                                             "invalid "
                                             "placement in layer "
                                          << layer_idx << std::endl;
                            }
                            continue;
                        }
                        double new_ratio = simulate_adjusted_placement_ratio(
                            temp_g, temp_expert_loads, remaining_hot_ranks,
                            avg_load, layer_idx, num_ranks, max_slots_per_rank,
                            num_experts);
                        int diff_increment = 1;
                        int comm_increment = (rank_a != rank_b) ? 2 : 1;
                        if (total_diff_count + diff_increment <= diff_limit &&
                            rank_comm_counts[rank_a] + 1 <= comm_limit &&
                            rank_comm_counts[rank_b] + 1 <= comm_limit) {
                            ChangeInstruction instr = {
                                layer_idx, OperationType::ADD,
                                rank_a,    expert_a,
                                pos_a,     rank_b,
                                -1,        pos_b};
                            adjustments.emplace_back(
                                2, rank_a, pos_a, expert_a, rank_b, pos_b, -1,
                                new_ratio, diff_increment, instr);
                            if (ENABLE_DEBUG) {
                                std::cout
                                    << "Generated ADD (replace) candidate: "
                                       "ratio="
                                    << new_ratio << ", source=(rank=" << rank_a
                                    << ", id=" << expert_a << ", pos=" << pos_a
                                    << ")"
                                    << ", target=(rank=" << rank_b
                                    << ", pos=" << pos_b << ")" << std::endl;
                            }
                            break; // 每张最热卡至多一条 ADD 指令
                        }
                    }
                    if (!adjustments.empty() &&
                        std::get<0>(adjustments.back()) == 2)
                        break; // 已有 ADD 指令
                }
                if (!adjustments.empty() &&
                    std::get<0>(adjustments.back()) == 2)
                    break; // 已有 ADD 指令
            }
            if (!adjustments.empty())
                break; // 每张最热卡至多一条指令
        }

        // 选择最佳调整
        double total_load_current = total_load;
        double max_load_current = max_load;
        double avg_load_current = total_load_current / num_ranks;
        double current_ratio =
            (avg_load_current > 0) ? max_load_current / avg_load_current : 1.0;

        double best_ratio = current_ratio;
        std::tuple<int, int, int, int, int, int, int, double, int,
                   ChangeInstruction>
            best_adjustment = {-1, -1, -1, -1,
                               -1, -1, -1, std::numeric_limits<double>::max(),
                               0,  {}};
        const double ratio_threshold = 0.01;
        for (const auto &adj : adjustments) {
            auto &[type, rank_a, pos_a, expert_a, rank_b, pos_b, expert_b,
                   new_ratio, diff_increment, instr] = adj;
            if (new_ratio < best_ratio - ratio_threshold) {
                best_ratio = new_ratio;
                best_adjustment = adj;
            } else if (std::abs(new_ratio - best_ratio) < ratio_threshold &&
                       diff_increment < std::get<8>(best_adjustment)) {
                best_ratio = new_ratio;
                best_adjustment = adj;
            }
        }

        if (std::get<0>(best_adjustment) != -1 && best_ratio < current_ratio) {
            selected_adjustments.push_back(best_adjustment);
            if (ENABLE_DEBUG) {
                auto [type, rank_a, pos_a, expert_a, rank_b, pos_b, expert_b,
                      new_ratio, diff_increment, instr] = best_adjustment;
                std::cout << "Selected adjustment for rank " << rank_a
                          << ": type=" << type << ", ratio=" << new_ratio
                          << ", diff_increment=" << diff_increment << std::endl;
            }
        }
    }

    // 更新状态并应用所有选中的调整
    if (!selected_adjustments.empty()) {
        // 备份当前状态以便验证
        std::vector<int> temp_g = g;
        std::unordered_map<int, double> temp_expert_loads = expert_loads;
        std::unordered_map<int, int> temp_deployments = expert_deployments;
        std::unordered_map<int, int> temp_expert_counts = expert_counts;
        std::vector<int> temp_rank_expert_counts = rank_expert_counts;
        std::vector<int> temp_rank_comm_counts = rank_comm_counts;
        int temp_total_diff_count = total_diff_count;

        // 应用所有调整
        for (const auto &[type, rank_a, pos_a, expert_a, rank_b, pos_b,
                          expert_b, new_ratio, diff_increment, instr] :
             selected_adjustments) {
            if (type == 0) { // SWAP
                std::swap(temp_g[pos_a], temp_g[pos_b]);
                temp_rank_comm_counts[rank_a]++;
                if (rank_a != rank_b) {
                    temp_rank_comm_counts[rank_b]++;
                }
            } else if (type == 1) { // REMOVE
                temp_g[pos_a] = -1;
                temp_expert_counts[expert_a]--;
                temp_rank_expert_counts[rank_a]--;
                temp_deployments[expert_a]--;
                if (temp_deployments[expert_a] > 0) {
                    temp_expert_loads[expert_a] =
                        temp_expert_loads.at(expert_a) *
                        (temp_deployments[expert_a] + 1) /
                        temp_deployments[expert_a];
                } else {
                    temp_expert_loads.erase(expert_a);
                }
            } else if (type == 2) { // ADD
                int old_expert = temp_g[pos_b];
                temp_g[pos_b] = expert_a;
                temp_expert_counts[expert_a]++;
                temp_rank_expert_counts[rank_b]++;
                if (old_expert != -1) {
                    temp_expert_counts[old_expert]--;
                    temp_deployments[old_expert]--;
                    if (temp_deployments[old_expert] > 0) {
                        temp_expert_loads[old_expert] =
                            temp_expert_loads.at(old_expert) *
                            (temp_deployments[old_expert] + 1) /
                            temp_deployments[old_expert];
                    } else {
                        temp_expert_loads.erase(old_expert);
                    }
                }
                temp_rank_comm_counts[rank_a]++;
                if (rank_a != rank_b) {
                    temp_rank_comm_counts[rank_b]++;
                }
                temp_deployments[expert_a]++;
                temp_expert_loads[expert_a] = temp_expert_loads.at(expert_a) *
                                              (temp_deployments[expert_a] - 1) /
                                              temp_deployments[expert_a];
            }
            temp_total_diff_count += diff_increment;
        }

        // 验证应用所有调整后的 placement
        if (!validate_unique_expert_ids(temp_g, layer_idx, num_ranks,
                                        max_slots_per_rank) ||
            !validate_all_experts_present(temp_g, layer_idx, num_ranks,
                                          max_slots_per_rank, num_experts) ||
            !validate_expert_counts(temp_g, num_ranks, max_slots_per_rank,
                                    num_experts_per_rank,
                                    num_redundant_per_rank, layer_idx)) {
            if (ENABLE_DEBUG) {
                std::cout << "Invalid placement after applying multiple "
                             "adjustments in layer "
                          << layer_idx << std::endl;
            }
            return {};
        }

        // 检查通信次数
        for (int r = 0; r < num_ranks; ++r) {
            if (temp_rank_comm_counts[r] > comm_limit) {
                if (ENABLE_DEBUG) {
                    std::cout << "Communication limit exceeded for rank " << r
                              << " after applying multiple adjustments"
                              << std::endl;
                }
                return {};
            }
        }

        // 检查差异限制
        if (temp_total_diff_count > diff_limit) {
            if (ENABLE_DEBUG) {
                std::cout << "Total diff count " << temp_total_diff_count
                          << " exceeds limit " << diff_limit
                          << " after applying multiple adjustments"
                          << std::endl;
            }
            return {};
        }

        // 应用调整到实际状态
        g = temp_g;
        expert_loads = temp_expert_loads;
        expert_deployments = temp_deployments;
        expert_counts = temp_expert_counts;
        rank_expert_counts = temp_rank_expert_counts;
        rank_comm_counts = temp_rank_comm_counts;
        total_diff_count = temp_total_diff_count;

        // 更新 rank_loads
        rank_loads.assign(num_ranks, 0.0);
        for (int rank = 0; rank < num_ranks; ++rank) {
            int rank_offset = rank * max_slots_per_rank;
            for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
                int idx = rank_offset + slot_idx;
                int expert_id = g[idx];
                if (expert_id != -1 && expert_loads.count(expert_id)) {
                    rank_loads[rank] +=
                        expert_loads[expert_id] / expert_deployments[expert_id];
                }
            }
        }

        return selected_adjustments;
    }

    return {};
}

std::vector<int> ExpertLoadBalancer::generate_constrained_placement(
    const std::vector<std::vector<ExpertInformation>> &layer_experts,
    const std::vector<int> &input_placement,
    const std::vector<int64_t> &input_activations, int layer_idx, int num_ranks,
    int num_experts_per_rank, int num_redundant_per_rank,
    int expert_redundant_limit, int diff_limit,
    std::vector<ChangeInstruction> &instructions,
    std::vector<std::tuple<double, int, ChangeInstruction>>
        &candidate_adjustments) {
    int max_slots_per_rank = num_experts_per_rank + num_redundant_per_rank;
    int num_experts = num_ranks * num_experts_per_rank;

    int layer_offset = layer_idx * num_ranks * max_slots_per_rank;
    std::vector<int> f(input_placement.begin() + layer_offset,
                       input_placement.begin() + layer_offset +
                           num_ranks * max_slots_per_rank);
    std::vector<int64_t> layer_activations(
        input_activations.begin() + layer_offset,
        input_activations.begin() + layer_offset +
            num_ranks * max_slots_per_rank);

    if (!validate_expert_counts(f, num_ranks_, max_slots_per_rank,
                                num_experts_per_rank, num_redundant_per_rank,
                                layer_idx) ||
        !validate_unique_expert_ids(f, layer_idx, num_ranks_,
                                    max_slots_per_rank) ||
        !validate_all_experts_present(f, layer_idx, num_ranks_,
                                      max_slots_per_rank, num_experts)) {
        return {};
    }

    std::vector<int> g = f;
    std::vector<int> last_valid_g = g;
    std::vector<ChangeInstruction> last_valid_instructions;
    auto expert_loads =
        compute_expert_loads(layer_experts[layer_idx], num_experts);
    std::unordered_map<int, int> expert_counts =
        compute_expert_counts(g, num_ranks_, max_slots_per_rank);
    std::vector<int> rank_expert_counts(num_ranks, 0);
    std::vector<int> rank_comm_counts(num_ranks, 0);
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int i = 0; i < max_slots_per_rank; ++i) {
            if (g[rank_offset + i] != -1) {
                rank_expert_counts[rank]++;
            }
        }
    }

    std::vector<std::set<int>> to_remove(num_ranks), to_add(num_ranks);
    int total_diff_count = 0;
    std::vector<ChangeInstruction> adjustment_instructions;

    std::vector<double> rank_loads(num_ranks, 0.0);
    std::unordered_map<int, int> expert_deployments;
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = g[idx];
            if (expert_id != -1) {
                expert_deployments[expert_id]++;
            }
        }
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
        int rank_offset = rank * max_slots_per_rank;
        for (int slot_idx = 0; slot_idx < max_slots_per_rank; ++slot_idx) {
            int idx = rank_offset + slot_idx;
            int expert_id = g[idx];
            if (expert_id != -1 && expert_loads.count(expert_id)) {
                rank_loads[rank] +=
                    expert_loads[expert_id] / expert_deployments[expert_id];
            }
        }
    }

    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations && total_diff_count < diff_limit;
         ++iter) {
        // 调用 find_next_candidate 获取多条指令
        auto adjustments = find_next_candidate(
            g, layer_idx, rank_loads, expert_loads, expert_deployments,
            expert_counts, rank_expert_counts, rank_comm_counts, num_ranks_,
            max_slots_per_rank, num_experts, num_experts_per_rank,
            num_redundant_per_rank, expert_redundant_limit, comm_limit_,
            total_diff_count, diff_limit);

        if (adjustments.empty()) {
            if (ENABLE_DEBUG) {
                std::cout << "No valid adjustments found, stopping "
                             "optimization for layer "
                          << layer_idx << std::endl;
            }
            break;
        }

        // 添加所有调整指令
        for (const auto &[type, rank_a, pos_a, expert_a, rank_b, pos_b,
                          expert_b, new_ratio, diff_increment, instr] :
             adjustments) {
            adjustment_instructions.push_back(instr);
            candidate_adjustments.emplace_back(new_ratio, diff_increment,
                                               instr);
            total_diff_count += diff_increment;

            // 更新 to_remove 和 to_add
            if (type == 0) { // SWAP
                to_remove[rank_a].insert(expert_a);
                to_add[rank_b].insert(expert_a);
                to_remove[rank_b].insert(expert_b);
                to_add[rank_a].insert(expert_b);
            } else if (type == 1) { // REMOVE
                to_remove[rank_a].insert(expert_a);
            } else if (type == 2) { // ADD
                to_add[rank_b].insert(expert_a);
                if (g[pos_b] != -1) {
                    to_remove[rank_b].insert(g[pos_b]);
                }
            }
        }

        // 检查通信次数是否超限
        bool comm_limit_exceeded = false;
        for (int r = 0; r < num_ranks; ++r) {
            if (rank_comm_counts[r] > comm_limit_) {
                std::cerr << "Error: Communication count for rank " << r << " ("
                          << rank_comm_counts[r] << ") exceeds limit ("
                          << comm_limit_ << ") in layer_idx " << layer_idx
                          << "\n";
                comm_limit_exceeded = true;
            }
        }
        if (comm_limit_exceeded) {
            if (ENABLE_DEBUG) {
                std::cout << "Reverting to last valid state due to "
                             "communication limit "
                             "exceeded"
                          << std::endl;
            }
            g = last_valid_g;
            adjustment_instructions = last_valid_instructions;
            candidate_adjustments.clear();
            break;
        }

        // 验证当前 placement
        if (!validate_expert_counts(g, num_ranks_, max_slots_per_rank,
                                    num_experts_per_rank,
                                    num_redundant_per_rank, layer_idx) ||
            !validate_unique_expert_ids(g, layer_idx, num_ranks_,
                                        max_slots_per_rank) ||
            !validate_all_experts_present(g, layer_idx, num_ranks_,
                                          max_slots_per_rank, num_experts)) {
            std::cerr << "Error: Invalid placement after applying adjustments "
                         "in layer "
                      << layer_idx << "\n";
            g = last_valid_g;
            adjustment_instructions = last_valid_instructions;
            candidate_adjustments.clear();
            break;
        }

        if (total_diff_count > diff_limit) {
            std::cerr << "Error: Total diff count " << total_diff_count
                      << " exceeds limit " << diff_limit << "\n";
            g = last_valid_g;
            adjustment_instructions = last_valid_instructions;
            candidate_adjustments.clear();
            break;
        }

        last_valid_g = g;
        last_valid_instructions = adjustment_instructions;
    }

    if (ENABLE_DEBUG) {
        std::cout << "Final communication counts for layer " << layer_idx
                  << ": ";
        for (int r = 0; r < num_ranks; ++r) {
            std::cout << "Rank " << r << ": " << rank_comm_counts[r] << " ";
        }
        std::cout << std::endl;
    }

    instructions = adjustment_instructions;
    return g;
}

// find_best_swap 保持不变，因为交换逻辑与 Python 一致
std::optional<std::pair<int, int>> ExpertLoadBalancer::find_best_swap(
    std::vector<int> &placement1, std::vector<int64_t> &activations1,
    std::vector<int> &placement2, std::vector<int64_t> &activations2,
    int max_slots_per_rank) {
    int64_t sum1 =
        std::accumulate(activations1.begin(), activations1.end(), int64_t(0));
    int64_t sum2 =
        std::accumulate(activations2.begin(), activations2.end(), int64_t(0));
    int64_t delta = sum1 - sum2;
    int64_t best_diff = std::abs(delta);
    std::pair<int, int> best_pos = {-1, -1};

    for (size_t i = 0; i < placement1.size(); ++i) {
        if (placement1[i] == -1)
            continue;
        for (size_t j = 0; j < placement2.size(); ++j) {
            if (placement2[j] == -1)
                continue;
            int64_t a = activations1[i];
            int64_t b = activations2[j];
            int64_t new_delta = delta + 2 * (b - a);
            if (std::abs(new_delta) < best_diff) {
                best_diff = std::abs(new_delta);
                best_pos = {static_cast<int>(i), static_cast<int>(j)};
            }
        }
    }

    if (best_pos.first != -1) {
        std::swap(placement1[best_pos.first], placement2[best_pos.second]);
        std::swap(activations1[best_pos.first], activations2[best_pos.second]);
        return best_pos;
    }
    return std::nullopt;
}

void ExpertLoadBalancer::buck_balance(
    std::vector<int> &placement, std::vector<int64_t> &activations, int left,
    int right, int layer_idx, int max_slots_per_rank, int num_ranks,
    std::vector<std::pair<ChangeInstruction, int>> &instruction_depths,
    int depth) {
    // 终止条件：区间长度等于 max_slots_per_rank
    if (right - left + 1 == max_slots_per_rank)
        return;

    // 动态二分划分
    int middle = (left + right) / 2;

    // 提取两个 buckets
    std::vector<int> bucket1_placement(placement.begin() + left,
                                       placement.begin() + middle + 1);
    std::vector<int64_t> bucket1_activations(activations.begin() + left,
                                             activations.begin() + middle + 1);
    std::vector<int> bucket2_placement(placement.begin() + middle + 1,
                                       placement.begin() + right + 1);
    std::vector<int64_t> bucket2_activations(activations.begin() + middle + 1,
                                             activations.begin() + right + 1);

    // 平衡两个 buckets
    while (true) {
        auto best_pos = find_best_swap(bucket1_placement, bucket1_activations,
                                       bucket2_placement, bucket2_activations,
                                       max_slots_per_rank);
        if (!best_pos.has_value())
            break;

        // 计算层内位置
        int pos_a = left + best_pos->first;
        int pos_b = middle + 1 + best_pos->second;

        // 计算 rank
        int rank_a = pos_a / max_slots_per_rank;
        int rank_b = pos_b / max_slots_per_rank;

        // 验证 rank 的合法性
        if (rank_a < 0 || rank_a >= num_ranks || rank_b < 0 ||
            rank_b >= num_ranks) {
            std::cerr << "Error: Invalid rank calculated in layer " << layer_idx
                      << ": rank_a=" << rank_a << ", rank_b=" << rank_b << "\n";
            continue;
        }

        // 使用层内完整位置
        if (pos_a < 0 || pos_a >= num_ranks * max_slots_per_rank || pos_b < 0 ||
            pos_b >= num_ranks * max_slots_per_rank) {
            std::cerr << "Error: Invalid position in layer " << layer_idx
                      << ": pos_a=" << pos_a << ", pos_b=" << pos_b << "\n";
            continue;
        }

        // 生成 SWAP 指令，使用层内完整位置
        ChangeInstruction instr{
            layer_idx, OperationType::SWAP, rank_a, placement[pos_a], pos_a,
            rank_b,    placement[pos_b],    pos_b};
        instruction_depths.emplace_back(instr, depth); // 记录指令和当前深度

        // 更新 bucket1 和 bucket2
        placement[pos_a] = bucket1_placement[best_pos->first];
        placement[pos_b] = bucket2_placement[best_pos->second];
        activations[pos_a] = bucket1_activations[best_pos->first];
        activations[pos_b] = bucket2_activations[best_pos->second];
    }

    // 递归处理左右子区间
    if (middle - left + 1 >= max_slots_per_rank) {
        buck_balance(placement, activations, left, middle, layer_idx,
                     max_slots_per_rank, num_ranks, instruction_depths,
                     depth + 1); // 深度加 1
    }
    if (right - (middle + 1) + 1 >= max_slots_per_rank) {
        buck_balance(placement, activations, middle + 1, right, layer_idx,
                     max_slots_per_rank, num_ranks, instruction_depths,
                     depth + 1); // 深度加 1
    }
}

BucketBalancedResult ExpertLoadBalancer::generate_bucket_balanced_placement(
    const std::vector<int> &layer_placement,
    const std::vector<int64_t> &layer_activations, int layer_idx, int num_ranks,
    int num_experts_per_rank) {
    BucketBalancedResult result;
    int max_slots_per_rank = num_experts_per_rank; // 无冗余槽位
    size_t expected_size = static_cast<size_t>(num_ranks) * max_slots_per_rank;
    int num_experts = num_ranks * num_experts_per_rank;

    // 验证输入大小
    if (layer_placement.size() != expected_size ||
        layer_activations.size() != expected_size) {
        std::cerr << "Error: Invalid input size for layer " << layer_idx
                  << ": placement_size=" << layer_placement.size()
                  << ", activations_size=" << layer_activations.size()
                  << ", expected=" << expected_size << "\n";
        return result;
    }

    // 复制输入数据以进行修改
    std::vector<int> placement = layer_placement;
    std::vector<int64_t> activations = layer_activations;

    // 验证输入
    if (!validate_unique_expert_ids(placement, layer_idx, num_ranks,
                                    max_slots_per_rank) ||
        !validate_all_experts_present(placement, layer_idx, num_ranks,
                                      max_slots_per_rank, num_experts) ||
        !validate_expert_counts(placement, num_ranks, max_slots_per_rank,
                                num_experts_per_rank, 0, layer_idx)) {
        std::cerr << "Error: Invalid input placement for layer " << layer_idx
                  << "\n";
        return result;
    }

    std::vector<std::pair<ChangeInstruction, int>> instruction_depths;
    buck_balance(placement, activations, 0, num_ranks * max_slots_per_rank - 1,
                 layer_idx, max_slots_per_rank, num_ranks, instruction_depths,
                 0);

    // 提取指令
    std::vector<ChangeInstruction> instructions;
    for (const auto &[instr, depth] : instruction_depths) {
        instructions.push_back(instr);
    }

    // 验证最终 placement
    if (!validate_unique_expert_ids(placement, layer_idx, num_ranks,
                                    max_slots_per_rank) ||
        !validate_all_experts_present(placement, layer_idx, num_ranks,
                                      max_slots_per_rank, num_experts) ||
        !validate_expert_counts(placement, num_ranks, max_slots_per_rank,
                                num_experts_per_rank, 0, layer_idx)) {
        std::cerr << "Error: Invalid final placement for layer " << layer_idx
                  << "\n";
        return result;
    }

    result.instructions = instructions;
    result.placement = placement;
    result.instruction_depths = instruction_depths;
    return result;
}

std::vector<ChangeInstruction>
ExpertLoadBalancer::optimize_and_generate_instructions(
    const std::vector<int> &input_placement,
    const std::vector<int64_t> &input_activations) {
    std::vector<ChangeInstruction> all_instructions;
    std::vector<LayerDebugInfo> layer_debug_info;

    std::chrono::high_resolution_clock::time_point start_time;
    if (ENABLE_DEBUG && rank_ == 0) {
        start_time = std::chrono::high_resolution_clock::now();
        std::time_t start_t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::cout << "Optimization started at: " << std::ctime(&start_t);
    }

    int max_slots_per_rank = num_experts_per_rank_ + num_redundant_per_rank_;
    int num_experts = num_ranks_ * num_experts_per_rank_;
    int diff_limit = num_ranks_ * 2;

    if (!validate_input_size(input_placement, input_activations, num_layers_,
                             num_ranks_, max_slots_per_rank)) {
        std::cerr << "Error: Invalid input size\n";
        return {};
    }

    if (num_redundant_per_rank_ == 0) {
        max_slots_per_rank = num_experts_per_rank_;

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank;

            std::vector<int> layer_placement(
                input_placement.begin() + layer_offset,
                input_placement.begin() + layer_offset +
                    num_ranks_ * max_slots_per_rank);
            std::vector<int64_t> layer_activations(
                input_activations.begin() + layer_offset,
                input_activations.begin() + layer_offset +
                    num_ranks_ * max_slots_per_rank);

            if (!validate_expert_counts(layer_placement, num_ranks_,
                                        max_slots_per_rank,
                                        num_experts_per_rank_, 0, layer_idx) ||
                !validate_unique_expert_ids(layer_placement, layer_idx,
                                            num_ranks_, max_slots_per_rank) ||
                !validate_all_experts_present(layer_placement, layer_idx,
                                              num_ranks_, max_slots_per_rank,
                                              num_experts)) {
                std::cerr << "Error: Invalid initial placement for layer "
                          << layer_idx << "\n";
                return {};
            }

            // 调用 bucket 平衡算法
            auto result = generate_bucket_balanced_placement(
                layer_placement, layer_activations, layer_idx, num_ranks_,
                num_experts_per_rank_);

            if (result.placement.empty()) {
                std::cerr << "Error: Failed to generate bucket balanced "
                             "placement for layer "
                          << layer_idx << "\n";
                return {};
            }

            // 计算输入排布的 ratio
            double input_ratio = compute_placement_ratio(
                layer_placement, layer_activations, layer_idx, num_ranks_,
                max_slots_per_rank, num_experts);
            if (ENABLE_DEBUG && rank_ == 0) {
                std::cout << "Input Placement Ratio for layer " << layer_idx
                          << ": " << std::fixed << std::setprecision(6)
                          << input_ratio << "\n";
                std::cout << "Verifying initial placement for layer "
                          << layer_idx << "\n";
            }

            std::vector<std::pair<ChangeInstruction, int>>
                sorted_instruction_depths = result.instruction_depths;
            std::stable_sort(sorted_instruction_depths.begin(),
                             sorted_instruction_depths.end(),
                             [](const auto &a, const auto &b) {
                                 return a.second <
                                        b.second; // 按深度从小到大排序
                             });

            std::vector<ChangeInstruction> sorted_instructions;
            for (const auto &[instr, depth] : sorted_instruction_depths) {
                sorted_instructions.push_back(instr);
            }

            // 对 sorted_instructions 调用 transfer 函数进行优先级排序
            transfer(sorted_instructions);

            // 新增：将 sorted_instructions 的 prior 值回写到
            // sorted_instruction_depths
            for (size_t i = 0; i < sorted_instructions.size(); ++i) {
                sorted_instruction_depths[i].first.prior =
                    sorted_instructions[i].prior;
            }

            // 计算输出排布的 ratio
            std::unordered_map<int, double> expert_loads;
            std::unordered_map<int, int> expert_counts;
            for (size_t i = 0; i < layer_placement.size(); ++i) {
                int expert_id = layer_placement[i];
                if (expert_id == -1)
                    continue;
                if (expert_id < 0 || expert_id >= num_experts) {
                    std::cerr << "Warning: Invalid expert ID " << expert_id
                              << " in layer " << layer_idx << ", position " << i
                              << "\n";
                    continue;
                }
                expert_loads[expert_id] +=
                    static_cast<double>(layer_activations[i]);
                expert_counts[expert_id]++;
                if (expert_counts[expert_id] > expert_redundant_limit_ + 1) {
                    std::cerr << "Warning: Expert ID " << expert_id
                              << " exceeds redundant limit "
                              << expert_redundant_limit_ << " in layer "
                              << layer_idx << "\n";
                    expert_loads[expert_id] = 0.0;
                }
            }
            double output_ratio = simulate_placement_ratio(
                result.placement, expert_loads, layer_idx, num_ranks_,
                max_slots_per_rank, num_experts);
            if (ENABLE_DEBUG && rank_ == 0) {
                std::cout << "Output Placement Ratio for layer " << layer_idx
                          << ": " << std::fixed << std::setprecision(6)
                          << output_ratio << "\n";
            }

            // 新增：比较 output_ratio 和 input_ratio
            if (output_ratio < input_ratio - 0.1) {
                all_instructions.insert(all_instructions.end(),
                                        sorted_instructions.begin(),
                                        sorted_instructions.end());
            } else {
                if (ENABLE_DEBUG && rank_ == 0) {
                    std::cout << "Skipping layer " << layer_idx
                              << ": output_ratio (" << output_ratio
                              << ") is not sufficiently improved compared to "
                                 "input_ratio ("
                              << input_ratio << ")\n";
                }
                // 清空 sorted_instructions 和
                // sorted_instruction_depths，以反映不生成指令
                sorted_instructions.clear();
                sorted_instruction_depths.clear();
            }

            // 收集调试信息
            LayerDebugInfo info;
            info.layer_idx = layer_idx;
            info.input_placement = layer_placement;
            info.input_activations = layer_activations;
            info.initial_placement = layer_placement;
            info.optimized_placement =
                (output_ratio < input_ratio - 0.1)
                    ? result.placement
                    : layer_placement; // 使用优化后的 placement 仅当满足条件
            info.output_placement = info.optimized_placement;
            info.instructions = sorted_instructions; // 仅包含满足条件的指令
            info.adjustment_instructions = {};
            info.instruction_depths =
                sorted_instruction_depths; // 仅包含满足条件的指令深度
            info.candidate_adjustments = {};
            info.total_diff_count =
                sorted_instructions.size(); // 反映实际指令数量
            layer_debug_info.push_back(info);
        }
    } else {
        // 原有逻辑保持不变
        auto layer_experts = extract_expert_info(
            input_placement, input_activations, num_layers_, num_ranks_,
            num_experts_per_rank_, num_redundant_per_rank_,
            expert_redundant_limit_);
        if (layer_experts.empty()) {
            std::cerr << "Error: Failed to extract expert information\n";
            return {};
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank;
            std::vector<int> f(input_placement.begin() + layer_offset,
                               input_placement.begin() + layer_offset +
                                   num_ranks_ * max_slots_per_rank);
            std::vector<int64_t> layer_activations(
                input_activations.begin() + layer_offset,
                input_activations.begin() + layer_offset +
                    num_ranks_ * max_slots_per_rank);

            if (!validate_expert_counts(f, num_ranks_, max_slots_per_rank,
                                        num_experts_per_rank_,
                                        num_redundant_per_rank_, layer_idx) ||
                !validate_unique_expert_ids(f, layer_idx, num_ranks_,
                                            max_slots_per_rank) ||
                !validate_all_experts_present(f, layer_idx, num_ranks_,
                                              max_slots_per_rank,
                                              num_experts)) {
                std::cerr << "Error: Invalid initial placement for layer "
                          << layer_idx << "\n";
                return {};
            }

            std::vector<ChangeInstruction> layer_instructions;
            std::vector<std::tuple<double, int, ChangeInstruction>>
                candidate_adjustments;
            std::vector<int> g = generate_constrained_placement(
                layer_experts, input_placement, input_activations, layer_idx,
                num_ranks_, num_experts_per_rank_, num_redundant_per_rank_,
                expert_redundant_limit_, diff_limit, layer_instructions,
                candidate_adjustments);
            if (g.empty()) {
                std::cerr << "Error: Failed to generate constrained placement "
                             "for layer "
                          << layer_idx << "\n";
                return {};
            }

            all_instructions.insert(all_instructions.end(),
                                    layer_instructions.begin(),
                                    layer_instructions.end());

            std::vector<std::set<int>> to_remove(num_ranks_),
                to_add(num_ranks_);
            int total_diff_count = 0;
            auto S_f = compute_rank_sets(f, num_ranks_, max_slots_per_rank);
            auto S_g = compute_rank_sets(g, num_ranks_, max_slots_per_rank);
            for (int r = 0; r < num_ranks_; ++r) {
                std::set_difference(
                    S_f[r].begin(), S_f[r].end(), S_g[r].begin(), S_g[r].end(),
                    std::inserter(to_remove[r], to_remove[r].begin()));
                std::set_difference(
                    S_g[r].begin(), S_g[r].end(), S_f[r].begin(), S_f[r].end(),
                    std::inserter(to_add[r], to_add[r].begin()));
                total_diff_count += to_remove[r].size() + to_add[r].size();
            }

            LayerDebugInfo info;
            info.layer_idx = layer_idx;
            info.input_placement = f;
            info.input_activations = layer_activations;
            info.initial_placement = f;
            info.optimized_placement = g;
            info.output_placement = g;
            info.instructions = layer_instructions;
            info.adjustment_instructions = layer_instructions;
            info.instruction_depths = {}; // 非 bucket 平衡模式，无深度信息
            info.candidate_adjustments = candidate_adjustments;
            info.total_diff_count = total_diff_count;
            layer_debug_info.push_back(info);
        }
    }

    if (rank_ == 0) {
        print_debug_info(layer_debug_info, ENABLE_DEBUG, start_time);
    }
    return all_instructions;
}

// 打印 Placement
void ExpertLoadBalancer::print_placement(const vector<int> &placement,
                                         int layer_idx, const string &label) {
    cout << label << " for layer " << layer_idx << ": ";
    for (int x : placement) {
        cout << x << " ";
    }
    cout << "\n";
}

// 打印指令
void ExpertLoadBalancer::print_instruction(const ChangeInstruction &instr,
                                           int depth) {
    cout << "  Instruction: Type=" << static_cast<int>(instr.type)
         << ", Source=(rank=" << instr.source_rank
         << ", id=" << instr.source_expert_id
         << ", pos=" << instr.source_global_position << ")"
         << ", Target=(rank=" << instr.target_rank
         << ", id=" << instr.target_expert_id
         << ", pos=" << instr.target_global_position << ")";
    if (depth >= 0) {
        cout << ", Recursive Depth=" << depth;
    }
    cout << ", Priority=" << instr.prior; // 新增：打印优先级
    cout << "\n";
}

// 打印激活值
void ExpertLoadBalancer::print_activations(const vector<int64_t> &activations,
                                           int layer_idx) {
    cout << "Input Activations for layer " << layer_idx << ": ";
    for (int64_t x : activations) {
        cout << x << " ";
    }
    cout << "\n";
}

void ExpertLoadBalancer::print_debug_info(
    const std::vector<LayerDebugInfo> &layer_info, bool enable_debug,
    const std::chrono::high_resolution_clock::time_point &start_time) {
    if (!enable_debug)
        return;

    std::cout << "=== Debug Information ===\n";
    std::cout << "=== Step 1: Extract expert information ===\n";

    std::cout << "\n=== Step 2: Generate optimized placement and instructions "
                 "per layer ===\n";
    for (const auto &info : layer_info) {
        int layer_idx = info.layer_idx;
        std::cout << "\nProcessing layer " << layer_idx << "\n";

        print_placement(info.input_placement, layer_idx, "Input Placement");
        print_activations(info.input_activations, layer_idx);

        double input_ratio = compute_placement_ratio(
            info.input_placement, info.input_activations, layer_idx, num_ranks_,
            get_max_slots_per_rank(), num_ranks_ * num_experts_per_rank_);
        std::cout << "Input Placement Ratio for layer " << layer_idx << ": "
                  << std::fixed << std::setprecision(6) << input_ratio << "\n";

        std::cout << "Verifying initial placement for layer " << layer_idx
                  << "\n";

        print_placement(info.initial_placement, layer_idx,
                        "Layer " + std::to_string(layer_idx) +
                            " initial placement (f)");
        print_placement(info.optimized_placement, layer_idx,
                        "Layer " + std::to_string(layer_idx) +
                            " optimized placement (g)");

        std::cout << "\nAdjustment Instructions for layer " << layer_idx
                  << ":\n";
        if (info.adjustment_instructions.empty()) {
            std::cout << "  No adjustment instructions generated.\n";
        } else {
            if (info.instruction_depths.empty()) {
                for (const auto &instr : info.adjustment_instructions) {
                    print_instruction(instr, -1);
                }
            } else {
                for (const auto &[instr, depth] : info.instruction_depths) {
                    print_instruction(instr, depth);
                }
            }
        }

        std::cout << "\nCandidate Adjustments for layer " << layer_idx << ":\n";
        if (info.candidate_adjustments.empty()) {
            std::cout << "  No candidate adjustments generated.\n";
        } else {
            for (const auto &[ratio, diff, instr] :
                 info.candidate_adjustments) {
                std::cout << "  Ratio: " << std::fixed << std::setprecision(6)
                          << ratio << ", Diff Increment: " << diff << ", ";
                print_instruction(instr, -1);
            }
        }

        std::cout << "Total Diff Count (to_remove + to_add): "
                  << info.total_diff_count << "\n";

        std::cout << "\nLayer " << layer_idx << " Instructions:\n";
        if (info.instructions.empty()) {
            std::cout << "  No instructions needed.\n";
        } else {
            if (info.instruction_depths.empty()) {
                for (const auto &instr : info.instructions) {
                    print_instruction(instr, -1);
                }
            } else {
                for (const auto &[instr, depth] : info.instruction_depths) {
                    print_instruction(instr, depth);
                }
            }
        }

        std::unordered_map<int, double> expert_loads;
        std::unordered_map<int, int> expert_counts;
        int num_experts = num_ranks_ * num_experts_per_rank_;
        for (size_t i = 0; i < info.input_placement.size(); ++i) {
            int expert_id = info.input_placement[i];
            if (expert_id == -1)
                continue;
            if (expert_id < 0 || expert_id >= num_experts) {
                std::cerr << "Warning: Invalid expert ID " << expert_id
                          << " in layer " << layer_idx << ", position " << i
                          << "\n";
                continue;
            }
            expert_loads[expert_id] +=
                static_cast<double>(info.input_activations[i]);
            expert_counts[expert_id]++;
            if (expert_counts[expert_id] > expert_redundant_limit_ + 1) {
                std::cerr << "Warning: Expert ID " << expert_id
                          << " exceeds redundant limit "
                          << expert_redundant_limit_ << " in layer "
                          << layer_idx << "\n";
                expert_loads[expert_id] = 0.0;
            }
        }
        double output_ratio = simulate_placement_ratio(
            info.output_placement, expert_loads, layer_idx, num_ranks_,
            get_max_slots_per_rank(), num_ranks_ * num_experts_per_rank_);

        std::cout << "Output Placement Ratio for layer " << layer_idx << ": "
                  << std::fixed << std::setprecision(6) << output_ratio << "\n";

        print_placement(info.output_placement, layer_idx, "Output Placement");
    }

    if (enable_debug) {
        std::chrono::high_resolution_clock::time_point end_time =
            std::chrono::high_resolution_clock::now();
        std::time_t end_t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::cout << "\nOptimization ended at: " << std::ctime(&end_t);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();
        std::cout << "Total optimization time: " << duration
                  << " milliseconds\n";

        int total_instructions = 0;
        for (const auto &info : layer_info) {
            total_instructions += info.instructions.size();
        }
        std::cout << "Total instructions generated across all layers: "
                  << total_instructions << "\n";
    }
}
