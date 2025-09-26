// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "dynamic_eplb_greedy.h"

GreedyExpertLoadBalancer::GreedyExpertLoadBalancer(
    int num_layers, int world_size, int num_experts, int num_deployed_experts,
    int expert_redundant_limit, int rank)
    : num_layers_(num_layers), world_size_(world_size),
      num_experts_(num_experts), num_deployed_experts_(num_deployed_experts),
      expert_redundant_limit_(expert_redundant_limit), rank_(rank) {
    if (num_layers <= 0 || world_size <= 0 || num_experts <= 0 ||
        num_deployed_experts < 0 || expert_redundant_limit < 0 || rank < 0) {
        throw std::runtime_error("Invalid initialization parameters");
    }
    num_deployed_experts_per_rank_ = num_deployed_experts_ / world_size_;

    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        for (int rank = 0; rank < world_size_; ++rank) {
            rankinfo_ptrs_.emplace_back(new RankActivateInformation(
                rank, num_deployed_experts_per_rank_));
        }
        for (int expert_id = 0; expert_id < num_experts_; ++expert_id) {
            logit_expert_info_ptrs_.emplace_back(new LogitExpertInformation);
        }
    }
}

void GreedyExpertLoadBalancer::reset_infomation() {
    for (size_t idx = 0; idx < logit_expert_info_ptrs_.size(); ++idx)
        delete logit_expert_info_ptrs_[idx];
    for (size_t idx = 0; idx < rankinfo_ptrs_.size(); ++idx)
        delete rankinfo_ptrs_[idx];

    logit_expert_info_ptrs_.clear();
    rankinfo_ptrs_.clear();

    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        for (int rank = 0; rank < world_size_; ++rank) {
            rankinfo_ptrs_.emplace_back(new RankActivateInformation(
                rank, num_deployed_experts_per_rank_));
        }
        for (int expert_id = 0; expert_id < num_experts_; ++expert_id) {
            logit_expert_info_ptrs_.emplace_back(new LogitExpertInformation);
        }
    }
}

void GreedyExpertLoadBalancer::init_infomation(
    int layer_id, const std::vector<int> &placement,
    const std::vector<int64_t> &activations) {

    std::vector<int> allRanksValue;
    for (int rank = 0; rank < world_size_; ++rank) {
        int value = 0;
        for (int position_id = 0; position_id < num_deployed_experts_per_rank_;
             ++position_id) {
            int global_position_id =
                rank * num_deployed_experts_per_rank_ + position_id;
            int position_offset =
                layer_id * num_deployed_experts_ + global_position_id;
            int expert_id = placement[position_offset];
            int activated_value = activations[position_offset];
            if (expert_id != -1)
                value = value + activated_value;
        }
        allRanksValue.emplace_back(value);
    }
    float ratio = getUnbalancedRatio(allRanksValue);
    if (isInvalidOffloadRatio(ratio)) {
        if (rank_ == 0)
            std::cout
                << "[Warning]: The unbalanced ratio of this time activation "
                   "value is [" +
                       std::to_string(ratio) +
                       "], which abnormal and passed \n";
        return;
    }

    // 输入的 激活值是0 的清空， 不进行已有的激活值累加更新
    if (ratio == 1)
        return;
    reset_infomation();
    // float keep_ratio = 1-1/ratio; // 新来的数据越不均衡，
    // 原数据保留的越多；预防波动
    float keep_ratio = alpha_;

    for (int rank = 0; rank < world_size_; ++rank) {
        RankActivateInformation *rank_info_ptr = getRankInfos(layer_id, rank);
        for (int position_id = 0; position_id < num_deployed_experts_per_rank_;
             ++position_id) {
            int global_position_id =
                rank * num_deployed_experts_per_rank_ + position_id;
            int position_offset =
                layer_id * num_deployed_experts_ + global_position_id;
            int expert_id = placement[position_offset];
            int activated_value = activations[position_offset];
            if (expert_id != -1) {
                LogitExpertInformation *logit_expert_infos =
                    getLogitExpertInfo(layer_id, expert_id);
                logit_expert_infos->update(global_position_id, activated_value,
                                           keep_ratio);
                rank_info_ptr->update(
                    position_id, expert_id,
                    logit_expert_infos->getActivateValue(global_position_id));
            }
        }
    }
}

GreedyExpertLoadBalancer::~GreedyExpertLoadBalancer() {
    for (size_t idx = 0; idx < logit_expert_info_ptrs_.size(); ++idx)
        delete logit_expert_info_ptrs_[idx];
    for (size_t idx = 0; idx < rankinfo_ptrs_.size(); ++idx)
        delete rankinfo_ptrs_[idx];
}

std::vector<int> GreedyExpertLoadBalancer::getAllRanksValue(int layer_id) {
    std::vector<int> result;
    for (int rank = 0; rank < world_size_; ++rank) {
        int rank_offset = layer_id * world_size_ + rank;
        RankActivateInformation *rank_info_ptr = rankinfo_ptrs_[rank_offset];
        result.emplace_back(rank_info_ptr->get_value());
    }
    return result;
}

RankActivateInformation *GreedyExpertLoadBalancer::getTheHighestOffloadRank(
    int layer_id, const std::vector<int> &exclude_ranks) {
    RankActivateInformation *result = nullptr;
    int max_value = -1;
    for (int rank = 0; rank < world_size_; ++rank) {
        if (std::find(exclude_ranks.begin(), exclude_ranks.end(), rank) !=
            exclude_ranks.end())
            continue;
        int rank_offset = layer_id * world_size_ + rank;
        RankActivateInformation *rank_info_ptr = rankinfo_ptrs_[rank_offset];
        int value = rank_info_ptr->get_value();
        if (value > max_value) {
            max_value = value;
            result = rank_info_ptr;
        }
    }
    return result;
}

RankActivateInformation *GreedyExpertLoadBalancer::getTheLowestOffloadRank(
    int layer_id, const std::vector<int> &exclude_ranks) {
    RankActivateInformation *result = nullptr;
    int min = std::numeric_limits<int>::max();
    for (int rank = 0; rank < world_size_; ++rank) {
        if (std::find(exclude_ranks.begin(), exclude_ranks.end(), rank) !=
            exclude_ranks.end())
            continue;
        int rank_offset = layer_id * world_size_ + rank;
        RankActivateInformation *rank_info_ptr = rankinfo_ptrs_[rank_offset];
        int value = rank_info_ptr->get_value();
        if (value < min) {
            min = value;
            result = rank_info_ptr;
        }
    }
    return result;
}

std::vector<ChangeInstruction>
GreedyExpertLoadBalancer::generate_remove_instructions_per_layer(
    int layer_id, int bestEP_value) {
    // 每次取负载最高的Rank， 且该Rank还没完成
    std::vector<int> exclude_ranks;
    std::vector<ChangeInstruction> instructions;
    while (true) {
        RankActivateInformation *thehighestrank = getTheHighestOffloadRank(
            layer_id,
            exclude_ranks); // exclude_ranks
                            // 会把大于bestEP_value的rank都处理一次
        if (thehighestrank->get_value() <= bestEP_value)
            break;
        for (int position_id = 0;
             position_id < thehighestrank->getNumPosition(); ++position_id) {
            if (thehighestrank->get_value() < bestEP_value)
                break;
            if (thehighestrank->is_empty(position_id))
                continue; // exclude the position with -1  expert_id
            int expert_id = thehighestrank->getExpertID(position_id);
            LogitExpertInformation *logit_expert_infos =
                getLogitExpertInfo(layer_id, expert_id);
            if (!logit_expert_infos->is_redundant())
                continue;

            std::vector<int> allRanksValue = getAllRanksValue(layer_id);
            float unbalancedRatio = getUnbalancedRatio(allRanksValue);

            int current_rank = thehighestrank->get_rank_id();
            int global_position_id =
                current_rank * num_deployed_experts_per_rank_ + position_id;
            int decrease_value =
                logit_expert_infos->getActivateValue(global_position_id);
            int add_value =
                decrease_value / (logit_expert_infos->getNumRedundants() - 1);
            std::vector<int> redundant_ranks_this_expert =
                logit_expert_infos->getRedundantRanks(
                    num_deployed_experts_per_rank_);

            for (size_t idx = 0; idx < redundant_ranks_this_expert.size();
                 ++idx) {
                int rank = redundant_ranks_this_expert[idx];
                if (rank != current_rank)
                    allRanksValue[rank] = allRanksValue[rank] + add_value;
                else
                    allRanksValue[rank] = allRanksValue[rank] - decrease_value;
            }

            if (getUnbalancedRatio(allRanksValue) < unbalancedRatio) {
                ChangeInstruction instruction;
                instruction.type = OperationType::REMOVE;
                instruction.layer_idx = layer_id;
                instruction.source_rank = -1;
                instruction.source_expert_id = -1;
                instruction.source_global_position = -1;
                instruction.target_rank = current_rank;
                instruction.target_expert_id = expert_id;
                instruction.target_global_position = global_position_id;
                update(instruction);
                instructions.emplace_back(instruction);
            }
        }

        if (thehighestrank->get_value() > bestEP_value)
            exclude_ranks.emplace_back(
                thehighestrank
                    ->get_rank_id()); // 移除后大于均值，则不再考虑移除其上面的冗余专家；
                                      // 小于均值，有可能其他卡冗余移除，从而提高该卡负载
    }

    return instructions;
}
ChangeInstruction
GreedyExpertLoadBalancer::optimizeTheHighestOffload(int layer_id) {
    RankActivateInformation *thehighestrank =
        getTheHighestOffloadRank(layer_id, {});
    std::vector<int> allRanksValue = getAllRanksValue(layer_id);
    float minUnbalancedRatio = getUnbalancedRatio(allRanksValue);
    float unbalancedRatio;
    ChangeInstruction instruction;
    instruction.type = OperationType::EMPTY;
    for (int s_position_idx = 0;
         s_position_idx < num_deployed_experts_per_rank_; ++s_position_idx) {
        int s_global_position =
            thehighestrank->get_rank_id() * num_deployed_experts_per_rank_ +
            s_position_idx;
        int s_expert_id = thehighestrank->getExpertID(s_position_idx);
        if (s_expert_id == -1)
            continue;
        LogitExpertInformation *s_expert_info =
            getLogitExpertInfo(layer_id, s_expert_id);
        for (int t_rank = 0; t_rank < world_size_; ++t_rank) {
            if (t_rank == thehighestrank->get_rank_id())
                continue;
            RankActivateInformation *t_rank_info =
                getRankInfos(layer_id, t_rank);
            for (int t_position_idx = 0;
                 t_position_idx < num_deployed_experts_per_rank_;
                 ++t_position_idx) {
                int t_global_position = t_rank_info->get_rank_id() *
                                            num_deployed_experts_per_rank_ +
                                        t_position_idx;
                int t_expert_id = t_rank_info->getExpertID(t_position_idx);
                if (s_expert_id == t_expert_id)
                    continue;
                std::vector<int> temptAllRanksValue = allRanksValue;

                // Swap
                int s_value = thehighestrank->get_value(s_position_idx);
                int t_value = t_rank_info->get_value(t_position_idx);
                if (t_value > s_value)
                    continue;
                temptAllRanksValue[thehighestrank->get_rank_id()] =
                    temptAllRanksValue[thehighestrank->get_rank_id()] -
                    s_value + t_value;
                temptAllRanksValue[t_rank] =
                    temptAllRanksValue[t_rank] - t_value + s_value;
                unbalancedRatio = getUnbalancedRatio(temptAllRanksValue);
                if (unbalancedRatio < minUnbalancedRatio) {
                    minUnbalancedRatio = unbalancedRatio;
                    instruction.type = OperationType::SWAP;
                    instruction.layer_idx = layer_id;
                    instruction.source_rank = thehighestrank->get_rank_id();
                    instruction.source_expert_id = s_expert_id;
                    instruction.source_global_position = s_global_position;
                    instruction.target_rank = t_rank;
                    instruction.target_expert_id = t_expert_id;
                    instruction.target_global_position = t_global_position;
                }

                // Add
                // target rank 支持被覆盖
                if (t_expert_id != -1 &&
                    !getLogitExpertInfo(layer_id, t_expert_id)->is_redundant())
                    continue;
                // 当前expert没有超出冗余数量限制
                if (s_expert_info->getNumRedundants() > expert_redundant_limit_)
                    continue;
                temptAllRanksValue = allRanksValue;
                int t_increase_value = s_expert_info->getActivateValue() /
                                       (s_expert_info->getNumRedundants() + 1);
                int s_decrease_value =
                    t_increase_value / s_expert_info->getNumRedundants();
                std::vector<int> s_redundant_ranks =
                    s_expert_info->getRedundantRanks(
                        num_deployed_experts_per_rank_);
                for (size_t idx = 0; idx < s_redundant_ranks.size(); ++idx) {
                    int rank = s_redundant_ranks[idx];
                    temptAllRanksValue[rank] =
                        temptAllRanksValue[rank] - s_decrease_value;
                }
                temptAllRanksValue[t_rank] =
                    temptAllRanksValue[t_rank] + t_increase_value;
                if (t_expert_id != -1) {
                    LogitExpertInformation *t_expert_info =
                        getLogitExpertInfo(layer_id, t_expert_id);
                    std::vector<int> t_redundant_ranks =
                        t_expert_info->getRedundantRanks(
                            num_deployed_experts_per_rank_);
                    int t_decrease_value =
                        t_expert_info->getActivateValue(t_global_position);
                    for (size_t idx = 0; idx < t_redundant_ranks.size();
                         ++idx) {
                        int rank = t_redundant_ranks[idx];
                        if (rank == t_rank) {
                            temptAllRanksValue[rank] =
                                temptAllRanksValue[rank] - t_decrease_value;
                        } else {
                            temptAllRanksValue[rank] =
                                temptAllRanksValue[rank] +
                                (t_decrease_value /
                                 (t_redundant_ranks.size() - 1));
                        }
                    }
                }
                // Calculate the unbalanced ratio
                unbalancedRatio = getUnbalancedRatio(temptAllRanksValue);
                if (unbalancedRatio < minUnbalancedRatio) {
                    minUnbalancedRatio = unbalancedRatio;
                    instruction.type = OperationType::ADD;
                    instruction.layer_idx = layer_id;
                    instruction.source_rank = thehighestrank->get_rank_id();
                    instruction.source_expert_id = s_expert_id;
                    instruction.source_global_position = s_global_position;
                    instruction.target_rank = t_rank;
                    instruction.target_expert_id = t_expert_id;
                    instruction.target_global_position = t_global_position;
                }
            }
        }
    }
    if (instruction.type != OperationType::EMPTY) {
        update(instruction);
    }
    return instruction;
}
ChangeInstruction
GreedyExpertLoadBalancer::optimizeTheLowestOffload(int layer_id) {
    RankActivateInformation *thelowestrank =
        getTheLowestOffloadRank(layer_id, {});
    std::vector<int> allRanksValue = getAllRanksValue(layer_id);
    float minUnbalancedRatio = getUnbalancedRatio(allRanksValue);
    float unbalancedRatio;
    ChangeInstruction instruction;
    instruction.type = OperationType::EMPTY;

    for (int s_position_idx = 0;
         s_position_idx < num_deployed_experts_per_rank_; ++s_position_idx) {
        int s_global_position =
            thelowestrank->get_rank_id() * num_deployed_experts_per_rank_ +
            s_position_idx;
        int s_expert_id = thelowestrank->getExpertID(s_position_idx);
        for (int t_rank = 0; t_rank < world_size_; ++t_rank) {
            if (t_rank == thelowestrank->get_rank_id())
                continue;
            RankActivateInformation *t_rank_info =
                getRankInfos(layer_id, t_rank);
            for (int t_position_idx = 0;
                 t_position_idx < num_deployed_experts_per_rank_;
                 ++t_position_idx) {
                int t_global_position = t_rank_info->get_rank_id() *
                                            num_deployed_experts_per_rank_ +
                                        t_position_idx;
                int t_expert_id = t_rank_info->getExpertID(t_position_idx);
                if (s_expert_id == t_expert_id || t_expert_id == -1)
                    continue;
                std::vector<int> temptAllRanksValue = allRanksValue;
                int s_value = thelowestrank->get_value(s_position_idx);
                int t_value = t_rank_info->get_value(t_position_idx);
                if (t_value < s_value)
                    continue; // 换了负载劣化，跳过
                temptAllRanksValue[thelowestrank->get_rank_id()] =
                    temptAllRanksValue[thelowestrank->get_rank_id()] - s_value +
                    t_value;
                temptAllRanksValue[t_rank] =
                    temptAllRanksValue[t_rank] - t_value + s_value;
                unbalancedRatio = getUnbalancedRatio(temptAllRanksValue);
                if (unbalancedRatio < minUnbalancedRatio) {
                    minUnbalancedRatio = unbalancedRatio;
                    instruction.type = OperationType::SWAP;
                    instruction.layer_idx = layer_id;
                    instruction.source_rank = thelowestrank->get_rank_id();
                    instruction.source_expert_id = s_expert_id;
                    instruction.source_global_position = s_global_position;
                    instruction.target_rank = t_rank;
                    instruction.target_expert_id = t_expert_id;
                    instruction.target_global_position = t_global_position;
                }
            }
        }
    }
    if (instruction.type != OperationType::EMPTY) {
        update(instruction);
    }
    return instruction;
}
std::vector<ChangeInstruction>
GreedyExpertLoadBalancer::generate_swap_add_instructions_per_layer(
    int layer_id) {
    std::vector<ChangeInstruction> instructions;
    bool high_flag = true;
    int maxinstructionNums = 32;
    float threadshold = 1.1;
    int break_flag = 0;

    std::vector<int> allRanksValue = getAllRanksValue(layer_id);
    float unbalancedRatio = getUnbalancedRatio(allRanksValue);

    for (int idx = 0; idx < maxinstructionNums; ++idx) {
        if (break_flag == 2)
            break;
        if (unbalancedRatio < threadshold)
            break;

        ChangeInstruction instruction;
        if (high_flag) {
            high_flag = false;
            instruction = optimizeTheHighestOffload(layer_id);
        } else {
            high_flag = true;
            instruction = optimizeTheLowestOffload(layer_id);
        }
        if (instruction.type == OperationType::EMPTY)
            break_flag = break_flag + 1;
        else {
            instructions.emplace_back(instruction);
            allRanksValue = getAllRanksValue(layer_id);
            float ratio = getUnbalancedRatio(allRanksValue);
            if (ratio > unbalancedRatio)
                break_flag = break_flag + 1;
            else
                break_flag = 0;
            unbalancedRatio = ratio;
        }
    }

    return instructions;
}

void GreedyExpertLoadBalancer::update(ChangeInstruction instruction) {
    if (instruction.type == OperationType::REMOVE) {
        if (instruction.target_expert_id == -1) {
            throw std::runtime_error("[Error]-Instruction-rank[" +
                                     std::to_string(rank_) +
                                     "]  type[remove] target_expert_id[-1]");
        }
        LogitExpertInformation *logit_expert_infos = getLogitExpertInfo(
            instruction.layer_idx, instruction.target_expert_id);
        if (!logit_expert_infos->is_redundant()) {
            throw std::runtime_error(
                "[Error]-Instruction-rank[" + std::to_string(rank_) +
                "]  type[remove] target_expert_id[" +
                std::to_string(instruction.target_expert_id) +
                "] is not redundant!");
        }
        int decrease_value = logit_expert_infos->getActivateValue(
            instruction.target_global_position);

        logit_expert_infos->remove(instruction.target_global_position);
        logit_expert_infos->update(decrease_value); // 均分减少的激活量
        RankActivateInformation *rank_info_ptr =
            getRankInfos(instruction.layer_idx, instruction.target_rank);
        int local_position_idx =
            instruction.target_global_position % num_deployed_experts_per_rank_;

        rank_info_ptr->update(local_position_idx, -1, 0);

        std::vector<int> redundant_ranks_this_expert =
            logit_expert_infos->getRedundantRanks(
                num_deployed_experts_per_rank_);
        std::vector<int> global_position_idxs =
            logit_expert_infos->getGlobalPositionIdx();

        for (size_t idx = 0; idx < redundant_ranks_this_expert.size(); ++idx) {
            int rank = redundant_ranks_this_expert[idx];
            int global_position_idx = global_position_idxs[idx];
            local_position_idx =
                global_position_idx % num_deployed_experts_per_rank_;

            rank_info_ptr = getRankInfos(instruction.layer_idx, rank);
            rank_info_ptr->update(
                local_position_idx, instruction.target_expert_id,
                logit_expert_infos->getActivateValue(global_position_idx));
        }

    } else if (instruction.type == OperationType::ADD) {
        LogitExpertInformation *s_logit_expert_infos = getLogitExpertInfo(
            instruction.layer_idx, instruction.source_expert_id);
        s_logit_expert_infos->update(instruction.target_global_position, 0,
                                     0); // 新增一个位置
        s_logit_expert_infos->update();  // 均分激活信息

        std::vector<int> s_ranks = s_logit_expert_infos->getRedundantRanks(
            num_deployed_experts_per_rank_);
        std::vector<int> s_local_positions =
            s_logit_expert_infos->getLocalPositionIdx(
                num_deployed_experts_per_rank_);
        std::vector<int> s_global_position_idxs =
            s_logit_expert_infos->getGlobalPositionIdx();

        for (size_t idx = 0; idx < s_ranks.size(); ++idx) {
            int rank = s_ranks[idx];
            int local_position = s_local_positions[idx];
            int global_position_idx = s_global_position_idxs[idx];
            getRankInfos(instruction.layer_idx, rank)
                ->update(local_position, instruction.source_expert_id,
                         s_logit_expert_infos->getActivateValue(
                             global_position_idx));
            ;
        }

        if (instruction.target_expert_id != -1) {
            LogitExpertInformation *t_logit_expert_infos = getLogitExpertInfo(
                instruction.layer_idx, instruction.target_expert_id);
            int t_expert_value_per_position_before_delete =
                t_logit_expert_infos->getActivateValue(
                    instruction.target_global_position);
            t_logit_expert_infos->remove(
                instruction.target_global_position); // 删除
            t_logit_expert_infos->update(
                t_expert_value_per_position_before_delete); // 更新到剩余位置

            std::vector<int> t_ranks = t_logit_expert_infos->getRedundantRanks(
                num_deployed_experts_per_rank_);
            std::vector<int> t_local_positions =
                t_logit_expert_infos->getLocalPositionIdx(
                    num_deployed_experts_per_rank_);
            std::vector<int> t_global_position_idxs =
                t_logit_expert_infos->getGlobalPositionIdx();
            for (size_t idx = 0; idx < t_ranks.size(); ++idx) {
                int rank = t_ranks[idx];
                int local_position = t_local_positions[idx];
                int global_position_idx = t_global_position_idxs[idx];
                getRankInfos(instruction.layer_idx, rank)
                    ->update(local_position, instruction.target_expert_id,
                             t_logit_expert_infos->getActivateValue(
                                 global_position_idx));
                ;
            }
        }
    } else if (instruction.type == OperationType::SWAP) {

        RankActivateInformation *s_rank_info_ptr =
            getRankInfos(instruction.layer_idx, instruction.source_rank);
        RankActivateInformation *t_rank_info_ptr =
            getRankInfos(instruction.layer_idx, instruction.target_rank);
        int source_local_position =
            instruction.source_global_position % num_deployed_experts_per_rank_;
        int target_local_position =
            instruction.target_global_position % num_deployed_experts_per_rank_;
        int s_value = s_rank_info_ptr->get_value(source_local_position);
        int t_value = t_rank_info_ptr->get_value(target_local_position);
        s_rank_info_ptr->update(source_local_position,
                                instruction.target_expert_id, t_value);
        t_rank_info_ptr->update(target_local_position,
                                instruction.source_expert_id, s_value);

        if (instruction.source_expert_id != -1) {
            LogitExpertInformation *s_logit_expert_infos = getLogitExpertInfo(
                instruction.layer_idx, instruction.source_expert_id);
            s_logit_expert_infos->remove(instruction.source_global_position);
            s_logit_expert_infos->update(instruction.target_global_position,
                                         s_value, 0);
        }

        if (instruction.target_expert_id != -1) {
            LogitExpertInformation *t_logit_expert_infos = getLogitExpertInfo(
                instruction.layer_idx, instruction.target_expert_id);
            t_logit_expert_infos->remove(instruction.target_global_position);
            t_logit_expert_infos->update(instruction.source_global_position,
                                         t_value, 0);
        }
    }
}

int GreedyExpertLoadBalancer::getTheHighestOffload(
    const std::vector<int> &allRanksValues, int bestEPValue,
    const std::vector<int> &include_ranks) {
    int result = 0;
    if (include_ranks.size() == 0) {
        for (size_t rank = 0; rank < allRanksValues.size(); ++rank) {
            int bias = allRanksValues[rank] - bestEPValue;
            if (bias < 0)
                continue;
            result = std::max(result, bias);
        }
    } else {
        for (size_t idx = 0; idx < include_ranks.size(); ++idx) {
            int rank = include_ranks[idx];
            int bias = allRanksValues[rank] - bestEPValue;
            if (bias < 0)
                continue;
            result = std::max(result, bias);
        }
    }
    return result;
}

float GreedyExpertLoadBalancer::getUnbalancedRatio(
    const std::vector<int> &all_ranks_values) {
    assert(all_ranks_values.size() == world_size_ &&
           "size of input parameters is not equals to world_Size");
    int max = 1;
    int min = std::numeric_limits<int>::max();
    for (size_t rank = 0; rank < all_ranks_values.size(); ++rank) {
        max = std::max(max, all_ranks_values[rank]);
        min = std::min(min, all_ranks_values[rank]);
    }
    min = std::max(1, min);
    return (float)max / (float)min;
}

int GreedyExpertLoadBalancer::getBestEPValue(
    const std::vector<int> &all_ranks_values) {
    assert(all_ranks_values.size() == world_size_ &&
           "size of input parameters is not equals to world_Size");
    int result = 0;
    for (size_t rank = 0; rank < all_ranks_values.size(); ++rank) {
        result = result + all_ranks_values[rank];
    }
    return result / world_size_;
}

std::string floatToString(float num) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << num;
    return oss.str();
}

std::vector<ChangeInstruction>
GreedyExpertLoadBalancer::optimize_and_generate_instructions(
    const std::vector<int> &placement,
    const std::vector<int64_t> &activations) {
    std::vector<ChangeInstruction> instructions;
    std::string logging = "";
    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        init_infomation(layer_id, placement, activations);
        std::vector<int> allRanksValue = getAllRanksValue(layer_id);
        int bestEP_value = getBestEPValue(allRanksValue);
        float unbalancedRatio = getUnbalancedRatio(allRanksValue);
        logging = logging + "layer[" + std::to_string(layer_id) +
                  "]: " + floatToString(unbalancedRatio) + "->";
        std::vector<ChangeInstruction> tmp_instructions =
            generate_remove_instructions_per_layer(layer_id, bestEP_value);
        instructions.insert(instructions.end(), tmp_instructions.begin(),
                            tmp_instructions.end());
        allRanksValue = getAllRanksValue(layer_id);
        bestEP_value = getBestEPValue(allRanksValue);
        unbalancedRatio = getUnbalancedRatio(allRanksValue);
        logging = logging + floatToString(unbalancedRatio) + "[" +
                  std::to_string(tmp_instructions.size()) + "]->";
        tmp_instructions = generate_swap_add_instructions_per_layer(layer_id);
        instructions.insert(instructions.end(), tmp_instructions.begin(),
                            tmp_instructions.end());
        allRanksValue = getAllRanksValue(layer_id);
        unbalancedRatio = getUnbalancedRatio(allRanksValue);
        logging = logging + floatToString(unbalancedRatio) + "[" +
                  std::to_string(tmp_instructions.size()) + "] \n";
    }
    if (rank_ == 0)
        std::cout << logging << std::endl;
    return instructions;
}

// Apply instructions to current placement for a layer
std::vector<int> GreedyExpertLoadBalancer::apply_layer_instructions(
    const std::vector<int> &current_placement,
    const std::vector<ChangeInstruction> &instructions) {
    std::vector<int> new_placement = current_placement;
    for (const auto &instr : instructions) {
        if (instr.type == OperationType::SWAP) {
            int source_idx = instr.source_global_position;
            int target_idx = instr.target_global_position;
            std::swap(new_placement[source_idx], new_placement[target_idx]);
        } else if (instr.type == OperationType::ADD) {
            int target_idx = instr.target_global_position;
            new_placement[target_idx] = instr.source_expert_id;
        } else if (instr.type == OperationType::REMOVE) {
            int target_idx = instr.target_global_position;
            new_placement[target_idx] = -1;
        }
    }
    return new_placement;
}

// New method that returns optimized placement instead of instructions
std::vector<int> GreedyExpertLoadBalancer::optimize_placement(
    const std::vector<int> &placement,
    const std::vector<int64_t> &activations) {

    // Call existing optimize_and_generate_instructions to get all instructions
    std::vector<ChangeInstruction> all_instructions =
        optimize_and_generate_instructions(placement, activations);

    // Group instructions by layer
    std::vector<std::vector<ChangeInstruction>> instructions_per_layer(
        num_layers_);
    for (const auto &instr : all_instructions) {
        if (instr.layer_idx >= 0 && instr.layer_idx < num_layers_) {
            instructions_per_layer[instr.layer_idx].push_back(instr);
        }
    }

    // Apply instructions to get optimized placement
    std::vector<int> optimized_placement = placement;
    for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        int layer_offset = layer_id * num_deployed_experts_;
        std::vector<int> layer_placement(
            optimized_placement.begin() + layer_offset,
            optimized_placement.begin() + layer_offset + num_deployed_experts_);

        // Apply this layer's instructions
        std::vector<int> new_layer_placement = apply_layer_instructions(
            layer_placement, instructions_per_layer[layer_id]);

        // Copy back to optimized_placement
        for (int i = 0; i < num_deployed_experts_; ++i) {
            optimized_placement[layer_offset + i] = new_layer_placement[i];
        }
    }

    return optimized_placement;
}