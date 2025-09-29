// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_load_balancer.h"
#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>

class ExpertLoadBalancerTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_layers_ = 2;
        num_ranks_ = 4;
        num_experts_per_rank_ = 2;
        num_redundant_per_rank_ = 0;
        expert_redundant_limit_ = 0;
        max_slots_per_rank_ = num_experts_per_rank_ + num_redundant_per_rank_;
        num_experts_ = num_ranks_ * num_experts_per_rank_;
    }

    std::string OperationTypeToString(OperationType type) {
        switch (type) {
            case OperationType::SWAP: return "swap";
            case OperationType::ADD: return "add";
            case OperationType::REMOVE: return "remove";
            default: return "unknown";
        }
    }

    void RunOptimizeTest(const std::string& test_name,
                         const std::vector<int>& input_placement,
                         const std::vector<int64_t>& input_activations,
                         int expected_instruction_count,
                         const std::set<OperationType>& expected_instruction_types) {
        ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                   num_redundant_per_rank_, expert_redundant_limit_, 0);
        std::cout << "Test: " << test_name << "\n";

        EXPECT_TRUE(balancer.get_validate_input_size(input_placement, input_activations,
                                                    num_layers_, num_ranks_, max_slots_per_rank_))
            << "Invalid input size in " << test_name;

        auto instructions = balancer.optimize_and_generate_instructions(input_placement, input_activations);
        EXPECT_EQ(instructions.size(), expected_instruction_count)
            << "Unexpected instruction count in " << test_name;

        std::set<OperationType> actual_types;
        for (const auto& instr : instructions) {
            actual_types.insert(instr.type);
        }
        EXPECT_EQ(actual_types, expected_instruction_types)
            << "Instruction types mismatch in " << test_name;

        std::vector<std::vector<int>> layer_placements(num_layers_);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
            layer_placements[layer_idx] = std::vector<int>(
                input_placement.begin() + layer_offset,
                input_placement.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);
        }

        for (const auto& instr : instructions) {
            int layer_idx = instr.layer_idx;
            auto& current_placement = layer_placements[layer_idx];
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
            size_t source_pos = static_cast<size_t>(instr.source_global_position - layer_offset);
            size_t target_pos = static_cast<size_t>(instr.target_global_position - layer_offset);

            EXPECT_TRUE(source_pos < current_placement.size())
                << "Invalid source_global_position: " << source_pos << " for layer " << instr.layer_idx;
            EXPECT_TRUE(target_pos < current_placement.size())
                << "Invalid target_global_position: " << target_pos << " for layer " << instr.layer_idx;

            if (instr.type == OperationType::ADD) {
                EXPECT_EQ(current_placement[source_pos], instr.source_expert_id)
                    << "Source position does not contain source_expert_id in layer " << instr.layer_idx;
            }

            if (instr.type == OperationType::SWAP) {
                std::swap(current_placement[source_pos], current_placement[target_pos]);
            } else if (instr.type == OperationType::ADD) {
                current_placement[target_pos] = instr.target_expert_id;
            } else if (instr.type == OperationType::REMOVE) {
                current_placement[target_pos] = -1;
            }

            std::cout << "Applied " << OperationTypeToString(instr.type) << " instruction: layer=" << instr.layer_idx
                      << ", source_pos=" << source_pos << ", target_pos=" << target_pos
                      << ", placement: ";
            for (int x : current_placement) std::cout << x << " ";
            std::cout << std::endl;
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            EXPECT_TRUE(balancer.get_validate_unique_expert_ids(layer_placements[layer_idx], layer_idx, num_ranks_, max_slots_per_rank_))
                << "Duplicate expert IDs in final placement for layer " << layer_idx << " in " << test_name;
            EXPECT_TRUE(balancer.get_validate_all_experts_present(layer_placements[layer_idx], layer_idx, num_ranks_, max_slots_per_rank_, num_experts_))
                << "Not all experts present in final placement for layer " << layer_idx << " in " << test_name;
        }
    }

    std::vector<int> CreateUniformPlacement() {
        std::vector<int> placement(num_layers_ * num_ranks_ * max_slots_per_rank_, -1);
        for (int layer = 0; layer < num_layers_; ++layer) {
            int layer_offset = layer * num_ranks_ * max_slots_per_rank_;
            for (int r = 0; r < num_ranks_; ++r) {
                int rank_offset = layer_offset + r * max_slots_per_rank_;
                for (int i = 0; i < num_experts_per_rank_; ++i) {
                    placement[rank_offset + i] = r * num_experts_per_rank_ + i;
                }
            }
        }
        return placement;
    }

    std::vector<int64_t> CreateUniformActivations(int64_t value) {
        return std::vector<int64_t>(num_layers_ * num_ranks_ * max_slots_per_rank_, value);
    }

    int num_layers_;
    int num_ranks_;
    int num_experts_per_rank_;
    int num_redundant_per_rank_;
    int expert_redundant_limit_;
    int max_slots_per_rank_;
    int num_experts_;
};

// Constructor Tests
TEST_F(ExpertLoadBalancerTest, Constructor_ValidParameters) {
    ExpertLoadBalancer balancer(2, 4, 2, 1, 1, 0);
    EXPECT_EQ(balancer.get_num_layers(), 2);
    EXPECT_EQ(balancer.get_num_ranks(), 4);
    EXPECT_EQ(balancer.get_num_experts_per_rank(), 2);
    EXPECT_EQ(balancer.get_num_redundant_per_rank(), 1);
    EXPECT_EQ(balancer.get_expert_redundant_limit(), 1);
    EXPECT_EQ(balancer.get_max_slots_per_rank(), 3);
    EXPECT_EQ(balancer.get_num_experts(), 8);
}

TEST_F(ExpertLoadBalancerTest, Constructor_InvalidParameters) {
    EXPECT_THROW(ExpertLoadBalancer(0, 4, 2, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 0, 2, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 0, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, -1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, 1, -1, 0), std::runtime_error);
}



// get_validate_input_size Test
TEST_F(ExpertLoadBalancerTest, ValidateInputSize) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    std::vector<int> placement(num_layers_ * num_ranks_ * max_slots_per_rank_, 0);
    std::vector<int64_t> activations(num_layers_ * num_ranks_ * max_slots_per_rank_, 100);
    EXPECT_TRUE(balancer.get_validate_input_size(placement, activations, num_layers_, num_ranks_, max_slots_per_rank_));

    placement.resize(num_layers_ * num_ranks_ * max_slots_per_rank_ - 1);
    EXPECT_FALSE(balancer.get_validate_input_size(placement, activations, num_layers_, num_ranks_, max_slots_per_rank_));
}

// get_validate_unique_expert_ids Test
TEST_F(ExpertLoadBalancerTest, ValidateUniqueExpertIds) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1,
                                  0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1};
    EXPECT_TRUE(balancer.get_validate_unique_expert_ids(placement, 0, num_ranks_, max_slots_per_rank_));

    placement[0] = 1;
    EXPECT_FALSE(balancer.get_validate_unique_expert_ids(placement, 0, num_ranks_, max_slots_per_rank_));
}


// get_compute_expert_loads Test
TEST_F(ExpertLoadBalancerTest, ComputeExpertLoads) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    std::vector<ExpertInformation> experts = {
        {0, 0, 0, 100, 0, 2},
        {0, 1, 0, 50, 3, 2},
        {0, 2, 1, 200, 6, 1}
    };
    auto loads = balancer.get_compute_expert_loads(experts, num_experts_);
    EXPECT_DOUBLE_EQ(loads[0], 150.0);
    EXPECT_DOUBLE_EQ(loads[1], 200.0);
    EXPECT_DOUBLE_EQ(loads[2], 0.0);
}

// get_compute_expert_total_activations Test
TEST_F(ExpertLoadBalancerTest, ComputeExpertTotalActivations) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    std::vector<std::vector<ExpertInformation>> layer_experts = {
        {
            {0, 0, 0, 100, 0, 2},
            {0, 1, 0, 50, 3, 2},
            {0, 2, 1, 200, 6, 1}
        },
        {
            {1, 0, 0, 150, 12, 1},
            {1, 1, 1, 250, 15, 1}
        }
    };
    auto total_activations = balancer.get_compute_expert_total_activations(layer_experts, num_layers_, num_ranks_, num_experts_per_rank_);
    EXPECT_EQ(total_activations[0], 150);
    EXPECT_EQ(total_activations[1], 200);
    EXPECT_EQ(total_activations[8], 150);
    EXPECT_EQ(total_activations[9], 250);
}



// get_extract_expert_info Test
TEST_F(ExpertLoadBalancerTest, ExtractExpertInfo) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);
    auto layer_experts = balancer.get_extract_expert_info(placement, activations, num_layers_, num_ranks_,
                                                         num_experts_per_rank_, num_redundant_per_rank_, expert_redundant_limit_);
    EXPECT_EQ(layer_experts.size(), num_layers_);
    EXPECT_EQ(layer_experts[0].size(), num_ranks_ * num_experts_per_rank_);
}

// Optimization Tests
TEST_F(ExpertLoadBalancerTest, Optimize_BalancedNoChanges) {
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);
    RunOptimizeTest("BalancedNoChanges", placement, activations, 0, {});
}

TEST_F(ExpertLoadBalancerTest, Optimize_ImbalancedTriggersAdd) {
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);
    activations[0] = 1000;
    activations[1] = 1000;
    RunOptimizeTest("ImbalancedTriggersAdd", placement, activations, 1, {OperationType::SWAP});
}




// get_compute_rank_sets Test
TEST_F(ExpertLoadBalancerTest, ComputeRankSets) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement，所有位置填充专家，无冗余（-1）
    std::vector<int> placement = {
        0, 1, 2, 3, 4, 5, 6, 7, // layer 0: 每个rank分配2个专家
        0, 1, 2, 3, 4, 5, 6, 7  // layer 1: 相同分布
    };
    auto rank_sets = balancer.get_compute_rank_sets(placement, num_ranks_, max_slots_per_rank_);
    EXPECT_EQ(rank_sets.size(), num_ranks_);
    EXPECT_EQ(rank_sets[0], std::set<int>({0, 1}));
    EXPECT_EQ(rank_sets[1], std::set<int>({2, 3}));
    EXPECT_EQ(rank_sets[2], std::set<int>({4, 5}));
    EXPECT_EQ(rank_sets[3], std::set<int>({6, 7}));
}

// get_find_position_with_expert Test
TEST_F(ExpertLoadBalancerTest, FindPositionWithExpert) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement，所有位置填充专家
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(balancer.get_find_position_with_expert(placement, 0, 0, max_slots_per_rank_), 0);
    EXPECT_EQ(balancer.get_find_position_with_expert(placement, 1, 3, max_slots_per_rank_), 3);
    EXPECT_EQ(balancer.get_find_position_with_expert(placement, 2, 99, max_slots_per_rank_), -1);
}

// get_find_empty_position Test
TEST_F(ExpertLoadBalancerTest, FindEmptyPosition) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 无冗余位置，所有位置已分配专家，找不到空位置
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(balancer.get_find_empty_position(placement, 0, max_slots_per_rank_), -1);
    EXPECT_EQ(balancer.get_find_empty_position(placement, 2, max_slots_per_rank_), -1);
    EXPECT_EQ(balancer.get_find_empty_position(placement, 3, max_slots_per_rank_), -1);
}

// get_compute_expert_counts Test
TEST_F(ExpertLoadBalancerTest, ComputeExpertCounts) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement，专家均匀分布，无冗余
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    auto counts = balancer.get_compute_expert_counts(placement, num_ranks_, max_slots_per_rank_);
    for (int i = 0; i < num_experts_; ++i) {
        EXPECT_EQ(counts[i], 1) << "Expert " << i << " should appear exactly once";
    }
}

// get_validate_all_experts_present Test
TEST_F(ExpertLoadBalancerTest, ValidateAllExpertsPresent) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement，包含所有专家
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_TRUE(balancer.get_validate_all_experts_present(placement, 0, num_ranks_, max_slots_per_rank_, num_experts_));

    // 测试缺少专家的情况
    placement[0] = 1; // 引入重复，缺少专家0
    EXPECT_FALSE(balancer.get_validate_all_experts_present(placement, 0, num_ranks_, max_slots_per_rank_, num_experts_));
}

// get_compute_placement_ratio Test
TEST_F(ExpertLoadBalancerTest, ComputePlacementRatio) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement和activations，所有专家均匀负载
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> activations = {100, 100, 100, 100, 100, 100, 100, 100};
    double ratio = balancer.get_compute_placement_ratio(placement, activations, 0, num_ranks_, max_slots_per_rank_, num_experts_);
    EXPECT_NEAR(ratio, 1.0, 0.1); // 均匀分布，ratio应接近1
}

// get_extract_layer_expert_info Test
TEST_F(ExpertLoadBalancerTest, ExtractLayerExpertInfo) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);
    // 构造placement和activations，无冗余
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> activations = {100, 200, 300, 400, 500, 600, 700, 800};
    auto experts = balancer.get_extract_layer_expert_info(placement, activations, 0, num_ranks_, max_slots_per_rank_,
                                                         num_experts_, expert_redundant_limit_);
    EXPECT_EQ(experts.size(), num_experts_);
    for (size_t i = 0; i < experts.size(); ++i) {
        EXPECT_EQ(experts[i].expert_id, static_cast<int>(i));
        EXPECT_EQ(experts[i].activations, activations[i]);
        EXPECT_EQ(experts[i].total_count, 1);
    }
}

// Optimize_RedundantExpertsTriggersRemove Test
TEST_F(ExpertLoadBalancerTest, Optimize_RedundantExpertsTriggersRemove) {
    std::vector<int> placement = CreateUniformPlacement();
    // 无冗余专家，无法触发REMOVE操作，期望无指令
    std::vector<int64_t> activations = CreateUniformActivations(100);
    RunOptimizeTest("RedundantExpertsTriggersRemove", placement, activations, 0, {});
}

// Optimize_MismatchedPlacementTriggersSwap Test
TEST_F(ExpertLoadBalancerTest, Optimize_MismatchedPlacementTriggersSwap) {
    std::vector<int> placement = CreateUniformPlacement();
    // 交换两个专家位置，触发SWAP
    std::swap(placement[0], placement[2]);
    std::vector<int64_t> activations = CreateUniformActivations(100);
    RunOptimizeTest("MismatchedPlacementTriggersSwap", placement, activations,0, {});
}

// Optimize_ComplexCaseAllInstructions Test
TEST_F(ExpertLoadBalancerTest, Optimize_ComplexCaseAllInstructions) {
    std::vector<int> placement = CreateUniformPlacement();
    // 修改激活值，触发SWAP（无冗余，无法触发ADD/REMOVE）
    std::vector<int64_t> activations = CreateUniformActivations(100);
    activations[0] = 1000; // 使专家0负载不均
    RunOptimizeTest("ComplexCaseAllInstructions", placement, activations,0, {});
}

TEST_F(ExpertLoadBalancerTest, CompareLoadRatioBeforeAndAfterCustom) {
    // 设置参数
    num_layers_ = 1;
    num_ranks_ = 32;
    num_experts_per_rank_ = 8;
    num_redundant_per_rank_ = 0;
    expert_redundant_limit_ = 0;
    max_slots_per_rank_ = num_experts_per_rank_;
    num_experts_ = num_ranks_ * num_experts_per_rank_;

    // 设置输入 placement 和 activations
    std::vector<int> input_placement = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
        64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
        96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
        128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
        160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
        192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
        224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255};
    
    std::vector<int64_t> input_activations = {
        6484,5972,6027,1382,2391,5241,104817,1483,6778,104231,1471,5643,6397,5655,2133,1769,103606,2065,876,1168,6854,4253,3139,11858,7736,28672,3863,3995,2468,6631,19528,14646,4870,5198,8444,24971,12674,22674,3723,4677,4125,19438,16776,1491,14120,5592,13923,12103,13237,12361,6333,15183,6331,6257,16019,10908,10839,5927,968,6860,6452,14958,23482,15707,7195,5330,24040,10974,15186,14306,4439,4577,10546,2142,8720,29276,4840,7025,4408,19899,13161,5320,12930,12831,13140,5926,7070,15636,5050,7308,15968,6891,12029,3387,6941,30217,13538,5618,5972,8487,11636,16164,16164,7952,15019,13532,5477,8005,5157,16754,16200,6979,18600,2691,15518,20291,3149,8387,5904,12676,14668,3357,7651,13016,7027,11787,15453,13496,3146,3815,3380,109200,4961,4660,4640,3683,18875,5542,18086,5601,5651,6125,7160,14148,4750,4192,4481,3869,3600,4721,4629,104850,14617,10485,16393,5269,5659,14198,4805,13378,4536,107883,4207,3324,5648,4877,4407,4254,11671,7042,12639,10989,7783,14010,8405,8088,3103,5110,5338,107437,3467,3998,3306,2948,10484,50291,5846,2386,4118,4133,4345,5461,5435,11402,7744,12924,15123,14912,7313,5211,3861,4074,3936,3554,4443,5014,4848,110810,6962,5709,6198,5707,6206,28775,21816,5407,13810,13076,5793,16943,5757,5617,12553,13638,3311,26788,6700,5628,3077,5512,14514,21120,4185,12398,15018,3581,19179,14472,14173,3241,17005,28326,7744,12096,6763,6077,3576,5131,6453,8273,33922,5599,4854,13285,8685,4554
    };

    // 验证输入 placement 的唯一性
    std::set<int> input_experts(input_placement.begin(), input_placement.end());
    ASSERT_EQ(input_experts.size(), num_experts_) << "input_placement does not contain exactly 256 unique experts";
    for (int i = 0; i < num_experts_; ++i) {
        ASSERT_TRUE(input_experts.count(i)) << "Expert ID " << i << " missing in input_placement";
    }

    // 创建 ExpertLoadBalancer 对象
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);

    // 验证输入大小
    EXPECT_TRUE(balancer.get_validate_input_size(input_placement, input_activations,
                                                num_layers_, num_ranks_, max_slots_per_rank_))
        << "Invalid input size for input_placement";

    // 验证初始 placement 的唯一性和完整性
    std::vector<int> initial_layer_placement(input_placement.begin(),
                                            input_placement.begin() + num_ranks_ * max_slots_per_rank_);
    EXPECT_TRUE(balancer.get_validate_unique_expert_ids(initial_layer_placement, 0, num_ranks_, max_slots_per_rank_))
        << "Duplicate expert IDs in initial placement";
    EXPECT_TRUE(balancer.get_validate_all_experts_present(initial_layer_placement, 0, num_ranks_, max_slots_per_rank_, num_experts_))
        << "Not all experts present in initial placement";

    // 计算每层
    std::vector<double> initial_ratios(num_layers_);
    std::vector<double> final_ratios(num_layers_);
    std::vector<std::vector<int>> final_layer_placements(num_layers_);

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        // 提取当前层的数据
        int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
        std::vector<int> layer_placement(
            input_placement.begin() + layer_offset,
            input_placement.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);
        std::vector<int64_t> layer_activations(
            input_activations.begin() + layer_offset,
            input_activations.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);

        // 计算初始 placement 的比率
        initial_ratios[layer_idx] = balancer.get_compute_placement_ratio(
            layer_placement, layer_activations, layer_idx, num_ranks_, max_slots_per_rank_, num_experts_);

        // 生成优化 placement 和指令
        auto [instructions, optimized_placement, instruction_depths] = balancer.get_generate_bucket_balanced_placement(
            layer_placement, layer_activations, layer_idx, num_ranks_, num_experts_per_rank_);
        ASSERT_FALSE(optimized_placement.empty())
            << "Failed to generate optimized placement for layer " << layer_idx;
        final_layer_placements[layer_idx] = optimized_placement;

        // 打印指令数量
        std::cout << "Layer " << layer_idx << ": Instruction Count: " << instructions.size() << "\n";

        // 验证指令的 rank 和 pos
        for (const auto& instr : instructions) {
            EXPECT_GE(instr.source_rank, 0) << "Source rank should be non-negative in layer " << layer_idx;
            EXPECT_LT(instr.source_rank, num_ranks_) << "Source rank exceeds num_ranks in layer " << layer_idx;
            EXPECT_GE(instr.target_rank, 0) << "Target rank should be non-negative in layer " << layer_idx;
            EXPECT_LT(instr.target_rank, num_ranks_) << "Target rank exceeds num_ranks in layer " << layer_idx;
            EXPECT_GE(instr.source_global_position, 0) << "Source position should be non-negative in layer " << layer_idx;
            EXPECT_LT(instr.source_global_position, num_ranks_*max_slots_per_rank_) << "Source position exceeds max_slots_per_rank in layer " << layer_idx;
            EXPECT_GE(instr.target_global_position, 0) << "Target position should be non-negative in layer " << layer_idx;
            EXPECT_LT(instr.target_global_position, num_ranks_*max_slots_per_rank_) << "Target position exceeds max_slots_per_rank in layer " << layer_idx;
        }

        // 计算优化后的激活值
        std::unordered_map<int, int64_t> expert_activations;
        for (size_t idx = 0; idx < layer_placement.size(); ++idx) {
            int expert_id = layer_placement[idx];
            if (expert_id != -1) {
                expert_activations[expert_id] = layer_activations[idx];
            }
        }

        std::vector<int64_t> adjusted_activations(num_ranks_ * max_slots_per_rank_, 0);
        for (size_t i = 0; i < final_layer_placements[layer_idx].size(); ++i) {
            int expert_id = final_layer_placements[layer_idx][i];
            if (expert_id != -1 && expert_activations.count(expert_id)) {
                adjusted_activations[i] = expert_activations[expert_id];
            }
        }

        // 计算优化后的比率
        final_ratios[layer_idx] = balancer.get_compute_placement_ratio(
            final_layer_placements[layer_idx], adjusted_activations, layer_idx, num_ranks_, max_slots_per_rank_, num_experts_);
    }

    // 打印比率对比
    std::cout << "\nLoad Ratio Comparison:\n";
    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        std::cout << "Layer " << layer_idx << ":\n";
        std::cout << "  Initial Load Ratio:   " << std::fixed << std::setprecision(4) << initial_ratios[layer_idx] << "\n";
        std::cout << "  Optimized Load Ratio: " << std::fixed << std::setprecision(4) << final_ratios[layer_idx] << "\n";
    }

    // 验证优化后的 placement
    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        EXPECT_TRUE(balancer.get_validate_unique_expert_ids(final_layer_placements[layer_idx], 0, num_ranks_, max_slots_per_rank_))
            << "Duplicate expert IDs in optimized placement for layer " << layer_idx;
        EXPECT_TRUE(balancer.get_validate_all_experts_present(final_layer_placements[layer_idx], 0, num_ranks_, max_slots_per_rank_, num_experts_))
            << "Not all experts present in optimized placement for layer " << layer_idx;
    }

    // 验证优化比率是否改善
    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        EXPECT_LE(final_ratios[layer_idx], initial_ratios[layer_idx] * 1.05)
            << "Optimized load ratio did not improve sufficiently for layer " << layer_idx;
    }


    // // 调用 optimize_and_generate_instructions 并打印指令数量
    // auto all_instructions = balancer.optimize_and_generate_instructions(input_placement, input_activations);
    // std::cout << "Total Instruction Count from optimize_and_generate_instructions: " << all_instructions.size() << "\n";


}


class ExpertLoadBalancerPlacementActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_layers_ = 1;
        num_ranks_ = 8;
        num_experts_per_rank_ = 4;
        num_redundant_per_rank_ = 0;
        expert_redundant_limit_ = 0;
        max_slots_per_rank_ = num_experts_per_rank_ + num_redundant_per_rank_;
        num_experts_ = num_ranks_ * num_experts_per_rank_;
    }

        // 创建初始顺序排布（所有层，冗余位置为 -1）
        std::vector<int> CreateSequentialPlacement() {
            std::vector<int> placement(num_layers_ * num_ranks_ * max_slots_per_rank_, -1);
            for (int layer = 0; layer < num_layers_; ++layer) {
                int layer_offset = layer * num_ranks_ * max_slots_per_rank_;
                for (int r = 0; r < num_ranks_; ++r) {
                    int rank_offset = layer_offset + r * max_slots_per_rank_;
                    // 分配 8 个专家
                    for (int i = 0; i < num_experts_per_rank_; ++i) {
                        int expert_id = r * num_experts_per_rank_ + i;
                        placement[rank_offset + i] = expert_id;
                    }
                    // 冗余位置设置为 -1
                    for (int i = num_experts_per_rank_; i < max_slots_per_rank_; ++i) {
                        placement[rank_offset + i] = -1;
                    }
                }
            }
            return placement;
        }

        // 创建激活矩阵（所有层）
        std::vector<int64_t> CreateActivations() {
            std::vector<int64_t> activations(num_layers_ * num_ranks_ * max_slots_per_rank_, 0);
            for (int layer = 0; layer < num_layers_; ++layer) {
                int layer_offset = layer * num_ranks_ * max_slots_per_rank_;
                for (int r = 0; r < num_ranks_; ++r) {
                    int rank_offset = layer_offset + r * max_slots_per_rank_;
                    for (int i = 0; i < max_slots_per_rank_; ++i) {
                        activations[rank_offset + i] = static_cast<int64_t>(r * 1000 + layer * 10 + i);
                    }
                }
            }
            return activations;
        }

    int num_layers_;
    int num_ranks_;
    int num_experts_per_rank_;
    int num_redundant_per_rank_;
    int expert_redundant_limit_;
    int max_slots_per_rank_;
    int num_experts_;
};


TEST_F(ExpertLoadBalancerPlacementActivationTest, SequentialPlacementAndActivations) {
    std::ofstream log_file("expert_load_balancer_log_bucket_sort.txt");
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file" << std::endl;
        GTEST_FAIL();
        return;
    }

    std::streambuf* cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(log_file.rdbuf());

    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                               num_redundant_per_rank_, expert_redundant_limit_, 0);

    auto placement = CreateSequentialPlacement();
    auto activations = CreateActivations();

    ASSERT_TRUE(balancer.get_validate_input_size(placement, activations, num_layers_, num_ranks_, max_slots_per_rank_))
        << "Invalid input size for placement or activations";

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
        std::vector<int> layer_placement(
            placement.begin() + layer_offset,
            placement.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);
        ASSERT_TRUE(balancer.get_validate_unique_expert_ids(layer_placement, layer_idx, num_ranks_, max_slots_per_rank_))
            << "Duplicate expert IDs in initial placement for layer " << layer_idx;
        ASSERT_TRUE(balancer.get_validate_all_experts_present(layer_placement, layer_idx, num_ranks_, max_slots_per_rank_, num_experts_))
            << "Not all experts present in initial placement for layer " << layer_idx;
    }

    std::cout << "Running optimization for all layers..." << std::endl;
    auto instructions = balancer.optimize_and_generate_instructions(placement, activations);
    std::cout << "Optimization completed, total instruction count: " << instructions.size() << std::endl;

    std::vector<int> final_placement = placement;
    for (const auto& instr : instructions) {
        int layer_offset = instr.layer_idx * num_ranks_ * max_slots_per_rank_;
        size_t source_pos = static_cast<size_t>(instr.source_global_position) + layer_offset;
        size_t target_pos = static_cast<size_t>(instr.target_global_position) + layer_offset;

        ASSERT_TRUE(source_pos < final_placement.size()) << "Invalid source_global_position: " << instr.source_global_position << " in layer " << instr.layer_idx;
        ASSERT_TRUE(target_pos < final_placement.size()) << "Invalid target_global_position: " << instr.target_global_position << " in layer " << instr.layer_idx;
        ASSERT_EQ(final_placement[source_pos], instr.source_expert_id) << "Source position does not contain source_expert_id: pos=" << source_pos << ", expected_id=" << instr.source_expert_id;

        if (instr.type == OperationType::SWAP) {
            std::swap(final_placement[source_pos], final_placement[target_pos]);
        } else if (instr.type == OperationType::ADD) {
            final_placement[target_pos] = instr.target_expert_id;
        } else if (instr.type == OperationType::REMOVE) {
            final_placement[target_pos] = -1;
        }

        // 中间验证
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
            std::vector<int> layer_placement(
                final_placement.begin() + layer_offset,
                final_placement.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);
            ASSERT_TRUE(balancer.get_validate_unique_expert_ids(layer_placement, layer_idx, num_ranks_, max_slots_per_rank_))
                << "Duplicate expert IDs after applying instruction in layer " << layer_idx;
        }
    }

    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
        std::vector<int> layer_placement(
            final_placement.begin() + layer_offset,
            final_placement.begin() + layer_offset + num_ranks_ * max_slots_per_rank_);
        ASSERT_TRUE(balancer.get_validate_unique_expert_ids(layer_placement, layer_idx, num_ranks_, max_slots_per_rank_))
            << "Duplicate expert IDs in final placement for layer " << layer_idx;
        ASSERT_TRUE(balancer.get_validate_all_experts_present(layer_placement, layer_idx, num_ranks_, max_slots_per_rank_, num_experts_))
            << "Not all experts present in final placement for layer " << layer_idx;
    }

    std::cout.rdbuf(cout_buf);
    log_file.close();
}
