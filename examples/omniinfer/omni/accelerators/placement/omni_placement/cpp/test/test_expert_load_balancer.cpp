// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_load_balancer.h"
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

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

    // Helper to compute load ratio for a given placement and activations
    double GetPlacementRatio(ExpertLoadBalancer &balancer,
                             const std::vector<int> &placement,
                             const std::vector<int64_t> &activations,
                             int layer_idx) {
        return balancer.ut_compute_placement_ratio_combined(
            placement, activations, layer_idx, num_ranks_, max_slots_per_rank_,
            num_experts_, 0 /* type 0 for max/avg */);
    }

    // Rewritten RunOptimizeTest to work with the new `optimize_placement` API
    void RunOptimizeTest(const std::string &test_name,
                         const std::vector<int> &input_placement,
                         const std::vector<int64_t> &input_activations,
                         bool expect_placement_to_change) {
        // Use a low threshold to trigger optimization easily
        ExpertLoadBalancer balancer(
            num_layers_, num_ranks_, num_experts_per_rank_,
            num_redundant_per_rank_, expert_redundant_limit_, 0, 1.05);

        std::cout << "Test: " << test_name << "\n";

        ASSERT_TRUE(balancer.ut_validate_input_size(
            input_placement, input_activations, num_layers_, num_ranks_,
            max_slots_per_rank_))
            << "Invalid input size in " << test_name;

        // The new core function call
        auto optimized_placement =
            balancer.optimize_placement(input_placement, input_activations);

        // Validate the size of the output
        ASSERT_EQ(optimized_placement.size(), input_placement.size())
            << "Optimized placement has incorrect size in " << test_name;

        // Check if the placement changed as expected
        if (expect_placement_to_change) {
            EXPECT_NE(input_placement, optimized_placement)
                << "Placement was expected to change but did not in "
                << test_name;
        } else {
            EXPECT_EQ(input_placement, optimized_placement)
                << "Placement was not expected to change but it did in "
                << test_name;
        }

        // Validate the integrity of the final placement for each layer
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
            int layer_offset = layer_idx * num_ranks_ * max_slots_per_rank_;
            std::vector<int> final_layer_placement(
                optimized_placement.begin() + layer_offset,
                optimized_placement.begin() + layer_offset +
                    num_ranks_ * max_slots_per_rank_);

            EXPECT_TRUE(balancer.ut_validate_unique_expert_ids(
                final_layer_placement, layer_idx, num_ranks_,
                max_slots_per_rank_))
                << "Duplicate expert IDs in final placement for layer "
                << layer_idx << " in " << test_name;
            EXPECT_TRUE(balancer.ut_validate_all_experts_present(
                final_layer_placement, layer_idx, num_ranks_,
                max_slots_per_rank_, num_experts_))
                << "Not all experts present in final placement for layer "
                << layer_idx << " in " << test_name;
        }
    }

    // Create uniform placement across layers
    std::vector<int> CreateUniformPlacement() {
        std::vector<int> placement(
            num_layers_ * num_ranks_ * max_slots_per_rank_, -1);
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

    // Create uniform activations with specified value
    std::vector<int64_t> CreateUniformActivations(int64_t value) {
        return std::vector<int64_t>(
            num_layers_ * num_ranks_ * max_slots_per_rank_, value);
    }

    int num_layers_;
    int num_ranks_;
    int num_experts_per_rank_;
    int num_redundant_per_rank_;
    int expert_redundant_limit_;
    int max_slots_per_rank_;
    int num_experts_;
};

// Test case: Constructor with valid parameters (updated for new API)
TEST_F(ExpertLoadBalancerTest, Constructor_ValidParameters) {
    ExpertLoadBalancer balancer(2, 4, 2, 1, 1, 0, 1.1, 0.05, 8, 300.0);
    EXPECT_EQ(balancer.ut_num_layers(), 2);
    EXPECT_EQ(balancer.ut_num_ranks(), 4);
    EXPECT_EQ(balancer.ut_num_experts_per_rank(), 2);
    EXPECT_EQ(balancer.ut_num_redundant_per_rank(), 1);
    EXPECT_EQ(balancer.ut_expert_redundant_limit(), 1);
    EXPECT_EQ(balancer.ut_max_slots_per_rank(), 3);
    EXPECT_EQ(balancer.ut_num_experts(), 8);
    EXPECT_EQ(balancer.ut_num_ranks_per_host(), 8);
    EXPECT_NEAR(balancer.ut_high_low_ratio_threshold(), 300.0, 1e-9);
}

// Test case: Constructor with invalid parameters (updated for new API)
TEST_F(ExpertLoadBalancerTest, Constructor_InvalidParameters) {
    EXPECT_THROW(ExpertLoadBalancer(0, 4, 2, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 0, 2, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 0, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, -1, 1, 0), std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, 1, -1, 0), std::runtime_error);
    // New checks from the new constructor
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, 1, 1, 0, 1.1, 0.05, 0),
                 std::runtime_error);
    EXPECT_THROW(ExpertLoadBalancer(2, 4, 2, 1, 1, 0, 1.1, 0.05, 8, 0.0),
                 std::runtime_error);
}

// Test case: Validate input size (updated for new API)
TEST_F(ExpertLoadBalancerTest, ValidateInputSize) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement(num_layers_ * num_ranks_ * max_slots_per_rank_,
                               0);
    std::vector<int64_t> activations(
        num_layers_ * num_ranks_ * max_slots_per_rank_, 100);
    EXPECT_TRUE(balancer.ut_validate_input_size(
        placement, activations, num_layers_, num_ranks_, max_slots_per_rank_));

    placement.resize(num_layers_ * num_ranks_ * max_slots_per_rank_ - 1);
    EXPECT_FALSE(balancer.ut_validate_input_size(
        placement, activations, num_layers_, num_ranks_, max_slots_per_rank_));
}

// Test case: Validate unique expert IDs (updated for new API)
TEST_F(ExpertLoadBalancerTest, ValidateUniqueExpertIds) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    // A single layer's placement for validation
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_TRUE(balancer.ut_validate_unique_expert_ids(placement, 0, num_ranks_,
                                                       max_slots_per_rank_));

    placement[0] = 1; // Introduce a duplicate
    EXPECT_FALSE(balancer.ut_validate_unique_expert_ids(
        placement, 0, num_ranks_, max_slots_per_rank_));
}

// Test case: Compute expert loads (updated for new API)
TEST_F(ExpertLoadBalancerTest, ComputeExpertLoads) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<ExpertInformation> experts = {
        {0, 0, 0, 100, 0, 2}, {0, 1, 0, 50, 3, 2}, {0, 2, 1, 200, 6, 1}};
    auto loads = balancer.ut_compute_expert_loads(experts, num_experts_);
    EXPECT_DOUBLE_EQ(loads[0], 150.0);
    EXPECT_DOUBLE_EQ(loads[1], 200.0);
    EXPECT_DOUBLE_EQ(loads.count(2),
                     0); // Expert 2 has no activations in this list
}

// Test case: Extract expert information (updated for new API)
TEST_F(ExpertLoadBalancerTest, ExtractExpertInfo) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);
    auto layer_experts = balancer.ut_extract_expert_info(
        placement, activations, num_layers_, num_ranks_, num_experts_per_rank_,
        num_redundant_per_rank_, expert_redundant_limit_);
    EXPECT_EQ(layer_experts.size(), num_layers_);
    EXPECT_EQ(layer_experts[0].size(), num_ranks_ * num_experts_per_rank_);
}

// Test case: Optimize with balanced load, no changes expected (Rewritten)
TEST_F(ExpertLoadBalancerTest, Optimize_BalancedNoChanges) {
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);
    // With a balanced load, the placement should not change.
    RunOptimizeTest("BalancedNoChanges", placement, activations, false);
}

// Test case: Optimize with imbalanced load, expects changes (Rewritten with
// Redundancy)
TEST_F(ExpertLoadBalancerTest, Optimize_ImbalancedTriggersChange) {
    // Override SetUp() values to introduce redundancy for this specific test.

    num_redundant_per_rank_ = 1;
    expert_redundant_limit_ = 1;
    max_slots_per_rank_ = num_experts_per_rank_ + num_redundant_per_rank_;

    // Create placement and activations based on the new redundant setup.
    std::vector<int> placement = CreateUniformPlacement();
    std::vector<int64_t> activations = CreateUniformActivations(100);

    // Create a significant imbalance to trigger optimization.
    activations[0] = 10000;
    // The index for the second layer needs to account for the new
    // max_slots_per_rank_
    int layer_2_offset = num_ranks_ * max_slots_per_rank_;
    activations[layer_2_offset] = 10000; // Imbalance in layer 1 too

    ExpertLoadBalancer balancer(
        num_layers_, num_ranks_, num_experts_per_rank_, num_redundant_per_rank_,
        expert_redundant_limit_, 0, 1.1 /* low threshold */);

    // Extract layer 0 for ratio calculation
    std::vector<int> initial_layer_placement_L0(
        placement.begin(),
        placement.begin() + num_ranks_ * max_slots_per_rank_);
    std::vector<int64_t> initial_layer_activations_L0(
        activations.begin(),
        activations.begin() + num_ranks_ * max_slots_per_rank_);
    auto initial_ratio_L0 = GetPlacementRatio(
        balancer, initial_layer_placement_L0, initial_layer_activations_L0, 0);

    auto optimized_placement =
        balancer.optimize_placement(placement, activations);

    auto final_layer_placement_L0 = std::vector<int>(
        optimized_placement.begin(),
        optimized_placement.begin() + num_ranks_ * max_slots_per_rank_);

    // Create adjusted activations map to calculate final ratio correctly
    std::unordered_map<int, int64_t> expert_activations_map;
    for (size_t i = 0; i < placement.size(); ++i) {
        if (placement[i] != -1) {
            // Use the activation at the original position of the expert
            expert_activations_map[placement[i]] = activations[i];
        }
    }
    std::vector<int64_t> final_activations_L0(num_ranks_ * max_slots_per_rank_,
                                              0);
    for (size_t i = 0; i < final_layer_placement_L0.size(); ++i) {
        if (final_layer_placement_L0[i] != -1) {
            final_activations_L0[i] =
                expert_activations_map[final_layer_placement_L0[i]];
        }
    }

    auto final_ratio_L0 = GetPlacementRatio(balancer, final_layer_placement_L0,
                                            final_activations_L0, 0);

    // With redundancy, the placement is now expected to change to balance the
    // load.
    EXPECT_NE(placement, optimized_placement)
        << "Placement should have changed because redundancy is available.";

    // The new ratio should be better (lower) than the initial one.
    EXPECT_LT(final_ratio_L0, initial_ratio_L0)
        << "Optimized ratio should be better than initial ratio.";
}

// Test case: Compute rank sets (updated for new API)
TEST_F(ExpertLoadBalancerTest, ComputeRankSets) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7}; // Single layer
    auto rank_sets = balancer.ut_compute_rank_sets(placement, num_ranks_,
                                                   max_slots_per_rank_);
    EXPECT_EQ(rank_sets.size(), num_ranks_);
    EXPECT_EQ(rank_sets[0], std::set<int>({0, 1}));
    EXPECT_EQ(rank_sets[1], std::set<int>({2, 3}));
    EXPECT_EQ(rank_sets[2], std::set<int>({4, 5}));
    EXPECT_EQ(rank_sets[3], std::set<int>({6, 7}));
}

// Test case: Find position with expert (updated for new API)
TEST_F(ExpertLoadBalancerTest, FindPositionWithExpert) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(balancer.ut_find_position_with_expert(placement, 0, 0,
                                                    max_slots_per_rank_),
              0);
    EXPECT_EQ(balancer.ut_find_position_with_expert(placement, 1, 3,
                                                    max_slots_per_rank_),
              3);
    EXPECT_EQ(balancer.ut_find_position_with_expert(placement, 2, 99,
                                                    max_slots_per_rank_),
              -1);
}

// Test case: Find empty position (updated for new API)
TEST_F(ExpertLoadBalancerTest, FindEmptyPosition) {
    num_redundant_per_rank_ = 1; // Need redundant slots to have empty positions
    max_slots_per_rank_ = num_experts_per_rank_ + num_redundant_per_rank_;
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);

    std::vector<int> placement = {0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1};
    EXPECT_EQ(
        balancer.ut_find_empty_position(placement, 0, max_slots_per_rank_), 2);
    EXPECT_EQ(
        balancer.ut_find_empty_position(placement, 1, max_slots_per_rank_), 5);

    std::vector<int> full_placement = {0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11};
    EXPECT_EQ(
        balancer.ut_find_empty_position(full_placement, 0, max_slots_per_rank_),
        -1);
}

// Test case: Compute expert counts (Corrected)
TEST_F(ExpertLoadBalancerTest, ComputeExpertCounts) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);

    // The vector now has a size of 8, which matches the expected processing
    // size (num_ranks_ * max_slots_per_rank_ = 4 * 2 = 8). Duplicates (0 and 1)
    // are created within this range.
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 0, 1};

    auto counts = balancer.ut_compute_expert_counts(placement, num_ranks_,
                                                    max_slots_per_rank_);

    // Now the assertions are correct based on the input {0, 1, 2, 3, 4, 5, 0,
    // 1}
    EXPECT_EQ(counts[0], 2);       // Expert 0 appears twice
    EXPECT_EQ(counts[1], 2);       // Expert 1 appears twice
    EXPECT_EQ(counts[2], 1);       // Expert 2 appears once
    EXPECT_EQ(counts[5], 1);       // Expert 5 appears once
    EXPECT_EQ(counts.count(8), 0); // Expert 8 does not appear
}

// Test case: Validate all experts present (updated for new API)
TEST_F(ExpertLoadBalancerTest, ValidateAllExpertsPresent) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_TRUE(balancer.ut_validate_all_experts_present(
        placement, 0, num_ranks_, max_slots_per_rank_, num_experts_));

    placement[0] = 1; // Introduce duplicate, missing expert 0
    EXPECT_FALSE(balancer.ut_validate_all_experts_present(
        placement, 0, num_ranks_, max_slots_per_rank_, num_experts_));
}

// Test case: Compute placement ratio (updated for new API)
TEST_F(ExpertLoadBalancerTest, ComputePlacementRatio) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> activations = {100, 100, 100, 100, 100, 100, 100, 100};
    double ratio = balancer.ut_compute_placement_ratio_combined(
        placement, activations, 0, num_ranks_, max_slots_per_rank_,
        num_experts_, 0); // type 0: max/avg
    EXPECT_NEAR(ratio, 1.0,
                1e-9); // Uniform distribution, ratio should be exactly 1
}

// Test case: Extract layer expert information (updated for new API)
TEST_F(ExpertLoadBalancerTest, ExtractLayerExpertInfo) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0);
    std::vector<int> placement = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> activations = {100, 200, 300, 400, 500, 600, 700, 800};

    // ut_extract_layer_expert_info takes the full placement vector
    std::vector<int> full_placement = placement;
    std::vector<int64_t> full_activations = activations;
    // Pad to full size if needed for the test setup
    full_placement.resize(num_layers_ * num_ranks_ * max_slots_per_rank_, -1);
    full_activations.resize(num_layers_ * num_ranks_ * max_slots_per_rank_, 0);

    auto experts = balancer.ut_extract_layer_expert_info(
        full_placement, full_activations, 0, num_ranks_, max_slots_per_rank_,
        num_experts_, expert_redundant_limit_);

    EXPECT_EQ(experts.size(), num_experts_);
    for (size_t i = 0; i < experts.size(); ++i) {
        EXPECT_EQ(experts[i].expert_id, static_cast<int>(i));
        EXPECT_EQ(experts[i].activations, activations[i]);
        EXPECT_EQ(experts[i].total_count, 1);
    }
}

// This test is now a more complex end-to-end check.
// It uses the same large custom input and verifies that the optimization
// improves the load balance ratio.
TEST_F(ExpertLoadBalancerTest, CompareLoadRatioBeforeAndAfterCustom) {
    // Set parameters
    num_layers_ = 1;
    num_ranks_ = 32;
    num_experts_per_rank_ = 8;
    num_redundant_per_rank_ = 0;
    expert_redundant_limit_ = 0;
    max_slots_per_rank_ = num_experts_per_rank_;
    num_experts_ = num_ranks_ * num_experts_per_rank_;

    // Input data from the original test
    std::vector<int> input_placement = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
        154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
        168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
        182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
        196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
        252, 253, 254, 255};

    std::vector<int64_t> input_activations = {
        6484,   5972,  6027,  1382,  2391,  5241,   104817, 1483,   6778,
        104231, 1471,  5643,  6397,  5655,  2133,   1769,   103606, 2065,
        876,    1168,  6854,  4253,  3139,  11858,  7736,   28672,  3863,
        3995,   2468,  6631,  19528, 14646, 4870,   5198,   8444,   24971,
        12674,  22674, 3723,  4677,  4125,  19438,  16776,  1491,   14120,
        5592,   13923, 12103, 13237, 12361, 6333,   15183,  6331,   6257,
        16019,  10908, 10839, 5927,  968,   6860,   6452,   14958,  23482,
        15707,  7195,  5330,  24040, 10974, 15186,  14306,  4439,   4577,
        10546,  2142,  8720,  29276, 4840,  7025,   4408,   19899,  13161,
        5320,   12930, 12831, 13140, 5926,  7070,   15636,  5050,   7308,
        15968,  6891,  12029, 3387,  6941,  30217,  13538,  5618,   5972,
        8487,   11636, 16164, 16164, 7952,  15019,  13532,  5477,   8005,
        5157,   16754, 16200, 6979,  18600, 2691,   15518,  20291,  3149,
        8387,   5904,  12676, 14668, 3357,  7651,   13016,  7027,   11787,
        15453,  13496, 3146,  3815,  3380,  109200, 4961,   4660,   4640,
        3683,   18875, 5542,  18086, 5601,  5651,   6125,   7160,   14148,
        4750,   4192,  4481,  3869,  3600,  4721,   4629,   104850, 14617,
        10485,  16393, 5269,  5659,  14198, 4805,   13378,  4536,   107883,
        4207,   3324,  5648,  4877,  4407,  4254,   11671,  7042,   12639,
        10989,  7783,  14010, 8405,  8088,  3103,   5110,   5338,   107437,
        3467,   3998,  3306,  2948,  10484, 50291,  5846,   2386,   4118,
        4133,   4345,  5461,  5435,  11402, 7744,   12924,  15123,  14912,
        7313,   5211,  3861,  4074,  3936,  3554,   4443,   5014,   4848,
        110810, 6962,  5709,  6198,  5707,  6206,   28775,  21816,  5407,
        13810,  13076, 5793,  16943, 5757,  5617,   12553,  13638,  3311,
        26788,  6700,  5628,  3077,  5512,  14514,  21120,  4185,   12398,
        15018,  3581,  19179, 14472, 14173, 3241,   17005,  28326,  7744,
        12096,  6763,  6077,  3576,  5131,  6453,   8273,   33922,  5599,
        4854,   13285, 8685,  4554};

    // Create ExpertLoadBalancer with a low threshold to ensure optimization
    // runs
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0, 1.1);

    // Compute initial placement ratio
    double initial_ratio =
        GetPlacementRatio(balancer, input_placement, input_activations, 0);

    // Generate optimized placement
    std::vector<int> optimized_placement =
        balancer.optimize_placement(input_placement, input_activations);

    // Compute activations for the optimized placement
    // We need to map the original activations to the new expert locations
    std::unordered_map<int, int64_t> expert_activations_map;
    for (size_t i = 0; i < input_placement.size(); ++i) {
        if (input_placement[i] != -1) {
            expert_activations_map[input_placement[i]] = input_activations[i];
        }
    }
    std::vector<int64_t> adjusted_activations(optimized_placement.size());
    for (size_t i = 0; i < optimized_placement.size(); ++i) {
        if (optimized_placement[i] != -1) {
            adjusted_activations[i] =
                expert_activations_map[optimized_placement[i]];
        }
    }

    // Compute optimized placement ratio
    double final_ratio = GetPlacementRatio(balancer, optimized_placement,
                                           adjusted_activations, 0);

    // Print load ratio comparison
    std::cout << "\nLoad Ratio Comparison:\n";
    std::cout << "  Initial Load Ratio:   " << std::fixed
              << std::setprecision(4) << initial_ratio << "\n";
    std::cout << "  Optimized Load Ratio: " << std::fixed
              << std::setprecision(4) << final_ratio << "\n";

    // Validate optimized placement
    EXPECT_TRUE(balancer.ut_validate_unique_expert_ids(
        optimized_placement, 0, num_ranks_, max_slots_per_rank_))
        << "Duplicate expert IDs in optimized placement";
    EXPECT_TRUE(balancer.ut_validate_all_experts_present(
        optimized_placement, 0, num_ranks_, max_slots_per_rank_, num_experts_))
        << "Not all experts present in optimized placement";

    // Verify improvement in optimized load ratio
    EXPECT_LE(final_ratio, initial_ratio)
        << "Optimized load ratio should be less than or equal to the initial "
           "ratio.";
}

// This test class is simplified as the instruction-applying logic is obsolete.
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

    std::vector<int> CreateSequentialPlacement() {
        std::vector<int> placement(num_ranks_ * max_slots_per_rank_);
        for (int r = 0; r < num_ranks_; ++r) {
            for (int i = 0; i < num_experts_per_rank_; ++i) {
                placement[r * max_slots_per_rank_ + i] =
                    r * num_experts_per_rank_ + i;
            }
        }
        return placement;
    }

    std::vector<int64_t> CreateActivations() {
        std::vector<int64_t> activations(num_ranks_ * max_slots_per_rank_);
        for (int r = 0; r < num_ranks_; ++r) {
            for (int i = 0; i < max_slots_per_rank_; ++i) {
                activations[r * max_slots_per_rank_ + i] =
                    static_cast<int64_t>(r * 1000 + i);
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

// Test case: Sequential placement and activations (Rewritten)
TEST_F(ExpertLoadBalancerPlacementActivationTest,
       SequentialPlacementAndActivations) {
    ExpertLoadBalancer balancer(num_layers_, num_ranks_, num_experts_per_rank_,
                                num_redundant_per_rank_,
                                expert_redundant_limit_, 0, 1.1);

    auto placement = CreateSequentialPlacement();
    auto activations = CreateActivations();

    ASSERT_TRUE(balancer.ut_validate_input_size(
        placement, activations, num_layers_, num_ranks_, max_slots_per_rank_))
        << "Invalid input size for placement or activations";

    ASSERT_TRUE(balancer.ut_validate_unique_expert_ids(placement, 0, num_ranks_,
                                                       max_slots_per_rank_))
        << "Duplicate expert IDs in initial placement";
    ASSERT_TRUE(balancer.ut_validate_all_experts_present(
        placement, 0, num_ranks_, max_slots_per_rank_, num_experts_))
        << "Not all experts present in initial placement";

    std::cout << "Running optimization..." << std::endl;
    auto final_placement = balancer.optimize_placement(placement, activations);
    std::cout << "Optimization completed." << std::endl;

    // The main check is to ensure the final placement is valid.
    ASSERT_TRUE(balancer.ut_validate_unique_expert_ids(
        final_placement, 0, num_ranks_, max_slots_per_rank_))
        << "Duplicate expert IDs in final placement";
    ASSERT_TRUE(balancer.ut_validate_all_experts_present(
        final_placement, 0, num_ranks_, max_slots_per_rank_, num_experts_))
        << "Not all experts present in final placement";
}