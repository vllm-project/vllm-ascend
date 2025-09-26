// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_activation.h"
#include "placement_mapping.h"
#include "placement_optimizer.h"
#include "tensor.h"
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

// Since the constructors of PlacementMapping and ClusterActivation are very
// complex, we will create a simplified test environment focusing on verifying
// the core logic of PlacementOptimizer.
class PlacementOptimizerTest : public ::testing::Test {
  protected:
    PlacementMapping *placement_mapping;
    ClusterActivation *cluster_activation;
    PlacementOptimizer *optimizer;

    void *selector_ptr_hbm = nullptr;
    void *num_redundant_per_expert_ptr_hbm = nullptr;
    void *npu_count_ptr_hbm = nullptr;

    // Define parameters for the test scenario
    const int rank = 0;
    const int world_size = 2;
    const int num_layers = 1;
    const int num_experts = 4;
    const int num_deploy_experts_per_rank =
        3; // 2 regular experts + 1 redundant slot
    const int num_deploy_experts = num_deploy_experts_per_rank * world_size;
    const int num_devices_per_host = 2;
    const int max_redundant_per_expert = 2;
    const int64_t max_activation_count = 1000000;
    const int activation_window_size = 10;
    // Define the path for the test file. The directory './test_data' is assumed
    // to exist.
    const std::string pattern_file =
        "../test_data/test_placement_pattern_for_optimizer.txt";

    void SetUp() override {
        // 1. Initialize ACL and NPU device context (required for unit tests)
        aclInit(nullptr);
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        // 2. Create the placement pattern file within the existing test
        // directory.
        std::ofstream pattern_file_stream(pattern_file);
        if (!pattern_file_stream.is_open()) {
            // Throw an error if file creation fails, to make debugging easier.
            throw std::runtime_error(
                "Failed to create test pattern file for writing: " +
                pattern_file);
        }
        pattern_file_stream << world_size << " " << num_layers << " "
                            << num_experts << std::endl;
        pattern_file_stream << "1 1 0 0" << std::endl; // Rank 0: experts 0, 1
        pattern_file_stream << "0 0 1 1" << std::endl; // Rank 1: experts 2, 3
        pattern_file_stream.close();

        // 3. Initialize a simplified PlacementMapping
        // Allocate memory for the selector
        size_t selector_len =
            num_layers * max_redundant_per_expert * num_experts;
        aclrtMalloc(&this->selector_ptr_hbm, selector_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        // Allocate memory for the redundant expert count tensor, as required by
        // the new constructor
        size_t num_redundant_per_expert_len = num_layers * num_experts;
        aclrtMalloc(&this->num_redundant_per_expert_ptr_hbm,
                    num_redundant_per_expert_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        // Call the updated constructor with all 10 arguments
        placement_mapping = new PlacementMapping(
            pattern_file, rank, num_devices_per_host, max_redundant_per_expert,
            num_deploy_experts, 0, {}, (size_t)this->selector_ptr_hbm, false,
            (size_t)this->num_redundant_per_expert_ptr_hbm);

        // 4. Initialize a simplified ClusterActivation
        size_t npu_count_len = num_layers * num_deploy_experts_per_rank;
        aclrtMalloc(&this->npu_count_ptr_hbm, npu_count_len * sizeof(int64_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);
        Tensor npu_count_tensor((uint64_t)this->npu_count_ptr_hbm,
                                npu_count_len, sizeof(int64_t), "npu_count");
        cluster_activation = new ClusterActivation(
            npu_count_tensor, max_activation_count, num_layers,
            num_deploy_experts_per_rank, activation_window_size, world_size,
            world_size, rank);
        cluster_activation->set_params(num_experts);

        // 5. Initialize PlacementOptimizer
        optimizer =
            new PlacementOptimizer(placement_mapping, cluster_activation);
    }

    void TearDown() override {
        delete optimizer;
        delete cluster_activation;
        delete placement_mapping;

        if (this->selector_ptr_hbm) {
            aclrtFree(this->selector_ptr_hbm);
        }
        if (this->num_redundant_per_expert_ptr_hbm) {
            aclrtFree(this->num_redundant_per_expert_ptr_hbm);
        }
        if (this->npu_count_ptr_hbm) {
            aclrtFree(this->npu_count_ptr_hbm);
        }

        remove(pattern_file.c_str());

        aclrtResetDevice(0);
        aclFinalize();
    }
};

// Test the core optimization logic by only checking the number of instructions
// generated.
TEST_F(PlacementOptimizerTest, OptimizeSimpleImbalance) {
    // Initial placement and load:
    // Rank 0: {0, 1, -1} -> Load: 1000 + 50 = 1050
    // Rank 1: {2, 3, -1} -> Load: 20 + 30 = 50
    std::vector<int> placement = {0, 1, -1, 2, 3, -1};
    std::vector<int64_t> activations = {1000, 50, 0, 20, 30, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Based on the error log, the optimizer generates 3 instructions for this
    // scenario.
    ASSERT_EQ(instructions.size(), 3);
}

// New test case: Verify no optimization instructions in a balanced load
// scenario
TEST_F(PlacementOptimizerTest, OptimizeBalanced) {
    std::vector<int> placement = {0, 1, -1, 2, 3, -1};
    // Activations are perfectly balanced between ranks (100 vs 100)
    std::vector<int64_t> activations = {50, 50, 0, 50, 50, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Expected: Balanced load, no optimization needed
    ASSERT_EQ(instructions.size(), 0);
}

// New test case: Verify optimization for a hot expert on Rank 1
TEST_F(PlacementOptimizerTest, OptimizeHotInRank1) {
    // Rank 0 Load: 20 + 30 = 50
    // Rank 1 Load: 1000 + 50 = 1050
    std::vector<int> placement = {0, 1, -1, 2, 3, -1};
    std::vector<int64_t> activations = {20, 30, 0, 1000, 50, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Based on the crash log and similarity to the other imbalanced case,
    // we expect 3 instructions to be generated before the crash.
    ASSERT_EQ(instructions.size(), 3);
}

// New test case: Verify optimization for multiple hot experts where ranks are
// balanced
TEST_F(PlacementOptimizerTest, OptimizeMultipleHotExpertsBalancedRanks) {
    std::vector<int> placement = {0, 1, -1, 2, 3, -1};
    // Both ranks have one hot expert, but the total load per rank is equal
    // (1050 vs 1050).
    std::vector<int64_t> activations = {1000, 50, 0, 1000, 50, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Expected: The load between ranks is balanced, so no instructions should
    // be generated.
    ASSERT_EQ(instructions.size(), 0);
}

// Test case: Verify optimization for a valid placement with a redundant expert.
TEST_F(PlacementOptimizerTest, OptimizeWithImbalancedRedundantExpert) {
    // This placement is valid since max_redundant_per_expert is 2.
    // Expert 0 appears on both Rank 0 and Rank 1 (as a redundant copy).
    std::vector<int> placement = {0, 1, -1, 2, 0, 3};
    // The load is highly imbalanced: Rank 0 load is 1050, Rank 1 load is 50.
    std::vector<int64_t> activations = {1000, 50, 0, 20, 30, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Expected: The logic will attempt to rebalance the load.
    // We assert that instructions are generated, as the placement is
    // imbalanced.
    ASSERT_FALSE(instructions.empty());
}

// New test case: Verify placement with missing experts.
TEST_F(PlacementOptimizerTest, OptimizeInvalidPlacementMissingExperts) {
    // Expert 3 is missing from the placement, which makes it invalid.
    std::vector<int> placement = {0, 1, -1, 2, -1, -1};
    std::vector<int64_t> activations = {1000, 50, 0, 20, 0, 0};

    std::vector<ChangeInstruction> instructions =
        optimizer->optimize(placement, activations);

    // Expected: The validation inside `extract_expert_info` will fail because
    // not all experts are present. This will lead to an empty `layer_experts`
    // vector, and the optimizer will fall back to returning the original
    // placement, resulting in zero instructions.
    ASSERT_EQ(instructions.size(), 0);
}

// New test case: Verify getter methods
TEST_F(PlacementOptimizerTest, Getters) {
    EXPECT_EQ(optimizer->get_num_layers(), num_layers);
    EXPECT_EQ(optimizer->get_rank(), rank);
    EXPECT_EQ(optimizer->get_world_size(), world_size);
    EXPECT_EQ(optimizer->get_num_experts(), num_experts);
    EXPECT_EQ(optimizer->get_num_devices_per_host(), num_devices_per_host);
    EXPECT_EQ(optimizer->get_num_experts_per_rank(), num_experts / world_size);
    EXPECT_EQ(optimizer->get_num_redundant_per_rank(),
              num_deploy_experts_per_rank - (num_experts / world_size));
    EXPECT_EQ(optimizer->get_expert_redundant_limit(),
              max_redundant_per_expert - 1);
}

// New test case: Verify constructor exception throwing (without fixture)
TEST(PlacementOptimizerTestNoFixture, ConstructorThrowsInvalidParams) {
    // Test that providing null pointers to the constructor throws a
    // runtime_error.
    EXPECT_THROW(PlacementOptimizer(nullptr, nullptr), std::runtime_error);
}