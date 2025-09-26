// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dynamic_eplb_greedy.h"

// Base Fixture class containing common logic
class BaseGreedyExpertLoadBalancerTest : public ::testing::Test {
  protected:
    GreedyExpertLoadBalancer *balancer_ = nullptr;

    virtual void SetUp() override = 0; // Implemented by subclasses
    void TearDown() override { delete balancer_; }

    // Helper function: Generate placement and activations data
    virtual std::vector<int> GenerateSimplePlacement() = 0;
    std::vector<int64_t> GenerateBalancedActivations(int num_deploy_experts) {
        std::vector<int64_t> activations(num_layers_ * num_deploy_experts, 0);
        for (int layer = 0; layer < num_layers_; ++layer) {
            int offset = layer * num_deploy_experts;
            for (int i = 0; i < num_deploy_experts; ++i) {
                int pos = offset + i;
                activations[pos] = 50;
            }
        }
        return activations;
    }

    std::vector<int64_t> GenerateImbalancedActivations(int num_deploy_experts) {
        std::vector<int64_t> activations(num_layers_ * num_deploy_experts, 0);
        for (int layer = 0; layer < num_layers_; ++layer) {
            int offset = layer * num_deploy_experts;
            activations[offset + 0] = 1000; // Expert 0 (hot)
            for (int i = 1; i < num_deploy_experts; ++i) {
                activations[offset + i] = 50;
            }
        }
        return activations;
    }

    // Test parameters, set by subclasses
    int rank_;
    int world_size_;
    int num_layers_;
    int num_experts_;
    int num_deploy_experts_per_rank_;
    int num_deploy_experts_;
    int expert_redundant_limit_;
};

// Fixture class: num_deploy_experts_per_rank_ = 3
class GreedyExpertLoadBalancerTestWithRedundantSlot
    : public BaseGreedyExpertLoadBalancerTest {
  protected:
    void SetUp() override {
        rank_ = 1;
        world_size_ = 2;
        num_layers_ = 2;
        num_experts_ = 4;
        num_deploy_experts_per_rank_ = 3; // 2 experts + 1 redundant slot
        num_deploy_experts_ = num_deploy_experts_per_rank_ * world_size_;
        expert_redundant_limit_ = 2;

        balancer_ = new GreedyExpertLoadBalancer(
            num_layers_, world_size_, num_experts_, num_deploy_experts_,
            expert_redundant_limit_, rank_);
    }

    std::vector<int> GenerateSimplePlacement() override {
        // Rank 0: pos 0 (expert 0), pos 1 (expert 1), pos 2 (empty, -1)
        // Rank 1: pos 3 (expert 2), pos 4 (expert 3), pos 5 (empty, -1)
        std::vector<int> placement(num_layers_ * num_deploy_experts_, -1);
        for (int layer = 0; layer < num_layers_; ++layer) {
            int offset = layer * num_deploy_experts_;
            placement[offset + 0] = 0; // Rank 0, expert 0
            placement[offset + 1] = 1; // Rank 0, expert 1
            placement[offset + 3] = 2; // Rank 1, expert 2
            placement[offset + 4] = 3; // Rank 1, expert 3
        }
        return placement;
    }
};

// Fixture class: num_deploy_experts_per_rank_ = 2
class GreedyExpertLoadBalancerTestNoRedundantSlot
    : public BaseGreedyExpertLoadBalancerTest {
  protected:
    void SetUp() override {
        rank_ = 1;
        world_size_ = 2;
        num_layers_ = 2;
        num_experts_ = 4;
        num_deploy_experts_per_rank_ = 2; // 2 experts, no redundant slot
        num_deploy_experts_ = num_deploy_experts_per_rank_ * world_size_;
        expert_redundant_limit_ = 2;

        balancer_ = new GreedyExpertLoadBalancer(
            num_layers_, world_size_, num_experts_, num_deploy_experts_,
            expert_redundant_limit_, rank_);
    }

    std::vector<int> GenerateSimplePlacement() override {
        // Rank 0: pos 0 (expert 0), pos 1 (expert 1)
        // Rank 1: pos 2 (expert 2), pos 3 (expert 3)
        std::vector<int> placement(num_layers_ * num_deploy_experts_, -1);
        for (int layer = 0; layer < num_layers_; ++layer) {
            int offset = layer * num_deploy_experts_;
            placement[offset + 0] = 0; // Rank 0, expert 0
            placement[offset + 1] = 1; // Rank 0, expert 1
            placement[offset + 2] = 2; // Rank 1, expert 2
            placement[offset + 3] = 3; // Rank 1, expert 3
        }
        return placement;
    }
};

// Test case: Balanced load (with redundant slot)
TEST_F(GreedyExpertLoadBalancerTestWithRedundantSlot, OptimizeBalancedLoad) {
    auto placement = GenerateSimplePlacement();
    auto activations = GenerateBalancedActivations(num_deploy_experts_);
    auto instructions =
        balancer_->optimize_and_generate_instructions(placement, activations);
    EXPECT_TRUE(instructions.empty())
        << "No instructions should be generated for balanced load.";
}

// Test case: Imbalanced load (with redundant slot, ADD instruction)
TEST_F(GreedyExpertLoadBalancerTestWithRedundantSlot,
       OptimizeImbalancedLoadAdd) {
    auto placement = GenerateSimplePlacement();
    auto activations = GenerateImbalancedActivations(num_deploy_experts_);
    auto instructions =
        balancer_->optimize_and_generate_instructions(placement, activations);

    ASSERT_EQ(instructions.size(), 2); // One instruction per layer
    for (const auto &instr : instructions) {
        EXPECT_EQ(instr.type, OperationType::ADD);
        EXPECT_EQ(instr.source_expert_id, 0);  // Hot expert
        EXPECT_EQ(instr.target_expert_id, -1); // Target is empty slot
        EXPECT_TRUE(instr.target_global_position == 2 ||
                    instr.target_global_position == 5 ||
                    instr.target_global_position == 8 ||
                    instr.target_global_position == 11); // Empty slot
    }
}

// Test case: Balanced load (no redundant slot)
TEST_F(GreedyExpertLoadBalancerTestNoRedundantSlot, OptimizeBalancedLoad) {
    auto placement = GenerateSimplePlacement();
    auto activations = GenerateBalancedActivations(num_deploy_experts_);
    auto instructions =
        balancer_->optimize_and_generate_instructions(placement, activations);
    EXPECT_TRUE(instructions.empty())
        << "No instructions should be generated for balanced load.";
}

// Test case: SWAP instruction (no redundant slot)
TEST_F(GreedyExpertLoadBalancerTestNoRedundantSlot, OptimizeSwap) {
    std::vector<int> placement = GenerateSimplePlacement();
    std::vector<int64_t> activations(num_layers_ * num_deploy_experts_, 0);
    for (int layer = 0; layer < num_layers_; ++layer) {
        int offset = layer * num_deploy_experts_;
        activations[offset + 0] = 1000; // Expert 0
        activations[offset + 1] = 150;  // Expert 1
        activations[offset + 2] = 50;   // Expert 2
        activations[offset + 3] = 50;   // Expert 3
    }
    auto instructions =
        balancer_->optimize_and_generate_instructions(placement, activations);

    EXPECT_FALSE(instructions.empty());
    bool found_swap = false;
    for (const auto &instr : instructions) {
        if (instr.type == OperationType::SWAP) {
            found_swap = true;
            EXPECT_EQ(instr.source_expert_id, 0); // Hot expert
            EXPECT_TRUE(instr.target_expert_id == 2 ||
                        instr.target_expert_id == 3); // Cold expert
        }
    }
    EXPECT_TRUE(found_swap) << "Expected at least one SWAP instruction.";
}

// Test case: Imbalanced load (with redundant slot, ADD2 instruction)
TEST_F(GreedyExpertLoadBalancerTestWithRedundantSlot,
       OptimizeImbalancedLoadADD2) {
    std::vector<int> placement(num_layers_ * num_deploy_experts_, -1);
    for (int layer = 0; layer < num_layers_; ++layer) {
        int offset = layer * num_deploy_experts_;
        placement[offset + 0] = 0;  // Rank 0, expert 0
        placement[offset + 1] = 1;  // Rank 0, expert 1
        placement[offset + 2] = 0;  // Rank 0, expert 0 (redundant)
        placement[offset + 3] = 2;  // Rank 1, expert 2
        placement[offset + 4] = 3;  // Rank 1, expert 3
        placement[offset + 5] = -1; // Rank 1, empty slot
    }
    auto activations = GenerateImbalancedActivations(num_deploy_experts_);
    auto instructions =
        balancer_->optimize_and_generate_instructions(placement, activations);

    // Expected: Two instructions per layer
    // First instruction: REMOVE, remove redundant expert 0 from Rank 0
    // (position 2) Second instruction: ADD, add expert 0 to empty slot on Rank
    // 1 (position 5)
    ASSERT_EQ(instructions.size(), 4); // Two layers, two instructions per layer

    // Verify instructions per layer
    for (int layer = 0; layer < num_layers_; ++layer) {
        // Find instructions for the current layer
        std::vector<ChangeInstruction> layer_instructions;
        for (const auto &instr : instructions) {
            if (instr.layer_idx == layer) {
                layer_instructions.push_back(instr);
            }
        }
        ASSERT_EQ(layer_instructions.size(), 2)
            << "Layer " << layer << " should have exactly two instructions.";

        // Verify first instruction: REMOVE
        EXPECT_EQ(layer_instructions[0].type, OperationType::REMOVE);
        EXPECT_EQ(layer_instructions[0].target_expert_id, 0)
            << "Layer " << layer
            << " first instruction should remove expert 0.";
        EXPECT_EQ(layer_instructions[0].target_global_position, 2)
            << "Layer " << layer
            << " first instruction should target position 2.";

        // Verify second instruction: ADD
        EXPECT_EQ(layer_instructions[1].type, OperationType::ADD);
        EXPECT_EQ(layer_instructions[1].source_expert_id, 0)
            << "Layer " << layer << " second instruction should add expert 0.";
        EXPECT_EQ(layer_instructions[1].target_expert_id, -1)
            << "Layer " << layer
            << " second instruction should target an empty slot.";
        EXPECT_EQ(layer_instructions[1].target_global_position, 5)
            << "Layer " << layer
            << " second instruction should target position 5.";
    }
}

// Verify if optimize_placement correctly generates the final placement under
// imbalanced load.
TEST_F(GreedyExpertLoadBalancerTestWithRedundantSlot,
       OptimizePlacementImbalancedLoad) {
    // 1. Prepare the initial placement and imbalanced activations.
    auto initial_placement = GenerateSimplePlacement();
    auto activations = GenerateImbalancedActivations(num_deploy_experts_);

    // 2. Call the function under test.
    auto optimized_placement =
        balancer_->optimize_placement(initial_placement, activations);

    // 3. Determine the expected final placement.
    // Based on the logic from the OptimizeImbalancedLoadAdd test, we expect the
    // hottest expert (expert 0) to be duplicated (ADDed) to an empty slot to
    // balance the load. The optimal choice is to copy it to the less loaded
    // Rank 1. Initial placement (Layer 0): [0, 1, -1, 2, 3, -1] (Rank0: pos
    // 0,1,2; Rank1: pos 3,4,5) The empty slot on Rank 1 is at global
    // position 5. Therefore, we expect expert 0 to be added to position 5 in
    // Layer 0 and position 11 in Layer 1.
    std::vector<int> expected_placement = initial_placement;

    // Layer 0: Expect expert 0 to be added to global position 5.
    int layer0_target_pos = 5;
    expected_placement[layer0_target_pos] = 0; // source_expert_id is 0

    // Layer 1: Expect expert 0 to be added to global position 11 (5 +
    // num_deploy_experts_).
    int layer1_target_pos = 5 + num_deploy_experts_;
    expected_placement[layer1_target_pos] = 0; // source_expert_id is 0

    // 4. Verify the result.
    ASSERT_EQ(optimized_placement.size(), expected_placement.size());
    for (size_t i = 0; i < optimized_placement.size(); ++i) {
        EXPECT_EQ(optimized_placement[i], expected_placement[i])
            << "Optimized placement mismatch at index " << i;
    }
}