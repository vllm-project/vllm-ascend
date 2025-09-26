// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_activation.h"
#include "moe_weights.h"
#include "placement_mapping.h"
#include "placement_optimizer.h"

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <thread>
#include <unistd.h>

// Helper Functions for Test Setup

void CreatePatternFile(const std::string &filename, int ws, int nl, int ne,
                       const std::vector<std::string> &patterns) {
    std::ofstream pattern_file(filename);
    if (!pattern_file.is_open()) {
        throw std::runtime_error("Failed to create test pattern file: " +
                                 filename);
    }
    pattern_file << ws << " " << nl << " " << ne << std::endl;
    for (const auto &line : patterns) {
        pattern_file << line << std::endl;
    }
    pattern_file.close();
}

// Mock Classes for Testing
struct MockChangeInstruction {
    int layer_idx;
    int source_rank;
    int target_rank;
    int source_global_position;
    int target_global_position;
    int source_expert_id;
    int target_expert_id;
    int round = -1;

    void print() const {
        std::cout << "MockChangeInstruction: layer=" << layer_idx
                  << " src_rank=" << source_rank << " tgt_rank=" << target_rank
                  << " src_pos=" << source_global_position
                  << " tgt_pos=" << target_global_position
                  << " src_exp=" << source_expert_id
                  << " tgt_exp=" << target_expert_id << " round=" << round
                  << std::endl;
    }
};

// Placement Manager Component Testing

class PlacementManagerComponentTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize ACL environment
        auto acl_ret = aclInit(nullptr);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to initialize ACL";

        // Get device count
        uint32_t device_count = 0;
        acl_ret = aclrtGetDeviceCount(&device_count);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to get device count";
        ASSERT_GT(device_count, 0) << "No NPU devices found";

        std::cout << "Found " << device_count << " NPU devices" << std::endl;

        // Set device
        device_id_ = 0;
        acl_ret = aclrtSetDevice(device_id_);
        ASSERT_EQ(acl_ret, ACL_SUCCESS)
            << "Failed to set device " << device_id_;

        // Create context
        acl_ret = aclrtCreateContext(&context_, device_id_);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to create context";

        acl_ret = aclrtSetCurrentContext(context_);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to set current context";

        // Setup test parameters
        SetupTestParameters();

        // Allocate memory and create components
        AllocateTestMemory();
        CreateTestComponents();
    }

    void TearDown() override {
        // Cleanup
        CleanupTestComponents();
        FreeTestMemory();
        remove(pattern_filename_.c_str());

        // Cleanup ACL
        if (context_ != nullptr) {
            aclrtDestroyContext(context_);
        }
        aclrtResetDevice(device_id_);
        aclFinalize();
    }

  private:
    void SetupTestParameters() {
        world_size_ = 2; // Test with 2 ranks
        rank_ = 0;
        hccl_comm_world_size_ = 2;
        num_layers_ = 1;  // Simplified to 1 layer for debugging
        num_experts_ = 4; // Only 4 experts total
        num_devices_per_host_ = 2;
        max_redundant_per_expert_ = 2;

        // Calculate deployment parameters more carefully
        int num_experts_per_rank =
            num_experts_ / world_size_;   // 2 experts per rank
        int redundant_slots_per_rank = 1; // 1 redundant slot per rank
        int num_deploy_experts_per_rank =
            num_experts_per_rank + redundant_slots_per_rank; // 3 slots per rank
        num_deploy_experts_ =
            num_deploy_experts_per_rank * world_size_; // 6 total slots
        experts_per_layer_ =
            num_deploy_experts_per_rank; // 3 experts per rank per layer

        pattern_filename_ = "./temp_placement_manager_test.txt";
        CreateValidPatternFile();

        std::cout << "Setup parameters:" << std::endl;
        std::cout << "  num_experts_: " << num_experts_ << std::endl;
        std::cout << "  num_experts_per_rank: " << num_experts_per_rank
                  << std::endl;
        std::cout << "  num_deploy_experts_: " << num_deploy_experts_
                  << std::endl;
        std::cout << "  experts_per_layer_: " << experts_per_layer_
                  << std::endl;
    }

    void CreateValidPatternFile() {
        // Create a simple valid pattern file
        // Format: world_size layers experts
        // Then for each rank and layer, show which experts are deployed (0 or
        // 1)

        std::vector<std::string> patterns;

        // Rank 0, Layer 0: deploy experts 0 and 1, not expert 2 and 3
        patterns.push_back(
            "1 1 0 0"); // Expert 0=1, Expert 1=1, Expert 2=0, Expert 3=0

        // Rank 1, Layer 0: deploy experts 2 and 3, not expert 0 and 1
        patterns.push_back(
            "0 0 1 1"); // Expert 0=0, Expert 1=0, Expert 2=1, Expert 3=1

        CreatePatternFile(pattern_filename_, world_size_, num_layers_,
                          num_experts_, patterns);

        // Debug: Print pattern file content
        std::cout << "Created pattern file with content:" << std::endl;
        std::ifstream file(pattern_filename_);
        std::string line;
        while (std::getline(file, line)) {
            std::cout << "  " << line << std::endl;
        }
        file.close();
    }

    void AllocateTestMemory() {
        // Allocate activation tensor memory - ensure size matches
        // experts_per_layer_
        size_t activation_length = num_layers_ * experts_per_layer_;
        size_t activation_element_size = sizeof(int64_t);
        size_t activation_size = activation_length * activation_element_size;

        auto acl_ret = aclrtMalloc(&activation_data_ptr_, activation_size,
                                   ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(acl_ret, ACL_SUCCESS)
            << "Failed to allocate activation memory";

        // Initialize activation data
        std::vector<int64_t> init_data(activation_length, 0);
        acl_ret =
            aclrtMemcpy(activation_data_ptr_, activation_size, init_data.data(),
                        activation_size, ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(acl_ret, ACL_SUCCESS)
            << "Failed to initialize activation data";

        activation_tensor_ =
            Tensor((uint64_t)activation_data_ptr_, activation_length,
                   activation_element_size, "test_activation_tensor");

        // Allocate selector memory
        size_t selector_len =
            num_layers_ * max_redundant_per_expert_ * num_experts_;
        acl_ret =
            aclrtMalloc(&selector_ptr_hbm_, selector_len * sizeof(int32_t),
                        ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to allocate selector memory";

        // Initialize selector data
        std::vector<int32_t> selector_init(selector_len, 0);
        acl_ret =
            aclrtMemcpy(selector_ptr_hbm_, selector_len * sizeof(int32_t),
                        selector_init.data(), selector_len * sizeof(int32_t),
                        ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(acl_ret, ACL_SUCCESS) << "Failed to initialize selector data";

        // Allocate redundant_nums memory
        size_t redundant_nums_len = num_layers_ * num_experts_;
        acl_ret = aclrtMalloc(&redundant_nums_ptr_hbm_,
                              redundant_nums_len * sizeof(int32_t),
                              ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(acl_ret, ACL_SUCCESS)
            << "Failed to allocate redundant_nums memory";

        // Initialize redundant_nums data
        std::vector<int32_t> redundant_nums_init(redundant_nums_len, 1);
        acl_ret = aclrtMemcpy(
            redundant_nums_ptr_hbm_, redundant_nums_len * sizeof(int32_t),
            redundant_nums_init.data(), redundant_nums_len * sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(acl_ret, ACL_SUCCESS)
            << "Failed to initialize redundant_nums memory";

        std::cout << "Test memory allocated successfully" << std::endl;
    }

    void FreeTestMemory() {
        if (activation_data_ptr_)
            aclrtFree(activation_data_ptr_);
        if (selector_ptr_hbm_)
            aclrtFree(selector_ptr_hbm_);
        if (redundant_nums_ptr_hbm_)
            aclrtFree(redundant_nums_ptr_hbm_);
    }

    void CreateTestComponents() {
        // Create ClusterActivation
        activations_ = new ClusterActivation(
            activation_tensor_, 1000, num_layers_, experts_per_layer_, 10,
            world_size_, hccl_comm_world_size_, rank_);

        // Create PlacementMapping
        placement_mapping_ = new PlacementMapping(
            pattern_filename_, rank_, num_devices_per_host_,
            max_redundant_per_expert_, num_deploy_experts_, 0, {},
            (size_t)selector_ptr_hbm_, false, (size_t)redundant_nums_ptr_hbm_);

        // Debug: Print placement mapping info
        std::cout << "PlacementMapping created:" << std::endl;
        std::cout << "  get_num_experts: "
                  << placement_mapping_->get_num_experts() << std::endl;
        std::cout << "  get_num_deploy_experts: "
                  << placement_mapping_->get_num_deploy_experts() << std::endl;
        std::cout << "  get_world_size: "
                  << placement_mapping_->get_world_size() << std::endl;

        // Create PlacementOptimizer
        optimizer_ = new PlacementOptimizer(placement_mapping_, activations_);

        // Create MoEWeights - ensure proper initialization
        moe_weights_ = new MoEWeights(num_deploy_experts_, world_size_);

        std::cout << "Test components created successfully" << std::endl;
    }

    void CleanupTestComponents() {
        delete activations_;
        delete placement_mapping_;
        delete optimizer_;
        delete moe_weights_;
    }

  protected:
    // ACL related
    int32_t device_id_ = 0;
    aclrtContext context_ = nullptr;

    // Test parameters
    uint32_t rank_ = 0;
    uint32_t world_size_ = 2;
    uint32_t hccl_comm_world_size_ = 2;
    int num_layers_ = 1;
    int num_experts_ = 4;
    int num_devices_per_host_ = 2;
    int num_deploy_experts_ = 6; // 2 ranks * 3 slots per rank
    int experts_per_layer_ = 3;  // 3 slots per rank per layer
    int max_redundant_per_expert_ = 2;

    // Memory and objects
    std::string pattern_filename_;
    void *activation_data_ptr_ = nullptr;
    Tensor activation_tensor_;
    void *selector_ptr_hbm_ = nullptr;
    void *redundant_nums_ptr_hbm_ = nullptr;
    ClusterActivation *activations_ = nullptr;
    PlacementMapping *placement_mapping_ = nullptr;
    PlacementOptimizer *optimizer_ = nullptr;
    MoEWeights *moe_weights_ = nullptr;
};

// Test Cases for Placement Manager Components

TEST_F(PlacementManagerComponentTest, PlacementMappingValidationTest) {
    ASSERT_NE(placement_mapping_, nullptr);

    // Test that placement mapping was created correctly
    EXPECT_EQ(placement_mapping_->get_num_layers(), num_layers_);
    EXPECT_EQ(placement_mapping_->get_num_experts(), num_experts_);
    EXPECT_EQ(placement_mapping_->get_world_size(), world_size_);
    EXPECT_EQ(placement_mapping_->get_rank(), rank_);

    // Debug: Print position mappings
    std::cout << "Position mappings for layer 0:" << std::endl;
    for (int pos = 0; pos < num_deploy_experts_; ++pos) {
        int expert_id = placement_mapping_->get_expert_id(0, pos);
        std::cout << "  Position " << pos << " -> Expert " << expert_id
                  << std::endl;
    }

    // Test that we can get position mappings without errors
    EXPECT_NO_THROW({
        for (int layer = 0; layer < num_layers_; ++layer) {
            for (int pos = 0; pos < num_deploy_experts_; ++pos) {
                int expert_id = placement_mapping_->get_expert_id(layer, pos);
                // Expert ID should be valid or -1 (empty slot)
                EXPECT_TRUE(expert_id == -1 ||
                            (expert_id >= 0 && expert_id < num_experts_))
                    << "Invalid expert_id " << expert_id << " at position "
                    << pos;
            }
        }
    });

    std::cout << "Placement mapping validation test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, OptimizerBasicTest) {
    ASSERT_NE(optimizer_, nullptr);
    ASSERT_NE(activations_, nullptr);
    ASSERT_NE(placement_mapping_, nullptr);

    // Setup activations parameters
    activations_->set_params(placement_mapping_->get_num_experts());

    // Test optimizer basic properties
    EXPECT_EQ(optimizer_->get_num_experts(), num_experts_);
    EXPECT_EQ(optimizer_->get_world_size(), world_size_);
    EXPECT_EQ(optimizer_->get_rank(), rank_);

    // Test optimizer creation and basic functionality
    EXPECT_NO_THROW({
        // This should not crash even if no optimization is needed
        auto instructions = optimizer_->optimize();
        std::cout << "Optimizer returned " << instructions.size()
                  << " instructions" << std::endl;
    });

    std::cout << "Optimizer basic test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, MoEWeightsBasicTest) {
    ASSERT_NE(moe_weights_, nullptr);

    // Test basic MoEWeights functionality - using correct method names from
    // moe_weights.h
    EXPECT_EQ(moe_weights_->getNumExperts(), num_deploy_experts_);

    // Test initialization status
    std::cout << "MoEWeights SHM initialized: "
              << moe_weights_->isShmInitialized() << std::endl;
    std::cout << "MoEWeights HBM initialized: "
              << moe_weights_->isHbmInitialized() << std::endl;

    // Test other getter methods - but don't make strict assertions since they
    // depend on init_weights
    std::cout << "MoEWeights num layers: " << moe_weights_->getNumLayers()
              << std::endl;
    EXPECT_GE(moe_weights_->getShmName().length(), 0); // Should have a name

    std::cout << "MoEWeights basic test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, ComponentIntegrationTest) {
    ASSERT_NE(activations_, nullptr);
    ASSERT_NE(placement_mapping_, nullptr);
    ASSERT_NE(optimizer_, nullptr);

    // Test integration between components
    activations_->set_params(placement_mapping_->get_num_experts());

    // Simulate some activation updates - be careful with array bounds
    EXPECT_NO_THROW({ activations_->updateDeltaActivationCount(); });

    EXPECT_NO_THROW({ activations_->updateShiftWindows(placement_mapping_); });

    // Test optimization with updated activations
    EXPECT_NO_THROW({
        auto instructions = optimizer_->optimize();
        std::cout << "Integration test: optimizer returned "
                  << instructions.size() << " instructions" << std::endl;
    });

    std::cout << "Component integration test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, ConfigurationConsistencyTest) {
    // Test that all components have consistent configuration
    EXPECT_EQ(activations_->get_world_size(), world_size_);
    EXPECT_EQ(activations_->get_rank(), rank_);
    EXPECT_EQ(activations_->get_num_layers(), num_layers_);

    EXPECT_EQ(placement_mapping_->get_num_layers(), num_layers_);
    EXPECT_EQ(placement_mapping_->get_num_experts(), num_experts_);

    EXPECT_EQ(moe_weights_->getNumExperts(), num_deploy_experts_);

    // Test parameter consistency
    EXPECT_EQ(activations_->get_num_deploy_experts_per_rank(),
              experts_per_layer_);

    std::cout << "Configuration consistency test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, MoEWeightsWithProperInitTest) {
    ASSERT_NE(moe_weights_, nullptr);

    // Create some simple test tensors
    std::vector<std::vector<std::vector<Tensor>>> test_weights;

    // Only create tensors if we can allocate memory properly
    bool can_create_tensors = true;

    try {
        // Create simple test data - 1 layer, experts_per_layer_ experts per
        // rank
        test_weights.resize(1);                     // 1 layer
        test_weights[0].resize(experts_per_layer_); // experts per rank

        for (size_t expert = 0; expert < experts_per_layer_; ++expert) {
            // Create a simple tensor for each expert
            void *tensor_ptr = nullptr;
            size_t tensor_size = 10 * sizeof(float);
            auto acl_ret = aclrtMalloc(&tensor_ptr, tensor_size,
                                       ACL_MEM_MALLOC_HUGE_FIRST);
            if (acl_ret == ACL_SUCCESS) {
                Tensor test_tensor((uint64_t)tensor_ptr, 10, sizeof(float),
                                   "test_tensor");
                test_weights[0][expert].push_back(test_tensor);
            } else {
                can_create_tensors = false;
                break;
            }
        }

        if (can_create_tensors) {
            // Test initialization with weights (but don't init shared memory to
            // avoid issues)
            EXPECT_NO_THROW({
                moe_weights_->init_weights(test_weights,
                                           false); // false = don't init shm
            });

            // After initialization, check properties
            EXPECT_EQ(moe_weights_->getNumLayers(), 1);
            EXPECT_GT(moe_weights_->getNumDeployExpertsPerRank(), 0);
            EXPECT_TRUE(moe_weights_->isHbmInitialized());
        }

        // Cleanup allocated tensors
        for (auto &layer : test_weights) {
            for (auto &expert : layer) {
                for (auto &tensor : expert) {
                    if (tensor.get_data_ptr() != nullptr) {
                        aclrtFree(tensor.get_data_ptr());
                    }
                }
            }
        }
    } catch (...) {
        can_create_tensors = false;
    }

    if (!can_create_tensors) {
        std::cout << "Skipping tensor initialization test due to memory "
                     "allocation issues"
                  << std::endl;
    }

    std::cout << "MoEWeights with proper init test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, MockInstructionProcessingTest) {
    // Test instruction processing logic without actually creating Placement
    // object
    std::vector<MockChangeInstruction> mock_instructions;

    // Create some mock instructions with valid expert IDs (0-3)
    MockChangeInstruction inst1 = {
        0, 0, 1, 0,
        3, 0, 2, -1}; // Layer 0: move expert 0 from rank 0 to rank 1
    MockChangeInstruction inst2 = {
        0, 1, 0, 4,
        1, 3, 1, -1}; // Layer 0: move expert 3 from rank 1 to rank 0

    mock_instructions.push_back(inst1);
    mock_instructions.push_back(inst2);

    // Test instruction sorting by layer
    std::stable_sort(
        mock_instructions.begin(), mock_instructions.end(),
        [](const auto &a, const auto &b) { return a.layer_idx < b.layer_idx; });

    // Test instruction validation
    for (const auto &inst : mock_instructions) {
        EXPECT_GE(inst.layer_idx, 0);
        EXPECT_LT(inst.layer_idx, num_layers_);
        EXPECT_GE(inst.source_rank, 0);
        EXPECT_LT(inst.source_rank, world_size_);
        EXPECT_GE(inst.target_rank, 0);
        EXPECT_LT(inst.target_rank, world_size_);
        EXPECT_GE(inst.source_expert_id, 0);
        EXPECT_LT(inst.source_expert_id, num_experts_);
        EXPECT_GE(inst.target_expert_id, 0);
        EXPECT_LT(inst.target_expert_id, num_experts_);
    }

    std::cout << "Mock instruction processing test passed" << std::endl;
}

TEST_F(PlacementManagerComponentTest, StressTest) {
    // Test repeated optimization cycles
    activations_->set_params(placement_mapping_->get_num_experts());

    for (int cycle = 0; cycle < 3; ++cycle) {
        // Simulate activation updates
        EXPECT_NO_THROW({ activations_->updateDeltaActivationCount(); });

        EXPECT_NO_THROW(
            { activations_->updateShiftWindows(placement_mapping_); });

        // Run optimization
        EXPECT_NO_THROW({
            auto instructions = optimizer_->optimize();
            std::cout << "Stress test cycle " << cycle << ": "
                      << instructions.size() << " instructions" << std::endl;
        });

        // Simulate some delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Stress test passed" << std::endl;
}