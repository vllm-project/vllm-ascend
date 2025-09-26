// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "config.h"
#include "distribution.h"
#include "expert_activation.h"
#include "placement_mapping.h"
#include <acl/acl.h>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>

OmniConfig config;

// ============================================================================
// Test fixture for ExpertActivation
// ============================================================================
class ExpertActivationTest : public ::testing::Test {
  protected:
    void SetUp() override { ea = std::make_unique<ExpertActivation>(); }

    std::unique_ptr<ExpertActivation> ea;
};

// Test 1: Verify activation updates accumulate correctly
TEST_F(ExpertActivationTest, SumsActivations) {
    ea->update(1);
    ea->update(2);
    EXPECT_EQ(ea->getTotalValue(), 3); // 1 + 2
}

// Test 2: Verify oldest activation is overwritten when capacity is reached
TEST_F(ExpertActivationTest, EjectsOldestWhenMaxReached) {
    for (int i = 1; i <= 20; ++i) { // length_ = 20
        ea->update(1);
    }
    EXPECT_EQ(ea->getTotalValue(), 20);
    ea->update(2);                      // Overwrites the first '1'
    EXPECT_EQ(ea->getTotalValue(), 21); // 19 * 1 + 2
}

// Test 3: Verify total count correctness
TEST_F(ExpertActivationTest, TotalCountCorrect) {
    ea->update(1);
    ea->update(2);
    ea->update(3);
    EXPECT_EQ(ea->getTotalValue(), 6); // 1 + 2 + 3
    ea->update(4);
    EXPECT_EQ(ea->getTotalValue(), 10); // 1 + 2 + 3 + 4
}

// Test 4: Verify total count when exceeding capacity
TEST_F(ExpertActivationTest, TotalCountExceedCapacityCorrect) {
    for (int i = 0; i < 20; ++i) {
        ea->update(1);
    }
    ea->update(2);                      // Overwrites first '1'
    ea->update(3);                      // Overwrites second '1'
    EXPECT_EQ(ea->getTotalValue(), 23); // 18 * 1 + 2 + 3
}

// Test 5: Verify empty state
TEST_F(ExpertActivationTest, EmptyState) { EXPECT_EQ(ea->getTotalValue(), 0); }

// Test 6: Verify get_last_value works correctly
TEST_F(ExpertActivationTest, GetLastValue) {
    ea->update(10);
    EXPECT_EQ(ea->get_last_value(), 10);
    ea->update(25);
    EXPECT_EQ(ea->get_last_value(), 25);

    // Fill the buffer to test wrap-around
    for (int i = 0; i < 20; ++i) {
        ea->update(i);
    }
    EXPECT_EQ(ea->get_last_value(), 19);

    ea->update(99); // idx becomes 1, last value is at index 0
    EXPECT_EQ(ea->get_last_value(), 99);
}

// ============================================================================
// Test fixture for ClusterActivation
// ============================================================================
class ClusterActivationTest : public ::testing::Test {
  protected:
    // Test parameters
    const int64_t max_activation_count = 10000;
    const size_t num_layers = 2;
    const size_t num_deploy_experts_per_rank = 4;
    const int activation_window_size = 20;
    const size_t world_size = 2;
    const size_t hccl_comm_world_size = 2;
    const size_t rank = 0;

    // For PlacementMapping
    const char *placement_filename = "test_placement_pattern.txt";
    void *selector_ptr = nullptr;
    void *num_redundant_ptr = nullptr;

    int old_activation_quiesce;
    Tensor npu_count_tensor;
    std::unique_ptr<ClusterActivation> ca;
    std::unique_ptr<PlacementMapping> pm;

    void SetUp() override {
        ACLCHECK(aclInit(nullptr));
        ACLCHECK(aclrtSetDevice(0));
        old_activation_quiesce = config.activation_quiesce;
        config.activation_quiesce = 0;

        // Create a placement pattern file for initializing PlacementMapping
        std::ofstream placement_file(placement_filename);
        const int pm_num_experts = 2; // Logical experts for the pattern
        placement_file << world_size << " " << num_layers << " "
                       << pm_num_experts << "\n";
        // R0, L0: deploy E0, E1
        placement_file << "1 1\n";
        // R0, L1: deploy E0
        placement_file << "1 0\n";
        // R1, L0: deploy E0, E1
        placement_file << "1 1\n";
        // R1, L1: deploy E1
        placement_file << "0 1\n";
        placement_file.close();

        // Allocate memory for pointers required by PlacementMapping
        const int max_redundant_per_expert = 4;
        const int max_num_deployed_expert =
            world_size * num_deploy_experts_per_rank;
        size_t selector_size = num_layers * pm_num_experts * sizeof(int32_t);
        size_t redundant_size = num_layers * pm_num_experts * sizeof(int32_t);

        ACLCHECK(aclrtMalloc(&selector_ptr, selector_size,
                             ACL_MEM_MALLOC_HUGE_FIRST));
        ACLCHECK(aclrtMalloc(&num_redundant_ptr, redundant_size,
                             ACL_MEM_MALLOC_HUGE_FIRST));

        ACLCHECK(aclrtMemset(selector_ptr, selector_size, 0, selector_size));
        ACLCHECK(
            aclrtMemset(num_redundant_ptr, redundant_size, 0, redundant_size));

        // Initialize PlacementMapping
        pm = std::make_unique<PlacementMapping>(
            placement_filename, rank, world_size, max_redundant_per_expert,
            max_num_deployed_expert, 0, std::vector<int64_t>{},
            (size_t)selector_ptr, true, (size_t)num_redundant_ptr);

        // Create a valid tensor for the ClusterActivation constructor
        npu_count_tensor =
            CreateTensor(num_layers, num_deploy_experts_per_rank);

        // Default initialization for ClusterActivation
        ca = std::make_unique<ClusterActivation>(
            npu_count_tensor, max_activation_count, num_layers,
            num_deploy_experts_per_rank, activation_window_size, world_size,
            hccl_comm_world_size, rank);
    }

    void TearDown() override {
        config.activation_quiesce = old_activation_quiesce;

        // Free allocated memory
        ACLCHECK(aclrtFree(selector_ptr));
        ACLCHECK(aclrtFree(num_redundant_ptr));
        if (npu_count_tensor.get_data_ptr()) {
            ACLCHECK(aclrtFree(npu_count_tensor.get_data_ptr()));
        }

        // Clean up created files
        remove(placement_filename);
        if (access("./test_dump_dir", F_OK) == 0) {
            rmdir("./test_dump_dir");
        }
        if (access("test_activations.txt", F_OK) == 0) {
            remove("test_activations.txt");
        }

        ACLCHECK(aclrtResetDevice(0));
        ACLCHECK(aclFinalize());
    }

    Tensor CreateTensor(size_t layers, size_t experts_per_rank,
                        std::string name = "test_tensor") {
        size_t element_size = sizeof(int64_t);
        size_t length = layers * experts_per_rank;
        size_t size = length * element_size;
        void *data_ptr = nullptr; // 初始化为 nullptr

        ACLCHECK(aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        ACLCHECK(aclrtMemset(data_ptr, size, 0, size));

        return Tensor(data_ptr, length, element_size, name);
    }
};

// Test constructor validation
TEST_F(ClusterActivationTest, ConstructorValidation) {
    auto dummy_tensor = CreateTensor(1, 1);
    EXPECT_THROW(ClusterActivation(dummy_tensor, 0, 1, 1, 1, 1, 1, 0),
                 std::invalid_argument); // max_activation_count <= 0
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 0, 1, 1, 1, 1, 0),
                 std::invalid_argument); // num_layers == 0
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 1, 0, 1, 1, 1, 0),
                 std::invalid_argument); // num_deploy_experts_per_rank == 0
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 1, 1, 1, 0, 1, 0),
                 std::invalid_argument); // world_size == 0
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 1, 1, 1, 1, 0, 0),
                 std::invalid_argument); // hccl_comm_world_size == 0
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 1, 1, 1, 2, 1, 0),
                 std::invalid_argument); // hccl_comm_world_size < world_size
    EXPECT_THROW(ClusterActivation(dummy_tensor, 100, 1, 1, 1, 2, 2, 2),
                 std::runtime_error); // rank >= hccl_comm_world_size
    ACLCHECK(aclrtFree(dummy_tensor.get_data_ptr()));
}

// Test updateDeltaActivationCount logic
TEST_F(ClusterActivationTest, UpdateDeltaActivationCount) {
    int64_t *last_count = static_cast<int64_t *>(ca->get_last_count_ptr());
    int64_t *deployed_counts =
        static_cast<int64_t *>(ca->get_deployed_experts_counts_host_ptr());

    // --- Scenario 1: Simple increment ---
    last_count[0] = 10;
    deployed_counts[0] = 15; // delta = 5
    last_count[1] = 20;
    deployed_counts[1] = 30; // delta = 10

    ca->updateDeltaActivationCount();

    EXPECT_EQ(ca->getExpertActivationCount(0, 0), 5);  // layer 0, expert 0
    EXPECT_EQ(ca->getExpertActivationCount(0, 1), 10); // layer 0, expert 1

    // --- Scenario 2: Test wraparound ---
    // last_count is now {15, 30}
    last_count = static_cast<int64_t *>(ca->get_last_count_ptr());
    EXPECT_EQ(last_count[0], 15);
    EXPECT_EQ(last_count[1], 30);

    deployed_counts[0] = 5;  // Current (5) < Last (15), so current becomes 5 +
                             // 10000. Delta = 9990
    deployed_counts[1] = 25; // Current (25) < Last (30), so current becomes 25
                             // + 10000. Delta = 9995

    ca->updateDeltaActivationCount();

    EXPECT_EQ(ca->getExpertActivationCount(0, 0), max_activation_count - 10);
    EXPECT_EQ(ca->getExpertActivationCount(0, 1), max_activation_count - 5);
}

// Test collecting activation data from a text file
TEST_F(ClusterActivationTest, CollectFromTxt) {
    const char *filename = "test_activations.txt";
    std::ofstream test_file(filename);
    // layer 0: 4 experts for rank 0, 4 for rank 1
    test_file << "10\t20\t30\t40\t50\t60\t70\t80\n";
    // layer 1: 4 experts for rank 0, 4 for rank 1
    test_file << "11\t21\t31\t41\t51\t61\t71\t81\n";
    test_file.close();

    ca->collect_from_txt(filename);

    // Verify values for layer 0
    EXPECT_EQ(ca->getExpertActivationCount(0, 0), 10); // rank 0, local expert 0
    EXPECT_EQ(ca->getExpertActivationCount(0, 3), 40); // rank 0, local expert 3
    EXPECT_EQ(ca->getExpertActivationCount(0, 4), 50); // rank 1, local expert 0
    EXPECT_EQ(ca->getExpertActivationCount(0, 7), 80); // rank 1, local expert 3

    // Verify values for layer 1
    EXPECT_EQ(ca->getExpertActivationCount(1, 0), 11); // rank 0, local expert 0
    EXPECT_EQ(ca->getExpertActivationCount(1, 4), 51); // rank 1, local expert 0

    remove(filename);
}

// Test dump directory setting
TEST_F(ClusterActivationTest, SetDumpDirAndStopDump) {
    const char *dirname = "./test_dump_dir";

    // Initially, dump is disabled
    EXPECT_FALSE(ca->is_dump_enabled());

    // Set a new directory
    ca->setDumpDir(dirname);
    EXPECT_TRUE(ca->is_dump_enabled());
    EXPECT_EQ(ca->get_dump_dir(), dirname);

    // Check if directory was created
    struct stat info;
    EXPECT_EQ(stat(dirname, &info), 0);
    EXPECT_TRUE(info.st_mode & S_IFDIR);

    // Stop dumping
    ca->stopDump();
    EXPECT_FALSE(ca->is_dump_enabled());

    // Clean up
    rmdir(dirname);
}

// Test sliding window update and logit activation calculation using real
// PlacementMapping
TEST_F(ClusterActivationTest, SlidingWindowAndLogitActivation) {
    const size_t num_logical_experts =
        pm->get_num_experts(); // Should be 2 from file
    ca->set_params(num_logical_experts);

    // --- First update cycle ---
    int64_t *delta_counts =
        static_cast<int64_t *>(ca->get_delta_experts_counts_ptr());
    size_t delta_counts_size =
        num_layers * world_size * num_deploy_experts_per_rank * sizeof(int64_t);
    ACLCHECK(
        aclrtMemset(delta_counts, delta_counts_size, 0, delta_counts_size));

    // Manually set delta counts for layer 0 based on the placement pattern
    // From pattern: L0 mapping is: pos 0->E0, pos 1->E1, pos 4->E0, pos 5->E1
    // Global positions are calculated as: rank * num_deploy_experts_per_rank +
    // local_pos For layer 0: Rank 0 deploys E0 (local_pos 0 -> global_pos 0)
    // and E1 (local_pos 1 -> global_pos 1) Rank 1 deploys E0 (local_pos 0 ->
    // global_pos 4) and E1 (local_pos 1 -> global_pos 5)
    delta_counts[pm->getGlobalPositionOffset(0, 0)] =
        10; // L0, global pos 0 (E0)
    delta_counts[pm->getGlobalPositionOffset(0, 1)] =
        20; // L0, global pos 1 (E1)
    delta_counts[pm->getGlobalPositionOffset(0, 4)] =
        30; // L0, global pos 4 (E0)
    delta_counts[pm->getGlobalPositionOffset(0, 5)] =
        40; // L0, global pos 5 (E1)

    ca->updateShiftWindows(pm.get());

    // Verify activations for layer 0
    // Logical expert 0 activation = 10 (from pos 0) + 30 (from pos 4) = 40
    // Logical expert 1 activation = 20 (from pos 1) + 40 (from pos 5) = 60
    int redundant_count_e0 = pm->get_redundant_count(0, 0); // Should be 2
    int redundant_count_e1 = pm->get_redundant_count(0, 1); // Should be 2

    EXPECT_EQ(redundant_count_e0, 2);
    EXPECT_EQ(redundant_count_e1, 2);

    EXPECT_EQ(ca->getLogitExpertShiftActivateion(0, 0, redundant_count_e0), 5);
    EXPECT_EQ(ca->getLogitExpertShiftActivateion(0, 1, redundant_count_e1), 10);
}