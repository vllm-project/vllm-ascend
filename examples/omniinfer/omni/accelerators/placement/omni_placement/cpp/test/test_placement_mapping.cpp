// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "placement_mapping.h"
#include "tensor.h"
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// This fixture tests the default behavior where enable_rank_round_robin is
// false.
class PlacementMappingTest : public ::testing::Test {
  protected:
    // Test parameters
    const int rank_ = 0;
    const int world_size_ = 2;
    const int num_layers_ = 1;
    const int num_experts_ = 4;
    const int num_deploy_experts_per_rank_ = 3; // 2 experts + 1 redundant slot
    const int num_deploy_experts_ = num_deploy_experts_per_rank_ * world_size_;
    const int num_devices_per_host_ = 2;
    const int max_redundant_per_expert_ = 2;
    const bool enable_rank_round_robin_ = false; // Test default path
    const std::string pattern_filename_ = "./test_placement_pattern_simple.txt";

    PlacementMapping *pm_ = nullptr;
    void *selector_ptr_hbm_ = nullptr;
    void *redundant_nums_ptr_hbm_ = nullptr;

    // Helper function to create the pattern file
    void CreatePatternFile(const std::string &filename, int ws, int nl, int ne,
                           const std::vector<std::string> &patterns) {
        std::ofstream pattern_file(filename);
        if (!pattern_file.is_open()) {
            throw std::runtime_error("Failed to create test pattern file.");
        }
        pattern_file << ws << " " << nl << " " << ne << std::endl;
        for (const auto &line : patterns) {
            pattern_file << line << std::endl;
        }
        pattern_file.close();
    }

    void SetUp() override {
        aclInit(nullptr);
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        // Dynamically create the pattern file for the test
        CreatePatternFile(pattern_filename_, world_size_, num_layers_,
                          num_experts_, {"1 1 0 0", "0 0 1 1"});

        // Allocate memory for selector
        size_t selector_len =
            num_layers_ * max_redundant_per_expert_ * num_experts_;
        aclrtMalloc(&selector_ptr_hbm_, selector_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        // Allocate memory for the new redundant_nums tensor
        size_t redundant_nums_len = num_layers_ * num_experts_;
        aclrtMalloc(&redundant_nums_ptr_hbm_,
                    redundant_nums_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        // Constructor call updated to match the new interface
        pm_ = new PlacementMapping(
            pattern_filename_, rank_, num_devices_per_host_,
            max_redundant_per_expert_, num_deploy_experts_,
            0,  // placement_pattern_ptr (use file instead)
            {}, // pattern_shape (use file instead)
            (size_t)selector_ptr_hbm_, enable_rank_round_robin_,
            (size_t)redundant_nums_ptr_hbm_);
    }

    void TearDown() override {
        delete pm_;
        pm_ = nullptr;

        aclrtFree(selector_ptr_hbm_);
        selector_ptr_hbm_ = nullptr;
        aclrtFree(redundant_nums_ptr_hbm_);
        redundant_nums_ptr_hbm_ = nullptr;

        remove(pattern_filename_.c_str());

        aclrtResetDevice(0);
        aclFinalize();
    }
};

// Test constructor and basic getters
TEST_F(PlacementMappingTest, ConstructorAndBasicGetters) {
    ASSERT_NE(pm_, nullptr);
    EXPECT_EQ(pm_->get_rank(), rank_);
    EXPECT_EQ(pm_->get_world_size(), world_size_);
    EXPECT_EQ(pm_->get_num_layers(), num_layers_);
    EXPECT_EQ(pm_->get_num_experts(), num_experts_);
    EXPECT_EQ(pm_->get_num_deploy_experts(), num_deploy_experts_);
    EXPECT_EQ(pm_->get_num_devices_per_host(), num_devices_per_host_);
    EXPECT_EQ(pm_->get_max_redundant_per_expert(), max_redundant_per_expert_);
}

// Test get_position_expert_id - Verified to be correct with new implementation
TEST_F(PlacementMappingTest, GetPositionExpertId) {
    EXPECT_EQ(pm_->get_position_expert_id(0, 0), 0);
    EXPECT_EQ(pm_->get_position_expert_id(0, 1), 1);
    EXPECT_EQ(pm_->get_position_expert_id(0, 2), -1);
    EXPECT_EQ(pm_->get_position_expert_id(0, 3), 2);
    EXPECT_EQ(pm_->get_position_expert_id(0, 4), 3);
    EXPECT_EQ(pm_->get_position_expert_id(0, 5), -1);
}

// Test mapping completeness - Verified to be correct
TEST_F(PlacementMappingTest, MappingCompleteness) {
    std::unordered_set<int32_t> seen_experts;
    for (int pos = 0; pos < num_deploy_experts_; ++pos) {
        int32_t expert_id = pm_->get_position_expert_id(0, pos);
        if (expert_id != -1) {
            seen_experts.insert(expert_id);
        }
    }
    EXPECT_EQ(seen_experts.size(), num_experts_)
        << "Not all experts were found in the mapping.";
}

// Test get_redundant_count - Verified to be correct
TEST_F(PlacementMappingTest, GetRedundantCount) {
    for (int expert_id = 0; expert_id < num_experts_; ++expert_id) {
        EXPECT_EQ(pm_->get_redundant_count(0, expert_id), 1);
    }
}

// Test invalid input handling - Verified to be correct
TEST_F(PlacementMappingTest, InvalidInputHandling) {
    EXPECT_THROW(pm_->get_position_expert_id(-1, 0), std::out_of_range);
    EXPECT_THROW(pm_->get_position_expert_id(num_layers_, 0),
                 std::out_of_range);
    EXPECT_THROW(pm_->get_position_expert_id(0, -1), std::out_of_range);
    EXPECT_THROW(pm_->get_position_expert_id(0, num_deploy_experts_),
                 std::out_of_range);

    EXPECT_THROW(pm_->get_redundant_count(-1, 0), std::out_of_range);
    EXPECT_THROW(pm_->get_redundant_count(num_layers_, 0), std::out_of_range);
    EXPECT_THROW(pm_->get_redundant_count(0, -1), std::out_of_range);
    EXPECT_THROW(pm_->get_redundant_count(0, num_experts_), std::out_of_range);
}

// Test update_selector (for enable_rank_round_robin = false)
TEST_F(PlacementMappingTest, UpdateSelector) {
    std::vector<bool> all_layers_update(num_layers_, true);
    pm_->update_selector(all_layers_update);

    size_t selector_len =
        num_layers_ * max_redundant_per_expert_ * num_experts_;
    std::vector<int32_t> selector_host(selector_len);
    aclrtMemcpy(selector_host.data(), selector_len * sizeof(int32_t),
                pm_->get_selector().get_data_ptr(),
                selector_len * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    EXPECT_EQ(selector_host[0 * max_redundant_per_expert_ + 0], 0);
    EXPECT_EQ(selector_host[1 * max_redundant_per_expert_ + 0], 1);
    EXPECT_EQ(selector_host[2 * max_redundant_per_expert_ + 0], 3);
    EXPECT_EQ(selector_host[3 * max_redundant_per_expert_ + 0], 4);
}

// Test init_selector (which calls update_selector)
TEST_F(PlacementMappingTest, InitSelector) {
    auto selector = pm_->get_selector();
    EXPECT_EQ(selector.get_length(),
              num_layers_ * max_redundant_per_expert_ * num_experts_);

    std::vector<int32_t> selector_host(selector.get_length());
    aclrtMemcpy(selector_host.data(), selector_host.size() * sizeof(int32_t),
                selector.get_data_ptr(), selector_host.size() * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    EXPECT_EQ(selector_host[0 * max_redundant_per_expert_ + 0], 0);
    EXPECT_EQ(selector_host[1 * max_redundant_per_expert_ + 0], 1);
    EXPECT_EQ(selector_host[2 * max_redundant_per_expert_ + 0], 3);
    EXPECT_EQ(selector_host[3 * max_redundant_per_expert_ + 0], 4);
}

// checkPositionIsConsistency - Verified to be correct
TEST_F(PlacementMappingTest, CheckPositionIsConsistency) {
    EXPECT_TRUE(pm_->checkPositionIsConsistency(0, 0, 0));
    EXPECT_FALSE(pm_->checkPositionIsConsistency(0, 0, 1));
    EXPECT_TRUE(pm_->checkPositionIsConsistency(0, 2, -1));
}

// checkUpdateIsValied - Verified to be correct
TEST_F(PlacementMappingTest, CheckUpdateIsValied) {
    EXPECT_TRUE(pm_->checkUpdateIsValied(0, 0, 1));
    EXPECT_EQ(pm_->get_redundant_count(0, 0), 2);
    EXPECT_TRUE(pm_->checkUpdateIsValied(0, 0, -2));
    EXPECT_EQ(pm_->get_redundant_count(0, 0), 0);
    EXPECT_THROW(pm_->checkUpdateIsValied(0, -1, 1), std::runtime_error);
}

// New fixture for testing complex placement patterns
class PlacementMappingComplexTest : public PlacementMappingTest {
  protected:
    const std::string pattern_filename_ =
        "./test_placement_pattern_complex.txt";

    void SetUp() override {
        aclInit(nullptr);
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        CreatePatternFile(pattern_filename_, world_size_, num_layers_,
                          num_experts_, {"1 1 1 0", "0 1 1 1"});

        size_t selector_len =
            num_layers_ * max_redundant_per_expert_ * num_experts_;
        aclrtMalloc(&selector_ptr_hbm_, selector_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        size_t redundant_nums_len = num_layers_ * num_experts_;
        aclrtMalloc(&redundant_nums_ptr_hbm_,
                    redundant_nums_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        pm_ = new PlacementMapping(
            pattern_filename_, rank_, num_devices_per_host_,
            max_redundant_per_expert_, num_deploy_experts_, 0, {},
            (size_t)selector_ptr_hbm_, enable_rank_round_robin_,
            (size_t)redundant_nums_ptr_hbm_);
    }

    // ***** FIX: Added an explicit TearDown method for this class *****
    void TearDown() override {
        delete pm_;
        pm_ = nullptr;

        aclrtFree(selector_ptr_hbm_);
        selector_ptr_hbm_ = nullptr;
        aclrtFree(redundant_nums_ptr_hbm_);
        redundant_nums_ptr_hbm_ = nullptr;

        // This now correctly refers to the filename defined in this class
        remove(pattern_filename_.c_str());

        aclrtResetDevice(0);
        aclFinalize();
    }
};

// Test redundancy counts for a more complex pattern.
TEST_F(PlacementMappingComplexTest, GetRedundantCountComplex) {
    EXPECT_EQ(pm_->get_redundant_count(0, 0), 1);
    EXPECT_EQ(pm_->get_redundant_count(0, 1), 2);
    EXPECT_EQ(pm_->get_redundant_count(0, 2), 2);
    EXPECT_EQ(pm_->get_redundant_count(0, 3), 1);
}

// New fixture for redundancy tests
class PlacementMappingRedundancyTest : public ::testing::Test {
  protected:
    const int rank_ = 0;
    const int world_size_ = 2;
    const int num_layers_ = 2;
    const int num_experts_ = 4;
    const int num_deploy_experts_per_rank_ = 4;
    const int num_deploy_experts_ = num_deploy_experts_per_rank_ * world_size_;
    const int num_devices_per_host_ = 8;
    const int max_redundant_per_expert_ = 2;
    const bool enable_rank_round_robin_ = false;
    const std::string pattern_filename_ =
        "./test_placement_pattern_redundancy.txt";

    PlacementMapping *pm_ = nullptr;
    void *selector_ptr_hbm_ = nullptr;
    void *redundant_nums_ptr_hbm_ = nullptr;

    void CreatePatternFile(const std::string &filename, int ws, int nl, int ne,
                           const std::vector<std::string> &patterns) {
        std::ofstream pattern_file(filename);
        pattern_file << ws << " " << nl << " " << ne << std::endl;
        for (const auto &line : patterns) {
            pattern_file << line << std::endl;
        }
        pattern_file.close();
    }

    void SetUp() override {
        aclInit(nullptr);
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        CreatePatternFile(pattern_filename_, world_size_, num_layers_,
                          num_experts_,
                          {
                              "1 0 1 1", // Rank 0 Layer 0
                              "0 1 1 1", // Rank 0 Layer 1
                              "0 1 1 1", // Rank 1 Layer 0
                              "1 0 1 1"  // Rank 1 Layer 1
                          });

        size_t selector_len =
            num_layers_ * max_redundant_per_expert_ * num_experts_;
        aclrtMalloc(&selector_ptr_hbm_, selector_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        size_t redundant_nums_len = num_layers_ * num_experts_;
        aclrtMalloc(&redundant_nums_ptr_hbm_,
                    redundant_nums_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        pm_ = new PlacementMapping(
            pattern_filename_, rank_, num_devices_per_host_,
            max_redundant_per_expert_, num_deploy_experts_, 0, {},
            (size_t)selector_ptr_hbm_, enable_rank_round_robin_,
            (size_t)redundant_nums_ptr_hbm_);
    }

    void TearDown() override {
        delete pm_;
        aclrtFree(selector_ptr_hbm_);
        aclrtFree(redundant_nums_ptr_hbm_);
        remove(pattern_filename_.c_str());
        aclrtResetDevice(0);
        aclFinalize();
    }
};

// Test redundancy mapping construction.
TEST_F(PlacementMappingRedundancyTest,
       ConstructPerRedundancyEpidMappingToPosition) {
    // Layer 0 verification
    EXPECT_EQ(pm_->get_position_expert_id(0, 0), 0);
    EXPECT_EQ(pm_->get_position_expert_id(0, 1), 2);
    EXPECT_EQ(pm_->get_position_expert_id(0, 2), 3);
    EXPECT_EQ(pm_->get_position_expert_id(0, 3), -1);
    EXPECT_EQ(pm_->get_position_expert_id(0, 4), 1);
    EXPECT_EQ(pm_->get_position_expert_id(0, 5), 2);
    EXPECT_EQ(pm_->get_position_expert_id(0, 6), 3);
    EXPECT_EQ(pm_->get_position_expert_id(0, 7), -1);

    // Layer 1 verification
    EXPECT_EQ(pm_->get_position_expert_id(1, 0), 1);
    EXPECT_EQ(pm_->get_position_expert_id(1, 1), 2);
    EXPECT_EQ(pm_->get_position_expert_id(1, 2), 3);
    EXPECT_EQ(pm_->get_position_expert_id(1, 3), -1);
    EXPECT_EQ(pm_->get_position_expert_id(1, 4), 0);
    EXPECT_EQ(pm_->get_position_expert_id(1, 5), 2);
    EXPECT_EQ(pm_->get_position_expert_id(1, 6), 3);
    EXPECT_EQ(pm_->get_position_expert_id(1, 7), -1);

    // Verify redundancy counts for Layer 0
    EXPECT_EQ(pm_->get_redundant_count(0, 0), 1);
    EXPECT_EQ(pm_->get_redundant_count(0, 1), 1);
    EXPECT_EQ(pm_->get_redundant_count(0, 2), 2);
    EXPECT_EQ(pm_->get_redundant_count(0, 3), 2);

    // Verify redundancy counts for Layer 1
    EXPECT_EQ(pm_->get_redundant_count(1, 0), 1);
    EXPECT_EQ(pm_->get_redundant_count(1, 1), 1);
    EXPECT_EQ(pm_->get_redundant_count(1, 2), 2);
    EXPECT_EQ(pm_->get_redundant_count(1, 3), 2);
}

// New fixture to test the Rank Round Robin selector logic
class PlacementMappingRRTest : public PlacementMappingComplexTest {
  protected:
    // Inherits from ComplexTest to use its pattern, but overrides the RR flag
    const bool enable_rank_round_robin_ = true;

    void SetUp() override {
        // We need to call the base class's setup but with a modified RR flag.
        aclInit(nullptr);
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        CreatePatternFile(pattern_filename_, world_size_, num_layers_,
                          num_experts_, {"1 1 1 0", "0 1 1 1"});

        // For RR logic, selector size is num_layers * num_experts
        size_t selector_len = num_layers_ * num_experts_;
        aclrtMalloc(&selector_ptr_hbm_, selector_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        size_t redundant_nums_len = num_layers_ * num_experts_;
        aclrtMalloc(&redundant_nums_ptr_hbm_,
                    redundant_nums_len * sizeof(int32_t),
                    ACL_MEM_MALLOC_HUGE_FIRST);

        // Call constructor with enable_rank_round_robin = true
        pm_ = new PlacementMapping(
            pattern_filename_, rank_, num_devices_per_host_,
            max_redundant_per_expert_, num_deploy_experts_, 0, {},
            (size_t)selector_ptr_hbm_, enable_rank_round_robin_,
            (size_t)redundant_nums_ptr_hbm_);
    }
    // No TearDown needed here, it will inherit the corrected one from
    // PlacementMappingComplexTest
};

// Test the update_selector logic when enable_rank_round_robin is true
TEST_F(PlacementMappingRRTest, UpdateSelectorWithRR) {
    auto selector = pm_->get_selector();
    std::vector<int32_t> selector_host(selector.get_length());
    aclrtMemcpy(selector_host.data(), selector_host.size() * sizeof(int32_t),
                selector.get_data_ptr(), selector_host.size() * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    EXPECT_EQ(selector.get_length(), num_layers_ * num_experts_);
    EXPECT_EQ(selector_host[0], 0); // Expert 0 -> Pos 0
    EXPECT_EQ(selector_host[1], 1); // Expert 1 -> Pos 1 (local rank preference)
    EXPECT_EQ(selector_host[2], 2); // Expert 2 -> Pos 2 (local rank preference)
    EXPECT_EQ(selector_host[3], 5); // Expert 3 -> Pos 5
}