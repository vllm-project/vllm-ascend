// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include <stdexcept>
#include <memory>
#include <vector>
#include <fstream>
#include <cstdint>
#include <acl/acl.h>
#include <numeric>
#include "placement_optimizer.h"

/** @typedef memcpy_fun_t
 *  @brief Function pointer type for memory copy operations compatible with ACL runtime.
 */
typedef aclError (*memcpy_fun_t)(void*, size_t, const void*, size_t, aclrtMemcpyKind);

/** @brief Retrieves the current memory copy function pointer. */
memcpy_fun_t get_memcpy_fun();

/** @brief Sets a new memory copy function to be used. */
void set_memcpy_fun(memcpy_fun_t fun);

/** @brief Builds an expert-to-position mapping from a position pattern. */
std::vector<std::vector<int>> BuildEpMappingFromPP(
    const std::vector<std::vector<std::vector<int>>>& Position_pattern,
    const std::vector<std::vector<std::vector<int>>>& Position_pattern_unique);

/** @brief Constructs an expert mapping array with a frozen placement setting. */
int32_t* BuildExpertMappingWithFrozenSetting(
    const std::vector<std::vector<std::vector<int>>>& Position_pattern,
    int device_rank,
    int num_device_per_host);

/** @brief A no-op memory copy function that mimics ACL memcpy behavior. */
aclError my_memcpy_no_op_1(void* dst, size_t destMax, const void* src,
    size_t count, aclrtMemcpyKind kind) {
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}

/** @brief Loads a 3D array from a text file. */
int32_t* load3DArrayFromFileTZ(const std::string& filename, int64_t shape[3]) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return nullptr;
    }
    file >> shape[0] >> shape[1] >> shape[2];
    int totalSize = shape[0] * shape[1] * shape[2];
    if (totalSize <= 0) {
        std::cerr << "Invalid array dimensions: " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
        return nullptr;
    }
    int32_t* data = new int32_t[totalSize];
    if (!data) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return nullptr;
    }
    for (int i = 0; i < totalSize; ++i) {
        if (!(file >> data[i])) {
            std::cerr << "Data reading failed, insufficient data in file!" << std::endl;
            delete[] data;
            return nullptr;
        }
    }
    int extra;
    if (file >> extra) {
        std::cerr << "Warning: File contains extra data!" << std::endl;
    }
    file.close();
    return data;
}

/** @brief Constant representing the scalar type for integers. */
const int ScalarType_Int = 3;

/** @class PlacementOptimizerTest
 *  @brief Test fixture for testing PlacementOptimizer with large pattern.
 */
class PlacementOptimizerTest : public ::testing::Test {
protected:
    static PlacementMapping* shared_placement_mapping;
    static ClusterActivation* shared_cluster_activation;
    static int32_t* shared_placement_pattern_ptr;
    static int64_t cached_placement_pattern_shape[3];
    static int32_t* shared_expert_mapping_ptr;
    static int64_t* shared_mock_data_ptr;
    static bool is_initialized;
    PlacementOptimizer* opt = nullptr;
    memcpy_fun_t old_fun = nullptr;

    int rank = 0;
    int num_devices_per_host = 8;
    int64_t expert_mapping_shape[2] = {58, 256};
    int64_t mock_data_shape[2] = {58, 272};

    void SetUp() override {
        // old_fun = get_memcpy_fun();
        // set_memcpy_fun(&my_memcpy_no_op_1);

        if (!is_initialized) {
            // Load and cache placement pattern
            std::string filename = "./test_data/placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.txt";
            shared_placement_pattern_ptr = load3DArrayFromFileTZ(filename, cached_placement_pattern_shape);
            if (!shared_placement_pattern_ptr) {
                FAIL() << "Failed to load placement pattern from file.";
            }

            // Convert to 3D vector for expert mapping
            std::vector<std::vector<std::vector<int>>> placement_pattern;
            placement_pattern.resize(cached_placement_pattern_shape[0]);
            int index = 0;
            for (int i = 0; i < cached_placement_pattern_shape[0]; ++i) {
                placement_pattern[i].resize(cached_placement_pattern_shape[1]);
                for (int j = 0; j < cached_placement_pattern_shape[1]; ++j) {
                    placement_pattern[i][j].resize(cached_placement_pattern_shape[2]);
                    for (int k = 0; k < cached_placement_pattern_shape[2]; ++k) {
                        placement_pattern[i][j][k] = shared_placement_pattern_ptr[index++];
                    }
                }
            }

            // Initialize expert mapping
            shared_expert_mapping_ptr = BuildExpertMappingWithFrozenSetting(placement_pattern, 0, 8);

            // Initialize mock data
            int num_layers = 58, num_positions = 272;
            shared_mock_data_ptr = new int64_t[num_layers * num_positions];
            for (size_t i = 0; i < num_layers; ++i) {
                for (size_t j = 0; j < num_positions; ++j) {
                    size_t idx = i * num_positions + j;
                    if (i == 0) {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(j + 1);
                    } else if (i == 1) {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(1000 - j);
                    } else if (i == 2) {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(j < 10 ? 100 : 1);
                    } else if (i == 3) {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(j < 10 ? 100 : 50);
                    } else if (i == 4) {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(j < 10 ? 50 : 100);
                    } else {
                        shared_mock_data_ptr[idx] = static_cast<int64_t>(j + 1);
                    }
                }
            }

            // Create PlacementMapping
            shared_placement_mapping = new PlacementMapping(
                rank, num_devices_per_host,
                shared_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                shared_expert_mapping_ptr, expert_mapping_shape, ScalarType_Int,
                shared_placement_pattern_ptr, cached_placement_pattern_shape, ScalarType_Int
            );

            // Create ClusterActivation
            shared_cluster_activation = new ClusterActivation(
                Tensor(shared_mock_data_ptr, num_layers * num_positions, sizeof(int64_t), "int64_tensor"),
                num_layers, num_positions, 12, 16, 0
            );

            // Synchronize and collect activation data
            aclError ret = shared_cluster_activation->get_npu_count().to_host(shared_cluster_activation->get_total_count_ptr());
            if (ret != ACL_ERROR_NONE) {
                FAIL() << "to_host failed: " << ret;
            }
            int64_t* total_count_ptr = static_cast<int64_t*>(shared_cluster_activation->get_total_count_ptr());
            int64_t* last_count_ptr = static_cast<int64_t*>(shared_cluster_activation->get_last_count_ptr());
            for (size_t layer = 0; layer < num_layers; ++layer) {
                for (size_t expert = 0; expert < num_positions; ++expert) {
                    size_t idx = layer * num_positions + expert;
                    int64_t count = total_count_ptr[idx] - last_count_ptr[idx];
                    if (count > 0) {
                        shared_cluster_activation->collect_activation(layer, expert, count);
                        last_count_ptr[idx] = total_count_ptr[idx];
                    }
                }
            }

            is_initialized = true;
        }

        opt = new PlacementOptimizer(shared_placement_mapping, shared_cluster_activation);
    }

    void TearDown() override {
        set_memcpy_fun(old_fun);
        delete opt;
    }

    static void TearDownTestSuite() {
        delete shared_placement_mapping;
        delete shared_cluster_activation;
        delete[] shared_placement_pattern_ptr;
        delete[] shared_expert_mapping_ptr;
        delete[] shared_mock_data_ptr;
        shared_placement_mapping = nullptr;
        shared_cluster_activation = nullptr;
        shared_placement_pattern_ptr = nullptr;
        shared_expert_mapping_ptr = nullptr;
        shared_mock_data_ptr = nullptr;
        is_initialized = false;
    }
};

PlacementMapping* PlacementOptimizerTest::shared_placement_mapping = nullptr;
ClusterActivation* PlacementOptimizerTest::shared_cluster_activation = nullptr;
int32_t* PlacementOptimizerTest::shared_placement_pattern_ptr = nullptr;
int64_t PlacementOptimizerTest::cached_placement_pattern_shape[3];
int32_t* PlacementOptimizerTest::shared_expert_mapping_ptr = nullptr;
int64_t* PlacementOptimizerTest::shared_mock_data_ptr = nullptr;
bool PlacementOptimizerTest::is_initialized = false;

TEST_F(PlacementOptimizerTest, ConstructorThrowsOnNullPlacementMapping) {
    EXPECT_THROW(PlacementOptimizer(nullptr, shared_cluster_activation), std::runtime_error);
}

TEST_F(PlacementOptimizerTest, ConstructorThrowsOnNullClusterActivation) {
    EXPECT_THROW(PlacementOptimizer(shared_placement_mapping, nullptr), std::runtime_error);
}

TEST_F(PlacementOptimizerTest, InitializationSetsNumLayers) {
    EXPECT_EQ(opt->get_num_layers(), shared_placement_mapping->get_num_layers());
}

TEST_F(PlacementOptimizerTest, InitializationSetsWorldSize) {
    EXPECT_EQ(opt->get_world_size(), shared_placement_mapping->get_world_size());
}

TEST_F(PlacementOptimizerTest, InitializationSetsNumDeployExperts) {
    EXPECT_EQ(opt->get_num_deploy_experts(), shared_placement_mapping->get_num_deploy_experts());
}

TEST_F(PlacementOptimizerTest, InitializationSetsNumExperts) {
    EXPECT_EQ(opt->get_num_experts(), shared_placement_mapping->get_num_experts());
}

TEST_F(PlacementOptimizerTest, InitializationSetsRank) {
    EXPECT_EQ(opt->get_rank(), shared_placement_mapping->get_rank());
}

TEST_F(PlacementOptimizerTest, InitializationSetsNumDevicesPerHost) {
    EXPECT_EQ(opt->get_num_devices_per_host(), shared_placement_mapping->get_num_devices_per_host());
}

TEST_F(PlacementOptimizerTest, InitializationCalculatesNumDeployExpertsPerDevice) {
    int expected = (shared_placement_mapping->get_num_deploy_experts() + shared_placement_mapping->get_world_size() - 1) /
                   shared_placement_mapping->get_world_size();
    EXPECT_EQ(opt->get_num_deploy_experts_per_device(), expected);
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusThrowsOnNegativeLayer) {
    EXPECT_THROW(opt->get_layer_freq_status(-1), std::out_of_range);
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusThrowsOnLayerBeyondMax) {
    EXPECT_THROW(opt->get_layer_freq_status(opt->get_num_layers()), std::out_of_range);
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusReturnsCorrectVectorSize) {
    auto layer_freq = opt->get_layer_freq_status(0);
    EXPECT_EQ(layer_freq.size(), static_cast<size_t>(opt->get_world_size() * opt->get_num_deploy_experts_per_device()));
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusReturnsNonNegativeFrequenciesForDeployedExperts) {
    auto layer_freq = opt->get_layer_freq_status(0);
    for (int pos = 0; pos < opt->get_num_deploy_experts(); ++pos) {
        EXPECT_GE(layer_freq[pos], 0);
    }
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusReturnsZeroForUnusedPositions) {
    auto layer_freq = opt->get_layer_freq_status(0);
    int positions = opt->get_world_size() * opt->get_num_deploy_experts_per_device();
    for (int pos = opt->get_num_deploy_experts(); pos < positions; ++pos) {
        EXPECT_EQ(layer_freq[pos], 0);
    }
}

TEST_F(PlacementOptimizerTest, GetLayerFreqStatusMatchesClusterActivationForFirstLayer) {
    auto layer_freq = opt->get_layer_freq_status(0);
    for (int pos = 0; pos < opt->get_num_deploy_experts(); ++pos) {
        EXPECT_EQ(layer_freq[pos], shared_cluster_activation->getClusterTotalActivationCount(0, pos));
    }
}

TEST_F(PlacementOptimizerTest, GetHostDeviceParamsSetsCorrectCurrentHost) {
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    EXPECT_EQ(current_host, opt->get_rank() / opt->get_num_devices_per_host());
}

TEST_F(PlacementOptimizerTest, GetHostDeviceParamsSetsCorrectPositionsPerHost) {
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    int expected_positions = opt->get_num_deploy_experts_per_device() * opt->get_num_devices_per_host();
    EXPECT_EQ(positions_per_host, expected_positions);
}

TEST_F(PlacementOptimizerTest, GetHostDeviceParamsSetsCorrectHostStartPosition) {
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    int expected_positions = opt->get_num_deploy_experts_per_device() * opt->get_num_devices_per_host();
    EXPECT_EQ(host_start, current_host * expected_positions);
}

TEST_F(PlacementOptimizerTest, GetHostDeviceParamsSetsCorrectHostEndPosition) {
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    EXPECT_EQ(host_end, std::min(host_start + positions_per_host, opt->get_num_deploy_experts()));
}

TEST_F(PlacementOptimizerTest, GetDeviceLoadsComputesCorrectLoadsForFirstLayer) {
    auto layer_freq = opt->get_layer_freq_status(0);
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    opt->get_device_loads(layer_freq, host_start, device_loads);
    for (int i = 0; i < opt->get_num_devices_per_host(); ++i) {
        int start = host_start + i * opt->get_num_deploy_experts_per_device();
        int end = std::min(start + opt->get_num_deploy_experts_per_device(), static_cast<int>(layer_freq.size()));
        int expected = std::accumulate(layer_freq.begin() + start, layer_freq.begin() + end, 0);
        EXPECT_EQ(device_loads[i], expected);
    }
}

TEST_F(PlacementOptimizerTest, GetDeviceLoadsThrowsOnEmptyFrequencyVector) {
    std::vector<int64_t> empty_freq;
    std::vector<int64_t> device_loads(opt->get_num_devices_per_host(), 0);
    EXPECT_THROW(opt->get_device_loads(empty_freq, 0, device_loads), std::invalid_argument);
}

TEST_F(PlacementOptimizerTest, GetRedundantPositionsReturnsValidPositions) {
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    auto redundant_positions = opt->get_redundant_positions(0);
    for (int pos : redundant_positions) {
        EXPECT_GE(pos, 0);
        EXPECT_LT(pos, opt->get_num_deploy_experts());
    }
}

TEST_F(PlacementOptimizerTest, GetSortedRedundantLoadsSortsInAscendingOrder) {
    auto layer_freq = opt->get_layer_freq_status(0);
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    auto redundant_positions = opt->get_redundant_positions(0);
    auto sorted_loads = opt->get_sorted_redundant_loads(redundant_positions, layer_freq);
    for (size_t i = 1; i < sorted_loads.size(); ++i) {
        EXPECT_LE(sorted_loads[i - 1].second, sorted_loads[i].second);
    }
}

TEST_F(PlacementOptimizerTest, GetReplacementReturnsValidReplacement) {
    auto layer_freq = opt->get_layer_freq_status(0);
    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;
    opt->get_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);
    auto redundant_positions = opt->get_redundant_positions(0);
    auto sorted_loads = opt->get_sorted_redundant_loads(redundant_positions, layer_freq);
    auto [source_ep, target_pos, target_ep] = opt->get_replacement(sorted_loads, redundant_positions, 0, layer_freq, host_start);
    if (source_ep != -1) {
        EXPECT_GE(target_pos, host_start);
        EXPECT_LT(target_pos, host_end);
    }
}

TEST_F(PlacementOptimizerTest, GetRedundantEpReplacementReturnsValidValuesForLayerZero) {
    auto layer_freq = opt->get_layer_freq_status(0);
    auto [source_ep, target_pos, target_ep] = opt->get_redundant_ep_replacement(0, layer_freq);
    if (source_ep != -1) {
        EXPECT_GE(source_ep, 0);
        EXPECT_LT(source_ep, opt->get_num_experts());
        EXPECT_GE(target_pos, 0);
        EXPECT_LT(target_pos, opt->get_num_deploy_experts());
    }
}

TEST_F(PlacementOptimizerTest, Optimize) {
    // 测试异常情况
    EXPECT_THROW(opt->optimize(-1), std::out_of_range);
    EXPECT_THROW(opt->optimize(opt->get_num_layers()), std::out_of_range);

    // 测试多层优化结果
    struct ExpectedResult {
        int layer;
        int source_ep;
        int target_pos;
        int target_ep;
    };
    std::vector<ExpectedResult> expected = {
        {0, 127, 16, 42},
        {1, 0, 119, 8},
        {2, 0, 33, 55},
        {3, 0, 33, 98},
        {4, 16, 16, 48},
        {5, 127, 16, 76}
    };

    for (const auto& exp : expected) {
        auto [source_ep, target_pos, target_ep] = opt->optimize(exp.layer);
        if (source_ep != -1) {
            EXPECT_GE(source_ep, 0) << "Layer " << exp.layer;
            EXPECT_LT(source_ep, opt->get_num_experts()) << "Layer " << exp.layer;
            EXPECT_GE(target_pos, 0) << "Layer " << exp.layer;
            EXPECT_LT(target_pos, opt->get_num_deploy_experts()) << "Layer " << exp.layer;
            EXPECT_EQ(source_ep, exp.source_ep) << "Layer " << exp.layer;
            EXPECT_EQ(target_pos, exp.target_pos) << "Layer " << exp.layer;
            EXPECT_EQ(target_ep, exp.target_ep) << "Layer " << exp.layer;
        } else {
            std::cout << "No optimization needed for layer " << exp.layer << std::endl;
        }
    }
}

/** @class PlacementOptimizerTestWithSmallPattern
 *  @brief Test fixture for testing PlacementOptimizer with small pattern.
 */
class PlacementOptimizerTestWithSmallPattern : public ::testing::Test {
protected:
    static PlacementMapping* shared_placement_mapping_small;
    static ClusterActivation* shared_cluster_activation_small;
    static int32_t* shared_placement_pattern_ptr_small;
    static int64_t cached_placement_pattern_shape_small[3];
    static int32_t* shared_expert_mapping_ptr_small;
    static int64_t* shared_mock_data_ptr_small;
    static bool is_initialized_small;
    PlacementOptimizer* opt1 = nullptr;
    memcpy_fun_t old_fun = nullptr;

    int rank = 0;
    int num_devices_per_host = 4;
    int64_t expert_mapping_shape[2] = {20, 8};
    int64_t mock_data_shape[2] = {20, 12};

    void SetUp() override {
        old_fun = get_memcpy_fun();
        set_memcpy_fun(&my_memcpy_no_op_1);

        if (!is_initialized_small) {
            // Load and cache placement pattern
            std::string filename = "./test_data/placement_pattern_test_for_optimizer.txt";
            shared_placement_pattern_ptr_small = load3DArrayFromFileTZ(filename, cached_placement_pattern_shape_small);
            if (!shared_placement_pattern_ptr_small) {
                FAIL() << "Failed to load placement pattern from file.";
            }

            // Convert to 3D vector for expert mapping
            std::vector<std::vector<std::vector<int>>> placement_pattern;
            placement_pattern.resize(cached_placement_pattern_shape_small[0]);
            int index = 0;
            for (int i = 0; i < cached_placement_pattern_shape_small[0]; ++i) {
                placement_pattern[i].resize(cached_placement_pattern_shape_small[1]);
                for (int j = 0; j < cached_placement_pattern_shape_small[1]; ++j) {
                    placement_pattern[i][j].resize(cached_placement_pattern_shape_small[2]);
                    for (int k = 0; k < cached_placement_pattern_shape_small[2]; ++k) {
                        placement_pattern[i][j][k] = shared_placement_pattern_ptr_small[index++];
                    }
                }
            }

            // Initialize expert mapping
            shared_expert_mapping_ptr_small = BuildExpertMappingWithFrozenSetting(placement_pattern, 0, 4);

            // Initialize mock data
            int num_layers = 20, num_positions = 12;
            shared_mock_data_ptr_small = new int64_t[num_layers * num_positions];
            for (size_t i = 0; i < num_layers; ++i) {
                for (size_t j = 0; j < num_positions; ++j) {
                    size_t idx = i * num_positions + j;
                    if (i == 0) {
                        shared_mock_data_ptr_small[idx] = static_cast<int64_t>(j + 1);
                    } else if (i == 1) {
                        shared_mock_data_ptr_small[idx] = static_cast<int64_t>(1000 - j);
                    } else {
                        shared_mock_data_ptr_small[idx] = static_cast<int64_t>(j + 1);
                    }
                }
            }

            // Create PlacementMapping
            shared_placement_mapping_small = new PlacementMapping(
                rank, num_devices_per_host,
                shared_expert_mapping_ptr_small, expert_mapping_shape, ScalarType_Int,
                shared_expert_mapping_ptr_small, expert_mapping_shape, ScalarType_Int,
                shared_placement_pattern_ptr_small, cached_placement_pattern_shape_small, ScalarType_Int
            );

            // Create ClusterActivation
            shared_cluster_activation_small = new ClusterActivation(
                Tensor(shared_mock_data_ptr_small, num_layers * num_positions, sizeof(int64_t), "int64_tensor"),
                num_layers, num_positions, 12, 4, 0
            );

            // Synchronize and collect activation data
            aclError ret = shared_cluster_activation_small->get_npu_count().to_host(shared_cluster_activation_small->get_total_count_ptr());
            if (ret != ACL_ERROR_NONE) {
                FAIL() << "to_host failed: " << ret;
            }
            int64_t* total_count_ptr = static_cast<int64_t*>(shared_cluster_activation_small->get_total_count_ptr());
            int64_t* last_count_ptr = static_cast<int64_t*>(shared_cluster_activation_small->get_last_count_ptr());
            for (size_t layer = 0; layer < num_layers; ++layer) {
                for (size_t expert = 0; expert < num_positions; ++expert) {
                    size_t idx = layer * num_positions + expert;
                    int64_t count = total_count_ptr[idx] - last_count_ptr[idx];
                    if (count > 0) {
                        shared_cluster_activation_small->collect_activation(layer, expert, count);
                        last_count_ptr[idx] = total_count_ptr[idx];
                    }
                }
            }

            is_initialized_small = true;
        }

        opt1 = new PlacementOptimizer(shared_placement_mapping_small, shared_cluster_activation_small);
    }

    void TearDown() override {
        set_memcpy_fun(old_fun);
        delete opt1;
    }

    static void TearDownTestSuite() {
        delete shared_placement_mapping_small;
        delete shared_cluster_activation_small;
        delete[] shared_placement_pattern_ptr_small;
        delete[] shared_expert_mapping_ptr_small;
        delete[] shared_mock_data_ptr_small;
        shared_placement_mapping_small = nullptr;
        shared_cluster_activation_small = nullptr;
        shared_placement_pattern_ptr_small = nullptr;
        shared_expert_mapping_ptr_small = nullptr;
        shared_mock_data_ptr_small = nullptr;
        is_initialized_small = false;
    }
};

PlacementMapping* PlacementOptimizerTestWithSmallPattern::shared_placement_mapping_small = nullptr;
ClusterActivation* PlacementOptimizerTestWithSmallPattern::shared_cluster_activation_small = nullptr;
int32_t* PlacementOptimizerTestWithSmallPattern::shared_placement_pattern_ptr_small = nullptr;
int64_t PlacementOptimizerTestWithSmallPattern::cached_placement_pattern_shape_small[3];
int32_t* PlacementOptimizerTestWithSmallPattern::shared_expert_mapping_ptr_small = nullptr;
int64_t* PlacementOptimizerTestWithSmallPattern::shared_mock_data_ptr_small = nullptr;
bool PlacementOptimizerTestWithSmallPattern::is_initialized_small = false;

TEST_F(PlacementOptimizerTestWithSmallPattern, OptimizeReturnsValidValuesForSmallPattern) {
    struct ExpectedResult {
        int layer;
        int source_ep;
        int target_pos;
        int target_ep;
    };
    std::vector<ExpectedResult> expected = {
        {0, 6, 2, 5},
        {1, 1, 9, 2}
    };

    for (const auto& exp : expected) {
        auto [source_ep, target_pos, target_ep] = opt1->optimize(exp.layer);
        if (source_ep != -1) {
            EXPECT_GE(source_ep, 0) << "Layer " << exp.layer;
            EXPECT_LT(source_ep, opt1->get_num_experts()) << "Layer " << exp.layer;
            EXPECT_GE(target_pos, 0) << "Layer " << exp.layer;
            EXPECT_LT(target_pos, opt1->get_num_deploy_experts()) << "Layer " << exp.layer;
            EXPECT_EQ(source_ep, exp.source_ep) << "Layer " << exp.layer;
            EXPECT_EQ(target_pos, exp.target_pos) << "Layer " << exp.layer;
            EXPECT_EQ(target_ep, exp.target_ep) << "Layer " << exp.layer;
        } else {
            std::cout << "No optimization needed for layer " << exp.layer << std::endl;
        }
    }
}

TEST_F(PlacementOptimizerTestWithSmallPattern, OptimizeThrowsOnNegativeLayer) {
    EXPECT_THROW(opt1->optimize(-1), std::out_of_range);
}

TEST_F(PlacementOptimizerTestWithSmallPattern, OptimizeThrowsOnLayerBeyondMax) {
    EXPECT_THROW(opt1->optimize(opt1->get_num_layers()), std::out_of_range);
}

// BuildEpMappingFromPP 实现（保持原样）
std::vector<std::vector<int>> BuildEpMappingFromPP(
    const std::vector<std::vector<std::vector<int>>>& Position_pattern,
    const std::vector<std::vector<std::vector<int>>>& Position_pattern_unique) {
    int num_devices = Position_pattern_unique.size();
    int num_layers = num_devices > 0 ? Position_pattern_unique[0].size() : 0;
    int num_epids = (num_layers > 0 && num_devices > 0) ? Position_pattern_unique[0][0].size() : 0;

    int sum_positions = 0;
    for (int i = 0; i < num_devices; ++i) {
        for (int k = 0; k < num_epids; ++k) {
            sum_positions += Position_pattern[i][0][k];
        }
    }
    int positions_per_device = static_cast<int>(sum_positions / num_devices + 0.5);

    std::vector<std::vector<int>> expert_position(num_layers, std::vector<int>(num_epids, -1));

    for (int i = 0; i < num_devices; ++i) {
        for (int j = 0; j < num_layers; ++j) {
            std::vector<int> cumsum_expert(num_epids, 0);
            int sum = 0;
            for (int k = 0; k < num_epids; ++k) {
                sum += Position_pattern[i][j][k];
                cumsum_expert[k] = sum;
            }
            for (int k = 0; k < num_epids; ++k) {
                if (Position_pattern_unique[i][j][k] == 1) {
                    expert_position[j][k] = i * positions_per_device + cumsum_expert[k] - 1;
                }
            }
        }
    }
    return expert_position;
}

// BuildExpertMappingWithFrozenSetting 实现（保持原样）
int32_t* BuildExpertMappingWithFrozenSetting(
    const std::vector<std::vector<std::vector<int>>>& Position_pattern,
    int device_rank,
    int num_device_per_host) {
    int num_devices = Position_pattern.size();
    int num_layers = num_devices > 0 ? Position_pattern[0].size() : 0;
    int num_epids = (num_layers > 0 && num_devices > 0) ? Position_pattern[0][0].size() : 0;

    std::vector<std::vector<std::vector<int>>> Position_pattern_C(
        num_devices,
        std::vector<std::vector<int>>(
            num_layers,
            std::vector<int>(num_epids, 0)
        )
    );

    for (int j = 0; j < num_layers; ++j) {
        std::vector<std::vector<int>> expert_locations(num_epids);
        for (int epid = 0; epid < num_epids; ++epid) {
            for (int i = 0; i < num_devices; ++i) {
                if (Position_pattern[i][j][epid] == 1) {
                    expert_locations[epid].push_back(i);
                }
            }
        }
        for (int epid = 0; epid < num_epids; ++epid) {
            const auto& devices = expert_locations[epid];
            if (devices.size() == 1) {
                Position_pattern_C[devices[0]][j][epid] = 1;
            } else if (devices.size() >= 2) {
                bool this_group_has_it = false;
                for (int device_id : devices) {
                    if ((device_rank / num_device_per_host) == (device_id / num_device_per_host)) {
                        this_group_has_it = true;
                        break;
                    }
                }
                for (int device_id : devices) {
                    if (this_group_has_it) {
                        if (epid / (num_epids / num_devices) != device_id) {
                            Position_pattern_C[device_id][j][epid] = 1;
                        }
                    } else {
                        if (epid / (num_epids / num_devices) == device_id) {
                            Position_pattern_C[device_id][j][epid] = 1;
                        }
                    }
                }
            }
        }
    }
    std::vector<std::vector<int>> ExpertMapping_C = BuildEpMappingFromPP(Position_pattern, Position_pattern_C);
    int32_t* result = new int32_t[num_layers * num_epids];
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < num_epids; ++j) {
            result[i * num_epids + j] = ExpertMapping_C[i][j];
        }
    }
    return result;
}