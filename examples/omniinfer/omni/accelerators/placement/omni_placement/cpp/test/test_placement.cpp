// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

// test_placement.cpp
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include "placement.h"  // Contains MoEWeights, Weight, and WeightPtr declarations

// Test fixture for MoEWeights
class MoEWeightsConstructorTest : public ::testing::Test {
protected:
    void SetUp() override {
        shm_unlink("moe_thread_control");
        shm_unlink("moe_activations");
        shm_unlink("/moe_shm");
    }

    void TearDown() override {
        shm_unlink("moe_thread_control");
        shm_unlink("moe_activations");
        shm_unlink("/moe_shm");
    }

    std::vector<std::vector<std::vector<Weight>>> createSampleCPUWeights(
        int num_layers, int experts_per_layer, int weights_per_expert) {
        std::vector<std::vector<std::vector<Weight>>> npu_weights(num_layers);
        float* dummy_data = new float[10]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        for (int l = 0; l < num_layers; ++l) {
            npu_weights[l].resize(experts_per_layer);
            for (int e = 0; e < experts_per_layer; ++e) {
                npu_weights[l][e].resize(weights_per_expert);
                for (int w = 0; w < weights_per_expert; ++w) {
                    npu_weights[l][e][w] = Weight(e, dummy_data, 10, sizeof(float),
                                                  "weight_" + std::to_string(l) + "_" +
                                                  std::to_string(e) + "_" + std::to_string(w));
                }
            }
        }
        return npu_weights;
    }

    std::vector<std::vector<std::vector<WeightPtr>>> createSampleNPUWeightPtrs(
        int num_layers, int experts_per_layer, int weights_per_expert) {
        std::vector<std::vector<std::vector<WeightPtr>>> npu_weight_ptrs(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            npu_weight_ptrs[l].resize(experts_per_layer);
            for (int e = 0; e < experts_per_layer; ++e) {
                npu_weight_ptrs[l][e].resize(weights_per_expert);
                for (int w = 0; w < weights_per_expert; ++w) {
                    npu_weight_ptrs[l][e][w] = WeightPtr(nullptr,
                                                         "npu_weight_" + std::to_string(l) + "_" +
                                                         std::to_string(e) + "_" + std::to_string(w));
                }
            }
        }
        return npu_weight_ptrs;
    }
};

// Test successful construction
TEST_F(MoEWeightsConstructorTest, ConstructorSuccess) {
    auto npu_weights = createSampleCPUWeights(2, 3, 4);
    auto npu_weight_ptrs = createSampleNPUWeightPtrs(2, 3, 4);

    MoEWeights moe(272);
    ASSERT_NO_THROW({
        moe.init_weights(npu_weights);
    });

    EXPECT_EQ(moe.get_npu_weights().size(), 2);
}

// Test empty CPU weights vector
TEST_F(MoEWeightsConstructorTest, ConstructorEmptyCPUWeights) {
    std::vector<std::vector<std::vector<Weight>>> empty_npu_weights;
    auto npu_weight_ptrs = createSampleNPUWeightPtrs(1, 1, 1);

    MoEWeights moe(272);
    ASSERT_DEATH({
        moe.init_weights(empty_npu_weights);
    }, ".*num_layers_ > 0.*");
}

// Test empty experts per layer
// TEST_F(MoEWeightsConstructorTest, ConstructorEmptyExperts) {
//     std::vector<std::vector<std::vector<Weight>>> npu_weights(1);
//     auto npu_weight_ptrs = createSampleNPUWeightPtrs(1, 1, 1);

//     MoEWeights moe(272);
//     ASSERT_DEATH({
//         moe.init_weights(npu_weights);
//     }, ".*npu_weights_[0].size() > 0.*");
// }

// Test shared memory initialization
TEST_F(MoEWeightsConstructorTest, ConstructorSharedMemoryInit) {
    auto npu_weights = createSampleCPUWeights(1, 1, 1);
    auto npu_weight_ptrs = createSampleNPUWeightPtrs(1, 1, 1);

    MoEWeights moe(272);
    moe.init_weights(npu_weights);

    int fd = shm_open("moe_thread_control", O_RDONLY, 0666);
    EXPECT_EQ(fd, -1);
    close(fd);
}

/*
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/
