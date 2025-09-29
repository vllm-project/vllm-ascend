// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>
#include "placement_manager.h"
#include <acl/acl.h>
// Mock classes for dependencies
// class MockClusterActivation {
// public:
//     MOCK_METHOD0(get_status, int()); // Example method, adjust as needed
// };

class MockPlacementMapping {
public:
    MOCK_METHOD0(get_num_layers, int());
    MOCK_METHOD0(get_num_experts, int());
    MOCK_METHOD0(get_num_deploy_experts, int());
    MOCK_METHOD2(change_to_default_pos_id, void(int, int));
    MOCK_METHOD3(change_pos_id, void(int, int, int));
    MOCK_METHOD0(update, void());
};

class MockMoEWeights : public MoEWeights {
public:
    MockMoEWeights(size_t num_experts) : MoEWeights(num_experts) {} // 调用基类构造函数
    MOCK_METHOD0(is_shm_weights_ok, bool());
    MOCK_METHOD3(placement, void(int, int, int));
};

class MockPlacementOptimizer {
public:
    MOCK_METHOD1(optimize, std::tuple<int, int, int>(int));
};

// 读取 .txt 文件并恢复三维数组
#include <iostream>
#include <fstream>

// 返回一个连续内存的数组指针，并通过引用参数返回维度
int32_t* load3DArrayFromFile(const std::string& filename, std::vector<int64_t> shape) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "无法打开文件：" << filename << std::endl;
        return nullptr;  // 返回空指针表示失败
    }

    // 读取 shape
    file >> shape[0] >> shape[1] >> shape[2];

    // 计算总元素数量
    int totalSize = shape[0] * shape[1] * shape[2];
    if (totalSize <= 0) {
        std::cerr << "无效的数组维度：" << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
        return nullptr;
    }

    // 分配连续内存
    int32_t* data = new int32_t[totalSize];
    if (!data) {
        std::cerr << "内存分配失败！" << std::endl;
        return nullptr;
    }

    // 读取所有数据到连续内存
    for (int i = 0; i < totalSize; ++i) {
        if (!(file >> data[i])) {
            std::cerr << "数据读取失败，文件中的数据量不足！" << std::endl;
            delete[] data;  // 释放已分配的内存
            return nullptr;
        }
    }

    // 检查是否有多余数据
    int extra;
    if (file >> extra) {
        std::cerr << "警告：文件中包含多余数据！" << std::endl;
        // 可以选择是否因此失败
    }

    file.close();
    return data;
}

// 示例使用和释放函数
void free3DArray(int* array) {
    delete[] array;
}

// Test fixture for Placement class
class PlacementTest : public ::testing::Test {
protected:
    void SetUp() override {

        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        world_size = 16;
        experts_per_layer = 17;
        num_layers = 58;
        num_devices = 16;
        num_expert = 256;
        num_device_per_host = 8;
        num_deploy_experts = 272;

        size_t length = 58*272;
        size_t element_size = sizeof(int64_t);
        size_t size = length*element_size;
        std::string name = "int8_tensor";

        expert_mapping_shape = {num_layers, num_expert};
        placement_pattern_shap = {world_size, num_layers, num_expert};
        std::string filename = "placement_pattern_3d_v3_indevrrori0328_16devices_58moe_SC_TZNB.txt";
        placement_pattern_ptr = load3DArrayFromFile(filename, placement_pattern_shap);

        void* data_ptr;
        ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        Tensor t(data_ptr, length, element_size, name);

        activations_ = new ClusterActivation(t, 58, 272, 10, 16, 0);
        mapping_ = new MockPlacementMapping();
        moe_weight_ = new MockMoEWeights(num_expert);
        optimizer_ = new MockPlacementOptimizer();


        // Default expectations
        ON_CALL(*mapping_, get_num_layers()).WillByDefault(testing::Return(2));
        ON_CALL(*mapping_, get_num_experts()).WillByDefault(testing::Return(4));
        ON_CALL(*mapping_, get_num_deploy_experts()).WillByDefault(testing::Return(8));
    }

    void TearDown() override {
        delete activations_;
        delete mapping_;
        // delete placement_pattern_ptr;
        delete moe_weight_;
        delete optimizer_;
    }
    int world_size;
    int experts_per_layer;
    int num_layers;
    int num_devices;
    int num_expert;
    int num_device_per_host;
    int num_deploy_experts;
    int32_t * expert_mapping_ptr;
    std::vector<int64_t> expert_mapping_shape;

    int32_t * placement_pattern_ptr;
    std::vector<int64_t> placement_pattern_shap;

    ClusterActivation* activations_;
    MockPlacementMapping* mapping_;
    MockMoEWeights* moe_weight_;
    MockPlacementOptimizer* optimizer_;
};

// Test Placement constructor and initialization
TEST_F(PlacementTest, ConstructorNormalCase) {

    // Placement(int rank, int world_size, int num_devices_per_host, ClusterActivation* activation,
    //     size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
    //     size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);

    EXPECT_EQ(placement.get_rank(), 0);
    EXPECT_EQ(placement.get_world_size(), world_size);
    EXPECT_EQ(placement.get_num_devices_per_host(), num_device_per_host);
    EXPECT_EQ(placement.get_num_layers(), num_layers);
    EXPECT_EQ(placement.get_num_experts(), num_expert);
    EXPECT_EQ(placement.get_num_deploy_experts(), num_deploy_experts);
    EXPECT_EQ(placement.get_num_deploy_experts_per_rank(), num_deploy_experts / world_size);
}

// Test boundary case: world_size = 1
TEST_F(PlacementTest, ConstructorSingleRank) {
    int world_size_1 = 1;
    Placement placement(0, world_size_1, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_EQ(placement.get_world_size(), world_size_1);
    EXPECT_EQ(placement.get_num_deploy_experts_per_rank(), num_deploy_experts / world_size_1);
}

// Test edge case: num_devices_per_host = 0 (invalid)
TEST_F(PlacementTest, ConstructorInvalidDevicesPerHost) {
    int num_devices_per_host_0 = 0;
    EXPECT_THROW(Placement(0, world_size, num_devices_per_host_0, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 1), std::exception);
}

// Test check_shm_weights and thread start
TEST_F(PlacementTest, CheckShmWeightsSuccess) {
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    //EXPECT_CALL(*moe_weight_, is_shm_weights_ok());
        //.WillOnce(testing::Return(true));
    ON_CALL(*moe_weight_, is_shm_weights_ok()).WillByDefault(testing::Return(false));
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Allow thread to run
    // No assertion, just ensure no crash
}
/*
// Test placement_manager_per_layer normal case
TEST_F(PlacementTest, PlacementManagerPerLayerNormal) {
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*optimizer_, optimize(0))
        .WillOnce(testing::Return(std::make_tuple(1, 2, 3))); // src=1, dst=2, old=3
    EXPECT_CALL(*mapping_, change_to_default_pos_id(0, 3));
    EXPECT_CALL(*moe_weight_, placement(0, 1, 0)); // local_pos = 2 % 2
    EXPECT_CALL(*mapping_, change_pos_id(0, 1, 2));
    EXPECT_CALL(*mapping_, update());

    placement.placement_manager_per_layer(0);
}

// Test placement_manager_per_layer with boundary position
TEST_F(PlacementTest, PlacementManagerPerLayerBoundary) {
    // std::string filename = "placement_test.txt";
    // auto placement_pattern_ptr = load3DArrayFromFile(filename, 16, 58, 256);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*optimizer_, optimize(1))
        .WillOnce(testing::Return(std::make_tuple(0, 7, 2))); // Last position
    EXPECT_CALL(*mapping_, change_to_default_pos_id(1, 2));
    EXPECT_CALL(*moe_weight_, placement(1, 0, 1)); // local_pos = 7 % 2
    EXPECT_CALL(*mapping_, change_pos_id(1, 0, 7));
    EXPECT_CALL(*mapping_, update());

    placement.placement_manager_per_layer(1);
}
/*
// Test placement_manager full cycle
TEST_F(PlacementTest, PlacementManagerFullCycle) {
    // std::string filename = "placement_test.txt";
    // auto placement_pattern_ptr = load3DArrayFromFile(filename, 16, 58, 256);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*moe_weight_, is_shm_weights_ok()).WillOnce(testing::Return(true));
    EXPECT_CALL(*optimizer_, optimize(0)).WillOnce(testing::Return(std::make_tuple(1, 2, 3)));
    EXPECT_CALL(*optimizer_, optimize(1)).WillOnce(testing::Return(std::make_tuple(0, 7, 2)));
    EXPECT_CALL(*mapping_, change_to_default_pos_id(testing::_, testing::_)).Times(2);
    EXPECT_CALL(*moe_weight_, placement(testing::_, testing::_, testing::_)).Times(2);
    EXPECT_CALL(*mapping_, change_pos_id(testing::_, testing::_, testing::_)).Times(2);
    EXPECT_CALL(*mapping_, update()).Times(2);

    placement.start_thread();
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Allow one cycle
    placement.stop_thread();
}

// Test edge case: num_layers_ = 0
TEST_F(PlacementTest, PlacementManagerNoLayers) {
    // std::string filename = "placement_test.txt";
    // auto placement_pattern_ptr = load3DArrayFromFile(filename, 16, 58, 256);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*mapping_, get_num_layers()).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*moe_weight_, is_shm_weights_ok()).WillOnce(testing::Return(true));

    placement.start_thread();
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Should do nothing
    placement.stop_thread();
    // No calls to placement_manager_per_layer expected
}

// Test start_thread and stop_thread
TEST_F(PlacementTest, StartStopThread) {
    // std::string filename = "placement_test.txt";
    // auto placement_pattern_ptr = load3DArrayFromFile(filename, 16, 58, 256);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*moe_weight_, is_shm_weights_ok()).WillOnce(testing::Return(true));

    placement.start_thread();
    EXPECT_TRUE(placement.get_worker_thread().joinable());
    placement.stop_thread();
    EXPECT_FALSE(placement.get_worker_thread().joinable());
}

// Test edge case: double start_thread
TEST_F(PlacementTest, DoubleStartThread) {
    // std::string filename = "placement_test.txt";
    // auto placement_pattern_ptr = load3DArrayFromFile(filename, 16, 58, 256);
    Placement placement(0, world_size, num_device_per_host, activations_, 0, expert_mapping_shape, 3, reinterpret_cast<size_t>(placement_pattern_ptr), placement_pattern_shap, 3);
    EXPECT_CALL(*moe_weight_, is_shm_weights_ok()).WillOnce(testing::Return(true));

    placement.start_thread();
    auto first_thread_id = placement.get_worker_thread().get_id();
    placement.start_thread(); // Should not start a new thread
    EXPECT_EQ(placement.get_worker_thread().get_id(), first_thread_id);
    placement.stop_thread();
}
*/
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    aclInit(NULL); // 初始化 ACL
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtSetCurrentContext(context);
    return RUN_ALL_TESTS();
}