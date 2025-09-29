// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "config.h"
#include "expert_activation.h" // Assuming ExpertActivation is in a header file
#include <dirent.h> // Include for POSIX directory operations
#include <fstream>
#include "distribution.h"


aclError static my_mem_fun(void* dst, size_t destMax, const void* src,
                         size_t count, aclrtMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}

// Test fixture for ExpertActivation
class ExpertActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with max 3 activations and 1-second threshold
        ea = std::make_unique<ExpertActivation>(3, 1.0);
    }

    std::unique_ptr<ExpertActivation> ea;
};

// Helper function to simulate time passing
void sleepFor(double seconds) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

// Test 1: Adding activations within threshold sums them up
TEST_F(ExpertActivationTest, SumsActivationsWithinThreshold) {
    ea->addActivation(1);
    ea->addActivation(2); // Within 1s, should accumulate
    EXPECT_EQ(ea->getTotalActivationCount(), 3); // 1 + 2 pending
    // EXPECT_EQ(ea->activationArray.size(), 0);    // Nothing added to array yet
}

// Test 2: Adding activation after threshold adds to array
TEST_F(ExpertActivationTest, AddsToArrayAfterThreshold) {
    ea->addActivation(1);
    sleepFor(1.5); // Wait > 1s
    ea->addActivation(2);
    EXPECT_EQ(ea->getTotalActivationCount(), 3); // 1 in array, 2 pending
    // EXPECT_EQ(ea->activationArray.size(), 1);    // 1 activation in array
}

// Test 3: Ejects oldest activation when max is reached
TEST_F(ExpertActivationTest, EjectsOldestWhenMaxReached) {
    ea->addActivation(1);  // t=0
    sleepFor(1.5);
    ea->addActivation(2);  // t=1.5
    sleepFor(1.5);
    ea->addActivation(3);  // t=3
    sleepFor(1.5);
    ea->addActivation(4);  // t=4.5, should eject t=0
    EXPECT_EQ(ea->getTotalActivationCount(), 10); // 2 + 3 + 4
    // EXPECT_EQ(ea->activationArray.size(), 3);    // Max size maintained

    // Verify oldest (1) is gone by checking total
    // If 1 were still there, total would be 10, not 9
}

// Test 4: Total count includes pending and array activations
TEST_F(ExpertActivationTest, TotalCountCorrect) {
    ea->addActivation(1);  // t=0
    sleepFor(1.5);
    ea->addActivation(2);  // t=1.5
    ea->addActivation(3);  // t=1.5 + small, sums with 2
    EXPECT_EQ(ea->getTotalActivationCount(), 6); // 1 in array, 2+3 pending
    sleepFor(1.5);
    ea->addActivation(4);  // t=3, adds 2+3 to array
    EXPECT_EQ(ea->getTotalActivationCount(), 10); // 1 + (2+3) + 4
}

// Test 4-1: Total count includes pending and array activations
TEST_F(ExpertActivationTest, TotalCountExceedCapacityCorrect) {
    ea->addActivation(1);  // t=0
    sleepFor(3);
    ea->addActivation(2);  // t=3
    sleepFor(3);
    ea->addActivation(3);  // t=6
    EXPECT_EQ(ea->getTotalActivationCount(), 6); // 1 in array, 2+3 pending
    sleepFor(3);
    ea->addActivation(4);  // t=9
    sleepFor(3);
    ea->addActivation(5);  // t=12
    EXPECT_EQ(ea->getTotalActivationCount(), 14); // 2 + 3 + (4+5)
}

// Test 5: Empty state has zero count
TEST_F(ExpertActivationTest, EmptyState) {
    EXPECT_EQ(ea->getTotalActivationCount(), 0);
    // EXPECT_EQ(ea->activationArray.size(), 0);
}

// 测试fixture，用于设置测试环境
class ClusterActivationTest : public ::testing::Test {
protected:
    int old_activation_quiesce;
    void SetUp() override {
        old_activation_quiesce = config.activation_quiesce;
        config.activation_quiesce=0;
        set_memcpy_fun(&my_mem_fun);
        // 在每个测试用例前执行的初始化代码
    }

    void TearDown() override {
        // 在每个测试用例后执行的清理代码
        config.activation_quiesce = old_activation_quiesce;
    }

    // 辅助函数：创建Tensor对象
    Tensor CreateTensor(size_t num_layers, size_t num_experts, std::string name = "test_tensor") {
        size_t element_size = sizeof(int64_t);
        size_t size = num_layers * num_experts * element_size;
        size_t length = num_layers * num_experts;
        void* data_ptr = malloc(size);
        return Tensor(data_ptr, length, element_size, name);
    }
};

// 测试正常构造情况
TEST_F(ClusterActivationTest, ConstructorNormalCase) {
    // 创建测试数据
    size_t num_layers = 58;
    size_t num_deploy_experts = 272;
    int activation_window_size = 4;
    size_t world_size = 2;
    int rank = 0;

    Tensor npu_count = CreateTensor(num_layers, num_deploy_experts, "int8_tensor");

    // 测试构造函数
    ASSERT_NO_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts,
                           activation_window_size, world_size, rank);

        // 验证成员变量是否正确初始化
        EXPECT_EQ(ca.get_num_layers(), num_layers);
        EXPECT_EQ(ca.get_num_deploy_experts_per_rank(), num_deploy_experts);
    });
}

// Test 6: setDumpDir sets the correct directory and checks its existence
TEST_F(ClusterActivationTest, SetDumpDir) {
    size_t num_layers = 58;
    size_t num_deploy_experts = 272;
    int activation_window_size = 4;
    size_t world_size = 2;
    int rank = 0;
    Tensor npu_count = CreateTensor(num_layers, num_deploy_experts, "int8_tensor");

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts,
                        activation_window_size, world_size, rank);

    std::string test_dir = "./test_dump_dir";
    ca.setDumpDir(test_dir);

    // Check if the directory exists using POSIX function
    struct stat info;
    EXPECT_EQ(stat(test_dir.c_str(), &info), 0);
    EXPECT_TRUE(info.st_mode & S_IFDIR);

    // Clean up the created directory
    rmdir(test_dir.c_str());
}

// 测试npu_count长度不匹配的情况
TEST_F(ClusterActivationTest, ConstructorInvalidNpuCountLength) {
    size_t num_layers = 58;
    size_t num_deploy_experts = 272;
    int activation_window_size = 4;
    size_t world_size = 2;
    int rank = 0;

    // 创建一个长度不匹配的Tensor（故意多一个元素）
    size_t element_size = sizeof(int64_t);
    size_t wrong_length = num_layers * num_deploy_experts + 1;
    size_t size = wrong_length * element_size;
    void* data_ptr = malloc(size);
    Tensor npu_count(data_ptr, wrong_length, element_size, "wrong_tensor");

    // 验证是否抛出异常
    EXPECT_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts,
                           activation_window_size, world_size, rank);
    }, std::runtime_error);

    free(data_ptr); // 手动清理
}

// 测试边界情况：最小有效值
TEST_F(ClusterActivationTest, ConstructorMinimumValues) {
    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    int activation_window_size = 1;
    size_t world_size = 1;
    int rank = 0;

    Tensor npu_count = CreateTensor(num_layers, num_deploy_experts);

    ASSERT_NO_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts,
                           activation_window_size, world_size, rank);

        EXPECT_EQ(ca.get_num_layers(), 1);
        EXPECT_EQ(ca.get_num_deploy_experts(), 1);
    });
}

// 测试rank和world_size的边界情况
TEST_F(ClusterActivationTest, ConstructorRankWorldSizeBoundary) {
    size_t num_layers = 58;
    size_t num_deploy_experts = 272;
    int activation_window_size = 4;
    size_t world_size = 2;
    int rank = 1; // rank等于world_size-1

    Tensor npu_count = CreateTensor(num_layers, num_deploy_experts);

    ASSERT_NO_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts,
                           activation_window_size, world_size, rank);

        EXPECT_EQ(ca.get_num_layers(), num_layers);
        EXPECT_EQ(ca.get_num_deploy_experts_per_rank(), num_deploy_experts);
    });
}

class ClusterActivationCollectWrapperTest : public ::testing::Test {
protected:
    int old_activation_quiesce;
    void SetUp() override {
        // 初始化测试环境
        old_activation_quiesce = config.activation_quiesce;
        config.activation_quiesce=0;
        set_memcpy_fun(&my_mem_fun);
    }

    void TearDown() override {
        // 清理测试环境
        config.activation_quiesce = old_activation_quiesce;
    }

    // 辅助函数：创建Tensor对象
    Tensor CreateTensor(size_t num_layers, size_t num_experts, std::string name = "test_tensor") {
        size_t element_size = sizeof(int64_t); // 假设npu_count_存储int类型数据
        size_t size = num_layers * num_experts * element_size;
        size_t length = num_layers * num_experts;
        void* data_ptr = malloc(size);
        memset(data_ptr, 0, size); // 初始化为0
        return Tensor(data_ptr, length, element_size, name);
    }

    // // 辅助函数：填充Tensor数据
    // void FillTensor(Tensor& tensor, int value) {
    //     int* data = static_cast<int*>(tensor.data_ptr); // 假设Tensor有public data_ptr成员
    //     for (size_t i = 0; i < tensor.get_length(); ++i) {
    //         data[i] = value;
    //     }
    // }
};

// 已不再activation中起thread，UT待重新調整
/*
class ClusterActivationThreadTest : public ::testing::Test {
protected:
    int old_activation_quiesce;
    void SetUp() override {
        // 初始化测试环境
        old_activation_quiesce = config.activation_quiesce;
        config.activation_quiesce=0;
        set_memcpy_fun(&my_mem_fun);
        dump_dir = "./test_dump_dir";
    }

    void TearDown() override {
        config.activation_quiesce=old_activation_quiesce;
        remove_dir(dump_dir);
        // 清理测试环境
    }

    std::string dump_dir;

    void remove_dir(const std::string& dir) {
        DIR* dp = opendir(dir.c_str());
        if (dp != nullptr) {
            dirent* ep;
            while ((ep = readdir(dp))) {
                std::string filename = std::string(ep->d_name);
                if (filename != "." && filename != "..") {
                    std::string file_path = dir + "/" + filename;
                    unlink(file_path.c_str());
                }
            }
            closedir(dp);
            rmdir(dir.c_str());
        }
    }

};

// 测试1：单线程启动collect_wrapper
TEST_F(ClusterActivationThreadTest, SingleRankStartCollectWrapper) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 0;

    size_t layer_idx = 2;
    size_t deploy_expert_idx = 100;
    size_t index = layer_idx*num_deploy_experts+deploy_expert_idx;

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size,
                         world_size, rank);


    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[index] = 40;

    // 等待片刻，让线程运行
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 验证基本功能
    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 40); // 检查计数功能是否正常
}

// 测试用例 1: 不同 Rank 环境下启动和计数验证
TEST_F(ClusterActivationThreadTest, DiffRankStartCollectWrapper) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 1;

    size_t layer_idx = 2;
    size_t deploy_expert_idx = 100;
    size_t index = layer_idx*num_deploy_experts+deploy_expert_idx;


    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank);

    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[index] = 25;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_NO_THROW({
        ca.stop_thread();
    });

    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 25); // 验证 rank 1 的计数
}
// 单元测试
TEST_F(ClusterActivationThreadTest, ValidElementSize) {
    // 验证合法的 element_size
    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 1;
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;
    size_t length = num_layers * num_deploy_experts;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    EXPECT_NO_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank);
    });
}

TEST_F(ClusterActivationThreadTest, InvalidElementSize) {
    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 1;
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;
    size_t length = num_layers * num_deploy_experts;

    size_t element_size = sizeof(char);
    size_t size = num_layers * num_deploy_experts * element_size;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    EXPECT_THROW(
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank),std::invalid_argument);
}

//边界条件测试（最小值）
TEST_F(ClusterActivationThreadTest, BoundaryConditionMin) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 0;

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank);

    size_t layer_idx = 0;
    size_t deploy_expert_idx = 0;
    size_t index = layer_idx * num_deploy_experts + deploy_expert_idx;

    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[index] = 42;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_NO_THROW({
        ca.stop_thread();
    });

    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 42); // 验证边界条件下的计数
}

TEST_F(ClusterActivationThreadTest, BoundaryConditionMax) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const size_t rank = world_size-1;

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank);

    size_t layer_idx = num_layers - 1;
    size_t deploy_expert_idx = num_deploy_experts - 1;
    size_t index = layer_idx * num_deploy_experts + deploy_expert_idx;

    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[index] = 99;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_NO_THROW({
        ca.stop_thread();
    });

    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 99); // 验证边界条件下的计数
}

TEST_F(ClusterActivationThreadTest, NullPointerHandling) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = nullptr; // 模拟空指针
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 0;

    EXPECT_THROW({
        ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size, world_size, rank);
    }, std::invalid_argument); // 验证抛出异常
}

TEST_F(ClusterActivationThreadTest, MultiRankStartCollectWrapperV1) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;
    const int activation_window_size = 4;
    const size_t world_size = 4;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);

    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    std::vector<std::unique_ptr<ClusterActivation>> ca_vec(world_size);

    for (size_t i = 0; i < world_size; ++i) {
        ca_vec[i] = std::make_unique<ClusterActivation>(npu_count, num_layers, num_deploy_experts, activation_window_size,world_size, i);
    }

    // 修改数据以触发线程处理逻辑
    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[1] = 60;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (size_t i = 0; i < world_size; ++i) {
        // 验证基本功能
        int64_t count = ca_vec[i]->getClusterTotalActivationCount(0, 1);
        EXPECT_EQ(count, 60*world_size); // 检查计数功能是否正常
    }

}

TEST_F(ClusterActivationThreadTest, MultiRankStartCollectWrapperV2) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;
    const int activation_window_size = 100;
    const size_t world_size = 4;
    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;


    size_t layer_idx = 44;
    size_t deploy_expert_idx = 44;
    size_t index = layer_idx * num_deploy_experts + deploy_expert_idx;

    // 监听对象
    void* tmp_ptr= malloc(size);
    memset(tmp_ptr,0, size);
    Tensor tmp_tensor(tmp_ptr, length, element_size, "test_tensor");
    ClusterActivation ca(tmp_tensor, num_layers, num_deploy_experts, activation_window_size, world_size, 0);

    std::vector<int> counts_world = {0,1,2,3};
    std::vector<std::unique_ptr<ClusterActivation>> ca_vec(world_size);
    std::atomic<bool> error_occurred(false);
    // 模拟不同的Rank
    std::vector<std::thread> threads;
    for (size_t i = 0; i < world_size; ++i) {
        threads.emplace_back([&, i]() {
            try {
                void* data_ptr = malloc(size);
                memset(data_ptr,0, size);
                int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
                tmp_ptr[index] = counts_world[i];
                Tensor npu_count(data_ptr, length, element_size, "test_tensor");
                ca_vec[i] = std::make_unique<ClusterActivation>(npu_count, num_layers, num_deploy_experts, activation_window_size,world_size, i);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                int64_t count = ca_vec[i]->getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
                EXPECT_EQ(count, 6); // 检查计数功能是否正常
            } catch (...) {
                error_occurred = true;
            }
        });
    }
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    // 验证结果
    EXPECT_FALSE(error_occurred);
    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 6); // 验证边界条件下的计数
}


TEST_F(ClusterActivationThreadTest, MultiRankStartCollectWrapperV3) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;
    const int activation_window_size = 4;
    const size_t world_size = 4;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;


    std::vector<std::unique_ptr<ClusterActivation>> ca_vec(world_size);
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[1] = 60;
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");
    for (size_t i = 0; i < world_size; ++i) {
        ca_vec[i] = std::make_unique<ClusterActivation>(npu_count, num_layers, num_deploy_experts, activation_window_size,world_size, i);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (size_t i = 0; i < world_size; ++i) {
        // 验证基本功能
        int64_t count = ca_vec[i]->getClusterTotalActivationCount(0, 1);
        EXPECT_EQ(count, 60*world_size); // 检查计数功能是否正常
    }

}

TEST_F(ClusterActivationThreadTest, MultiRankStartCollectWrapperV4) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 8;
    const int activation_window_size = 10;
    const size_t world_size = 32;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;


    std::vector<std::unique_ptr<ClusterActivation>> ca_vec(world_size);
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[1] = 60;
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");
    for (size_t i = 0; i < world_size; ++i) {
        ca_vec[i] = std::make_unique<ClusterActivation>(npu_count, num_layers, num_deploy_experts, activation_window_size,world_size, i);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // for (size_t i = 0; i < world_size; ++i) {
    //     // 验证基本功能
    //     int64_t count = ca_vec[i]->getClusterTotalActivationCount(0, 1);
    //     EXPECT_EQ(count, 60*world_size); // 检查计数功能是否正常
    // }

}


// 测试1：单线程启动collect_wrapper连续改写
TEST_F(ClusterActivationThreadTest, SingleRankStartContinueChangeCollectWrapper) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 0;

    size_t layer_idx = 2;
    size_t deploy_expert_idx = 100;
    size_t index = layer_idx*num_deploy_experts+deploy_expert_idx;

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size,
                         world_size, rank);


    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    for(size_t i=0;i<100;++i)
    {
        tmp_ptr[index] = i;
    }

    // 等待片刻，让线程运行
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 验证基本功能
    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 99); // 检查计数功能是否正常
}

// 测试1：单线程启动collect_wrapper
TEST_F(ClusterActivationThreadTest, BaseDumpActivateOnThread) {
    const size_t num_layers = 58;
    const size_t num_deploy_experts = 272;

    size_t element_size = sizeof(int64_t);
    size_t size = num_layers * num_deploy_experts * element_size;
    size_t length = num_layers * num_deploy_experts;
    void* data_ptr = malloc(size);
    memset(data_ptr, 0, size);
    Tensor npu_count = Tensor(data_ptr, length, element_size, "test_tensor");

    const int activation_window_size = 4;
    const size_t world_size = 2;
    const int rank = 0;

    size_t layer_idx = 2;
    size_t deploy_expert_idx = 100;
    size_t index = layer_idx*num_deploy_experts+deploy_expert_idx;

    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, activation_window_size,
                         world_size, rank);


    int64_t* tmp_ptr = static_cast<int64_t*>(data_ptr);
    tmp_ptr[index] = 40;

    // 等待片刻，让线程运行
    std::this_thread::sleep_for(std::chrono::milliseconds(100));


    // 当没有调用setDumpDir,不应该创建文件夹
    struct stat info;
    EXPECT_NE(stat(dump_dir.c_str(), &info), 0); // dump_dir文件夹不存在此时为预期情况
    ca.setDumpDir(dump_dir);
    EXPECT_EQ(stat(dump_dir.c_str(), &info), 0); // dump_dir文件夹存在此时为预期情况
    EXPECT_TRUE(info.st_mode & S_IFDIR);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // 打开目录并计算文件数目
    DIR* dir = opendir(dump_dir.c_str());
    if (dir == nullptr) {
        FAIL() << "Failed to open directory: " << dump_dir;
    }
    int file_count = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // 检查是否为常规文件
            ++file_count;
        }
    }
    closedir(dir);
    EXPECT_GT(file_count, 0); // 文件数必须大于0

    // 验证基本功能
    int64_t count = ca.getClusterTotalActivationCount(layer_idx, deploy_expert_idx);
    EXPECT_EQ(count, 40); // 检查计数功能是否正常
    // remove_dir(dump_dir);
}
*/


// Test fixture for ClusterActivation
class ClusterActivationDumpTest : public ::testing::Test {
protected:
    std::string dump_dir = "./test_dump_dir";
    int old_activation_quiesce;
    void SetUp() override {
        // Initialize with sample parameters
        set_memcpy_fun(&my_mem_fun);
        // Clear the dump directory if it exists
        remove_dir(dump_dir);

        old_activation_quiesce = config.activation_quiesce;
        config.activation_quiesce=0;

    }

    void TearDown() override {
        // Clean up the created directory
        remove_dir(dump_dir);
        config.activation_quiesce = old_activation_quiesce;
    }

    void remove_dir(const std::string& dir) {
        DIR* dp = opendir(dir.c_str());
        if (dp != nullptr) {
            dirent* ep;
            while ((ep = readdir(dp))) {
                std::string filename = std::string(ep->d_name);
                if (filename != "." && filename != "..") {
                    std::string file_path = dir + "/" + filename;
                    unlink(file_path.c_str());
                }
            }
            closedir(dp);
            rmdir(dir.c_str());
        }
    }

    Tensor createTensor(size_t num_layers, size_t num_experts, std::string name = "test_tensor") {
        size_t element_size = sizeof(int64_t);
        size_t size = num_layers * num_experts * element_size;
        size_t length = num_layers * num_experts;
        void* data_ptr = malloc(size);
        memset(data_ptr, 0, size); // Initialize to zero
        return Tensor(data_ptr, length, element_size, name);
    }
};

// Test 1: Verify basic file creation and contents
TEST_F(ClusterActivationDumpTest, BasicDump) {

    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    size_t world_size = 4;
    size_t rank = 0;


    Tensor npu_count = createTensor(num_layers,num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    total_count_ptr[0] = 10;
    last_count_ptr[0] = 5;

    size_t dump_count = 1;
    ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);

    // Check if the file exists
    std::string filename = dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    ASSERT_TRUE(file != nullptr);
    fclose(file);

    // Check if the file contains the correct content
    std::ifstream inFile(filename);
    std::string line;
    std::getline(inFile, line);
    EXPECT_EQ(line, "5\t");

    delete[] total_count_ptr;
    delete[] last_count_ptr;
    remove_dir(dump_dir);
}

// Test 2: Verify file creation and contents for multiple layers and experts
TEST_F(ClusterActivationDumpTest, MultipleLayersExperts) {

    size_t num_layers = 3;
    size_t num_deploy_experts = 2;
    size_t world_size = 2;
    size_t rank = 0;

    Tensor npu_count = createTensor(num_layers, num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    for (size_t i = 0; i < size; ++i) {
        total_count_ptr[i] = i ;
        last_count_ptr[i] = i;
    }

    size_t dump_count = 1;
    ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);

    // Check if the file exists
    std::string filename = dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    ASSERT_TRUE(file != nullptr);
    fclose(file);

    // Check if the file contains the correct content
    std::ifstream inFile(filename);
    std::string expected_content = "0\t0\t\n0\t0\t\n0\t0\t\n";
    std::string actual_content((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    EXPECT_EQ(actual_content, expected_content);

    delete[] total_count_ptr;
    delete[] last_count_ptr;
    remove_dir(dump_dir);
}

// Test 3: Verify that the dump file is overwritten for the same dump count
TEST_F(ClusterActivationDumpTest, OverwriteDumpFile) {

    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    size_t world_size = 4;
    size_t rank = 0;

    Tensor npu_count = createTensor(num_layers, num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    // Dump first set of data
    total_count_ptr[0] = 10;
    last_count_ptr[0] = 5;

    size_t dump_count = 1;
    ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);

    // Dump second set of data with the same dump count
    total_count_ptr[0] = 8;
    last_count_ptr[0] = 8;

    ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);

    // Check if the file exists
    std::string filename = dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    ASSERT_TRUE(file != nullptr);
    fclose(file);

    // Check if the file contains the correct content
    std::ifstream inFile(filename);
    std::string line;
    std::getline(inFile, line);
    EXPECT_EQ(line, "0\t");

    delete[] total_count_ptr;
    delete[] last_count_ptr;
    remove_dir(dump_dir);
}

// Test 4: Verify that directory is created if it does not exist
TEST_F(ClusterActivationDumpTest, DirectoryCreation) {

    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    size_t world_size = 4;
    size_t rank = 0;

    std::string non_existing_dump_dir = "./non_existing_dir";
    Tensor npu_count = createTensor(num_layers, num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(non_existing_dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    total_count_ptr[0] = 10;
    last_count_ptr[0] = 5;

    size_t dump_count = 1;
    ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);

    // Check if the file exists
    std::string filename = non_existing_dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    ASSERT_TRUE(file != nullptr);
    fclose(file);

    // Check if the file contains the correct content
    std::ifstream inFile(filename);
    std::string line;
    std::getline(inFile, line);
    EXPECT_EQ(line, "5\t");

    delete[] total_count_ptr;
    delete[] last_count_ptr;
    remove_dir(non_existing_dump_dir);
}

// Test 5: Verify that an invalid directory does not cause crash
TEST_F(ClusterActivationDumpTest, InvalidDirectory) {

    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    size_t world_size = 4;
    size_t rank = 0;

    std::string invalid_dump_dir = "./invalid_dir/invalid_subdir";
    Tensor npu_count = createTensor(num_layers, num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(invalid_dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    total_count_ptr[0] = 10;
    last_count_ptr[0] = 5;

    size_t dump_count = 1;

    EXPECT_NO_THROW({
        ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);
    });

    // Check that in case of an invalid directory, no file is created
    std::string filename = invalid_dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    EXPECT_TRUE(file == nullptr);
    if (file != nullptr) {
        fclose(file);
    }

    delete[] total_count_ptr;
    delete[] last_count_ptr;
}

// Test 6: Verify the behavior when dump directory is removed during dump process
TEST_F(ClusterActivationDumpTest, DumpDirRemovedDuringDump) {

    size_t num_layers = 1;
    size_t num_deploy_experts = 1;
    size_t world_size = 4;
    size_t rank = 0;

    Tensor npu_count = createTensor(num_layers, num_deploy_experts);
    ClusterActivation ca(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
    ca.stop_thread();
    ca.setDumpDir(dump_dir);

    size_t size = num_layers * num_deploy_experts;
    int64_t* total_count_ptr = new int64_t[size];
    int64_t* last_count_ptr = new int64_t[size];

    total_count_ptr[0] = 10;
    last_count_ptr[0] = 5;

    size_t dump_count = 1;

    // Remove the directory while the dump process is in progress
    std::thread remove_thread([&](){
        remove_dir(dump_dir);
    });

    remove_thread.join();

    // Dump the data
    EXPECT_NO_THROW({
        ca.dumpActivationCounts(dump_count, total_count_ptr, last_count_ptr);
    });

    // Check if the file exists
    std::string filename = dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(ca.get_rank())+".txt";
    FILE* file = fopen(filename.c_str(), "r");
    EXPECT_TRUE(file == nullptr);
    if (file != nullptr) {
        fclose(file);
    }

    delete[] total_count_ptr;
    delete[] last_count_ptr;
}

// Test 2: Verify multi-threaded dump
TEST_F(ClusterActivationDumpTest, MultiThreadedDump) {
    size_t num_layers = 3;
    size_t num_deploy_experts = 4;
    const size_t world_size = 4; // Total number of threads
    const size_t num_threads = world_size; // Number of threads to create

    std::vector<std::thread> threads;
    std::vector<ClusterActivation*> cas(world_size);
    std::vector<int64_t*> total_count_ptrs(world_size);
    std::vector<int64_t*> last_count_ptrs(world_size);

    // Initialize ClusterActivation instances and count pointers for each thread
    for (size_t rank = 0; rank < world_size; ++rank) {
        Tensor npu_count = createTensor(num_layers, num_deploy_experts);
        cas[rank] = new ClusterActivation(npu_count, num_layers, num_deploy_experts, 1, world_size, rank);
        cas[rank]->stop_thread();
        cas[rank]->setDumpDir(dump_dir);
        size_t size = num_layers * num_deploy_experts;
        total_count_ptrs[rank] = new int64_t[size];
        last_count_ptrs[rank] = new int64_t[size];
        // Initialize to avoid undefined behavior
        for (size_t i = 0; i < size; ++i) {
            total_count_ptrs[rank][i] = rank;
            last_count_ptrs[rank][i] = 0;
        }
    }

    size_t dump_count = 1;

    // Start threads to dump activation counts
    for (size_t rank = 0; rank < num_threads; ++rank) {
        threads.emplace_back([&, rank] {
            cas[rank]->dumpActivationCounts(dump_count, total_count_ptrs[rank], last_count_ptrs[rank]);
        });
    }

    // Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }

    // Check if the files exist and contain the correct content
    for (size_t rank = 0; rank < world_size; ++rank) {
        std::string filename = dump_dir + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_" + std::to_string(rank) + ".txt";
        FILE* file = fopen(filename.c_str(), "r");
        ASSERT_TRUE(file != nullptr);
        fclose(file);

        std::ifstream inFile(filename);
        std::string line;
        while (std::getline(inFile, line)) {
            std::vector<std::string> expected = {
                "0\t0\t0\t0\t",
                "1\t1\t1\t1\t",
                "2\t2\t2\t2\t",
                "3\t3\t3\t3\t"
                };
            EXPECT_EQ(line, expected[rank]);
        }
    }

    // Clean up
    for (size_t rank = 0; rank < world_size; ++rank) {
        delete[] total_count_ptrs[rank];
        delete[] last_count_ptrs[rank];
        delete cas[rank];
    }
    remove_dir(dump_dir);
}
