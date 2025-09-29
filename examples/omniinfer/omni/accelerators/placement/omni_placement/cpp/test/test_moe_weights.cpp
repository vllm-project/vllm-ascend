// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <random>
#include "moe_weights.h"
#include <iostream>

class MoEWeightsTest : public ::testing::Test {
protected:
    MoEWeights* moeweights;
    size_t num_experts = 64;

    void SetUp() override {
        moeweights = new MoEWeights(num_experts);
    };
    void TearDown() override {}
};

// 初始化函数

// 基本功能测试
TEST_F(MoEWeightsTest, Constructor_BasicInitialization) {
    size_t num_experts = 8;
    MoEWeights weights(num_experts);

    EXPECT_EQ(weights.getNumExperts(), num_experts);
    EXPECT_EQ(weights.getShmPtr(), nullptr);
    EXPECT_FALSE(weights.isShmInitialized());
}

// 边界值测试
TEST_F(MoEWeightsTest, Constructor_ZeroExperts) {
    MoEWeights weights(0);

    EXPECT_EQ(weights.getNumExperts(), 0);
    EXPECT_EQ(weights.getShmPtr(), nullptr);
    EXPECT_FALSE(weights.isShmInitialized());
}

// 最大值测试
TEST_F(MoEWeightsTest, Constructor_MaxSizeT) {
    MoEWeights weights(std::numeric_limits<size_t>::max());
    EXPECT_EQ(weights.getNumExperts(), std::numeric_limits<size_t>::max());
    EXPECT_EQ(weights.getShmPtr(), nullptr);
    EXPECT_FALSE(weights.isShmInitialized());
}

// 并发测试
TEST_F(MoEWeightsTest, Constructor_ConcurrentCreation) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);

    // 创建多个线程同时构造 MoEWeights
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                weights_vec[i] = std::make_unique<MoEWeights>(i + 1,num_threads);
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
    for (int i = 0; i < num_threads; ++i) {
        ASSERT_NE(weights_vec[i], nullptr);
        EXPECT_EQ(weights_vec[i]->getNumExperts(), static_cast<size_t>(i + 1));
        EXPECT_EQ(weights_vec[i]->getShmPtr(), nullptr);
        EXPECT_FALSE(weights_vec[i]->isShmInitialized());
    }
}
class MoEWeightsShmInitTest : public ::testing::Test {
protected:
    void SetUp() override {
        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);
    }
    void TearDown() override{}
};

TEST_F(MoEWeightsShmInitTest, CreateNewSharedMemory) {
    MoEWeights weights(10);
    size_t shm_size = 1024; // 设置共享内存大小
    weights.unittest_for_init_shared_memory(shm_size);
    void* ptr = weights.getShmPtr();

    size_t data_length = 16; // 要拷贝的数据长度
    char data_to_copy[data_length] = "TestData12345678"; // 要拷贝的数据
    char buffer[data_length] = {0}; // 用于读取的数据缓冲区
    
    // 验证返回的指针不为空
    ASSERT_NE(ptr, nullptr);

    // 拷贝数据到共享内存末尾
    memcpy(static_cast<char*>(ptr) + shm_size - data_length, data_to_copy, data_length);

    // 从共享内存末尾读取数据
    memcpy(buffer, static_cast<char*>(ptr) + shm_size - data_length, data_length);

    // 验证读取的数据与写入的数据一致
    EXPECT_EQ(std::memcmp(data_to_copy, buffer, data_length), 0);

    std::string shm_name = weights.getShmName();
    // 验证共享内存文件存在
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    ASSERT_GE(fd, 0);
    close(fd);

    // 清理共享内存
    munmap(ptr, shm_size + sizeof(CountData));
    shm_unlink(shm_name.c_str());
}

// 测试多次连续写入数据并验证一致性
TEST_F(MoEWeightsShmInitTest, MultipleWritesToSharedMemory) {
    MoEWeights weights(10);
    size_t shm_size = 4096; // 共享内存大小
    size_t data_length = 16; // 每次写入的数据长度
    char data1[data_length] = "TestData12345678";
    char data2[data_length] = "AnotherTest12345";
    char buffer[data_length] = {0};

    // 初始化共享内存
    weights.unittest_for_init_shared_memory(shm_size);
    void* ptr = weights.getShmPtr();
    ASSERT_NE(ptr, nullptr);

    // 第一次写入数据
    memcpy(ptr, data1, data_length);

    // 第二次写入数据
    memcpy(static_cast<char*>(ptr) + data_length, data2, data_length);

    // 验证第一次写入的数据
    memcpy(buffer, ptr, data_length);
    EXPECT_EQ(std::memcmp(data1, buffer, data_length), 0);

    // 验证第二次写入的数据
    memcpy(buffer, static_cast<char*>(ptr) + data_length, data_length);
    EXPECT_EQ(std::memcmp(data2, buffer, data_length), 0);

    // 清理共享内存
    munmap(ptr, shm_size);
    shm_unlink(weights.getShmName().c_str());
}

// 测试共享内存未初始化时尝试写入数据
TEST_F(MoEWeightsShmInitTest, CreateMaxSharedMemory) {
    MoEWeights weights(10);
    // 引入了计数的8bit的空间，导致没有对其
    size_t shm_size = 163644440576; // 设置共享内存大小
    weights.unittest_for_init_shared_memory(shm_size);

    void* ptr = weights.getShmPtr();
    size_t data_length = 14336; // 要拷贝的数据长度
    char data_to_copy[data_length]; // 要拷贝的数据
    memset(data_to_copy, 'A', data_length); // 填充数据
    char buffer[data_length] = {0}; // 用于读取的数据缓冲区
    
    // 验证返回的指针不为空
    ASSERT_NE(ptr, nullptr);

    // 拷贝数据到共享内存末尾
    memcpy(static_cast<char*>(ptr) + shm_size - data_length, data_to_copy, data_length);

    // 从共享内存末尾读取数据
    memcpy(buffer, static_cast<char*>(ptr) + shm_size - data_length, data_length);

    // 验证读取的数据与写入的数据一致
    EXPECT_EQ(std::memcmp(data_to_copy, buffer, data_length), 0);

    std::string shm_name = weights.getShmName();
    // 验证共享内存文件存在
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    ASSERT_GE(fd, 0);
    close(fd);

    // 清理共享内存
    munmap(ptr, shm_size + sizeof(CountData));
    shm_unlink(shm_name.c_str());
}

// MoEWeights::init_weights  测试

class MoEWeightsInitTest : public ::testing::Test {
protected:
    void SetUp() override {
        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);
        moe = std::make_unique<MoEWeights>(4);
    }
    // 创建 Tensor 的辅助函数
    Tensor create_tensor(size_t length, float value, const std::string& name) {
        size_t element_size = sizeof(float);
        std::vector<float> device_data;
        device_data.resize(length, value);  // 用 1.0f 填充

        void* data_ptr = nullptr;
        size_t size = length * element_size;
        EXPECT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

        Tensor tensor((uint64_t)(data_ptr), length, element_size, name);
        EXPECT_EQ(ACL_ERROR_NONE, tensor.to_device(device_data.data()));
        return tensor;
    }

    // 验证 shm_ptr_

    // 验证 shm_ptr 是否正确的函数
    void verify_shm_ptr(const MoEWeights& moe,
                        const std::vector<std::vector<int>>& expert_ids,
                        std::string& error_message) {
        void* shm_ptr = moe.getShmPtr();
        ASSERT_NE(shm_ptr, nullptr) << "shm_ptr is null";

        char* shm_data = static_cast<char*>(shm_ptr);
        bool all_equal = true;
        error_message.clear();
        size_t expert_size = moe.getNpuWeights()[0][0].get_total_size();

        for (size_t layer_idx = 0; layer_idx < moe.getNumLayers(); ++layer_idx) {
            for (size_t expert_idx = 0; expert_idx < expert_ids[layer_idx].size(); ++expert_idx) {
                // 计算当前专家在共享内存中的偏移
                char* shm_ptr_current = shm_data +
                                    (layer_idx * moe.getNumExperts() +
                                    expert_ids[layer_idx][expert_idx]) * expert_size;

                // 分配临时缓冲区并提取专家数据
                auto ptr = std::make_unique<char[]>(expert_size);
                char* raw_ptr = ptr.get();
                ExpertWeights expert_weights = moe.getNpuWeights()[layer_idx][expert_idx];
                expert_weights.to_host(raw_ptr);

                // 比较内存内容
                bool is_equal = std::equal(raw_ptr, raw_ptr + expert_size, shm_ptr_current);
                if (!is_equal) {
                    all_equal = false;
                    error_message += "Mismatch at layer " + std::to_string(layer_idx) +
                                    ", expert " + std::to_string(expert_idx) +
                                    " (id: " + std::to_string(expert_ids[layer_idx][expert_idx]) + ")\n";
                }
            }
        }
        EXPECT_TRUE(all_equal) << "Shared memory contents verification failed:\n" << error_message;
    }
    std::unique_ptr<MoEWeights> moe;
};
// 正常初始化测试
TEST_F(MoEWeightsInitTest, NormalInitialization) {
    // 第一层0,1 专家， 第二层为 2,3专家
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2")},
         {create_tensor(2,1, "w3"), create_tensor(3,1, "w4")}},
        {{create_tensor(2,2, "w5"), create_tensor(3, 2,"w6")},
         {create_tensor(2,3, "w7"), create_tensor(3,3, "w8")}}
    };
    std::vector<std::vector<int>> expert_ids = {{0, 1}, {2, 3}};

    moe->init_weights(npu_weights, expert_ids,true);

    EXPECT_EQ(moe->getNumLayers(), 2);
    EXPECT_TRUE(moe->isShmInitialized());
    EXPECT_EQ(moe->getNpuWeights().size(), 2);
    EXPECT_EQ(moe->getNpuWeights()[0].size(), 2);
    EXPECT_EQ(moe->getNpuWeights()[0][0].get_expert_id(), 0);
    EXPECT_EQ(moe->getNpuWeights()[0][0].get_total_size(), 20);  // (2*4 + 3*4)
}


// 空输入测试
TEST_F(MoEWeightsInitTest, EmptyInput) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights;
    std::vector<std::vector<int>> expert_ids;

    EXPECT_THROW(moe->init_weights(npu_weights, expert_ids,true), std::runtime_error);
}

// 零长度张量测试
TEST_F(MoEWeightsInitTest, ZeroLengthTensor) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{Tensor()}}
    };
    std::vector<std::vector<int>> expert_ids = {{0}};

    EXPECT_THROW(moe->init_weights(npu_weights, expert_ids, true), std::runtime_error);
}

// 维度不匹配测试
TEST_F(MoEWeightsInitTest, MismatchedDimensions) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1")}}
    };
    std::vector<std::vector<int>> expert_ids = {{0, 1}};

    EXPECT_THROW(moe->init_weights(npu_weights, expert_ids, true), std::out_of_range);
}

// 边界条件，验证expert_ids为空
TEST_F(MoEWeightsInitTest, EmptyExpertIds) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1")}}
    };
    std::vector<std::vector<int>> expert_ids = {{}};

    EXPECT_THROW(moe->init_weights(npu_weights, expert_ids, true), std::out_of_range);
}

// 多权重测试
TEST_F(MoEWeightsInitTest, MultipleWeightsPerExpert) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2"), create_tensor(1, 0,"w3")},
         {create_tensor(2, 1,"w4"), create_tensor(3,1, "w5"), create_tensor(1,1, "w6")}}
    };
    std::vector<std::vector<int>> expert_ids = {{0, 1}};

    moe->init_weights(npu_weights, expert_ids, true);

    EXPECT_EQ(moe->getNpuWeights()[0][0].get_total_size(), 24);  // 2*4 + 3*4 + 1*4
}

// 多权重测试
TEST_F(MoEWeightsInitTest, LastRankForMultiWorldSize) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2"), create_tensor(1, 0,"w3")},
         {create_tensor(2, 1,"w4"), create_tensor(3,1, "w5"), create_tensor(1,1, "w6")}}
    };
    MoEWeights cur_moe(64,4);
    std::vector<std::vector<int>> expert_ids = {{62, 63}};

    cur_moe.init_weights(npu_weights, expert_ids,true);

    EXPECT_EQ(cur_moe.getNpuWeights()[0][0].get_total_size(), 24);  // 2*4 + 3*4 + 1*4
}

// 多权重测试，包括 shm_ptr_ 验证
TEST_F(MoEWeightsInitTest, MultipleWeightsWithShm) {
    std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
        {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2"), create_tensor(1, 0,"w3")},
         {create_tensor(2, 1,"w4"), create_tensor(3,1, "w5"), create_tensor(1,1, "w6")}}
    };
    std::vector<std::vector<int>> expert_ids = {{0, 1}};
    moe->init_weights(npu_weights, expert_ids,true);
    size_t expert_size = moe->getNpuWeights()[0][0].get_total_size();
    EXPECT_EQ(expert_size, 24);
    // 验证 shm_ptr_
    // 验证 shm_ptr_
    std::string error_message;
    verify_shm_ptr(*moe, expert_ids, error_message);

    // 验证共享内存size 64对齐
    EXPECT_EQ(moe->getShmSize() + sizeof(CountData) - moe->getNumLayers() * moe->getNumExperts() * expert_size, 64);
}

// 多线程并发往共享内存的相同Ptr写入
TEST_F(MoEWeightsInitTest, MultipleThreadsWriteToShmWithSamePlace) {

    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);

    // 创建多个线程同时构造 MoEWeights并调用 init_weights
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                aclInit(NULL); // 初始化 ACL
                aclrtContext context;
                aclrtCreateContext(&context, 0);
                aclrtSetCurrentContext(context);
                weights_vec[i] = std::make_unique<MoEWeights>(2);
                std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
                    {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2"), create_tensor(1, 0,"w3")},
                    {create_tensor(2, 1,"w4"), create_tensor(3,1, "w5"), create_tensor(1,1, "w6")}}
                };
                std::vector<std::vector<int>> expert_ids = {{0, 1}};
                weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                std::string error_message;
                verify_shm_ptr(*weights_vec[i], expert_ids, error_message);
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
}

// 多线程并发往共享内存的不同Ptr写入
TEST_F(MoEWeightsInitTest, MultipleThreadsWriteToShmWithDiffPlace) {

    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);

    // 创建多个线程同时构造 MoEWeights并调用 init_weights
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                aclInit(NULL); // 初始化 ACL
                aclrtContext context;
                aclrtCreateContext(&context, 0);
                aclrtSetCurrentContext(context);
                weights_vec[i] = std::make_unique<MoEWeights>(2*num_threads);
                std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
                    {{create_tensor(2,0, "w1"), create_tensor(3,0, "w2"), create_tensor(1, 0,"w3")},
                    {create_tensor(2, 1,"w4"), create_tensor(3,1, "w5"), create_tensor(1,1, "w6")}}
                };
                std::vector<std::vector<int>> expert_ids = {{2*i, 2*i+1}};
                weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                std::string error_message;
                verify_shm_ptr(*weights_vec[i], expert_ids, error_message);
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
}

TEST_F(MoEWeightsInitTest, VerifyAllThreadsCompleteWrite) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);
    std::atomic<int> completed_threads(0); // 计数器，用于验证线程完成

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                aclInit(NULL);
                aclrtContext context;
                aclrtCreateContext(&context, 0);
                aclrtSetCurrentContext(context);
                weights_vec[i] = std::make_unique<MoEWeights>(2 * num_threads);
                std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
                    {{create_tensor(2, 0, "w1"), create_tensor(3, 0, "w2"), create_tensor(1, 0, "w3")},
                     {create_tensor(2, 1, "w4"), create_tensor(3, 1, "w5"), create_tensor(1, 1, "w6")}}
                };
                std::vector<std::vector<int>> expert_ids = {{2 * i, 2 * i + 1}};
                weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                std::string error_message;
                verify_shm_ptr(*weights_vec[i], expert_ids, error_message);
                completed_threads.fetch_add(1); // 线程完成后递增计数器
            } catch (...) {
                error_occurred = true;
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证所有线程是否完成写入
    EXPECT_FALSE(error_occurred);
    EXPECT_EQ(completed_threads.load(), num_threads); // 检查是否所有线程都完成了写入

    // 进一步验证共享内存内容
    for (int i = 0; i < num_threads; ++i) {
        std::vector<std::vector<int>> expected_expert_ids = {{2 * i, 2 * i + 1}};
        std::string error_message;
        verify_shm_ptr(*weights_vec[i], expected_expert_ids, error_message);
    }
}

// 所有进程完成初始化，才returnTrue
TEST_F(MoEWeightsInitTest, AllProcessesInitCompleteReturnsTrue) {
        const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                aclInit(NULL);
                aclrtContext context;
                aclrtCreateContext(&context, 0);
                aclrtSetCurrentContext(context);
                weights_vec[i] = std::make_unique<MoEWeights>(2 * num_threads,num_threads);
                std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
                    {{create_tensor(2, 0, "w1"), create_tensor(3, 0, "w2"), create_tensor(1, 0, "w3")},
                     {create_tensor(2, 1, "w4"), create_tensor(3, 1, "w5"), create_tensor(1, 1, "w6")}}
                };
                std::vector<std::vector<int>> expert_ids = {{2 * i, 2 * i + 1}};
                weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                std::string error_message;
                verify_shm_ptr(*weights_vec[i], expert_ids, error_message);
            } catch (...) {
                error_occurred = true;
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证所有线程是否完成写入
    EXPECT_FALSE(error_occurred);

    // 检查是否所有线程都完成了写入
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_TRUE(weights_vec[i]->isShmInitialized());
    }
}

// 创建10个线程，有一个线程不执行init，看是否完成初始化
TEST_F(MoEWeightsInitTest, OneProcessesNotInitReturnsFalse) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<MoEWeights>> weights_vec(num_threads);
    std::atomic<bool> error_occurred(false);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                aclInit(NULL);
                aclrtContext context;
                aclrtCreateContext(&context, 0);
                aclrtSetCurrentContext(context);
                weights_vec[i] = std::make_unique<MoEWeights>(2 * num_threads,num_threads);
                std::vector<std::vector<std::vector<Tensor>>> npu_weights = {
                    {{create_tensor(2, 0, "w1"), create_tensor(3, 0, "w2"), create_tensor(1, 0, "w3")},
                     {create_tensor(2, 1, "w4"), create_tensor(3, 1, "w5"), create_tensor(1, 1, "w6")}}
                };
                std::vector<std::vector<int>> expert_ids = {{2 * i, 2 * i + 1}};
                if (i != num_threads-1){
                    weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                    std::string error_message;
                    verify_shm_ptr(*weights_vec[i], expert_ids, error_message);
                }
            } catch (...) {
                error_occurred = true;
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证所有线程是否完成写入
    EXPECT_FALSE(error_occurred);

    // 检查是否所有线程都完成了写入
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_FALSE(weights_vec[i]->isShmInitialized());
    }
}


class MoEWeightsReplacementTest : public ::testing::Test {
protected:
    std::vector<std::unique_ptr<MoEWeights>> weights_vec;
    const size_t num_threads_ = 10;
    void SetUp() override {
        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        // 开N个线程，每个线程创建一个两层两个专家的单元测试
        // 每层有 2N个专家, Tensor 数值为
        weights_vec.resize(num_threads_);
        std::atomic<bool> error_occurred(false);

        // 线程初始化
        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads_; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    aclInit(NULL);
                    aclrtContext context;
                    aclrtCreateContext(&context, 0);
                    aclrtSetCurrentContext(context);
                    weights_vec[i] = std::make_unique<MoEWeights>(2 * num_threads_,num_threads_);
                    std::vector<std::vector<std::vector<Tensor>>> npu_weights = construct_npu_weights(i);
                    std::vector<std::vector<int>> expert_ids = construct_expert_ids(i);
                    weights_vec[i]->init_weights(npu_weights, expert_ids,true);
                } catch (...) {
                    error_occurred = true;
                }
            });
        }

        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
        // 验证所有线程是否完成写入
        EXPECT_FALSE(error_occurred);

        // 检查是否所有线程都完成了写入
        for (size_t i = 0; i < num_threads_; ++i) {
            EXPECT_TRUE(weights_vec[i]->isShmInitialized());
        }

    }
    // 创建 Tensor 的辅助函数
    Tensor create_tensor(size_t length, float value, const std::string& name) {
        size_t element_size = sizeof(float);
        std::vector<float> device_data;
        device_data.resize(length, value);  // 用 1.0f 填充

        void* data_ptr = nullptr;
        size_t size = length * element_size;
        EXPECT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

        Tensor tensor((uint64_t)(data_ptr), length, element_size, name);
        EXPECT_EQ(ACL_ERROR_NONE, tensor.to_device(device_data.data()));
        return tensor;
    }

    std::vector<std::vector<std::vector<Tensor>>> construct_npu_weights(int rank_id){
        //构建两层，每层两个专家， 用于单元测试
        std::vector<std::vector<Tensor>> npu_weight =
            {{create_tensor(2, 2*rank_id, "w1"), create_tensor(3, 2*rank_id, "w2"), create_tensor(1, 2*rank_id, "w3")},
                {create_tensor(2, 2*rank_id+1, "w4"), create_tensor(3, 2*rank_id+1, "w5"), create_tensor(1, 2*rank_id+1, "w6")}};
        std::vector<std::vector<std::vector<Tensor>>> npu_weights(2,npu_weight);
        return npu_weights;
    }
    std::vector<std::vector<int>> construct_expert_ids(int rank_id){
        //构建两层，每层两个专家， 用于单元测试
        std::vector<int> expert_ids_per_layer = {2 * rank_id, 2 * rank_id + 1};
        std::vector<std::vector<int>>  expert_ids(2,expert_ids_per_layer);
        return expert_ids;
    }

    // 验证 shm_ptr_
    // 验证 shm_ptr 是否正确的函数
    void verify_replacement(ExpertWeights* expert_weights,float expert_id) {

        size_t expert_size = expert_weights->get_total_size();
        // 分配临时缓冲区并提取专家数据
        auto ptr = std::make_unique<char[]>(expert_size);
        char* raw_ptr = ptr.get();
        expert_weights->to_host(raw_ptr);

        // GoldenValue
        size_t float_size = expert_size/(sizeof(float)/sizeof(char));
        float* float_ptr = new float[float_size];
        std::fill(float_ptr, float_ptr + float_size, expert_id);
        char* golden_ptr = reinterpret_cast<char*>(float_ptr);

         // 比较内存内容
        EXPECT_EQ(memcmp(golden_ptr, raw_ptr, expert_size), 0);
    }
};

// 测试用例1：正常情况
TEST_F(MoEWeightsReplacementTest, NormalReplacement) {
    size_t source_expert_id = 1;
    size_t layer_idx = 0;
    size_t dst_expert_idx = 0;
    size_t thread_idx = 0;
    weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx);  // 将全局专家1的权重替换到第0层第0个专家
    ExpertWeights result = weights_vec[thread_idx]->getNpuWeights()[layer_idx][dst_expert_idx];
    verify_replacement(&result,1);
}

// 测试用例2：替换到不同层的不同专家
TEST_F(MoEWeightsReplacementTest, ReplaceToDifferentLayerAndExpert) {
    size_t source_expert_id = 2;
    size_t layer_idx = 1;
    size_t dst_expert_idx = 1;
    size_t thread_idx = 0;
    weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx);
    ExpertWeights result = weights_vec[thread_idx]->getNpuWeights()[layer_idx][dst_expert_idx];
    verify_replacement(&result, source_expert_id);
}

// 测试用例3：边界情况 - 最大层和最大专家
TEST_F(MoEWeightsReplacementTest, BoundaryMaxLayerAndExpert) {
    size_t thread_idx = 0;
    size_t source_expert_id = weights_vec[thread_idx]->getNumExperts() - 1;
    size_t layer_idx = weights_vec[thread_idx]->getNumLayers() - 1;
    size_t dst_expert_idx = 1; // 每个线程只有两个专家
    weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx);
    ExpertWeights result = weights_vec[thread_idx]->getNpuWeights()[layer_idx][dst_expert_idx];
    verify_replacement(&result, source_expert_id);
}

// 测试用例4：边界情况 - 最小索引
TEST_F(MoEWeightsReplacementTest, BoundaryMinIndices) {
    size_t source_expert_id = 0;
    size_t layer_idx = 0;
    size_t dst_expert_idx = 0;
    size_t thread_idx = 0;
    weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx);
    ExpertWeights result = weights_vec[thread_idx]->getNpuWeights()[layer_idx][dst_expert_idx];
    verify_replacement(&result, source_expert_id);
}

// 测试用例5：无效的 layer_idx
TEST_F(MoEWeightsReplacementTest, InvalidLayerIndex) {
    size_t thread_idx = 0;
    size_t source_expert_id = 1;
    size_t layer_idx = weights_vec[thread_idx]->getNumLayers();  // 超出范围
    size_t dst_expert_idx = 0;
    EXPECT_THROW(
        weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx),
        std::runtime_error
    );
}

// 测试用例6：无效的 src_global_expert_id
TEST_F(MoEWeightsReplacementTest, InvalidSourceExpertId) {
    size_t thread_idx = 0;
    size_t source_expert_id = weights_vec[thread_idx]->getNumExperts();  // 超出范围
    size_t layer_idx = 0;
    size_t dst_expert_idx = 0;
    EXPECT_THROW(
        weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx),
        std::runtime_error
    );
}

// 测试用例7：无效的 dst_local_expert_idx
TEST_F(MoEWeightsReplacementTest, InvalidDestExpertIdx) {
    size_t source_expert_id = 1;
    size_t layer_idx = 0;
    size_t dst_expert_idx = 2;  // 超出范围0~1
    size_t thread_idx = 0;
    EXPECT_THROW(
        weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx),
        std::runtime_error
    );
}


// 测试用例1：多线程同时替换到不同目标
TEST_F(MoEWeightsReplacementTest, ConcurrentReplacementDifferentTargets) {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads_; ++i) {
        threads.emplace_back([this, i]() {
            aclInit(NULL);
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            weights_vec[i]->replacement(0, i, 0);  // 每个线程替换到自己的第0层第0个专家
        });
    }
    for (auto& t : threads) t.join();

    for (size_t i = 0; i < num_threads_; ++i) {
        ExpertWeights result = weights_vec[i]->getNpuWeights()[0][0];
        verify_replacement(&result, i);
    }
}

// 测试用例2：多线程替换同一源到不同目标
TEST_F(MoEWeightsReplacementTest, ConcurrentReplacementSameSource) {
    std::vector<std::thread> threads;
    size_t target_layer = 1;
    size_t target_expert = 1;
    size_t source_expert_id = 5; //0~2N-1
    for (size_t i = 0; i < num_threads_; ++i) {
        threads.emplace_back([this, i, target_layer,source_expert_id, target_expert]() {
            aclInit(NULL);
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            weights_vec[i]->replacement(target_layer, source_expert_id, target_expert);  // 所有线程替换到同一个实例
        });
    }
    for (auto& t : threads) t.join();

    for (size_t i = 0; i < num_threads_; ++i) {
        ExpertWeights result = weights_vec[i]->getNpuWeights()[target_layer][target_expert];
        verify_replacement(&result, source_expert_id);
    }
}

// 测试用例3：高并发压力测试
TEST_F(MoEWeightsReplacementTest, ConcurrentStressTest) {
    const size_t iterations = 100;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads_; ++i) {
        threads.emplace_back([this, i, iterations]() {
            aclInit(NULL);
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            for (size_t j = 0; j < iterations; ++j) {
                weights_vec[i]->replacement(0, i, 0);
            }
        });
    }
    for (auto& t : threads) t.join();

    for (size_t i = 0; i < num_threads_; ++i) {
        ExpertWeights result = weights_vec[i]->getNpuWeights()[0][0];
        verify_replacement(&result, i);
    }
}

// 测试用例4：单线程边界线程索引并发
TEST_F(MoEWeightsReplacementTest, BoundaryThread) {
    size_t thread_idx = num_threads_-1;
    size_t source_expert_id = weights_vec[thread_idx]->getNumExperts()-1;
    size_t layer_idx = 1;
    size_t dst_expert_idx = 1;  // 超出范围0~1
    weights_vec[thread_idx]->replacement(layer_idx, source_expert_id, dst_expert_idx);
    ExpertWeights result = weights_vec[thread_idx]->getNpuWeights()[layer_idx][dst_expert_idx];
    verify_replacement(&result, weights_vec[thread_idx]->getNumExperts()-1);
}

// // 测试用例4：多线程边界索引并发
TEST_F(MoEWeightsReplacementTest, ConcurrentBoundaryIndices) {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads_; ++i) {
        threads.emplace_back([this, i]() {
            aclInit(NULL);
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            size_t layer = (i % 2);  // 交替使用第0层和第1层
            weights_vec[i]->replacement(layer, weights_vec[i]->getNumExperts()-1, 1);
        });
    }
    for (auto& t : threads) t.join();

    for (size_t i = 0; i < num_threads_; ++i) {
        size_t layer = (i % 2);
        ExpertWeights result = weights_vec[i]->getNpuWeights()[layer][1];
        verify_replacement(&result, weights_vec[i]->getNumExperts()-1);
    }
}

// 测试用例5：多线程异常情况
TEST_F(MoEWeightsReplacementTest, ConcurrentInvalidIndex) {
    std::vector<std::thread> threads;
    std::atomic<bool> exception_caught(false);
    for (size_t i = 0; i < num_threads_; ++i) {
        threads.emplace_back([this, i, &exception_caught]() {
            aclInit(NULL);
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            try {
                weights_vec[i]->replacement(weights_vec[i]->getNumLayers(), i % weights_vec[i]->getNumExperts(), 0);  // 无效 layer_idx
            } catch (const std::runtime_error&) {
                exception_caught = true;
            }
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_TRUE(exception_caught);
}