// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <random>
#include "tensor.h"
#include <iostream>

class TensorTest : public ::testing::Test {
protected:
    Tensor* tensor;
    size_t length = 3;
    size_t element_size = sizeof(float);
    std::vector<float> device_data;

    void SetUp() override {
        aclInit(NULL); // 初始化 ACL
        aclrtContext context;
        aclrtCreateContext(&context, 0);
        aclrtSetCurrentContext(context);

        device_data.resize(length, 8.0f);
        void* data_ptr;
        size_t size = length*element_size;
        ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        tensor = new Tensor(static_cast<uint64_t*>(data_ptr), length, element_size, "test_tensor");
        ASSERT_EQ(ACL_ERROR_NONE, tensor->to_device(device_data.data()));
    };
};

// 测试默认构造函数
TEST_F(TensorTest, DefaultConstructor) {
    Tensor t;
    ASSERT_EQ(t.get_length(), 0);
    ASSERT_EQ(t.get_element_size(), 0);
    ASSERT_EQ(t.get_data_ptr(), nullptr);
    ASSERT_EQ(t.get_total_size(), 0);
}

// 测试 int8_t 类型初始化
TEST_F(TensorTest, ParameterizedConstructor) {


    size_t length = 3;
    size_t element_size = sizeof(int8_t);
    size_t size = length*element_size;
    std::string name = "int8_tensor";

    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), length, element_size, name);

    EXPECT_EQ(t.get_length(), length);
    EXPECT_EQ(t.get_element_size(), element_size);
    EXPECT_EQ(t.get_data_ptr(), data_ptr);
    EXPECT_EQ(t.get_total_size(), length * element_size);
}

// 测试 bfloat16 类型初始化
TEST_F(TensorTest, BFloat16Initialization) {
    size_t length = 3;
    size_t element_size = sizeof(uint16_t);
    std::string name = "bfloat16_tensor";
    size_t size = length*element_size;

    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), length, element_size, name);

    EXPECT_EQ(t.get_length(), length);
    EXPECT_EQ(t.get_element_size(), element_size);
    EXPECT_EQ(t.get_data_ptr(), data_ptr);
    EXPECT_EQ(t.get_total_size(), length * element_size);

}

// 测试 float32 类型初始化
TEST_F(TensorTest, Float32Initialization) {
    size_t length = 3;
    size_t element_size = sizeof(float);
    std::string name = "bfloat16_tensor";
    size_t size = length*element_size;

    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), length, element_size, name);

    EXPECT_EQ(t.get_length(), length);
    EXPECT_EQ(t.get_element_size(), element_size);
    EXPECT_EQ(t.get_data_ptr(), data_ptr);
    EXPECT_EQ(t.get_total_size(), length * element_size);

}

// 测试 getter 方法
TEST_F(TensorTest, Getters) {
    size_t length = 3;
    size_t element_size = sizeof(int8_t);
    size_t size = length*element_size;
    std::string name = "int8_tensor";

    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), length, element_size, name);

    EXPECT_EQ(t.get_length(), length);
    EXPECT_EQ(t.get_element_size(), element_size);
    EXPECT_EQ(t.get_data_ptr(), data_ptr);
    EXPECT_EQ(t.get_total_size(), length * element_size);
}

// 测试空数据指针的情况
TEST_F(TensorTest, NullDataPointer) {
    Tensor t(nullptr, 5, sizeof(double), "null_tensor");

    EXPECT_EQ(t.get_length(), 5);
    EXPECT_EQ(t.get_element_size(), sizeof(double));
    EXPECT_EQ(t.get_data_ptr(), nullptr);
    EXPECT_EQ(t.get_total_size(), 5 * sizeof(double));
}

// 测试零长度的情况
TEST_F(TensorTest, ZeroLength) {
    size_t length = 3;
    size_t element_size = sizeof(int8_t);
    size_t size = length*element_size;
    std::string name = "zero_length_tensor";

    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), 0, element_size, name);

    EXPECT_EQ(t.get_length(), 0);
    EXPECT_EQ(t.get_element_size(), sizeof(int8_t));
    EXPECT_EQ(t.get_data_ptr(), data_ptr);
    EXPECT_EQ(t.get_total_size(), 0);
}

// 测试 to_host 正常情况
TEST_F(TensorTest, ToHost_Success) {
    std::vector<float> host_buffer(length,0.f);
    aclError ret = tensor->to_host(host_buffer.data());
    EXPECT_EQ(ret, ACL_ERROR_NONE);
    for (size_t i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(host_buffer[i], device_data[i]);
    }
}

// 测试 to_host 空指针异常
TEST_F(TensorTest, ToHost_NullHostPointer) {
    EXPECT_THROW(tensor->to_host(nullptr), std::runtime_error);
}

TEST_F(TensorTest, ToHost_NullDevicePointer) {
    Tensor null_tensor(nullptr, length, element_size, "null_tensor");
    float host_buffer[3] = {0.0f};
    EXPECT_THROW(null_tensor.to_host(host_buffer), std::runtime_error);
}

// 测试 to_device 正常情况
TEST_F(TensorTest, ToDevice_Success) {

    size_t length = 3;
    size_t element_size = sizeof(int8_t);
    size_t size = length*element_size;
    std::string name = "int8_tensor";
    void* data_ptr;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));

    Tensor t(static_cast<uint64_t*>(data_ptr), length, element_size, name);

    std::vector<int8_t> init_data(length, 1);

    aclError ret = t.to_device(init_data.data());
    EXPECT_EQ(ret, ACL_ERROR_NONE);

    std::vector<int8_t> device_data(length, 0);
    ret = t.to_host(device_data.data());
    EXPECT_EQ(ret, ACL_ERROR_NONE);
    for (size_t i = 0; i < length; ++i) {
        EXPECT_EQ(device_data[i], init_data[i]);
    }
}

// 测试 to_device 空指针异常
TEST_F(TensorTest, ToDevice_NullHostPointer) {
    EXPECT_THROW(tensor->to_device(nullptr), std::runtime_error);
}

TEST_F(TensorTest, ToDevice_NullDataPointer) {
    Tensor null_tensor(nullptr, length, element_size, "null_tensor");
    float host_buffer[3] = {0.0f};
    EXPECT_THROW(null_tensor.to_device(host_buffer), std::runtime_error);
}

// 多并发测试 to_host
TEST_F(TensorTest, ToHost_Concurrent) {
    const int thread_count = 4;
    std::vector<std::thread> threads;
    std::vector<float*> host_buffers(thread_count);


    for (int i = 0; i < thread_count; ++i) {
        host_buffers[i] = new float[length]{0.0f};
        threads.emplace_back([this, &host_buffers, i]() {
            aclInit(NULL); // 初始化 ACL
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            aclError ret = tensor->to_host(host_buffers[i]);
            EXPECT_EQ(ret, ACL_ERROR_NONE);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // 验证每个线程的结果
    for (int i = 0; i < thread_count; ++i) {
        for (size_t j = 0; j < length; ++j) {
            EXPECT_FLOAT_EQ(host_buffers[i][j], device_data[j]);
        }
        delete[] host_buffers[i];
    }
}

// 多并发测试 to_device
TEST_F(TensorTest, ToDevice_Concurrent) {
    const int thread_count = 4;
    std::vector<std::thread> threads;
    std::vector<float*> host_buffers(thread_count);
    void* data_ptr;
    size_t size = length*element_size;
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    Tensor* t;
    t = new Tensor(static_cast<uint64_t*>(data_ptr), length, element_size, "device_tensor");


    for (int i = 0; i < thread_count; ++i) {
        host_buffers[i] = new float[length]{static_cast<float>(i + 1), static_cast<float>(i + 2), static_cast<float>(i + 3)};
        threads.emplace_back([t, &host_buffers, i]() {
            aclInit(NULL); // 初始化 ACL
            aclrtContext context;
            aclrtCreateContext(&context, 0);
            aclrtSetCurrentContext(context);
            aclError ret = t->to_device(host_buffers[i]);
            EXPECT_EQ(ret, ACL_ERROR_NONE);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 注意：由于多线程同时写到同一块 device_data，结果取决于最后一个完成的线程
    // 这里仅验证操作成功，实际数据验证需要线程同步
}

