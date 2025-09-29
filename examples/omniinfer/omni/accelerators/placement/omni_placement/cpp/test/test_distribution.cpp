// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <random>
#include "distribution.h"
#include <iostream>



class DistributionTest : public ::testing::Test {
protected:
    const char* rank_table_file = "/home/kww/20250507/omni_infer/rank_tables/rank_table_superPod.json";

    void SetUp() override {
    };
    void TearDown() override {

    }
};

// 测试默认构造函数
TEST_F(DistributionTest, DefaultConstructor) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        aclrtContext context;
        ACLCHECK(aclrtCreateContext(&context, 0));
        ACLCHECK(aclrtSetCurrentContext(context));
        aclrtStream stream;
        ACLCHECK(aclrtCreateStream(&stream));
        Distribution* dist_ptr = new Distribution(0,4, rank_table_file,HcclCommInitType::RankTableFile);
        delete dist_ptr;
        ACLCHECK(aclrtSynchronizeStream(stream));
        ACLCHECK(aclrtDestroyStream(stream));
        ACLCHECK(aclrtDestroyContext(context));
        exit(0);
    }
    else{
        waitpid(pid, nullptr, 0);
    }
}

void multiThreadInitWorker(size_t device, size_t world_size,const char* rank_table_file){
    aclrtContext context;
    ACLCHECK(aclrtCreateContext(&context, device));
    ACLCHECK(aclrtSetCurrentContext(context));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    Distribution* dist_ptr = new Distribution(device, world_size, rank_table_file,HcclCommInitType::RankTableFile);
    delete dist_ptr;
    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyContext(context));
    return;
}

// 每个进程中只能建立一次连接
TEST_F(DistributionTest, DefaultConstructorWithMultiProcess) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 4; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            multiThreadInitWorker(i,num_devices,rank_table_file);
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

void swap_worker(size_t device,size_t world_size, const char* rank_table_file, size_t t_rank){
    aclrtContext context;
    ACLCHECK(aclrtCreateContext(&context, device));
    ACLCHECK(aclrtSetCurrentContext(context));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    Distribution* dist_ptr = new Distribution(device, world_size, rank_table_file,HcclCommInitType::RankTableFile);
    
    
    size_t data_size = sizeof(int);
    // std::cout<<"sizeof(int): "<<sizeof(int)<<std::endl;
    void* data_ptr;
    void* recv_buf;

    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&recv_buf, data_size, ACL_MEM_MALLOC_HUGE_FIRST));

    // 主机缓冲区用于初始化和验证
    void *host_buf;
    ACLCHECK(aclrtMallocHost((void**)&host_buf, data_size));
    int* tmpHostBuff = static_cast<int*>(host_buf);
    for (uint32_t i = 0; i < 1; ++i) {
        tmpHostBuff[i] = device;
    }

    ACLCHECK(aclrtMemcpy(data_ptr, data_size, host_buf, data_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtSynchronizeStream(stream));
    bool send_first = device<t_rank;
    dist_ptr->swap(data_ptr, recv_buf, 1, "int", t_rank, send_first, stream);
    // Multi-Times Exchanged !
    dist_ptr->swap(data_ptr, recv_buf, 1, "int", t_rank, send_first, stream);
    dist_ptr->swap(data_ptr, recv_buf, 1, "int", t_rank, send_first, stream);
    ACLCHECK(aclrtMemcpy((void*)host_buf, data_size, recv_buf, data_size, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_EQ(static_cast<int*>(host_buf)[0], t_rank);

    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFree(data_ptr));
    ACLCHECK(aclrtFree(recv_buf));

    
    delete dist_ptr;
    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyContext(context));
    // ACLCHECK(aclrtResetDevice(device));
}



// 每个进程中只能建立一次连接
TEST_F(DistributionTest, SwapWithIntDataWithTwoProcessForOnePairs) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 2; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            swap_worker(i, 4, rank_table_file, (i + 1) % num_devices);
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

// 每个进程中只能建立一次连接
TEST_F(DistributionTest, SwapWithIntDataWithTwoProcessForOnePairsRerun) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 2; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            swap_worker(i, 4, rank_table_file, (i + 1) % num_devices);
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

void swap_int8_worker(size_t device,size_t world_size, const char* rank_table_file, size_t t_rank){
    aclrtContext context;
    ACLCHECK(aclrtCreateContext(&context, device));
    ACLCHECK(aclrtSetCurrentContext(context));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    Distribution* dist_ptr = new Distribution(device, world_size, rank_table_file,HcclCommInitType::RankTableFile);
    
    
    size_t length =10;
    size_t data_size = sizeof(int8_t)*length;
    // std::cout<<"sizeof(int): "<<sizeof(int)<<std::endl;
    void* data_ptr;
    void* recv_buf;

    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&recv_buf, data_size, ACL_MEM_MALLOC_HUGE_FIRST));

    // 主机缓冲区用于初始化和验证
    void *host_buf;
    ACLCHECK(aclrtMallocHost((void**)&host_buf, data_size));
    int8_t* tmpHostBuff = static_cast<int8_t*>(host_buf);

    for (uint32_t i = 0; i < length; ++i) {
        tmpHostBuff[i] = static_cast<int8_t>(device + i * 10);
    }


    ACLCHECK(aclrtMemcpy(data_ptr, data_size, host_buf, data_size, ACL_MEMCPY_HOST_TO_DEVICE));

    bool send_first = device<t_rank;
    dist_ptr->swap(data_ptr, recv_buf, length, "int8", t_rank, send_first, stream);
    // Multi-Times Exchanged !
    dist_ptr->swap(data_ptr, recv_buf, length, "int8", t_rank, send_first, stream);
    dist_ptr->swap(data_ptr, recv_buf, length, "int8", t_rank, send_first, stream);
    ACLCHECK(aclrtMemcpy((void*)host_buf, data_size, recv_buf, data_size, ACL_MEMCPY_DEVICE_TO_HOST));


    tmpHostBuff = static_cast<int8_t*>(host_buf);
    for (uint32_t i = 0; i < length; ++i) {
        EXPECT_EQ(tmpHostBuff[i], static_cast<int8_t>(t_rank+i*10));
    }
    

    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFree(data_ptr));
    ACLCHECK(aclrtFree(recv_buf));

    
    delete dist_ptr;
    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyContext(context));
    // ACLCHECK(aclrtResetDevice(device));
}


// 每个进程中只能建立一次连接
TEST_F(DistributionTest, SwapWithInt8VectorDataWithTwoProcessForOnePairs) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 2; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            swap_int8_worker(i, 4, rank_table_file, (i + 1) % num_devices);
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

void enqueueWorker(size_t device,size_t world_size, const char* rank_table_file, std::vector<size_t> t_ranks){
    aclrtContext context;
    ACLCHECK(aclrtCreateContext(&context, device));
    ACLCHECK(aclrtSetCurrentContext(context));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));
    Distribution* dist_ptr = new Distribution(device, world_size, rank_table_file,HcclCommInitType::RankTableFile);
    
    size_t length =10;
    size_t data_size = sizeof(int8_t)*length;
    void* data_ptr;

    ASSERT_EQ(ACL_ERROR_NONE, aclrtMalloc(&data_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST));
    // 主机缓冲区用于初始化和验证
    void *host_buf;
    ACLCHECK(aclrtMallocHost((void**)&host_buf, data_size));
    int8_t* tmpHostBuff = static_cast<int8_t*>(host_buf);
    for (uint32_t i = 0; i < length; ++i) {
        tmpHostBuff[i] = static_cast<int8_t>(device + i * 10);
    }

    


    ACLCHECK(aclrtMemcpy(data_ptr, data_size, host_buf, data_size, ACL_MEMCPY_HOST_TO_DEVICE));

    for (auto& t_rank : t_ranks){
        TransDesc desc;
        std::vector<void*> pAddrs = { data_ptr };
        std::vector<size_t> lengths = { length };
        std::vector<size_t> sizes = { data_size };
        std::vector<std::string> dtypes = { "int8" };
        desc.address = pAddrs;
        desc.lengths = lengths;
        desc.sizes = sizes;
        desc.dtypes = dtypes;
        desc.instruction.rank_b=t_rank;
        dist_ptr->enqueue(&desc);
    }
    delete dist_ptr; // 保证同步

    ACLCHECK(aclrtMemcpy((void*)host_buf, data_size, data_ptr, data_size, ACL_MEMCPY_DEVICE_TO_HOST));

    size_t t_rank = t_ranks.back();
    tmpHostBuff = static_cast<int8_t*>(host_buf);
    for (uint32_t i = 0; i < length; ++i) {
        EXPECT_EQ(tmpHostBuff[i], static_cast<int8_t>(t_rank+i*10));
    }
    

    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFree(data_ptr));

    
    
    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyContext(context));
    // ACLCHECK(aclrtResetDevice(device));
}

// 每个进程中只能建立一次连接
TEST_F(DistributionTest, EnqueueWithInt8VectorDataWithTwoProcessForOnePairs) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 2; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            enqueueWorker(i, 4, rank_table_file, {(i + 1) % num_devices});
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

// 0 <-> 1; 0 <-> 2
TEST_F(DistributionTest, EnqueueWithInt8VectorDataWithTwoProcessForThreePairsOfFourDevices) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 4; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            if (i==0){
                enqueueWorker(i, 4, rank_table_file, {1,2,3});
            }
            else{
                enqueueWorker(i, 4, rank_table_file, {0});
            }
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

TEST_F(DistributionTest, EnqueueWithInt8VectorDataWithTwoProcessForThreePairsOfThreeDevices) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 3; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            if (i==0){
                enqueueWorker(i, 4, rank_table_file, {1,2});
            }
            else if (i==1){
                enqueueWorker(i, 4, rank_table_file, {2,0});
            }
            else{
                enqueueWorker(i, 4, rank_table_file, {0,1});
            }
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

TEST_F(DistributionTest, EnqueueWithInt8VectorDataWithTwoProcessFor12PairsOf4Devices) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 4; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            if (i==0){
                enqueueWorker(i, 4, rank_table_file, {1,2,3});
            }
            else if (i==1){
                enqueueWorker(i, 4, rank_table_file, {3,0,2});
            }
            else if (i==2){
                enqueueWorker(i, 4, rank_table_file, {1,3,0});
            }
            else{
                enqueueWorker(i, 4, rank_table_file, {0,2,1});
            }
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

TEST_F(DistributionTest, EnqueueWithInt8VectorDataWithTwoProcessFor2PairsOf4Devices) {
    std::vector<pid_t> processes;
    // Create multiple processes to initialize Distribution objects
    size_t num_devices = 4; // Assuming we have 4 devices for this test
    for (size_t i = 0; i < num_devices; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            if (i==0){
                enqueueWorker(i, 4, rank_table_file, {1});
            }
            else if (i==1){
                enqueueWorker(i, 4, rank_table_file, {0});
            }
            else if (i==2){
                enqueueWorker(i, 4, rank_table_file, {3});
            }
            else{
                enqueueWorker(i, 4, rank_table_file, {2});
            }
            exit(0);
        } else if (pid < 0) {
            // Error handling
            perror("fork");
            exit(EXIT_FAILURE);
        } else {
            // Parent process
            processes.push_back(pid);
        }
    }

    // Wait for all child processes
    for (pid_t pid : processes) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child process %d failed with status %d\n", pid, WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        }
    }
}

