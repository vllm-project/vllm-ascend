// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "moe_weights.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream> // 用于文件操作
#include <iomanip>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h> // For shared memory (POSIX-like)
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

// #include <torch/extension.h>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace py = pybind11;

void ExpertWeights::enqueueSwapInformation(Distribution *dist_ptr,
                                           size_t t_rank,
                                           bool need_enqueue_recv_buff) {
    std::vector<size_t> lengths;
    std::vector<std::string> dtypes;
    std::vector<void *> address;
    std::vector<size_t> sizes;
    std::vector<void *> recv_buffs;
    std::vector<size_t> t_ranks;
    for (auto &weight : weights_) {
        lengths.push_back(weight.get_length());
        dtypes.push_back(weight.get_dtype());
        address.push_back(weight.get_data_ptr());
        sizes.push_back(weight.get_total_size());
        recv_buffs.emplace_back(dist_ptr->get_recv_buff_address(
            need_enqueue_recv_buff, weight.get_total_size()));
        t_ranks.push_back(t_rank);
    }
    TransDesc expert_trans_desc;
    expert_trans_desc.address = address;
    expert_trans_desc.lengths = lengths;
    expert_trans_desc.dtypes = dtypes;
    expert_trans_desc.sizes = sizes;
    expert_trans_desc.recv_buffs = recv_buffs;
    expert_trans_desc.t_rank = t_ranks;

    dist_ptr->prepare_batch(need_enqueue_recv_buff, expert_trans_desc);
}

size_t MoEWeights::get_expert_itemnum() {
    ExpertWeights expert = getExpert(0, 0);
    return expert.get_weights().size();
}

void MoEWeights::replacement(Distribution *dist_ptr, size_t layer_idx,
                             size_t source_rank, size_t source_global_position,
                             size_t target_rank, size_t target_global_position,
                             bool need_enqueue_recv_buff) {
    size_t localExpPos =
        source_rank == rank_ ? source_global_position : target_global_position;
    localExpPos = localExpPos % getNumDeployExpertsPerRank();
    ExpertWeights expert = getExpert(layer_idx, localExpPos);
    size_t t_rank = (source_rank == rank_) ? target_rank : source_rank;
    expert.enqueueSwapInformation(dist_ptr, t_rank, need_enqueue_recv_buff);
}

MoEWeights::MoEWeights(size_t num_experts) : num_experts_(num_experts) {
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    world_size_ = 1;
    num_deploy_experts_per_rank_ = num_experts_ / world_size_;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::MoEWeights(size_t num_experts, size_t world_size)
    : world_size_(world_size), num_experts_(num_experts) {
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    num_deploy_experts_per_rank_ = num_experts_ / world_size_;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::MoEWeights(size_t num_experts, size_t rank, size_t world_size)
    : rank_(rank), world_size_(world_size), num_experts_(num_experts) {
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    num_deploy_experts_per_rank_ = num_experts_ / world_size_;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::MoEWeights(size_t num_experts, size_t rank, size_t world_size,
                       const char *rankTableFile)
    : rank_(rank), world_size_(world_size), num_experts_(num_experts) {
    dist_ptr_ =
        new Distribution(num_experts / world_size, rank_, world_size_,
                         rankTableFile, HcclCommInitType::RankTableFile);
    dist_ptr_->printCommInfo();
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    num_deploy_experts_per_rank_ = num_experts_ / world_size_;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::MoEWeights(size_t num_experts, size_t rank, size_t world_size,
                       Distribution *dist_ptr)
    : rank_(rank), world_size_(world_size), num_experts_(num_experts),
      dist_ptr_(dist_ptr) {
    dist_ptr_->printCommInfo();
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::~MoEWeights() {
    delete dist_ptr_;
    // 清理控制共享内存
    if (shm_ptr_) {
        munmap(shm_ptr_, shm_size_);
        munmap(count_ptr_, sizeof(CountData));
        shm_unlink(shm_name_.c_str());
    }
}

void MoEWeights::init_weights(
    const std::vector<std::vector<std::vector<Tensor>>> &npu_weights,
    bool init_shm) {

    if (npu_weights.size() == 0) {
        throw std::runtime_error(
            "npu_weights.size() is 0, which is the layer dimension");
    }

    if (npu_weights[0].size() == 0) {
        throw std::runtime_error(
            "npu_weights[0].size() is 0, which is the experts dimension");
    }

    if (npu_weights[0].size() != num_deploy_experts_per_rank_) {
        throw std::runtime_error("npu_weights[0].size() is: " +
                                 std::to_string(npu_weights[0].size()) +
                                 " while num_deploy_experts_per_rank_ is:" +
                                 std::to_string(num_deploy_experts_per_rank_));
    }

    npu_weights_.resize(npu_weights.size()); // 预分配层数
    for (size_t layer_idx = 0; layer_idx < npu_weights.size(); ++layer_idx) {
        std::vector<ExpertWeights> layer_experts;
        layer_experts.resize(npu_weights[layer_idx].size()); // 预分配专家数
        for (size_t expert_idx = 0; expert_idx < npu_weights[layer_idx].size();
             ++expert_idx) {
            // 为每个专家创建 ExpertWeights 对象
            layer_experts.at(expert_idx) =
                ExpertWeights(npu_weights[layer_idx][expert_idx]);
        }
        npu_weights_.at(layer_idx) = std::move(layer_experts);
    }

    if (npu_weights_.size() == 0 || npu_weights_[0].size() == 0) {
        throw std::runtime_error("Invalid nums of layer or expert of "
                                 "npu_weights_: size cannot be 0");
    }

    npu_weights_[0][0].info();

    num_layers_ = npu_weights_.size();
    size_t expert_size =
        npu_weights_[0][0].get_total_size(); // FIXEME: 每个专家的大小均一致
    if (expert_size == 0) {
        throw std::runtime_error("Invalid size: size cannot be 0");
    }

    std::cout << "The Bytes of one Experts is: " << expert_size << std::endl;
    size_t total_size = num_layers_ * num_experts_ * expert_size;

    // TODO: 根据Queue Size 先分配一块显存
    is_initilized_ = true;
    if (!init_shm) {
        std::unique_lock<std::mutex> lock = acquire_lock();
        lock.unlock();
        return;
    }

    init_shared_memory(total_size); // TODO: 不需要 初始化SHM
    replicate_to_shared_memory();   // Initial copy to shared memory

    // 拷贝完成计数
    CountData *count_ptr = static_cast<CountData *>(count_ptr_);
    count_ptr->completed_processes.fetch_add(1);
}

size_t MoEWeights::get_expert_size() {
    ExpertWeights expert = getExpert(0, 0);
    return expert.get_total_size();
}

bool MoEWeights::isHbmInitialized() const {
    std::unique_lock<std::mutex> lock = acquire_lock();
    bool result = is_initilized_;
    lock.unlock();
    return result;
}

bool MoEWeights::isShmInitialized() const {
    if (count_ptr_ == nullptr) {
        // 还没有在共享内存完成初始化
        return false;
    } else {
        CountData *count_ptr = static_cast<CountData *>(count_ptr_);
        size_t count = count_ptr->completed_processes.load();
        return count == world_size_;
    }
}

// // 创建或附加共享内存
void *MoEWeights::create_or_attach_shmem(const std::string &name, size_t size) {
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    bool is_creator;
    if (fd >= 0) {
        is_creator = true;
        if (ftruncate(fd, shm_size_) == -1) {
            close(fd);
            shm_unlink(name.c_str());
            throw std::runtime_error("ftruncate failed");
        }
    } else if (errno == EEXIST) {
        // 共享内存已存在，直接打开
        fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            throw std::runtime_error("Failed to open existing shared memory");
        }
        is_creator = false;
    } else {
        throw std::runtime_error("shm_open failed");
    }

    size_t total_size = size + sizeof(CountData); // 加入计数符号位
    void *ptr =
        mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED)
        throw std::runtime_error("mmap failed");

    count_ptr_ = ptr;
    ptr = static_cast<void *>(static_cast<char *>(ptr) + sizeof(CountData));

    CountData *count_ptr = static_cast<CountData *>(count_ptr_);
    if (is_creator) {
        count_ptr->completed_processes.store(0);
        count_ptr->init_flag.store(999);
    } else {
        // 非创建 进程等待完成初始化操作
        size_t check_shm_unrelease = 0;
        while (true) {
            if (count_ptr->init_flag.load() == 999) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // 异常退出导致共享内存未被正常释放提示
            ++check_shm_unrelease;
            if (check_shm_unrelease > 20) {
                std::cout << "Noticed: A Long SHM Init time cost: "
                          << check_shm_unrelease
                          << "s , SHM seems unreleased, Pls Check /dev/shm"
                          << std::endl;
            }
        }
    }
    return ptr;
}

// Initialize shared memory
void MoEWeights::init_shared_memory(size_t shm_size) {
    assert(sizeof(CountData) <= 64 && "sizeof(CountData) is larger than 64");
    shm_size_ = shm_size + 64 - sizeof(CountData);
    shm_ptr_ =
        static_cast<void *>(create_or_attach_shmem(shm_name_, shm_size_));
}

bool is_within_bounds(char *shm_ptr, size_t shm_size, char *shm_ptr_current,
                      size_t cp_size) {
    // 检查指针有效性
    if (shm_ptr == nullptr || shm_ptr_current == nullptr) {
        return false;
    }

    // 计算当前偏移量
    ptrdiff_t offset = shm_ptr_current - shm_ptr;

    // 检查是否在起始地址之前
    if (offset < 0) {
        return false; // 当前指针在 shm_ptr 之前，无效
    }

    // 转换为无符号类型，避免负数问题
    size_t unsigned_offset = static_cast<size_t>(offset);

    // 检查是否超出总大小
    if (unsigned_offset + cp_size > shm_size) {
        return false; // 超出范围
    }

    // 检查溢出（可选，size_tmp 过大时）
    if (unsigned_offset + cp_size < unsigned_offset) {
        return false; // cp_size 太大导致溢出
    }

    return true; // 在范围内
}

// 1. Helper to copy weights to shared memory
void MoEWeights::replicate_to_shared_memory() {
    // 确保共享内存指针已初始化
    assert(shm_ptr_ != nullptr && "Shared memory pointer is not initialized");

    char *shm_ptr = static_cast<char *>(shm_ptr_);
    char *shm_ptr_current = nullptr;
    int layer_idx = -1;
    int expert_id = 0;

    // 遍历两层结构: num_layers -> experts_per_layer
    for (const auto &layer : npu_weights_) {
        layer_idx++;
        assert(layer.size() > 0);
        for (const auto &expert_weights : layer) {
            expert_id = expert_weights.get_expert_id();
            assert(expert_id >= 0 && (size_t)expert_id < num_experts_);
            size_t expert_size = expert_weights.get_total_size();
            shm_ptr_current =
                shm_ptr + (layer_idx * num_experts_ + expert_id) *
                              expert_size; // FIXME: 所有专家默认字节大小都一样

            // 检查共享内存地址拷贝范围是否合法
            if (not is_within_bounds(shm_ptr, shm_size_, shm_ptr_current,
                                     expert_size)) {
                throw std::runtime_error(
                    "Target memory (shm_ptr_current) is unvalided!");
            }
            aclError ret = expert_weights.to_host(shm_ptr_current);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                         std::to_string(ret));
            }
            shm_ptr_current += expert_size;
        }
    }
}
