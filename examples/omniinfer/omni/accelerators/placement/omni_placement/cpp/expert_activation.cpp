// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "expert_activation.h"
#include "config.h"
#include "distribution.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>

namespace py = pybind11;

void ClusterActivation::stopDump() { enable_dump_ = false; }

void ClusterActivation::setDumpDir(const std::string &dump_dir) {
    struct stat info;
    if (stat(dump_dir.c_str(), &info) != 0) {
        if (mkdir(dump_dir.c_str(), 0755) != 0) {
            perror("Error creating directory");
            dump_dir_ = "";
            enable_dump_ = false;
        } else {
            dump_dir_ = dump_dir;
            enable_dump_ = true;
        }
    } else if (info.st_mode & S_IFDIR) {
        dump_dir_ = dump_dir;
        enable_dump_ = true;
    } else {
        dump_dir_ = "";
        enable_dump_ = false;
    }
}

ClusterActivation::ClusterActivation(
    Tensor npu_count, int64_t max_activation_count, size_t num_layers,
    size_t num_deploy_experts_per_rank, int activation_window_size,
    size_t world_size, size_t hccl_comm_world_size, size_t rank)
    : npu_count_(npu_count), max_activation_count_(max_activation_count),
      num_layers_(num_layers),
      num_deploy_experts_per_rank_(num_deploy_experts_per_rank),
      activation_window_size_(activation_window_size), world_size_(world_size),
      hccl_comm_world_size_(hccl_comm_world_size), rank_(rank) {
    // Validate max_activation_count
    if (max_activation_count <= 0) {
        throw std::invalid_argument("max_activation_count must be positive");
    }
    // Validate num_layers
    if (num_layers == 0) {
        throw std::invalid_argument("num_layers must be greater than zero");
    }
    // Validate num_deploy_experts_per_rank
    if (num_deploy_experts_per_rank == 0) {
        throw std::invalid_argument(
            "num_deploy_experts_per_rank must be greater than zero");
    }
    // Validate world_size
    if (world_size == 0) {
        throw std::invalid_argument("world_size must be greater than zero");
    }

    // Validate hccl_comm_world_size
    if (hccl_comm_world_size == 0) {
        throw std::invalid_argument(
            "hccl_comm_world_size must be greater than zero");
    }

    // Additional consistency checks
    if (hccl_comm_world_size < world_size) {
        throw std::invalid_argument(
            "hccl_comm_world_size cannot be less than world_size");
    }

    if (npu_count_.get_data_ptr() == nullptr) {
        throw std::invalid_argument(
            "Current Tensor npu_count_'s HBM address is nullptr, which may not "
            "be initialized.");
    }
    // 约束Tensor的 element_size 为 int
    if (npu_count_.get_element_size() != sizeof(int64_t)) {
        throw std::invalid_argument(
            "Current Each Count Tensor Element Size is: " +
            std::to_string(npu_count_.get_element_size()) +
            ", while only support element size: " +
            std::to_string(sizeof(int64_t)) + " now");
    }
    if (get_rank() >= get_hccl_comm_world_size()) {
        throw std::runtime_error(
            "Current Rank is: " + std::to_string(get_rank()) +
            " Current world_size is :" +
            std::to_string(get_hccl_comm_world_size()));
    }

    // Since local tokens global experts -> glocal tokens local experts,
    // npu_coun_.get_length() must be less than
    // get_num_layers()*(size_t)get_num_deploy_experts_per_rank()
    if (npu_count_.get_length() >
        get_num_layers() * (size_t)get_num_deploy_experts_per_rank()) {
        throw std::runtime_error(
            "npu_count's length is: " +
            std::to_string(npu_count_.get_length()) +
            " , which is larger than Current total layer and experts_per_layer "
            "is "
            ":" +
            std::to_string(get_num_layers()) + "*" +
            std::to_string(get_num_deploy_experts_per_rank()));
    }

    init_activation_hbm();
    // init_activation_shmem();
    size_t total_size = npu_count_.get_total_size(); // 用于分配Host内存
    total_count_ptr_ = malloc(total_size);
    memset(total_count_ptr_, 0, total_size);

    thread_state_ = ThreadState::INIT;

    // 启动线程监听并更新专家激活信息
    // start_thread();  //合并到placement_manager
}

ClusterActivation::~ClusterActivation() {
    // stop_thread(); // 析构时安全关闭线程
    if (act_shm_ptr_) {
        munmap(act_shm_ptr_, act_shm_size_);
        shm_unlink(act_shm_name_.c_str());
    }
    if (expert_activation_counts_ != nullptr) {
        ACLCHECK(aclrtFree(expert_activation_counts_->get_data_ptr()));
        delete expert_activation_counts_;
    }
    free(total_count_ptr_);
    free(last_count_ptr_);
    free(deployed_experts_counts_host_);
    free(delta_experts_counts_);
}

void ClusterActivation::set_params(size_t num_experts) {
    num_experts_ = num_experts;
    all_logit_experts_activation_.resize(num_layers_ * num_experts_);
}

// 创建或附加共享内存
void *ClusterActivation::create_or_attach_shmem(const std::string &name,
                                                size_t size) {
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1)
        throw std::runtime_error("shm_open failed");

    if (ftruncate(fd, size) == -1) {
        close(fd);
        throw std::runtime_error("ftruncate failed");
    }

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED)
        throw std::runtime_error("mmap failed");
    return ptr;
}

// 初始化激活共享内存
void ClusterActivation::init_activation_hbm() {

    num_deploy_experts_ = world_size_ * num_deploy_experts_per_rank_;
    void *data_ptr;
    size_t length =
        num_layers_ * get_hccl_comm_world_size() * num_deploy_experts_per_rank_;
    size_t total_size = length * sizeof(int64_t);
    ACLCHECK(aclrtMalloc(&data_ptr, total_size, ACL_MEM_MALLOC_HUGE_FIRST));
    expert_activation_counts_ =
        new Tensor(static_cast<uint64_t *>(data_ptr), length, sizeof(int64_t),
                   "expert_activation_counts_");
    expert_activation_counts_->set_zero();

    // Set Host
    deployed_experts_counts_host_ = malloc(total_size);
    delta_experts_counts_ = malloc(total_size);
    last_count_ptr_ = malloc(total_size);
    memset(deployed_experts_counts_host_, 0, total_size);
    memset(last_count_ptr_, 0, total_size);
    memset(delta_experts_counts_, 0, total_size);
}

void ClusterActivation::start_thread() {
    // TODO: 两个版本后废弃
    assert(thread_state_ == ThreadState::INIT);
    thread_state_ = ThreadState::RUNNING;
}

void ClusterActivation::stop_thread() {
    thread_state_ = ThreadState::STOPPED;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ClusterActivation::dumpActivationCountsPerRank(size_t dump_count,
                                                    int64_t *total_count_ptr) {
    std::cout << "EXPERT_COUNTS For RecordStep:" << std::to_string(dump_count)
              << std::endl;
    std::ostringstream oss;
    for (size_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        for (size_t j = 0; j < num_deploy_experts_per_rank_; ++j) {
            size_t offset = layer_id * num_deploy_experts_per_rank_ + j;
            int64_t countDiff =
                total_count_ptr[offset]; // 旧接口， 依旧保留使用，无滑动窗口
            oss << countDiff << "\t";
        }
        oss << std::endl;
    }
    std::cout << oss.str() << std::endl; // Print to screen
    if (!dump_dir_.empty()) {
        std::string filename = dump_dir_ +
                               "_perRank/activation_counts_recordstep_" +
                               std::to_string(dump_count) + "_rank_" +
                               std::to_string(get_rank()) + ".txt";
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename
                      << std::endl;
            return;
        }
        outFile << oss.str();
        outFile.close();
    }
}

void ClusterActivation::dumpActivationCounts(size_t dump_count) {

    if (rank_ == 0) {
        std::cout << "EXPERT_COUNTS For RecordStep:"
                  << std::to_string(dump_count) << std::endl;
    }

    std::ostringstream oss;
    for (size_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
        for (size_t rank_id = 0; rank_id < world_size_; ++rank_id) {
            for (size_t j = 0; j < num_deploy_experts_per_rank_; ++j) {
                int physical_pos = rank_id * num_deploy_experts_per_rank_ + j;
                int64_t countDiff = getExpertActivationCount(
                    layer_id, physical_pos,
                    delta_experts_counts_); // 旧接口， 依旧保留使用，无滑动窗口
                oss << countDiff << "\t";
            }
        }
        oss << std::endl;
    }

    if (rank_ == 0) {
        std::cout << oss.str() << std::endl; // Print to screen
    }

    if (!dump_dir_.empty() && rank_ == 0) {
        std::string filename = dump_dir_ + "/activation_counts_recordstep_" +
                               std::to_string(dump_count) + "_rank_" +
                               std::to_string(get_rank()) + ".txt";
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename
                      << std::endl;
            return;
        }
        outFile << oss.str();
        outFile.close();
    }
}

void ClusterActivation::collect(Distribution *dist_ptr, aclrtStream stream) {
    aclError ret = npu_count_.to_host(total_count_ptr_, stream);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                 std::to_string(ret));
    }

    dist_ptr->allgather(npu_count_.get_data_ptr(),
                        expert_activation_counts_->get_data_ptr(),
                        npu_count_.get_length(), npu_count_.get_dtype());

    expert_activation_counts_->to_host(deployed_experts_counts_host_, stream);

    updateDeltaActivationCount();
}

void ClusterActivation::dump_and_collect(Distribution *dist_ptr,
                                         aclrtStream stream,
                                         size_t dump_count) {

    collect(dist_ptr, stream);
    // 通信后进行打印
    if (is_enbale_dump()) {
        dumpActivationCounts(dump_count);
    }
}

void ClusterActivation::updateDeltaActivationCount() {
    size_t total_size =
        world_size_ * num_layers_ * num_deploy_experts_per_rank_;
    int64_t *counts_host =
        static_cast<int64_t *>(deployed_experts_counts_host_);
    int64_t *last_count = static_cast<int64_t *>(last_count_ptr_);
    int64_t *delta_counts = static_cast<int64_t *>(delta_experts_counts_);

    for (size_t offset = 0; offset < total_size; ++offset) {
        int64_t current = counts_host[offset];
        int64_t last = last_count[offset];
        if (current < last)
            current += max_activation_count_;
        delta_counts[offset] = current - last;
    }
    // Copy deployed_experts_counts_host_ to last_count_ptr_ using memcpy
    std::memcpy(last_count_ptr_, deployed_experts_counts_host_,
                total_size * sizeof(int64_t));
}
void ClusterActivation::updateShiftWindows(
    PlacementMapping *placement_mapping) {
    for (int layer_id = 0; layer_id < placement_mapping->get_num_layers();
         ++layer_id) {
        std::vector<int64_t> logit_expert_activations(num_experts_, 0);
        int64_t total_value = 0;
        for (int global_position_id = 0;
             global_position_id < placement_mapping->get_num_deploy_experts();
             ++global_position_id) {
            int32_t expert_id =
                placement_mapping->get_expert_id(layer_id, global_position_id);
            int64_t value = getExpertActivationCount(
                layer_id, global_position_id, delta_experts_counts_);
            total_value = total_value + value;
            if (expert_id == -1)
                continue;
            logit_expert_activations[expert_id] =
                logit_expert_activations[expert_id] + value;
        }

        if (total_value != 0) {
            for (size_t idx = 0; idx < num_experts_; ++idx) {
                size_t offset = layer_id * num_experts_ + idx;
                all_logit_experts_activation_[offset].update(
                    logit_expert_activations[idx]);
            }
        }
    }
}

// Get activations
int64_t ClusterActivation::getLayerActivationCount(size_t layer_idx) {
    if (layer_idx >= num_layers_) {
        throw std::runtime_error("Invalid layer or expert, which is: layer: " +
                                 std::to_string(layer_idx) + " : " +
                                 std::to_string(num_layers_));
    }

    int64_t layer_counts = 0;
    int64_t *deployed_experts_counts_host =
        static_cast<int64_t *>(deployed_experts_counts_host_);
    for (size_t rank = 0; rank < world_size_; rank++) {
        for (size_t pos = 0; pos < num_deploy_experts_per_rank_; pos++) {
            size_t idx = rank * num_layers_ * num_deploy_experts_per_rank_ +
                         layer_idx * num_deploy_experts_per_rank_ + pos;
            layer_counts += deployed_experts_counts_host[idx];
        }
    }
    return layer_counts;
}

// Get activations
int64_t
ClusterActivation::getClusterTotalActivationCount(size_t layer_idx,
                                                  size_t deploy_expert_idx) {
    // TODO Disused on Next Version
    if (layer_idx >= num_layers_ || deploy_expert_idx >= num_deploy_experts_) {
        throw std::runtime_error(
            "Invalid layer or expert, which is: layer: " +
            std::to_string(layer_idx) + " : " + std::to_string(num_layers_) +
            " expert_idx: " + std::to_string(deploy_expert_idx) + " : " +
            std::to_string(num_deploy_experts_));
    }
    return getExpertActivationCount(layer_idx, deploy_expert_idx,
                                    delta_experts_counts_);
}

// 读取 expert_activation_counts_[layer][expert] 的值
int64_t ClusterActivation::getExpertActivationCount(int32_t layer_id,
                                                    int32_t expert_id) {
    return getExpertActivationCount(layer_id, expert_id, delta_experts_counts_);
}
int64_t ClusterActivation::getExpertActivationCount(int32_t layer_id,
                                                    int32_t expert_id,
                                                    void *activation_ptr) {
    if (layer_id < 0 || layer_id >= (int32_t)num_layers_) {
        throw std::out_of_range("layer_id out of range: " +
                                std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= (int32_t)num_deploy_experts_) {
        throw std::out_of_range("expert_id out of range: " +
                                std::to_string(expert_id));
    }

    size_t t_rank = expert_id / num_deploy_experts_per_rank_;
    size_t local_expert_idx = expert_id % num_deploy_experts_per_rank_;
    size_t total_offset = t_rank * num_layers_ * num_deploy_experts_per_rank_ +
                          layer_id * num_deploy_experts_per_rank_ +
                          local_expert_idx;

    int64_t *activation = static_cast<int64_t *>(activation_ptr);

    int64_t value = activation[total_offset];
    return value;
}

int64_t ClusterActivation::getLogitExpertShiftActivateion(
    int32_t layer_id, int32_t expert_id, int this_expert_deployed_num) {

    if (layer_id < 0 || layer_id >= (int32_t)num_layers_) {
        throw std::out_of_range("layer_id out of range: " +
                                std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= (int32_t)num_experts_) {
        throw std::out_of_range("expert_id out of range: " +
                                std::to_string(expert_id));
    }

    int offset = layer_id * num_experts_ + expert_id;
    int64_t value = all_logit_experts_activation_[offset].getTotalValue();
    value = value / this_expert_deployed_num;
    return value;
}

void ClusterActivation::collect_from_txt(const std::string &txt_path) {

    // Open the file
    std::ifstream file(txt_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + txt_path);
    }

    std::string line;
    int layer_id = 0;

    int64_t *delta_counts = static_cast<int64_t *>(delta_experts_counts_);

    // Read the file line by line
    while (std::getline(file, line) && layer_id < 58) {
        std::vector<int64_t> row;
        row.reserve(320);

        std::istringstream iss(line);
        std::string token;
        int pos_id = 0;

        while (std::getline(iss, token, '\t')) {
            if (pos_id >= 320) {
                throw std::runtime_error("Row " + std::to_string(layer_id + 1) +
                                         " has more than 320 elements");
            }

            // Trim whitespace (optional, if needed)
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);

            size_t pos;
            int64_t value = std::stoll(token, &pos);
            size_t rank = pos_id / num_deploy_experts_per_rank_;

            size_t offset = rank * num_layers_ * num_deploy_experts_per_rank_ +
                            layer_id * num_deploy_experts_per_rank_ +
                            pos_id % num_deploy_experts_per_rank_;
            delta_counts[offset] = value;

            ++pos_id;
        }
        ++layer_id;
    }
    file.close();
}

// 打印线程不再直接访问成员变量
// FIXME: Plz Consider different rank
void ClusterActivation::print_activations() {}
