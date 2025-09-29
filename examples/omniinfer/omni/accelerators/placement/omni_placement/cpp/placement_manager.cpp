// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_manager.h"
#include "config.h"
#include "expert_activation.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "moe_weights.h"
#include "placement_optimizer.h"
#include "tensor.h"
#include <future>
#include <libgen.h>
#include <limits.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <sstream>

std::string getTimestamp() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为 time_t（秒级精度）和本地时间结构体 std::tm
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&time);

    // 提取毫秒或微秒部分
    auto since_epoch = now.time_since_epoch();
    // auto ms =
    // std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch) %
    // 1000;
    // // 毫秒
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(since_epoch) %
        1000000; // 微秒

    // 格式化输出
    std::ostringstream oss;
    oss << "[" << std::put_time(&localTime, "%T"); // 先输出 "HH:MM:SS"

    // 追加毫秒或微秒
    // oss << ":" << std::setfill('0') << std::setw(3) << ms.count() << "] "; //
    // 毫秒版
    oss << ":" << std::setfill('0') << std::setw(6) << us.count()
        << "] "; // 微秒版

    return oss.str(); // 返回 "HH:MM:SS:ms" 或 "HH:MM:SS:us"
}

struct TimeTracker {
    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> last;

    void check_and_print(const std::string &tag) {
        auto now = clock::now();
        auto duration = now - last;
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                .count();

        if (ms >= 1) { // 仅打印超过1ms的耗时
            std::cout << getTimestamp() << "[elapse warn] " << tag << ": " << ms
                      << "ms\n";
        }

        last = now;
    }

    static TimeTracker &instance() {
        static TimeTracker tracker;
        return tracker;
    }

    void reset() { last = clock::now(); }
};

#define TRACK_START() TimeTracker::instance().reset()
#define TRACK_POINT(tag) TimeTracker::instance().check_and_print(tag)

OmniConfig global_omni_config;

namespace py = pybind11;

/**
 * Constructor for Placement class
 *
 * @param rank Global device ID
 * @param world_size Number of devices in the world
 * @param num_devices_per_host Number of devices per host
 * @param activations Pointer to ClusterActivation object
 * @param expert_mapping_ptr Pointer to expert mapping data
 * @param shape Shape of expert mapping data
 * @param dtype Data type of expert mapping data
 * @param placement_pattern_ptr Pointer to placement pattern data
 * @param placement_shape Shape of placement pattern data
 * @param placement_dtype Data type of placement pattern data
 *
 * Calls initialize_components immediately and starts a separate thread to check
 * shared memory weights
 */
Placement::Placement(int rank, int world_size, int hccl_comm_world_size,
                     int num_devices_per_host, ClusterActivation *activations,
                     PlacementMapping *placement_mapping, char *root_info,
                     bool enable_dynamic)
    : rank_(rank), world_size_(world_size),
      hccl_comm_world_size_(hccl_comm_world_size),
      num_devices_per_host_(num_devices_per_host), activations_(activations),
      mapping_(placement_mapping), enable_dynamic_(enable_dynamic) {

    // Initialize components immediately
    initialize_components(root_info);
    activations_->set_params(num_experts_);
    is_layer_update.resize(num_layers_, false);

    // Start a separate thread to check shared memory weights
    // start_thread(); // 快速验证
    // Shm is instead of HCCL, No need to check weights is ready or not!
    // init_thread_ = std::thread(&Placement::check_shm_weights, this);
    // init_thread_.detach();
}

void Placement::initialize_components(char *root_info) {
    num_layers_ = mapping_->get_num_layers();
    num_experts_ = mapping_->get_num_experts();
    num_deploy_experts_ = mapping_->get_num_deploy_experts();
    num_deploy_experts_per_rank_ = num_deploy_experts_ / world_size_;

    dist_ptr_ = new Distribution(rank_, hccl_comm_world_size_, root_info,
                                 HcclCommInitType::RootInfoString);
    moe_weight_ = new MoEWeights(num_deploy_experts_, rank_, world_size_);
    optimizer_ = new PlacementOptimizer(mapping_, activations_);
}

/**
 * Constructor for Placement class
 *
 * @param rank Global device ID
 * @param world_size Number of devices in the world
 * @param num_devices_per_host Number of devices per host
 * @param activations Pointer to ClusterActivation object
 * @param expert_mapping_ptr Pointer to expert mapping data
 * @param shape Shape of expert mapping data
 * @param dtype Data type of expert mapping data
 * @param placement_pattern_ptr Pointer to placement pattern data
 * @param placement_shape Shape of placement pattern data
 * @param placement_dtype Data type of placement pattern data
 *
 * Calls initialize_components immediately and starts a separate thread to check
 * shared memory weights
 */
Placement::Placement(int rank, int world_size, int num_devices_per_host,
                     ClusterActivation *activations, size_t expert_mapping_ptr,
                     std::vector<int64_t> shape, int dtype,
                     size_t placement_pattern_ptr,
                     std::vector<int64_t> placement_shape, int placement_dtype)
    : rank_(rank), world_size_(world_size),
      num_devices_per_host_(num_devices_per_host), activations_(activations) {

    // Initialize components immediately
    initialize_components(expert_mapping_ptr, shape, dtype,
                          placement_pattern_ptr, placement_shape,
                          placement_dtype);

    // Start a separate thread to check shared memory weights
    init_thread_ = std::thread(&Placement::check_shm_weights, this);
    // init_thread_.detach();
}

void Placement::initialize_components(size_t expert_mapping_ptr,
                                      std::vector<int64_t> shape, int dtype,
                                      size_t placement_pattern_ptr,
                                      std::vector<int64_t> placement_shape,
                                      int placement_dtype) {

    assert(shape.size() == 2);

    assert(placement_shape.size() == 3);
    mapping_ = new PlacementMapping("", rank_, 1, num_devices_per_host_,
                                    placement_shape[2], placement_pattern_ptr,
                                    placement_shape, expert_mapping_ptr, true,
                                    placement_pattern_ptr);

    num_layers_ = mapping_->get_num_layers();
    num_experts_ = mapping_->get_num_experts();
    num_deploy_experts_ = mapping_->get_num_deploy_experts();
    num_deploy_experts_per_rank_ = num_deploy_experts_ / world_size_;

    moe_weight_ = new MoEWeights(num_deploy_experts_, world_size_);
    optimizer_ = new PlacementOptimizer(mapping_, activations_);
}

void Placement::check_shm_weights() {
    std::cout << "check_shm_weights start success." << std::endl;
    while (!should_stop_init_) { // 使用标志控制退出
        if (moe_weight_ && moe_weight_->isShmInitialized()) {
            start_thread();
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(
            global_omni_config.activation_quiesce)); // Check every 30s
    }
}

Placement::~Placement() {
    stop_thread();
    delete moe_weight_;
    // delete mapping_;
    delete optimizer_;
    // delete activations_;
    delete dist_ptr_;
}

// 等待合适的时机等待专家权重替换
void quiesce() {
    // wait 5s before move weights to new postion
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // TODO: triger by vLLM when token finish
}

std::string convertInstructionToString(ChangeInstruction instruction) {
    std::string result =
        "layer_idx: " + std::to_string(instruction.layer_idx) +
        " \t type: " + std::to_string((int)instruction.type) +
        " \t source_rank: " + std::to_string(instruction.source_rank) +
        " \t target_rank: " + std::to_string(instruction.target_rank) +
        " \t source_global_position: " +
        std::to_string(instruction.source_global_position) +
        " \t target_global_position: " +
        std::to_string(instruction.target_global_position) +
        " \t source_expert_id: " +
        std::to_string(instruction.source_expert_id) +
        " \t target_expert_id: " +
        std::to_string(instruction.target_expert_id) + "\n";
    return result;
}

void Placement::placement_manager(aclrtContext currentContext) {
    ACLCHECK(aclrtSetCurrentContext(currentContext));
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 设置Stream
    MoEWeights *moe_weights = get_moe_weights();
    Distribution *dist_ptr = get_distribution();
    dist_ptr->set_stream(stream);

    if (enable_dynamic_ && !is_redundant_share_expert_rank()) {
        while (!moe_weight_->isHbmInitialized()) {
            std::this_thread::sleep_for(
                std::chrono::seconds(1)); // Run every 1 seconds
        }
        size_t expert_size =
            moe_weight_->get_expert_size(); // 根据QueueSize 预分配 接受buffs

        dist_ptr->allocate_recv_buffs(expert_size);
    }

    // 获取更新的 Mapping
    PlacementMapping *mapping = get_mapping();

    std::cout << "placement worker thread started\n";

    size_t dump_count = 0;
    size_t collect_times = 60; // 6 mins  to collect information

    activations_->collect(dist_ptr,
                          stream); // update the last & delta activation

    while (!should_stop_) {
        dump_count++;
        activations_->dump_and_collect(dist_ptr, stream, dump_count);

        if (!enable_dynamic_) {
            std::this_thread::sleep_for(std::chrono::seconds(collect_times));
            continue;
        }

        std::string log_info = "";
        // 构建下发交换队列
        std::vector<ChangeInstruction>
            changeInstructions; // all rank total instructions
        std::vector<ChangeInstruction> changeInstructions_this_rank;

        changeInstructions = optimizer_->optimize();

        for (auto &instruction : changeInstructions) {
            // if (instruction.layer_idx!=0) continue;
            if (instruction.source_rank != rank_ &&
                instruction.target_rank != rank_)
                continue;
            if (instruction.source_rank == rank_ &&
                instruction.type == OperationType::REMOVE)
                continue;
            changeInstructions_this_rank.push_back(instruction);

            // Log Infomation
            if (instruction.source_rank == rank_) {
                log_info += " S-L[" + std::to_string(instruction.layer_idx) +
                            "]-T[" + std::to_string((int)instruction.type) +
                            "]-TR[" + std::to_string(instruction.target_rank) +
                            "]";
            } else if (instruction.target_rank == rank_) {
                log_info += " T-L[" + std::to_string(instruction.layer_idx) +
                            "]-T[" + std::to_string((int)instruction.type) +
                            "]-TR[" + std::to_string(instruction.target_rank) +
                            "]";
            }
        }

        std::string log_tile =
            "Rank: " + std::to_string(rank_) + " , Exchanged Info: with nums[" +
            std::to_string(changeInstructions_this_rank.size()) + "] \n";

        // 下发入队, 下发成功并完成同步置于完成队列中
        size_t idx = 0;
        int fail_handshake_count = 0;
        std::string rank_str = std::to_string(rank_);
        while (idx < changeInstructions_this_rank.size()) {
            if (should_stop_)
                break;
            ChangeInstruction instruction = changeInstructions_this_rank[idx];

            int layer = instruction.layer_idx;
            OperationType type = instruction.type;
            bool need_enqueue_recv_buff = true;
            int t_rank = (instruction.source_rank == rank_)
                             ? instruction.target_rank
                             : instruction.source_rank;
            int global_position_id_this_layer =
                (instruction.source_rank == rank_)
                    ? instruction.source_global_position
                    : instruction.target_global_position;
            int expert_id = (instruction.source_rank == rank_)
                                ? instruction.target_expert_id
                                : instruction.source_expert_id;

            // int expert_id = (instruction.rank_a == rank_) ?
            // instruction.expert_idx_a : instruction.expert_idx_b; // UnitTest:
            // 不要交换收益

            int position_offset = mapping->getGlobalPositionOffset(
                layer, global_position_id_this_layer);

            if (type == OperationType::ADD &&
                instruction.source_rank == rank_) {
                need_enqueue_recv_buff = false;
                position_offset = -1; // add的source端不更新
            }

            if (type == OperationType::REMOVE &&
                instruction.target_rank == rank_) {
                t_rank = rank_; // 自己跟自己握手
                expert_id = -1; // 告诉该位置专家id修改为-1
            }

            bool flag = dist_ptr->performGlobalHandshake(
                t_rank, position_offset, expert_id); // 对端握手成功， 准备下发
            // mapping 有一把琐 要加在 deployed_mapping更新，
            // 且保证权重入队完成后才解锁

            if (flag) {
                fail_handshake_count = 0;
                bool positionIsConsistency;
                if (type == OperationType::SWAP) {
                    positionIsConsistency = mapping->checkPositionIsConsistency(
                        layer, instruction.source_global_position,
                        instruction.source_expert_id);
                    if (!positionIsConsistency) {
                        throw std::runtime_error("[Error]-rank[" +
                                                 std::to_string(rank_) +
                                                 "]  positionIsConsistency");
                    }
                    positionIsConsistency = mapping->checkPositionIsConsistency(
                        layer, instruction.target_global_position,
                        instruction.target_expert_id);
                    if (!positionIsConsistency) {
                        throw std::runtime_error("[Error]-rank[" +
                                                 std::to_string(rank_) +
                                                 "]  positionIsConsistency");
                    }
                } else if (type == OperationType::ADD) {
                    positionIsConsistency = mapping->checkPositionIsConsistency(
                        layer, instruction.source_global_position,
                        instruction.source_expert_id);
                    if (!positionIsConsistency) {
                        throw std::runtime_error("[Error]-rank[" +
                                                 std::to_string(rank_) +
                                                 "]  positionIsConsistency");
                    }
                }
            } else {
                fail_handshake_count++;
            }
            std::unique_lock<std::mutex> lock = acquire_lock();
            sub_thread_is_changing_ = true;
            lock.unlock();

            bool isUpdateValied =
                mapping->update_globalDeployedPositionToLogisticsIdMapping(
                    dist_ptr->getHandshakeStatus(), dist_ptr->get_info_length(),
                    is_layer_update);
            if (!isUpdateValied) {
                std::cout << "[Error]-isUpdateValied \t layer_idx: "
                          << instruction.layer_idx
                          << " \t type: " << (int)instruction.type
                          << " \t source_rank: " << instruction.source_rank
                          << " \t target_rank: " << instruction.target_rank
                          << " \t source_global_position: "
                          << instruction.source_global_position
                          << " \t target_global_position: "
                          << instruction.target_global_position
                          << " \t source_expert_id: "
                          << instruction.source_expert_id
                          << " \t target_expert_id: "
                          << instruction.target_expert_id << std::endl;
                throw std::runtime_error("[Error]-rank[" +
                                         std::to_string(rank_) +
                                         "] \t isUpdateValied: false");
            }

            if (type == OperationType::REMOVE) {
                // remove no need to swap expert weights
                sub_thread_is_changing_ = false;
                idx++;
                continue;
            }

            if (flag) {
                void *recv_buff_address = dist_ptr->get_recv_buff_address();
                moe_weights->replacement(
                    dist_ptr, layer, instruction.source_rank,
                    instruction.source_global_position, instruction.target_rank,
                    instruction.target_global_position, recv_buff_address,
                    need_enqueue_recv_buff);
                sub_thread_is_changing_ = false;
                idx++;
            } else {
                sub_thread_is_changing_ = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        // 所有Rank 完成本次Optimizer优化的同步等待
        while (true) {
            if (should_stop_)
                break;
            bool flag = dist_ptr->performGlobalHandshake(
                -1, -1, -1); //所有rank 都是-1， return true
            std::unique_lock<std::mutex> lock = acquire_lock();
            sub_thread_is_changing_ = true;
            lock.unlock();
            bool isUpdateValied =
                mapping->update_globalDeployedPositionToLogisticsIdMapping(
                    dist_ptr->getHandshakeStatus(), dist_ptr->get_info_length(),
                    is_layer_update);
            sub_thread_is_changing_ = false;
            if (!isUpdateValied)
                throw std::runtime_error("[Error]-rank[" +
                                         std::to_string(rank_) +
                                         "] \t isUpdateValied: false");
            if (flag)
                break;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(10)); // wait other ranks
        }
        std::cout << getTimestamp() << "rank[" << rank_ << "] finished"
                  << std::endl;

        activations_->collect(dist_ptr,
                              stream); // Clear the old placement activations
        std::this_thread::sleep_for(std::chrono::seconds(collect_times));
    }

    dist_ptr->release_recv_buffs();
    aclrtDestroyStream(stream);
    std::cout << "placement worker thread stoped\n";
}

void Placement::start_thread() {
    if (!worker_thread_.joinable()) {
        should_stop_ = false;
        aclrtContext currentContext;
        ACLCHECK(aclrtGetCurrentContext(&currentContext));
        worker_thread_ =
            std::thread(&Placement::placement_manager, this, currentContext);
    }
}

void Placement::stop_thread() {
    should_stop_init_ = true; // 通知 init_thread_ 退出
    should_stop_ = true;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    if (init_thread_.joinable()) {
        init_thread_.join(); // 等待初始化线程完成
    }
}

void do_placement_optimizer(Placement &placement) {
    Distribution *dist_ptr = placement.get_distribution();
    PlacementMapping *mapping = placement.get_mapping();
    std::unique_lock<std::mutex> lock = placement.acquire_lock();
    if (!placement.get_subthread_is_changing()) {
        if (!placement.is_redundant_share_expert_rank())
            dist_ptr->copyAllFromCompletedQueueToHBM();
        mapping->update_selector(placement.get_layer_update());
        placement.reset_layer_update();
    }
}

pybind11::bytes GetPDRootInfo() {
    // 获取root节点，root节点用户可指定，并非只可以设置为0节点
    char *pRootInfo = new char[HCCL_ROOT_INFO_BYTES];
    for (uint32_t i = 0; i < HCCL_ROOT_INFO_BYTES; i++) {
        pRootInfo[i] = 0;
    }
    HcclGetRootInfo((HcclRootInfo *)pRootInfo);
    std::cout << "the hccl root info in c++ is " << pRootInfo << std::endl;
    pybind11::bytes results(pRootInfo, HCCL_ROOT_INFO_BYTES);
    delete[] pRootInfo;
    return results;
}

PYBIND11_MODULE(omni_placement, m) {
    m.doc() = "MoE weights management with shared memory";

    // 绑定 ut_memcpy_fun 函数
    m.def("set_ut_memcpy_fun", &set_ut_memcpy_fun,
          "Set the UT memcpy function");
    m.def("unset_ut_memcpy_fun", &unset_ut_memcpy_fun,
          "Unset the UT memcpy function");

    m.def("do_placement_optimizer", &do_placement_optimizer,
          py::arg("placement"));
    m.def("get_pd_rootinfo", &GetPDRootInfo, "get_pd_rootinfo");

    // 1. 绑定 PlacementMapping 类
    py::class_<PlacementMapping>(m, "PlacementMapping")
        .def(py::init<const std::string &, int, int, int, int, size_t,
                      std::vector<int64_t>, size_t, bool, size_t>(),
             py::arg("filename"), py::arg("rank"), py::arg("num_devices"),
             py::arg("max_deployed_num"), py::arg("max_deployed_num"),
             py::arg("pattern"), py::arg("pattern_shape"), py::arg("selector"),
             py::arg("enable_rank_round_robin"),
             py::arg("num_redundant_per_expert"));

    // 3. 绑定 MoEWeights 类
    py::class_<MoEWeights>(m, "MoEWeights")
        // 根据 moe_weights.h 文件修改了 构造函数的入参
        .def(py::init<size_t>(), py::arg("num_experts"))
        .def(py::init<size_t, size_t>(), py::arg("num_experts"),
             py::arg("world_size"))
        .def(py::init<size_t, size_t, size_t>(), py::arg("num_experts"),
             py::arg("rank"), py::arg("world_size"))
        .def(py::init<size_t, size_t, size_t, const char *>(),
             py::arg("num_experts"), py::arg("rank"), py::arg("world_size"),
             py::arg("rankTableFile"))
        .def("isShmInitialized", &MoEWeights::isShmInitialized)
        .def("init_weights", &MoEWeights::init_weights, py::arg("npu_weights"),
             py::arg("init_shm"), "Initialize with NPU weights");

    // 4. 绑定 Placement 类
    py::class_<Placement>(m, "Placement")
        .def(py::init<>())
        .def(py::init<int, int, int, int, ClusterActivation *,
                      PlacementMapping *, char *, bool>(),
             py::arg("rank"), py::arg("world_size"),
             py::arg("hccl_comm_world_size"), py::arg("num_devices_per_host"),
             py::arg("activation"), py::arg("placement_mapping"),
             py::arg("root_info"), py::arg("enable_dynamic"))
        .def(py::init<int, int, int, ClusterActivation *, size_t,
                      std::vector<int64_t>, int, size_t, std::vector<int64_t>,
                      int>(),
             py::arg("rank"), py::arg("world_size"),
             py::arg("num_devices_per_host"), py::arg("activation"),
             py::arg("expert_mapping_ptr"), py::arg("shape"), py::arg("dtype"),
             py::arg("placement_pattern_ptr"), py::arg("placement_shape"),
             py::arg("placement_dtype"))
        .def("get_moe_weights", &Placement::get_moe_weights,
             py::return_value_policy::reference)
        .def("start_thread", &Placement::start_thread, "");

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<uint64_t, size_t, size_t,
                      const std::string &>(), // 按实际构造函数参数补全
             py::arg("data_ptr"), py::arg("length"), py::arg("element_size"),
             py::arg("name"))
        .def(py::init<uint64_t, size_t, size_t, const std::string &,
                      const std::string &>(), // 按实际构造函数参数补全
             py::arg("data_ptr"), py::arg("length"), py::arg("element_size"),
             py::arg("dtype"), py::arg("name"));

    py::class_<ClusterActivation>(m, "ClusterActivation")
        .def(py::init<Tensor, int64_t, size_t, size_t, int, size_t, size_t,
                      size_t>(), // 按实际构造函数参数补全
             py::arg("npu_count"), py::arg("max_activation_count"),
             py::arg("layer"), py::arg("num_expert"), py::arg("window_size"),
             py::arg("world_size"), py::arg("hccl_comm_world_size"),
             py::arg("rank"), "Initialize with expert activation")
        .def("getClusterTotalActivationCount",
             &ClusterActivation::getClusterTotalActivationCount,
             py::arg("layer"), py::arg("expert"), "")
        .def("stop_thread", &ClusterActivation::stop_thread, "")
        .def("stopDump", &ClusterActivation::stopDump, "")
        .def("setDumpDir", &ClusterActivation::setDumpDir, py::arg("dump_dir"),
             "Set the dump path for the cluster activation");
}