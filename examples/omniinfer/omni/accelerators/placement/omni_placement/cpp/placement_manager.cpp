// SPDX-License-Identifier: MIT
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

    dist_ptr_ = new Distribution(num_deploy_experts_per_rank_, rank_,
                                 hccl_comm_world_size_, root_info,
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

bool Placement::is_mergeful_and_merged(std::vector<std::vector<int>> &rank_used,
                                       int layer_1, int layer_2) {
    for (size_t i = 0; i < world_size_; i++) {
        if (rank_used[layer_1][i] + rank_used[layer_2][i] >
            dist_ptr_->queue_size())
            return false;
    }

    for (size_t i = 0; i < world_size_; i++) {
        rank_used[layer_1][i] += rank_used[layer_2][i];
    }
    return true;
}

void Placement::merge_instructions(std::vector<ChangeInstruction> &in) {
    stable_sort(in.begin(), in.end(), [](const auto &a, const auto &b) {
        return a.layer_idx < b.layer_idx;
    });
    std::vector<std::vector<int>> rank_used(num_layers_,
                                            std::vector<int>(world_size_));
    size_t total_num = in.size();
    for (size_t i = 0; i < total_num; i++) {
        rank_used[in[i].layer_idx][in[i].target_rank]++;
        in[i].round = -1;
    }

    int layer_round[num_layers_] = {0};
    memset(layer_round, 0, sizeof(layer_round));
    int cur_round = 1;
    for (size_t start_layer = 0; start_layer < num_layers_; start_layer++) {
        if (layer_round[start_layer] != 0)
            continue;
        layer_round[start_layer] = cur_round;
        for (size_t merge_layer = start_layer + 1; merge_layer < num_layers_;
             merge_layer++) {
            if (layer_round[merge_layer] != 0)
                continue;
            if (is_mergeful_and_merged(rank_used, start_layer, merge_layer)) {
                layer_round[merge_layer] = cur_round;
            }
        }
        cur_round++;
    }

    for (size_t i = 0; i < total_num; i++) {
        in[i].round = layer_round[in[i].layer_idx];
    }
}

void Placement::reorder_instructions(std::vector<ChangeInstruction> in,
                                     std::vector<ChangeInstruction> &out) {
    merge_instructions(in);
    stable_sort(in.begin(), in.end(),
                [](const auto &a, const auto &b) { return a.round < b.round; });
    size_t total_num = in.size();
    if (total_num <= 0) {
        out = std::move(in);
        return;
    }

    size_t start_idx = 0;
    std::unordered_map<int, int> rank_used;
    std::vector<ChangeInstruction> round_inst;
    // start: out: , round_inst:
    // r1-r2,r1-r3,r1-r4,r2-r3,r2-r4,r3-r4,r2-r1,r4-r3, ps:queue size=2 round 0
    // one batch: out:r1-r2,r3-r4; round_inst:
    // r1-r3,r1-r4,r2-r3,r2-r4,r2-r1,r4-r3 one batch:
    // out:r1-r2,r3-r4,r1-r3,r2-r4; round_inst: r1-r4,r2-r3,r2-r1,r4-r3 round 1
    // one batch: out:r1-r2,r3-r4,r1-r3,r2-r4,r1-r4,r2-r3; round_inst:
    // r2-r1,r4-r3 one batch:
    // out:r1-r2,r3-r4,r1-r3,r2-r4,r1-r4,r2-r3,r2-r1,r4-r3; round_inst:; end
    for (size_t i = 0; i < total_num; i++) {
        ChangeInstruction inst = in[i];
        round_inst.emplace_back(inst);
        if (i == total_num - 1 || in[i + 1].round > inst.round) {
            // one round end
            size_t cur_idx = 0;
            size_t round_num = round_inst.size();
            size_t stop_idx = round_num - 1;
            rank_used.clear();
            while (cur_idx < round_num) {
                if (cur_idx > stop_idx) {
                    // one batch end
                    rank_used.clear();
                    stop_idx = round_num - 1;
                }
                ChangeInstruction one_inst = round_inst[cur_idx];
                // split batch
                if (rank_used.find(one_inst.source_rank) != rank_used.end() ||
                    rank_used.find(one_inst.target_rank) != rank_used.end()) {
                    // can not put into batch, so back to queue
                    round_inst.emplace_back(one_inst);
                    round_num++;
                } else {
                    // put into batch
                    out.emplace_back(one_inst);
                    rank_used[one_inst.source_rank] = 1;
                    rank_used[one_inst.target_rank] = 1;
                }
                cur_idx++;
            }
            round_inst.clear();
        }
    }
}

bool Placement::check_instructions(std::vector<ChangeInstruction> insts) {
    for (int idx = 0; idx < insts.size(); idx++) {
        ChangeInstruction instr = insts[idx];
        if (!mapping_->checkPositionIsConsistency(instr.layer_idx,
                                                  instr.source_global_position,
                                                  instr.source_expert_id) ||
            !mapping_->checkPositionIsConsistency(instr.layer_idx,
                                                  instr.target_global_position,
                                                  instr.target_expert_id) ||
            (instr.type != OperationType::ADD &&
             instr.type != OperationType::REMOVE)) {
            instr.print();
            return false;
        }
    }
    return true;
}

void Placement::placement_handle_one_batch(
    std::vector<ChangeInstruction> changeInstructions) {
    size_t instNum = changeInstructions.size();
    if (instNum <= 0)
        return;
    dist_ptr_->clear_hccl_buffs();
    size_t idx = 0;

    for (idx = 0; idx < instNum; idx++) {
        if (should_stop_) {
            break;
        }
        ChangeInstruction inst = changeInstructions[idx];
        int layer = inst.layer_idx;
        int tgt_position_offset = mapping_->getGlobalPositionOffset(
            layer, inst.target_global_position);
        is_layer_update[layer] = true;
        if (inst.type == OperationType::REMOVE) {
            mapping_->update_pos_to_ep(layer, tgt_position_offset, -1);
            continue;
        } else if (inst.type == OperationType::ADD) {
            mapping_->update_pos_to_ep(layer, tgt_position_offset,
                                       inst.source_expert_id);
        }

        if (inst.source_rank != rank_ && inst.target_rank != rank_)
            continue;

        bool need_enqueue_recv_buff =
            (inst.type == OperationType::ADD && inst.target_rank == rank_);

        moe_weight_->replacement(dist_ptr_, inst.layer_idx, inst.source_rank,
                                 inst.source_global_position, inst.target_rank,
                                 inst.target_global_position,
                                 need_enqueue_recv_buff);
    }

    if (dist_ptr_->hccl_buffs_size() > 0) {
        dist_ptr_->hccl_batch_send();
        dist_ptr_->clear_hccl_buffs();
    }
}

void Placement::placement_handle_instrucions(
    std::vector<ChangeInstruction> src_changeInstructions) {
    size_t instNum = src_changeInstructions.size();
    if (instNum <= 0)
        return;
    std::vector<ChangeInstruction> changeInstructions;
    reorder_instructions(src_changeInstructions, changeInstructions);
    dist_ptr_->reset_buff_cur();
    dist_ptr_->clear_queue();
    dist_ptr_->clear_hccl_buffs();
    size_t max_ins_one_batch_one_rank = 1;

    using clock = std::chrono::high_resolution_clock;
    bool need_wait_main = false;
    bool need_split_batch = false;
    size_t cur_round = changeInstructions[0].round, cur_batch = 0;
    int rank_used[world_size_] = {0};
    memset(rank_used, 0, sizeof(rank_used));

    int rank_src_used[world_size_] = {0};
    int rank_tgt_used[world_size_] = {0};
    memset(rank_src_used, 0, sizeof(rank_src_used));
    memset(rank_tgt_used, 0, sizeof(rank_tgt_used));

    std::vector<ChangeInstruction> changeInstructions_one_batch;
    for (size_t idx = 0; idx < instNum; idx++) {
        if (should_stop_) {
            break;
        }
        ChangeInstruction inst = changeInstructions[idx];

        changeInstructions_one_batch.emplace_back(inst);
        if (inst.type != OperationType::REMOVE) {
            rank_used[inst.source_rank]++;
            rank_used[inst.target_rank]++;
            rank_src_used[inst.source_rank]++;
            rank_tgt_used[inst.target_rank]++;
        }

        if (inst.round > cur_round)
            cur_batch = 0;
        cur_round = inst.round;

        if (idx >= (instNum - 1)) {
            need_split_batch = true;
            need_wait_main = true;
        } else {
            ChangeInstruction instNext = changeInstructions[idx + 1];
            if (instNext.round > cur_round) {
                need_split_batch = true;
                need_wait_main = true;
            } else if ((instNext.type != OperationType::REMOVE) &&
                       (rank_used[instNext.source_rank] >=
                            max_ins_one_batch_one_rank ||
                        rank_used[instNext.target_rank] >=
                            max_ins_one_batch_one_rank)) {
                need_split_batch = true;
            }
        }

        if (need_split_batch) {
            dist_ptr_->sync_round_shakehand(cur_round, cur_batch);
            placement_handle_one_batch(changeInstructions_one_batch);
            cur_batch++;
            memset(rank_used, 0, sizeof(rank_used));
            memset(rank_src_used, 0, sizeof(rank_src_used));
            memset(rank_tgt_used, 0, sizeof(rank_tgt_used));
            changeInstructions_one_batch.clear();
            need_split_batch = false;
        }

        if (need_wait_main) {
            bool expected = false;
            bool desired = true;
            bool rst =
                buf_ready_flag_.compare_exchange_strong(expected, desired);
            if (!rst)
                std::cout << "[handle ins][err] buf ready flag error 1."
                          << "\n";

            while (buf_ready_flag_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            need_wait_main = false;
            dist_ptr_->reset_buff_cur();
            memset(rank_used, 0, sizeof(rank_used));
            memset(rank_src_used, 0, sizeof(rank_src_used));
            memset(rank_tgt_used, 0, sizeof(rank_tgt_used));
        }
    }
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
        dist_ptr->init_hccl_buffs(moe_weight_->get_expert_itemnum());
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

        std::cout << "[handle ins] placement worker before handle "
                     "instructions. tatal cnt: "
                  << changeInstructions.size() << "\n";
        if (changeInstructions.size() > 0) {
            TRACK_START();
            if (check_instructions(changeInstructions))
                placement_handle_instrucions(changeInstructions);
            TRACK_POINT("[handle ins] process " +
                        std::to_string(changeInstructions.size()) +
                        " instructions.");
        }

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
    if (!placement.buf_ready_flag_.load())
        return;
    Distribution *dist_ptr = placement.get_distribution();
    PlacementMapping *mapping = placement.get_mapping();
    if (!placement.is_redundant_share_expert_rank())
        dist_ptr->copy_from_queue_to_hbm();
    mapping->update_selector(placement.get_layer_update());
    placement.reset_layer_update();

    bool expected = true;
    bool desired = false;
    bool rst =
        placement.buf_ready_flag_.compare_exchange_strong(expected, desired);
    if (!rst)
        std::cout << "[handle ins][err] buf ready flag error 3." << "\n";
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
        .def("init_recv_buf", &Placement::init_recv_buf, "")
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