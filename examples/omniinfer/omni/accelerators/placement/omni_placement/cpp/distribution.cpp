// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "distribution.h"
#include <string.h>

const std::map<std::string, HcclDataType> NAME2DATATYPE = {
    {"int", HCCL_DATA_TYPE_INT32},     {"int32", HCCL_DATA_TYPE_INT32},
    {"int16", HCCL_DATA_TYPE_INT16},   {"int8", HCCL_DATA_TYPE_INT8},
    {"int64", HCCL_DATA_TYPE_INT64},   {"float", HCCL_DATA_TYPE_FP32},
    {"float32", HCCL_DATA_TYPE_FP32},  {"float16", HCCL_DATA_TYPE_FP16},
    {"bfloat16", HCCL_DATA_TYPE_BFP16}};

Distribution::Distribution(size_t rank, const char *rankTableFile) {
    // 构建 HCCL 通信域
    std::cout << "rank TableFile is " << rankTableFile << std::endl;
    HCCLCHECK(HcclCommInitClusterInfo(rankTableFile, rank, &hcclComm_));
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank_));
}

Distribution::Distribution(size_t rank, size_t world_size, const char *infoStr,
                           HcclCommInitType type) {
    // 构建 HCCL 通信域
    if (type == HcclCommInitType::RootInfoString) {
        HcclRootInfo rootInfo;
        memcpy(rootInfo.internal, infoStr, HCCL_ROOT_INFO_BYTES);
        HcclCommConfig config;
        HcclCommConfigInit(&config);
        config.hcclBufferSize = 100;
        config.hcclOpExpansionMode = 1;
        HCCLCHECK(HcclCommInitRootInfoConfig(world_size, &rootInfo, rank,
                                             &config, &hcclComm_));
    } else {
        HCCLCHECK(HcclCommInitClusterInfo(infoStr, rank, &hcclComm_));
    }
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank_));
    HCCLCHECK(HcclGetRankSize(hcclComm_, &world_size_));
    if (world_size != world_size_) {
        std::cout << "[DynamicEplb-Error], The world size from rank tables "
                     "does not correspond with input parameters"
                  << std::endl;
        exit(0);
    }

    if (world_size == 0) {
        std::cout << "[DynamicEplb-Error], Invalid world_size_, which is 0"
                  << std::endl;
        exit(0);
    }

    warmup();

    void *data_ptr;

    hostHandshakeStatus_.resize(world_size_ * info_length_, 0);

    ACLCHECK(aclrtMalloc(&data_ptr, world_size_ * info_length_ * sizeof(int),
                         ACL_MEM_MALLOC_HUGE_FIRST));
    deviceHandshakeStatus_ =
        Tensor((uint64_t)data_ptr, world_size_ * info_length_, sizeof(int),
               "int", "constant tensor");
    deviceHandshakeStatus_.to_device(hostHandshakeStatus_.data());

    ACLCHECK(aclrtMalloc(&data_ptr, info_length_ * sizeof(int),
                         ACL_MEM_MALLOC_HUGE_FIRST));
    deviceCurrentStatus_ = Tensor((uint64_t)data_ptr, info_length_, sizeof(int),
                                  "int", "constant tensor");

    ACLCHECK(aclrtCreateStream(&memcopy_stream_));
}

Distribution::~Distribution() {
    // 销毁HCCL通信域
    HCCLCHECK(HcclCommDestroy(hcclComm_));
    deviceHandshakeStatus_.release();
    deviceCurrentStatus_.release();
    aclrtDestroyStream(stream_);
    aclrtDestroyStream(memcopy_stream_);
}

void Distribution::allocate_recv_buffs(size_t expert_size) {
    expert_size_ = expert_size;
    size_t total_size = expert_size_ * QUEUE_SIZE;
    ACLCHECK(aclrtMalloc(&recv_buff_, total_size, ACL_MEM_MALLOC_HUGE_FIRST));
}

bool Distribution::isCompletedQueueFull() {
    return completedSynchronizeQueue_.IsFull();
}

void *Distribution::get_recv_buff_address() {
    if (recv_buff_ == nullptr) {
        std::cout << "[DynamicEplb-Error], Pls initilization recv_buff_ by "
                     "allocate_recv_buffs"
                  << std::endl;
        exit(0);
    }
    size_t queue_idx = getCompletedQueueEnqueuePosition();
    size_t offset_idx = queue_idx * expert_size_;
    return static_cast<void *>(static_cast<uint8_t *>(recv_buff_) + offset_idx);
}

void Distribution::release_recv_buffs() { ACLCHECK(aclrtFree(recv_buff_)); }

void Distribution::enqueue(TransDesc *desc, size_t t_rank,
                           bool need_enqueue_recv_buff) {

    if (desc == nullptr) {
        std::cout
            << "[DynamicEplb-Error], Adding an empty TransDesc ptr to the Queue"
            << std::endl;
        exit(0);
    }
    bool send_first = rank_ < t_rank;
    desc->t_rank = t_rank;
    TransDesc *position_recv_desc =
        completedSynchronizeQueue_.GetRear(desc->localExpertPositionOfsset);

    std::vector<void *> send_address = (position_recv_desc == nullptr)
                                           ? desc->address
                                           : position_recv_desc->recv_buffs;

    for (size_t idx = 0; idx < desc->address.size(); ++idx) {
        swap(send_address[idx], desc->recv_buffs[idx], desc->lengths[idx],
             desc->dtypes[idx], t_rank, send_first,
             stream_); // 对端队列满了， 等待超时交换失败
    }

    if (need_enqueue_recv_buff)
        completedSynchronizeQueue_.Enqueue(desc);
}

// The global_position_id will be put a new expert with expert_id
bool Distribution::performGlobalHandshake(int t_rank, int position_offset,
                                          int expert_id) {
    // world_size 为 队列满， -1为当前Rank已完成所有下发
    bool is_full = isCompletedQueueFull();
    if (is_full && (t_rank != -1 && t_rank != (int)rank_))
        t_rank = (int)world_size_; // 队列已满，广播当前队列已满 (标志符)

    std::vector<int> hostCurrentStatus = {t_rank, position_offset, expert_id};

    deviceCurrentStatus_.to_device(hostCurrentStatus.data());

    allgather(deviceCurrentStatus_.get_data_ptr(),
              deviceHandshakeStatus_.get_data_ptr(), info_length_,
              "int"); // 广播当前的目标队列
    deviceHandshakeStatus_.to_host(hostHandshakeStatus_.data());

    if (t_rank == (int)world_size_) {
        return false; // 当前队列已满
    }

    else if (t_rank != -1) {
        //当前rank 还存在下发任务
        if (hostHandshakeStatus_[t_rank * info_length_] == (int)rank_)
            return true; // 对端跟自己握手
        return false;    // 对端不跟自己握手
    }

    // 剩下 t_trank == -1 的情况
    for (size_t idx = 0; idx < world_size_; ++idx) {
        if (hostHandshakeStatus_[idx * info_length_] != -1)
            return false; // 还有rank存在swap 下发任务（队列满了等待状态），
    }

    return true;
}

/**
 * @return 当前完成队列是否为空
 */
void Distribution::copyFromCompletedQueueToHBM() {

    if (completedSynchronizeQueue_.IsEmpty()) {
        return;
    }

    TransDesc *desc = completedSynchronizeQueue_.GetFront();
    completedSynchronizeQueue_.Dequeue();

    for (size_t idx = 0; idx < desc->recv_buffs.size(); ++idx) {
        ACLCHECK(aclrtMemcpyAsync(
            desc->address[idx], desc->sizes[idx], desc->recv_buffs[idx],
            desc->sizes[idx], ACL_MEMCPY_DEVICE_TO_DEVICE, memcopy_stream_));
    }
}

void Distribution::copyAllFromCompletedQueueToHBM() {
    while (!completedSynchronizeQueue_.IsEmpty()) {
        copyFromCompletedQueueToHBM();
    }
    ACLCHECK(aclrtSynchronizeStream(memcopy_stream_));
}

void Distribution::swap(void *src_addr, void *recv_addr, size_t length,
                        std::string dtype, uint32_t t_rank, bool send_first,
                        aclrtStream stream) {
    assert(stream == nullptr && "stream should not be nullptr");
    if (send_first) {

        HCCLCHECK(HcclSend(src_addr, length, NAME2DATATYPE.at(dtype), t_rank,
                           hcclComm_, stream));

        HCCLCHECK(HcclRecv(recv_addr, length, NAME2DATATYPE.at(dtype), t_rank,
                           hcclComm_, stream));

    } else {

        HCCLCHECK(HcclRecv(recv_addr, length, NAME2DATATYPE.at(dtype), t_rank,
                           hcclComm_, stream));

        HCCLCHECK(HcclSend(src_addr, length, NAME2DATATYPE.at(dtype), t_rank,
                           hcclComm_, stream));
    }

    ACLCHECK(aclrtSynchronizeStream(stream));
}

void Distribution::allgather(void *src_addr, void *recv_addr, size_t length,
                             std::string dtype) {

    assert(stream_ == nullptr && "stream_ should not be nullptr");
    HCCLCHECK(HcclAllGather(src_addr, recv_addr, length,
                            NAME2DATATYPE.at(dtype), hcclComm_, stream_));
    ACLCHECK(aclrtSynchronizeStream(stream_));
}

void Distribution::set_stream(aclrtStream stream) { stream_ = stream; }

void Distribution::warmup() {

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    std::string dtype = "int";
    size_t length = 1;
    size_t data_size = length * sizeof(int);
    void *data_ptr;
    void *recv_buf;
    ACLCHECK(aclrtMalloc(&data_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recv_buf, data_size * world_size_,
                         ACL_MEM_MALLOC_HUGE_FIRST));

    if (rank_ == 0) {
        HCCLCHECK(HcclSend(data_ptr, length, NAME2DATATYPE.at(dtype),
                           (world_size_ - 1), hcclComm_, stream));
    } else if (rank_ == (world_size_ - 1)) {
        HCCLCHECK(HcclRecv(data_ptr, length, NAME2DATATYPE.at(dtype), 0,
                           hcclComm_, stream));
    }

    ACLCHECK(aclrtSynchronizeStream(stream));

    HCCLCHECK(HcclAllGather(data_ptr, recv_buf, length, NAME2DATATYPE.at(dtype),
                            hcclComm_, stream));

    ACLCHECK(aclrtSynchronizeStream(stream));

    ACLCHECK(aclrtFree(data_ptr));
    ACLCHECK(aclrtFree(recv_buf));

    ACLCHECK(aclrtSynchronizeStream(stream));
    ACLCHECK(aclrtDestroyStream(stream));

    if (rank_ == 0)
        std::cout << "finished hcclcomm_ warmup!" << std::endl;
}

void Distribution::printCommInfo() {
    // 获取当前进程的秩（Rank）
    uint32_t rank = 0;
    HCCLCHECK(HcclGetRankId(hcclComm_, &rank));

    // 获取通信域的大小（Size）
    uint32_t size = 0;
    HCCLCHECK(HcclGetRankSize(hcclComm_, &size));

    // 打印通信域信息
    std::cout << "HCCL Communicator Info:\n";
    std::cout << "  Rank: " << rank << std::endl;
    std::cout << "  Size: " << size << std::endl;
}
