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

Distribution::Distribution(size_t num_deploy_experts_per_rank, size_t rank,
                           size_t world_size, const char *infoStr,
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
    host_round_status_.resize(world_size_ * round_info_length_, 0);

    ACLCHECK(aclrtMalloc(&data_ptr,
                         world_size_ * round_info_length_ * sizeof(int),
                         ACL_MEM_MALLOC_HUGE_FIRST));
    device_round_status_ =
        Tensor((uint64_t)data_ptr, world_size_ * round_info_length_,
               sizeof(int), "int", "constant tensor");
    device_round_status_.to_device(host_round_status_.data());
    ACLCHECK(aclrtMalloc(&data_ptr, round_info_length_ * sizeof(int),
                         ACL_MEM_MALLOC_HUGE_FIRST));
    device_cur_round_status_ = Tensor((uint64_t)data_ptr, round_info_length_,
                                      sizeof(int), "int", "constant tensor");

    ACLCHECK(aclrtCreateStream(&memcopy_stream_));
    num_deploy_experts_per_rank_ = num_deploy_experts_per_rank;
    queue_size_ = num_deploy_experts_per_rank_;
    queue_item_num_ = 0;
    completed_sync_queue_.resize(queue_size_);
    for (int i = 0; i < queue_size_; ++i) {
        completed_sync_queue_[i] = new TransDesc;
    }
}

Distribution::~Distribution() {
    // 销毁HCCL通信域
    HCCLCHECK(HcclCommDestroy(hcclComm_));
    aclrtDestroyStream(stream_);
    aclrtDestroyStream(memcopy_stream_);
    device_round_status_.release();
    device_cur_round_status_.release();
    for (int i = 0; i < queue_size_; ++i) {
        delete completed_sync_queue_[i];
    }
}

void Distribution::allocate_recv_buffs(size_t expert_size) {
    expert_size_ = expert_size;
    size_t total_size = expert_size_ * queue_size_;
    ACLCHECK(aclrtMalloc(&recv_buff_, total_size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void Distribution::init_hccl_buffs(size_t item_num) {
    expert_weight_num_ = item_num;
    hccl_batch_idx_ = 0;
    max_hccl_batch_size_ = item_num * queue_size_;
    send_recv_info_.sendRecvType.resize(max_hccl_batch_size_);
    send_recv_info_.address.resize(max_hccl_batch_size_);
    send_recv_info_.lengths.resize(max_hccl_batch_size_);
    send_recv_info_.sizes.resize(max_hccl_batch_size_);
    send_recv_info_.dtypes.resize(max_hccl_batch_size_);
    send_recv_info_.recv_buffs.resize(max_hccl_batch_size_);
    send_recv_info_.t_rank.resize(max_hccl_batch_size_);
}

void Distribution::release_recv_buffs() { ACLCHECK(aclrtFree(recv_buff_)); }

void Distribution::clear_hccl_buffs() { hccl_batch_idx_ = 0; }

size_t Distribution::hccl_buffs_size() { return hccl_batch_idx_; }

void Distribution::prepare_batch(bool need_enqueue, TransDesc &desc) {
    if (need_enqueue) {
        SwapDirection s_direction = (rank_ == desc.t_rank[0])
                                        ? SwapDirection::LOCAL
                                        : SwapDirection::RECV;
        add_to_batch(desc, s_direction);
        enqueue(&desc);
    } else {
        add_to_batch(desc, SwapDirection::SEND);
    }
}

void Distribution::add_to_batch(TransDesc &desc, int direction) {
    std::string rank_str = std::to_string(rank_);
    for (size_t idx = 0; idx < desc.address.size(); ++idx) {
        // send
        if (direction & SwapDirection::SEND) {
            send_recv_info_.sendRecvType[hccl_batch_idx_] = SwapDirection::SEND;
            send_recv_info_.address[hccl_batch_idx_] = desc.address[idx];
            send_recv_info_.lengths[hccl_batch_idx_] = desc.lengths[idx];
            send_recv_info_.dtypes[hccl_batch_idx_] = desc.dtypes[idx];
            send_recv_info_.t_rank[hccl_batch_idx_] = desc.t_rank[idx];
            hccl_batch_idx_++;
        }

        // recv
        if (direction & SwapDirection::RECV) {
            send_recv_info_.sendRecvType[hccl_batch_idx_] = SwapDirection::RECV;
            send_recv_info_.recv_buffs[hccl_batch_idx_] = desc.recv_buffs[idx];
            send_recv_info_.lengths[hccl_batch_idx_] = desc.lengths[idx];
            send_recv_info_.dtypes[hccl_batch_idx_] = desc.dtypes[idx];
            send_recv_info_.t_rank[hccl_batch_idx_] = desc.t_rank[idx];
            hccl_batch_idx_++;
        }

        // local
        if (direction & SwapDirection::LOCAL) {
            send_recv_info_.sendRecvType[hccl_batch_idx_] =
                SwapDirection::LOCAL;
            send_recv_info_.address[hccl_batch_idx_] = desc.address[idx];
            send_recv_info_.recv_buffs[hccl_batch_idx_] = desc.recv_buffs[idx];
            send_recv_info_.sizes[hccl_batch_idx_] = desc.sizes[idx];
            send_recv_info_.lengths[hccl_batch_idx_] = desc.lengths[idx];
            send_recv_info_.dtypes[hccl_batch_idx_] = desc.dtypes[idx];
            send_recv_info_.t_rank[hccl_batch_idx_] = desc.t_rank[idx];
            hccl_batch_idx_++;
        }
    }
}

void Distribution::hccl_batch_send() {
    for (int i = 0; i < hccl_batch_idx_; i++) {
        if (send_recv_info_.sendRecvType[i] == SwapDirection::SEND) {
            HCCLCHECK(HcclSend(send_recv_info_.address[i],
                               send_recv_info_.lengths[i],
                               NAME2DATATYPE.at(send_recv_info_.dtypes[i]),
                               send_recv_info_.t_rank[i], hcclComm_, stream_));
        } else if (send_recv_info_.sendRecvType[i] == SwapDirection::RECV) {
            HCCLCHECK(HcclRecv(send_recv_info_.recv_buffs[i],
                               send_recv_info_.lengths[i],
                               NAME2DATATYPE.at(send_recv_info_.dtypes[i]),
                               send_recv_info_.t_rank[i], hcclComm_, stream_));
        } else if (send_recv_info_.sendRecvType[i] == SwapDirection::LOCAL) {
            ACLCHECK(aclrtMemcpyAsync(
                send_recv_info_.recv_buffs[i], send_recv_info_.sizes[i],
                send_recv_info_.address[i], send_recv_info_.sizes[i],
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));
        }
        ACLCHECK(aclrtSynchronizeStream(stream_));
    }
}

void *Distribution::get_recv_buff_address(bool need_enqueue, size_t size) {
    if (recv_buff_ == nullptr) {
        throw std::runtime_error(
            "Pls initilization recv_buff_ by allocate_recv_buffs");
    }

    if (!need_enqueue)
        return nullptr;
    void *ptr = static_cast<void *>(static_cast<uint8_t *>(recv_buff_) +
                                    recv_buff_cur_);
    recv_buff_cur_ += size;
    return ptr;
}

bool Distribution::sync_round_shakehand(int cur_round, int curBatch) {
    std::vector<int> hostCurrentStatus = {cur_round, curBatch};
    device_cur_round_status_.to_device(hostCurrentStatus.data());
    allgather(device_cur_round_status_.get_data_ptr(),
              device_round_status_.get_data_ptr(), round_info_length_,
              "int"); // 广播当前的目标队列
    device_round_status_.to_host(host_round_status_.data());

    for (int i = 0; i < world_size_; i++) {
        if (host_round_status_[i * round_info_length_] != cur_round) {
            return false;
        }
        if (host_round_status_[i * round_info_length_ + 1] != curBatch) {
            return false;
        }
    }
    return true;
}

void Distribution::copy_from_queue_to_hbm() {
    for (size_t i = 0; i < queue_item_num_; i++) {
        TransDesc *desc = completed_sync_queue_[i];
        for (size_t idx = 0; idx < desc->address.size(); ++idx) {
            ACLCHECK(aclrtMemcpyAsync(desc->address[idx], desc->sizes[idx],
                                      desc->recv_buffs[idx], desc->sizes[idx],
                                      ACL_MEMCPY_DEVICE_TO_DEVICE,
                                      memcopy_stream_));
        }
    }
    ACLCHECK(aclrtSynchronizeStream(memcopy_stream_));
    queue_item_num_ = 0;
}

size_t Distribution::get_recv_buff_maxsize() {
    return expert_size_ * queue_size_;
}

void Distribution::allgather(void *src_addr, void *recv_addr, size_t length,
                             std::string dtype) {

    assert(stream_ != nullptr && "stream_ should not be nullptr");
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