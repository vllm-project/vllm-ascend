// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "distribution.h"
#include "tensor.h"
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

#ifndef HCCL_ROOT_INFO_BYTES
#define HCCL_ROOT_INFO_BYTES 512
#endif

//============================================================================
// Mock Layer
//============================================================================

struct MockLogger {
    int aclrtMalloc_calls = 0;
    int aclrtFree_calls = 0;
    int aclrtCreateStream_calls = 0;
    int aclrtDestroyStream_calls = 0;
    int aclrtMemcpyAsync_calls = 0;
    int aclrtSynchronizeStream_calls = 0;
    int HcclCommInitRootInfoConfig_calls = 0;
    int HcclGetRankId_calls = 0;
    int HcclGetRankSize_calls = 0;
    int HcclCommDestroy_calls = 0;
    int HcclSend_calls = 0;
    int HcclRecv_calls = 0;
    int HcclAllGather_calls = 0;
    void reset() { *this = {}; }
};

static MockLogger mock_logger;
uint32_t g_mock_rank = 0;
uint32_t g_mock_world_size = 4;

extern "C" {

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy) {
    mock_logger.aclrtMalloc_calls++;
    *devPtr = malloc(size);
    return *devPtr ? ACL_SUCCESS : ACL_ERROR_INVALID_PARAM;
}

aclError aclrtFree(void *devPtr) {
    mock_logger.aclrtFree_calls++;
    free(devPtr);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream) {
    mock_logger.aclrtCreateStream_calls++;
    *stream = reinterpret_cast<aclrtStream>(new int(1));
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream) {
    mock_logger.aclrtDestroyStream_calls++;
    if (stream)
        delete reinterpret_cast<int *>(stream);
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                          size_t count, aclrtMemcpyKind, aclrtStream) {
    mock_logger.aclrtMemcpyAsync_calls++;
    if (destMax >= count)
        memcpy(dst, src, count);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream) {
    mock_logger.aclrtSynchronizeStream_calls++;
    return ACL_SUCCESS;
}

HcclResult HcclCommInitRootInfoConfig(uint32_t, const HcclRootInfo *, uint32_t,
                                      const HcclCommConfig *, HcclComm *comm) {
    mock_logger.HcclCommInitRootInfoConfig_calls++;
    *comm = reinterpret_cast<HcclComm>(new int(1));
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm, uint32_t *rank) {
    mock_logger.HcclGetRankId_calls++;
    *rank = g_mock_rank;
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm, uint32_t *size) {
    mock_logger.HcclGetRankSize_calls++;
    *size = g_mock_world_size;
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm) {
    mock_logger.HcclCommDestroy_calls++;
    if (comm)
        delete reinterpret_cast<int *>(comm);
    return HCCL_SUCCESS;
}

HcclResult HcclSend(void *sendBuf, uint64_t, HcclDataType, uint32_t, HcclComm,
                    aclrtStream) {
    mock_logger.HcclSend_calls++;
    (void)sendBuf;
    return HCCL_SUCCESS;
}

HcclResult HcclRecv(void *recvBuf, uint64_t, HcclDataType, uint32_t, HcclComm,
                    aclrtStream) {
    mock_logger.HcclRecv_calls++;
    (void)recvBuf;
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t count,
                         HcclDataType, HcclComm, aclrtStream) {
    mock_logger.HcclAllGather_calls++;
    size_t type_size = sizeof(int);
    for (uint32_t i = 0; i < g_mock_world_size; ++i) {
        memcpy(static_cast<char *>(recvBuf) + i * count * type_size, sendBuf,
               count * type_size);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfo(const char *, uint32_t, HcclComm *comm) {
    *comm = reinterpret_cast<HcclComm>(new int(1));
    return HCCL_SUCCESS;
}
}

//============================================================================
// GTest Fixture
//============================================================================

class DistributionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        set_ut_memcpy_fun();
        mock_logger.reset();

        rank_ = 0;
        world_size_ = 4;
        g_mock_rank = rank_;
        g_mock_world_size = world_size_;

        num_experts_per_rank_ = 8;

        memset(info_str_buffer_, 0, sizeof(info_str_buffer_));

        dist_ = new Distribution(num_experts_per_rank_, rank_, world_size_,
                                 info_str_buffer_,
                                 HcclCommInitType::RootInfoString);

        aclrtStream stream;
        aclrtCreateStream(&stream);
        dist_->set_stream(stream);

        mock_logger.reset();
    }

    void TearDown() override {
        delete dist_;
        unset_ut_memcpy_fun();
    }

    Distribution *dist_;
    size_t rank_;
    size_t world_size_;
    size_t num_experts_per_rank_;
    char info_str_buffer_[HCCL_ROOT_INFO_BYTES];
};

TEST_F(DistributionTest, ConstructorInitializesCorrectly) {
    mock_logger.reset();
    Distribution local_dist(num_experts_per_rank_, rank_, world_size_,
                            info_str_buffer_, HcclCommInitType::RootInfoString);
    aclrtStream temp_stream;
    aclrtCreateStream(&temp_stream);
    local_dist.set_stream(temp_stream);

    EXPECT_EQ(mock_logger.HcclCommInitRootInfoConfig_calls, 1);
    EXPECT_EQ(mock_logger.HcclGetRankId_calls, 1);
    EXPECT_EQ(mock_logger.HcclGetRankSize_calls, 1);
    EXPECT_EQ(mock_logger.aclrtMalloc_calls, 4);
    EXPECT_EQ(mock_logger.aclrtCreateStream_calls, 3);
}

TEST_F(DistributionTest, BufferManagement) {
    const size_t expert_size = 1024;
    dist_->allocate_recv_buffs(expert_size);
    EXPECT_EQ(mock_logger.aclrtMalloc_calls, 1);

    void *addr1 = dist_->get_recv_buff_address(true, 100);
    ASSERT_NE(addr1, nullptr);
    EXPECT_EQ(dist_->get_buff_cur(), 100);

    dist_->reset_buff_cur();
    EXPECT_EQ(dist_->get_buff_cur(), 0);

    dist_->release_recv_buffs();
    EXPECT_EQ(mock_logger.aclrtFree_calls, 1);
}

TEST_F(DistributionTest, HcclBatchSend) {
    dist_->init_hccl_buffs(1);

    // Create sufficiently large buffers to prevent stack smashing during local
    // copy.
    const size_t buffer_size = 512;
    char source_buffer[buffer_size];
    char dest_buffer[buffer_size];

    // Initialize buffers to avoid using uninitialized data in memcpy.
    memset(source_buffer, 'S', buffer_size);
    memset(dest_buffer, 'D', buffer_size);

    TransDesc send_desc;
    send_desc.address = {source_buffer};
    send_desc.recv_buffs = {nullptr};
    send_desc.lengths = {128};
    send_desc.sizes = {0};
    send_desc.dtypes = {"int"};
    send_desc.t_rank = {1};

    TransDesc recv_desc;
    recv_desc.address = {nullptr};
    recv_desc.recv_buffs = {dest_buffer};
    recv_desc.lengths = {128};
    recv_desc.sizes = {0};
    recv_desc.dtypes = {"float"};
    recv_desc.t_rank = {2};

    TransDesc local_desc;
    local_desc.address = {source_buffer};
    local_desc.recv_buffs = {dest_buffer};
    local_desc.lengths = {buffer_size};
    local_desc.sizes = {buffer_size};
    local_desc.dtypes = {"char"};
    local_desc.t_rank = {0};

    dist_->add_to_batch(send_desc, SwapDirection::SEND);
    dist_->add_to_batch(recv_desc, SwapDirection::RECV);
    dist_->add_to_batch(local_desc, SwapDirection::LOCAL);

    dist_->hccl_batch_send();

    EXPECT_EQ(mock_logger.HcclSend_calls, 1);
    EXPECT_EQ(mock_logger.HcclRecv_calls, 1);
    EXPECT_EQ(mock_logger.aclrtMemcpyAsync_calls, 1);
    EXPECT_EQ(mock_logger.aclrtSynchronizeStream_calls, 3);
}

TEST_F(DistributionTest, SyncRoundShakehand) {
    bool result_true = dist_->sync_round_shakehand(0, 0);
    EXPECT_TRUE(result_true);
    EXPECT_EQ(mock_logger.HcclAllGather_calls, 1);
}