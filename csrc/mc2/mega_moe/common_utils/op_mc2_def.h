/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_mc2_def.h
 * \brief
 */
#ifndef OP_MC2_DEF_H_
#define OP_MC2_DEF_H_

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "hccl/hccl_types.h"
#include "opdev/common_types.h"

namespace op {
using s64 = signed long long;
using u32 = unsigned int;
using u64 = unsigned long long;
const int MAX_RANK_NUM = 8;
const int NUM_ONE = 1;
const int NUM_TWO = 2;
const int NUM_THREE = 3;
const int NUM_FOUR = 4;
const int NUM_FIVE = 5;
const int NUM_SIX = 6;
const int NUM_SEVEN = 7;
const int NUM_EIGHT = 8;
const int NUM_TEN = 10;
const int COMMUNICATION_DATA_SIZE = 10;  // 10*1024以上two shot
static constexpr char REDUCE_OP_SUM[] = "sum";
static constexpr size_t BIAS_SUPPORTED_DIMENSIONAL = 2U;
static constexpr size_t SUPPORTED_DIMENSIONAL = 3U;
static constexpr int64_t EXPERT_LOWER_LIMIT = 2L;
static constexpr int64_t EXPERT_UPPER_LIMIT = 512L;
static constexpr int64_t MATMUL_E_OVER_EP_LIMIT = 32L;
static constexpr int64_t MATMUL_K_LIMIT = 65535L;
static constexpr size_t HCCL_GROUP_NAME_MAX = 128U;

constexpr uint64_t WORLD_SIZE_PAIR = 2U;
constexpr uint64_t WORLD_SIZE_QUAD = 4U;
constexpr uint64_t WORLD_SIZE_OCTET = 8U;
constexpr uint64_t WORLD_SIZE_SEXTET = 16U;
constexpr uint64_t WORLD_SIZE_THIRTYTWO = 32U;

enum HcclOperatorType {
    HCCL_OP_TYPE_INVALID = 0,
    HCCL_OP_TYPE_BROADCAST = 1,
    HCCL_OP_TYPE_ALLREDUCE,
    HCCL_OP_TYPE_REDUCE,
    HCCL_OP_TYPE_SEND,
    HCCL_OP_TYPE_RECEIVE,
    HCCL_OP_TYPE_ALLGATHER,
    HCCL_OP_TYPE_REDUCE_SCATTER,
    HCCL_OP_TYPE_ALLTOALLV,
    HCCL_OP_TYPE_ALLTOALLVC,
    HCCL_OP_TYPE_GATHER,
    HCCL_OP_TYPE_MAX
};

struct HcomParam {
    void *context;
    void *aicpuStream;
    void *aicpuNotify;
};

typedef struct tagAicpuRpcServerModeLaunchOpAPI {
    HcclOperatorType comm_type;
    HcclReduceOp op_type;
    uint32_t rank_id;
    uint32_t rank_num;
    uint64_t windows[MAX_RANK_NUM];
    uint64_t no_ipc_notify_ids[MAX_RANK_NUM * 2];
    uint64_t ipc_notify_ids[MAX_RANK_NUM * 4];
    uint64_t stream_ids[MAX_RANK_NUM];
    uint64_t sq_ids[MAX_RANK_NUM];
    uint64_t workSpaceAddr;
    char kernel_name[32];
    char so_name[32];
} AicpuRpcServerModeLaunchOpAPI;

typedef struct tagHcclCombinOpParam {
    u32 rank_id{0};
    u32 rank_num{0};
    u64 bufferSize{0};
    u64 windows[8] = {0};
    u64 no_ipc_notify_ids[16] = {0};
    u64 ipc_notify_ids[32] = {0};
    s64 stream_ids[8] = {0};
    u64 sq_ids[8] = {0};
} HcclCombinOpParam;

typedef struct tagHcclAndAicpuResource {
    HcclCombinOpParam hccl_param = {0};
    void *aicpu_win_addr = nullptr;
} HcclAndAicpuResource;

// all_to_all allgather/reducescatter bmm算子X type支持fp16和bf16
static const std::initializer_list<op::DataType> MOE_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// all_to_all allgather/reducescatter bmm算子bias type支持fp16和fp32
static const std::initializer_list<op::DataType> MOE_BIAS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

}  // namespace op
#endif  // OP_MC2_DEF_H_