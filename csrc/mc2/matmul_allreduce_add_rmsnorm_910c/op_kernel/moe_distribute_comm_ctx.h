/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_comm_ctx.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMM_CTX_H
#define MOE_DISTRIBUTE_COMM_CTX_H

constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;

// A5 HCCL Context
struct HcclCombinOpParam {
    uint64_t workSpace; // client和server之间通信的地址
    uint64_t workSpaceSize; // client和server之间通信的空间大小
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 总卡数
    uint64_t winSize; // ccu不使用
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用, MTE 数据区
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // ccu不使用，MTE 状态区

    // for ccu
    uint64_t xnAddr; // Xn寄存器起始地址
    uint64_t ckeAddr; // CKE寄存器起始地址
    uint64_t msAddr; // MS地址，预留
    uint64_t msSize; // 可写的MS个数，预留
};

#endif
