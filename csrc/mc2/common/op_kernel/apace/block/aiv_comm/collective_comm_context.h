/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_context.h
 * \brief Hcomm 通信上下文结构体，Host 侧构造后通过 GM 下发到 kernel
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace Apace {
namespace AivComm {

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;
static constexpr uint32_t COMM_WORKSPACE_SIZE = 512U;

struct CommUdmaContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t channelHandles[COMM_MAX_RANK_NUM];
    uint64_t commBufferAddrs[COMM_MAX_RANK_NUM];
};

struct CommUbmemContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBufferAddrs[COMM_MAX_RANK_NUM];
};

} // namespace AivComm
} // namespace Apace
