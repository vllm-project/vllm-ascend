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
 * \file mc2_moe_context.h
 * \brief
 */
#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H
#include <cstdint>

namespace Mc2Aclnn {
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};
}


#endif // MC2_MOE_CONTEXT_H