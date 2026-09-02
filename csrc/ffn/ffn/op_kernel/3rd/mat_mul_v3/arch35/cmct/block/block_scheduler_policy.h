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
 * \file block_scheduler_policy.h
 * \brief
 */

#pragma once

namespace Cmct {
namespace Gemm {

// FULL_LOAD_MODE_表示全载模式 0: 非全载 2：B全载
template <uint64_t FULL_LOAD_MODE_ = 0>
struct BuiltInAswtScheduler {
    static constexpr uint64_t FULL_LOAD_MODE = FULL_LOAD_MODE_;
};
struct BuiltInStreamKScheduler {};
struct BuiltInIterBatchScheduler {};
struct BuiltInMergeBatchScheduler {};
struct BuiltInBatchMatmulToMulScheduler {};
struct IterateKScheduler {};
struct QuantIterateKScheduler {};
template <uint64_t fullLoadMode = 0>
struct QuantBatchMatmulV3Scheduler {
    static constexpr uint64_t FULL_LOAD_MODE = fullLoadMode;
};
struct BuiltInFlatQuantScheduler {};
struct BuiltInRotateQuantScheduler {};
struct BuiltInMulScheduler {};
struct BuiltInVectorScheduler {};
} // namespace Gemm
} // namespace Cmct
