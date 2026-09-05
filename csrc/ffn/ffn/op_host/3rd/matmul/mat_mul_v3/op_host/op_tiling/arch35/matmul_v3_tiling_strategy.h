/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_tiling_strategy.h
 * \brief
 */
#pragma once

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace matmul_v3_advanced {
namespace strategy {
constexpr int32_t MATMUL_INPUT_K_EQUAL_ZERO = 0;
constexpr int32_t TO_MUL = 1;
constexpr int32_t TO_MULTI_MUL = 2;
constexpr int32_t BASIC_STREAM_K = 3;
constexpr int32_t BASIC_ASWT = 4;
constexpr int32_t BASE = 999;

const static std::map<NpuArch, std::vector<int32_t>> MatMulV3PrioritiesMap = {
    {NpuArch::DAV_3510,
     {strategy::MATMUL_INPUT_K_EQUAL_ZERO, strategy::TO_MUL, strategy::TO_MULTI_MUL, strategy::BASIC_STREAM_K,
      strategy::BASIC_ASWT}},
    {NpuArch::DAV_RESV, {strategy::BASIC_ASWT}}, // supportMmadS8S4平台
};

inline std::vector<int32_t> GetMatMulV3Priorities(NpuArch npuArch)
{
    std::vector<int32_t> priorities = {};
    if (MatMulV3PrioritiesMap.find(npuArch) != MatMulV3PrioritiesMap.end()) {
        priorities = MatMulV3PrioritiesMap.at(npuArch);
    }
    return priorities;
};
} // namespace strategy
} // namespace matmul_v3_advanced
} // namespace optiling
