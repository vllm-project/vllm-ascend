/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#ifndef __OP_HOST_MATMUL_V3_STRATEGY_H__
#define __OP_HOST_MATMUL_V3_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
namespace strategy {
constexpr int32_t BASIC_STREAM_K = 0;
constexpr int32_t STREAM_K = 1;
constexpr int32_t BASIC_ASWT = 2;
constexpr int32_t FULL_LOAD_BASE = 3;
constexpr int32_t BASE = 999;

const static std::map<NpuArch, std::vector<int32_t>> MatMulV3PrioritiesMap = {
    {NpuArch::DAV_3510, {strategy::BASIC_STREAM_K, strategy::STREAM_K, strategy::BASIC_ASWT, strategy::FULL_LOAD_BASE}},
    {NpuArch::DAV_RESV, {strategy::BASE}}, // supportMmadS8S4平台
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
} // namespace mc2_matmul_v3_advanced
} // namespace optiling

#endif // __OP_HOST_MATMUL_V3_STRATEGY_H__
