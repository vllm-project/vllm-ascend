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
 * \file batch_matmul_v3_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MATMUL_V3_TILING_STRATEGY_H__
#define __OP_HOST_BATCH_MATMUL_V3_TILING_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
namespace strategy {
constexpr int32_t ITER_BATCH_BASICAPI = 0;
constexpr int32_t ITER_BATCH = 1;
constexpr int32_t AL1_FULL_LOAD = 2;
constexpr int32_t BL1_FULL_LOAD_BASIC = 3;
constexpr int32_t BL1_FULL_LOAD = 4;
constexpr int32_t ASW_BASIC = 5;
constexpr int32_t BASE = 999;

const static std::map<NpuArch, std::vector<int32_t>> BatchMatMulV3PrioritiesMap = {
    {NpuArch::DAV_3510,
     {strategy::ITER_BATCH_BASICAPI, strategy::ITER_BATCH, strategy::AL1_FULL_LOAD, strategy::BL1_FULL_LOAD_BASIC,
      strategy::BL1_FULL_LOAD, strategy::ASW_BASIC, strategy::BASE}},
    {NpuArch::DAV_RESV, {strategy::ITER_BATCH, strategy::BASE}}, // supportMmadS8S4平台
};

inline std::vector<int32_t> GetBatchMatMulV3Priorities(NpuArch npuArch)
{
    std::vector<int32_t> priorities = {};
    if (BatchMatMulV3PrioritiesMap.find(npuArch) != BatchMatMulV3PrioritiesMap.end()) {
        priorities = BatchMatMulV3PrioritiesMap.at(npuArch);
    }
    return priorities;
};
} // namespace strategy
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling

#endif // __OP_HOST_BATCH_MATMUL_V3_STRATEGY_H__
