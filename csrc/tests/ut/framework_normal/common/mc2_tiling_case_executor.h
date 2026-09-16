/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "mc2_hcom_topology_mocker.h"
#include "kernel_tiling/kernel_tiling.h"

constexpr uint64_t MC2_TILING_DATA_RESERVED_LEN = (sizeof(Mc2InitTiling) + sizeof(Mc2CcTiling)) / sizeof(uint64_t);

inline void Mc2ExecuteTestCase(const gert::TilingContextPara &tilingContextPara,
                               const Mc2Hcom::MockValues &hcomTopologyMockValues,
                               ge::graphStatus expectResult = ge::GRAPH_FAILED, uint64_t expectTilingKey = 0,
                               const std::string &expectTilingData = "",
                               const std::vector<size_t> &expectWorkspaces = {}, uint64_t tilingDataReservedLen = 0,
                               bool useHashTilingData = false)
{
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, expectTilingData, expectWorkspaces,
                    tilingDataReservedLen, useHashTilingData);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H
