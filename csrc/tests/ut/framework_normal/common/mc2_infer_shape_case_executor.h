/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_INFER_SHAPE_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_INFER_SHAPE_CASE_EXECUTOR_H

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "mc2_hcom_topology_mocker.h"

inline void Mc2ExecuteTestCase(gert::InfershapeContextPara &infershapeContextPara,
                               const Mc2Hcom::MockValues &hcomTopologyMockValues,
                               ge::graphStatus expectResult = ge::GRAPH_FAILED,
                               const std::vector<std::vector<int64_t>> &expectOutputShape = {})
{
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    ExecuteTestCase(infershapeContextPara, expectResult, expectOutputShape);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_INFER_SHAPE_CASE_EXECUTOR_H
