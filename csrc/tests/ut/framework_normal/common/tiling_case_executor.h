/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H

#include "tiling_context_faker.h"

using namespace std;

struct TilingInfo {
    int64_t tilingKey = -1;
    std::vector<int64_t> workspaceSizes;
    std::unique_ptr<uint8_t[]> tilingData;
    size_t tilingDataSize = 0;
    size_t blockNum = 0;
};

void ExecuteTestCase(const gert::TilingContextPara &tilingContextPara, ge::graphStatus expectResult = ge::GRAPH_FAILED,
                     uint64_t expectTilingKey = 0, const string &expectTilingData = "",
                     const std::vector<size_t> &expectWorkspaces = {}, uint64_t tilingDataReservedLen = 0,
                     bool useHashTilingData = false);

bool ExecuteTiling(const gert::TilingContextPara &tilingContextPara, TilingInfo &tilingInfo);

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
