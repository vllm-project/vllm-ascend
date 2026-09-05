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
 * \file matmul_tiling_cfg.h
 * \brief
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "matmul_v3_tiling_key.h"

namespace optiling {
using namespace optiling::matmul_v3_advanced;
struct TilingResult {
    uint64_t tilingKey;
    uint64_t numBlocks;
    std::shared_ptr<void> tilingData;
    size_t tilingDataSize;
    std::vector<size_t> workspaceSize;
};

class MatMulTilingCfg {
public:
    MatMulTilingCfg(bool needUpdateIn, const void* compileInfoIn, const void* argsIn,
                    MatMulV3TilingKey* tilingKeyObjIn = nullptr)
        : needUpdate(needUpdateIn), compileInfo(compileInfoIn), args(argsIn), tilingKeyObj(tilingKeyObjIn)
    {}

    virtual ~MatMulTilingCfg(){};

    virtual ge::graphStatus Update(const TilingResult& /*result*/) { return ge::GRAPH_SUCCESS; };

public:
    const bool needUpdate = false;     // true: Tiling结果通过Update返回； false: Tiling结果填充到context中
    const void* compileInfo = nullptr; // 编译信息，用于生成TilingKey
    const void* args = nullptr;        // 算子参数，用于生成TilingKey
    MatMulV3TilingKey* tilingKeyObj = nullptr; // tilingkey函数指针， 用于:生成TilingKey
};
} // namespace optiling
