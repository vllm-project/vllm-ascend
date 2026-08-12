/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_tiling_tool.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H

#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_log.h"

using AscendC::BLOCK_CUBE;
using AscendC::ONE_BLK_SIZE;
using matmul_tiling::MatrixTraverse;

namespace optiling {

constexpr uint64_t BASIC_BLOCK = 512UL;

template <typename T1, typename T2>
T2 Mc2CalcTailSize(T1 num1, T2 num2)
{
    if (num2 == 0) {
        return 0;
    }

    T1 mod = num1 % static_cast<T1>(num2);
    return mod != 0 ? static_cast<T2>(mod) : num2;
}

int64_t Mc2GetDtypeBits(ge::DataType dtype);

uint64_t Mc2GetBlockAlignSizeByDataType(ge::DataType dtype);

uint64_t Mc2GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype);

bool Mc2CheckOptionalInputByShape(const gert::StorageShape *storageShape);

matmul_tiling::DataType Mc2GetMatmulTilingDtype(ge::DataType dtype);

ge::Format Mc2GetInputStorageFormat(const gert::TilingContext *context, size_t id);
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H
