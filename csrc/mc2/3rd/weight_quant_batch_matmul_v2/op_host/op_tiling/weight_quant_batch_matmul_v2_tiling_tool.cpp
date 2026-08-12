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
 * \file weight_quant_batch_matmul_v2_tiling_tool.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_tiling_tool.h"

namespace optiling {

constexpr int64_t B32_BITS = 32;
constexpr int64_t B16_BITS = 16;
constexpr int64_t B8_BITS = 8;
constexpr int64_t B4_BITS = 4;

uint64_t Mc2GetBlockAlignSizeByDataType(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return ONE_BLK_SIZE + ONE_BLK_SIZE;
    } else {
        return ONE_BLK_SIZE / static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t Mc2GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4) {
        return (shapeSize + 1) >> 1;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

bool Mc2CheckOptionalInputByShape(const gert::StorageShape *storageShape)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetShapeSize() != 0;
}

int64_t Mc2GetDtypeBits(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return B4_BITS;
    } else if (dtype == ge::DT_INT8 || dtype == ge::DT_HIFLOAT8 || dtype == ge::DT_FLOAT8_E5M2 ||
               dtype == ge::DT_FLOAT8_E4M3FN) {
        return B8_BITS;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return B16_BITS;
    } else if (dtype == ge::DT_FLOAT) {
        return B32_BITS;
    } else {
        return 0;
    }
}

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT8_E8M0, matmul_tiling::DataType::DT_FLOAT8_E8M0},
    {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
    {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_FLOAT4_E2M1},
    {ge::DT_FLOAT4_E1M2, matmul_tiling::DataType::DT_FLOAT4_E1M2},
};

matmul_tiling::DataType Mc2GetMatmulTilingDtype(ge::DataType dtype)
{
    auto it = DTYPE_MAP.find(dtype);
    // impossible to get runtime error
    return it != DTYPE_MAP.end() ? it->second : matmul_tiling::DataType::DT_FLOAT16;
}

ge::Format Mc2GetInputStorageFormat(const gert::TilingContext *context, size_t id)
{
    auto desc = context->GetInputDesc(id);
    OP_TILING_CHECK(
        desc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT("weight_quant_batch_matmul_v2_tiling", ("input" + std::to_string(id)).c_str()),
        return ge::FORMAT_NULL);
    return static_cast<ge::Format>(GetPrimaryFormat(desc->GetStorageFormat()));
}
} // namespace optiling
